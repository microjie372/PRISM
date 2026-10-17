# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Precompute the geometric warp prior for every training scene.

Forward-warps the reference latent to all target views using monocular depth
and the camera extrinsics, writing one file per scene holding warp_latents,
warp_mask and ref_latent. Training reads these instead of recomputing them.
"""

import argparse
import os
import torch
import torch.nn.functional as F
from tqdm import tqdm
from omegaconf import OmegaConf

from prism.warp import forward_warp_latent_zbuffer as forward_warp_latent
from src.models.data import get_multi_dataloader
from src.models.utils.misc import load_and_merge_configs, seed_everything
from src.models.utils.model import load_vae, encode_video


def load_moge(device):
    from moge.model.v1 import MoGeModel
    model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device)
    model.eval()
    print("MoGe loaded.")
    return model


@torch.no_grad()
def estimate_depth(moge_model, img_chw_01):
    """
    Run MoGe on a single (C, H, W) image in [0, 1].
    Returns depth (1, 1, H, W) and mask (1, 1, H, W).
    """
    out   = moge_model.infer(img_chw_01)
    depth = out["depth"].unsqueeze(0).unsqueeze(0)
    mask  = out["mask"].unsqueeze(0).unsqueeze(0).float()
    depth = torch.nan_to_num(depth, nan=1e4).clamp(0, 1e4)
    depth = torch.where(mask == 0, torch.full_like(depth, 1000.0), depth)
    return depth, mask


def build_K(intr, device):
    """Convert [fx,fy,cx,cy] -> (B, V, 3, 3). Handles (B,1,V,4) and (B,V,4)."""
    if intr.shape[-1] != 4:
        return intr
    if intr.dim() == 4:
        intr = intr[:, 0]
    B_i, V_i = intr.shape[:2]
    K = torch.zeros(B_i, V_i, 3, 3, device=device, dtype=torch.float32)
    K[:, :, 0, 0] = intr[:, :, 0]; K[:, :, 1, 1] = intr[:, :, 1]
    K[:, :, 0, 2] = intr[:, :, 2]; K[:, :, 1, 2] = intr[:, :, 3]
    K[:, :, 2, 2] = 1.0
    return K


def main(cfg):
    seed_everything(cfg.get("seed", 42))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- MoGe ----
    moge_model = load_moge(device)

    # ---- VAE (for single-frame ref_latent encoding, matching online inference) ----
    vae = load_vae(cfg.vae_backbone, cfg.vae_path)
    vae = vae.to(device=device, dtype=torch.bfloat16)
    vae.eval()

    # ---- Scan all scene IDs from the dataset root ----
    from src.models.data.registry import dataset_registry
    dataset_name = cfg.data_mode[0][0]
    root_path = cfg.get("dataset_root", None) or dataset_registry[dataset_name]['kwargs']['root_path']
    root_path = os.path.abspath(os.path.expanduser(root_path))
    dataset_registry[dataset_name]['kwargs']['root_path'] = root_path
    rgb_dir = os.path.join(root_path, "0", "rgb")
    if not os.path.isdir(rgb_dir):
        raise FileNotFoundError(
            f"No trajectory 0 under {root_path}. Set dataset_root to the "
            f"Lyra SDG root, which holds directories 0 through 5; "
            f"see docs/setup.md."
        )
    all_scene_ids = sorted({os.path.splitext(f)[0] for f in os.listdir(rgb_dir)
                            if f.endswith(".mp4")})
    total_scenes = len(all_scene_ids)

    C_lat        = cfg.get("prism_latent_channels", 16)
    T_lat        = cfg.get("prism_latent_T", 16)
    latent_scale = cfg.get("latent_spat_compression", 8)
    overwrite    = cfg.get("overwrite", False)
    use_splat    = cfg.get("use_bilinear_splat", True)
    out_dir      = os.path.abspath(cfg.get("warp_precompute_dir", "outputs/warp_precompute"))
    os.makedirs(out_dir, exist_ok=True)

    # Pre-scan existing warps, find missing
    existing = {os.path.splitext(f)[0] for f in os.listdir(out_dir) if f.endswith(".pt")}
    missing = [s for s in all_scene_ids if s not in existing] if not overwrite else all_scene_ids
    print(f"Total scenes: {total_scenes}, existing: {len(existing)}, to process: {len(missing)}")

    if not missing:
        print("All warps exist. Nothing to do.")
        return

    # Only create dataloader with missing scenes via filter list
    filter_path = os.path.join(out_dir, "_missing_filter.txt")
    with open(filter_path, "w") as f:
        f.write("\n".join(missing))

    # ---- data config ----
    data_cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    data_cfg.load_latents  = True
    data_cfg.use_depth     = False
    data_cfg.batch_size    = 1
    data_cfg.num_workers   = cfg.get("num_workers", 2)
    data_cfg.num_input_multi_views = cfg.get("num_input_multi_views", 6)
    data_cfg.sample_num_input_multi_views = False
    data_cfg.shuffle = False
    data_cfg.subsample_data_train_val = False
    data_cfg.num_test_scenes = 0
    ref_frame        = cfg.get("prism_fixed_ref_frame", 0)
    ref_frame_local  = ref_frame // data_cfg.get("target_index_subsample", 1)
    data_cfg.prism_fixed_ref_frame = ref_frame

    # Inject filter_list_path into registry (only loads missing scenes)
    from copy import deepcopy
    orig_kwargs = deepcopy(dataset_registry[cfg.data_mode[0][0]]['kwargs'])
    dataset_registry[cfg.data_mode[0][0]]['kwargs']['filter_list_path'] = filter_path

    train_loader, val_loader = get_multi_dataloader(data_cfg)
    loader = train_loader

    # Restore registry
    dataset_registry[cfg.data_mode[0][0]]['kwargs'] = orig_kwargs

    V             = data_cfg.num_input_multi_views          # 6
    n_frames      = data_cfg.num_input_views                # 121
    t_stride      = (n_frames - 1) // (T_lat - 1)          # 8

    print(f"V={V}, T_lat={T_lat}, n_frames={n_frames}, t_stride={t_stride}")

    processed = 0
    done_scenes = set()
    debug_printed = False

    for batch in tqdm(loader, desc="Precomputing warps"):
        scene_name = batch["file_name"][0] if isinstance(batch["file_name"], (list, tuple)) \
                     else batch["file_name"]

        if scene_name in done_scenes or scene_name in existing:
            continue

        out_path = os.path.join(out_dir, f"{scene_name}.pt")

        # ---- tensors to device ----
        rgb_latents      = batch["rgb_latents"].to(device, dtype=torch.float32)
        c2ws_input       = batch["c2ws_input"].to(device, dtype=torch.float32)
        intrinsics_input = batch["intrinsics_input"].to(device, dtype=torch.float32)

        def ensure_batch(t, ndim):
            return t.unsqueeze(0) if t.dim() == ndim - 1 else t

        rgb_latents      = ensure_batch(rgb_latents,      5)
        c2ws_input       = ensure_batch(c2ws_input,       4)
        intrinsics_input = ensure_batch(intrinsics_input, 4)

        intrinsics_input = build_K(intrinsics_input, device)   # (B, V*n_frames, 3, 3)

        B = rgb_latents.shape[0]
        assert B == 1

        src_c2w    = c2ws_input[:, ref_frame]                # (B, 4, 4)  view0, fixed ref frame

        # ---- ref image (fixed frame) ----
        ref_img_01 = batch["images_output"][:, ref_frame_local].float()[0].to(device)  # (C, H, W)

        # ---- ref latent: single-frame VAE encode ----
        with torch.no_grad():
            ref_img_bf16 = (batch["images_output"][:, ref_frame_local].float().to(device) * 2 - 1) \
                           .to(torch.bfloat16).unsqueeze(1)   # (B, 1, C, H, W) bfloat16
            ref_latent = encode_video(
                vae, ref_img_bf16, cfg.vae_backbone,
            )                                                  # (B, 1, C_lat, H_lat, W_lat)
        ref_latent = ref_latent[:, 0].float()                 # (B, C_lat, H_lat, W_lat) float32
        H_lat = ref_latent.shape[-2]
        W_lat = ref_latent.shape[-1]

        # Intrinsics at latent resolution (view 0, fixed ref frame)
        K_lat = intrinsics_input[:, ref_frame].clone()    # (B, 3, 3)
        K_lat[:, :2] = K_lat[:, :2] / latent_scale

        # ---- MoGe depth on reference image ----
        ref_depth_full, _ = estimate_depth(moge_model, ref_img_01)
        ref_depth_full = ref_depth_full.to(device)
        ref_depth_lat  = F.interpolate(ref_depth_full, size=(H_lat, W_lat),
                                       mode='bilinear', align_corners=False)  # (1, 1, H, W)

        if not debug_printed:
            print("\n[Debug @ first batch]")
            print(f"  c2ws_input shape:   {tuple(c2ws_input.shape)}")
            print(f"  images_output shape:{tuple(batch['images_output'].shape)}")
            print(f"  ref_latent (VAE single-frame): {tuple(ref_latent.shape)}  "
                  f"mean={ref_latent[0].mean():.4f} std={ref_latent[0].std():.4f}")
            print(f"  ref_depth_lat:      {tuple(ref_depth_lat.shape)}")
            debug_printed = True

        # ---- Warp ref_latent to all (V, T_lat) positions ----
        # Result assembled to match GT shape: (B, V*C_lat, T_lat, H_lat, W_lat)
        warp_latents = torch.zeros(B, V * C_lat, T_lat, H_lat, W_lat,
                                   device=device, dtype=torch.float32)
        warp_mask    = torch.zeros(B, V, T_lat, H_lat, W_lat,
                                   device=device, dtype=torch.float32)

        for v in range(V):
            # Stack T_lat target cameras for this view
            tgt_c2ws_v = torch.stack(
                [c2ws_input[:, v * n_frames + t * t_stride] for t in range(T_lat)],
                dim=1)  # (B, T_lat, 4, 4)

            warped_v, mask_v = forward_warp_latent(
                ref_latent,      # (B, C, H, W)
                ref_depth_lat,   # (B, 1, H, W)
                src_c2w,         # (B, 4, 4)
                tgt_c2ws_v,      # (B, T_lat, 4, 4)
                K_lat,           # (B, 3, 3)
                use_splat=use_splat,
            )
            # warped_v: (B, T_lat, C, H, W)
            # mask_v:   (B, T_lat, 1, H, W)

            # Store channels for view v: transpose T and C, then assign
            # warp_latents[:, v*C:(v+1)*C, t, :, :] = warped_v[:, t, :, :, :]
            warp_latents[:, v * C_lat:(v + 1) * C_lat, :, :, :] = \
                warped_v.permute(0, 2, 1, 3, 4)  # (B, C, T_lat, H, W)
            warp_mask[:, v, :, :, :] = mask_v[:, :, 0, :, :]  # (B, T_lat, H, W)

        # ---- Quality diagnostics (first 3 scenes) ----
        if processed < 3:
            gt_lat = rgb_latents[0]     # (V*C, T_lat, H, W)
            warp_l = warp_latents[0]    # (V*C, T_lat, H, W)

            # --- identity warp check: warp[v=0, t=0] vs ref_latent ---
            w_id = warp_l[0:C_lat, 0].reshape(-1)
            r_id = ref_latent[0].reshape(-1)
            cos_id = F.cosine_similarity(w_id.unsqueeze(0), r_id.unsqueeze(0)).item()
            tqdm.write(f"\n  [identity check] warp(v=0,t=0) vs ref_latent: cosine={cos_id:.4f}  "
                       f"(should be ~1.0 if v=0,t=0 camera = src_c2w)")

            # --- warp[v,t] vs GT[v,t] full matrix ---
            tqdm.write("  [warp vs GT cosine matrix] (rows=view, cols=T):")
            header = "         " + " ".join(f"T={t:2d}" for t in range(T_lat))
            tqdm.write(f"  {header}")
            for v in range(V):
                row = []
                for t in range(T_lat):
                    w = warp_l[v*C_lat:(v+1)*C_lat, t].reshape(-1)
                    g = gt_lat[v*C_lat:(v+1)*C_lat, t].reshape(-1)
                    cos = F.cosine_similarity(w.unsqueeze(0), g.unsqueeze(0)).item()
                    row.append(f"{cos:+.2f}")
                tqdm.write(f"  view{v}: {' '.join(row)}")

            mse_all = F.mse_loss(warp_l, gt_lat).item()
            psnr_all = -10 * torch.log10(torch.tensor(mse_all + 1e-8)).item()

            # Per spatial view, T=0 (pure spatial warp quality)
            cos_spatial = []
            for v in range(1, V):  # skip view 0 (identity warp)
                w = warp_l[v * C_lat:(v + 1) * C_lat, 0].reshape(-1)
                g = gt_lat[v * C_lat:(v + 1) * C_lat, 0].reshape(-1)
                cos_spatial.append(F.cosine_similarity(w.unsqueeze(0), g.unsqueeze(0)).item())

            # Per temporal frame, view 0 (pure temporal warp quality)
            cos_temporal = []
            for t in range(1, T_lat):  # skip t=0 (identity)
                w = warp_l[0:C_lat, t].reshape(-1)
                g = gt_lat[0:C_lat, t].reshape(-1)
                cos_temporal.append(F.cosine_similarity(w.unsqueeze(0), g.unsqueeze(0)).item())

            coverage = warp_mask[0].mean().item()
            tqdm.write(
                f"\n  [warp quality @ scene {processed}]"
                f"\n    coverage={coverage:.2%}"
                f"\n    MSE(all)={mse_all:.4f}  PSNR(all)={psnr_all:.1f}dB"
                f"\n    cosine_spatial(v1..5 T=0): {[f'{x:.3f}' for x in cos_spatial]}"
                f"\n    cosine_temporal(v0 t1..15): {[f'{x:.3f}' for x in cos_temporal]}"
            )

        torch.save({
            "warp_latents": warp_latents[0].half().cpu(),   # (V*C, T_lat, H, W)
            "warp_mask":    warp_mask[0].half().cpu(),       # (V, T_lat, H, W)
            "ref_latent":   ref_latent[0].half().cpu(),      # (C, H, W)
        }, out_path)
        processed += 1
        done_scenes.add(scene_name)
        total_done = len(existing) + processed
        tqdm.write(f"  Processed {scene_name} ({total_done}/{total_scenes} total)")

        # All missing scenes done -> exit early
        if processed >= len(missing):
            break

    # Cleanup temp filter file
    if os.path.isfile(filter_path):
        os.remove(filter_path)

    print(f"\nDone. Processed={processed} new scenes. Total warps: {len(existing) + processed}/{total_scenes}")
    print(f"Warp files saved to: {out_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--overwrite", action="store_true")
    args, unknown = parser.parse_known_args()

    cfg = OmegaConf.load(args.config)
    if "config_path" in cfg:
        cfg = OmegaConf.merge(load_and_merge_configs(list(cfg.config_path)), cfg)
    if unknown:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(unknown))
    cfg.overwrite = args.overwrite

    main(cfg)
