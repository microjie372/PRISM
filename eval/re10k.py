# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate PRISM on RealEstate10K.

Monocular depth on the reference frame builds the warp prior over six
synthetic trajectories; the decoder then renders from the dataset's own camera
poses, and the rendered frames are scored against the ground truth.
"""

import argparse
import json
import random
from pathlib import Path

import lpips as lpips_lib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from prism.data import (
    compute_psnr,
    compute_scale_via_flow,
    compute_ssim,
    load_cameras_raw,
    load_images,
    run_moge_preloaded,
    weight_dtype,
)
from prism.pipeline import (
    load_encoder,
    load_latent_recon,
    run_encoder_all_views,
)
from src.models.utils.misc import load_and_merge_configs, seed_everything
from src.models.utils.model import encode_video, load_vae
from src.utils.visu import save_video


def eval_scene(
    scene_dir: Path,
    vae, encoder, transformer, moge_model,
    inf_config, main_config, lpips_fn,
    img_size, patch_size_out_factor,
    num_eval_frames: int,
    device,
    save_renders: bool = False,
    out_dir: Path = None,
    **kwargs,
):
    """
    Evaluate a single RE10K scene.

    - First frame = reference input
    - Evaluate first num_eval_frames frames after reference (default 14)
    - 6 synthetic trajectories for the encoder cross-view attention
    - RE10K GT cameras for rendering

    Returns dict with per-scene mean PSNR / SSIM / LPIPS.
    """
    img_h, img_w = img_size

    # ---- Load images & raw camera matrices ----
    frame_paths = load_images(scene_dir)
    raw_matrices, intr_norm = load_cameras_raw(scene_dir)
    assert len(frame_paths) == raw_matrices.shape[0], (
        f"{scene_dir.name}: {len(frame_paths)} frames but "
        f"{raw_matrices.shape[0]} cameras; the scene directory is inconsistent."
    )

    # Apply stride
    stride = kwargs.get("stride", 1)
    if stride > 1:
        # Keep frame 0 (ref) + every stride-th frame after
        stride_indices = [0] + list(range(stride, len(frame_paths), stride))
        frame_paths  = [frame_paths[i]  for i in stride_indices if i < len(frame_paths)]
        raw_matrices = raw_matrices[stride_indices[:len(frame_paths)]]

    # Need at least ref + num_eval_frames images
    N_available = min(len(frame_paths), raw_matrices.shape[0])
    N_eval = min(num_eval_frames, N_available - 1)  # frames after ref
    assert N_eval >= 1, f"Scene {scene_dir.name} has too few frames ({N_available})"

    # Only keep ref + eval frames (not all frames)
    N = N_eval + 1
    frame_paths  = frame_paths[:N]
    raw_matrices = raw_matrices[:N]

    num_model_views = 121

    # ---- MoGe depth + intrinsics on reference frame ----
    # Mirror eval/single_image.py: MoGe provides depth, intrinsics, and
    # initial_w2c (identity) - all in a self-consistent coordinate system.
    moge_image, moge_depth, moge_mask, moge_w2c, moge_intrinsics = run_moge_preloaded(
        str(frame_paths[0]), target_h=img_h, target_w=img_w,
        device=device, moge_model=moge_model,
    )
    ref_depth_hw = (moge_depth if isinstance(moge_depth, torch.Tensor)
                    else torch.tensor(moge_depth, dtype=torch.float32)).float().to(device)
    if ref_depth_hw.dim() == 2:   ref_depth_hw = ref_depth_hw.unsqueeze(0).unsqueeze(0)
    elif ref_depth_hw.dim() == 3: ref_depth_hw = ref_depth_hw.unsqueeze(0)

    # MoGe intrinsics - MUST be used for warp (consistent with depth)

    # ---- VAE encode reference image ----
    preprocess = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ref_pil    = Image.open(frame_paths[0]).convert("RGB")
    img_tensor = preprocess(ref_pil).unsqueeze(0).unsqueeze(0).to(device, dtype=weight_dtype)
    with torch.no_grad():
        ref_latent = encode_video(vae, img_tensor, main_config.vae_backbone)[:, 0]

    # ---- EXACT COPY of eval/single_image.py pipeline ----
    from prism.pipeline import MULTI_TRAJECTORIES
    from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
    initial_w2c        = moge_w2c[0, 0].to(device)
    initial_intrinsics = moge_intrinsics[0, 0].to(device)

    cameras_list = []
    for traj_name, traj_cfg in MULTI_TRAJECTORIES.items():
        cam_gen_kwargs = traj_cfg.get("camera_gen_kwargs", {})
        syn_w2cs, syn_K = generate_camera_trajectory(
            trajectory_type    = traj_name,
            initial_w2c        = initial_w2c,
            initial_intrinsics = initial_intrinsics,
            num_frames         = 121,
            movement_distance  = traj_cfg["movement_distance"],
            camera_rotation    = "center_facing",
            device             = device,
            **cam_gen_kwargs,
        )
        cameras_list.append((syn_w2cs, syn_K))

    coverage_out = {}
    pred_latents_all, batch_list = run_encoder_all_views(
        cameras_list, ref_latent, ref_depth_hw,
        vae, encoder, main_config, inf_config,
        img_size, patch_size_out_factor, device,
        coverage_out=coverage_out,
    )

    V_enc = pred_latents_all.shape[1]
    T_lat = pred_latents_all.shape[2]
    C_lat = pred_latents_all.shape[3]
    H_lat = pred_latents_all.shape[4]
    W_lat = pred_latents_all.shape[5]

    images_input_embed_full = pred_latents_all.reshape(1, V_enc * T_lat, C_lat, H_lat, W_lat)
    plucker_full = torch.cat([b["plucker_embedding"] for b in batch_list], dim=1)
    rays_os_full = torch.cat([b["rays_os"]           for b in batch_list], dim=1)
    rays_ds_full = torch.cat([b["rays_ds"]           for b in batch_list], dim=1)

    # ---- Build RE10K cam_view for rendering ----
    # The Gaussians live in MoGe's frame, where the reference camera is the
    # identity, so the dataset poses have to be expressed relative to frame 0.
    # The stored matrices are world-to-camera; invert to get c2w, whose
    # translations are then scaled to match the monocular depth.
    from prism.pipeline import build_camera_batch
    raw_dev = raw_matrices.to(device).float()
    c2ws_abs = torch.linalg.inv(raw_dev)
    c2ws_re10k = torch.linalg.inv(c2ws_abs[0:1]) @ c2ws_abs

    # Align depth to the dataset cameras using every frame after the reference.
    align_indices = list(range(1, N))
    align_tgt_paths = [frame_paths[i] for i in align_indices]
    align_c2ws = [c2ws_re10k[i] for i in align_indices]
    # MoGe depth at full resolution
    depth_full = ref_depth_hw.squeeze()
    while depth_full.dim() > 2:
        depth_full = depth_full[0]
    if depth_full.shape != (img_h, img_w):
        depth_full = F.interpolate(depth_full.unsqueeze(0).unsqueeze(0),
                                   size=(img_h, img_w), mode='bilinear',
                                   align_corners=False).squeeze()
    sf = compute_scale_via_flow(frame_paths[0], align_tgt_paths, depth_full,
                                initial_intrinsics, align_c2ws, img_h, img_w)
    c2ws_re10k[:, :3, 3] *= sf
    # Pad to 121
    if c2ws_re10k.shape[0] < num_model_views:
        pad_len = num_model_views - c2ws_re10k.shape[0]
        c2ws_re10k = torch.cat([c2ws_re10k, c2ws_re10k[-1:].expand(pad_len, -1, -1)], 0)
    # Convert back to w2c for build_camera_batch
    re10k_w2cs = torch.linalg.inv(c2ws_re10k).unsqueeze(0)  # (1, 121, 4, 4)
    re10k_K = initial_intrinsics.unsqueeze(0).unsqueeze(0).expand(1, num_model_views, -1, -1).clone()
    re10k_batch, _, _ = build_camera_batch(re10k_w2cs, re10k_K, img_size, device)

    b0 = batch_list[0]
    batch = {
        "images_input_embed":     images_input_embed_full,
        "plucker_embedding":      plucker_full,
        "rays_os":                rays_os_full,
        "rays_ds":                rays_ds_full,
        "num_input_multi_views":  V_enc,
        "cam_view":               re10k_batch["cam_view"],      # RE10K cameras
        "intrinsics":             re10k_batch["intrinsics"],    # MoGe intrinsics
        "time_embeddings":        b0["time_embeddings"],
        "time_embeddings_target": b0["time_embeddings_target"],
        "flip_flag":              b0["flip_flag"],
    }

    with torch.no_grad():
        model_output = transformer(batch)

    # pred_images: (1, T_render, 3, H, W) - LatentRecon 3DGS outputs [0, 1]
    pred_frames = model_output["images_pred"].float().cpu()
    pred_frames = pred_frames.clamp(0, 1)               # already [0, 1], just clamp
    pred_frames = pred_frames[0]                         # (T_render, 3, H, W)
    # T_render == 121 (all GT camera positions); frame 0 = reference view

    # ---- Evaluate frames 1..N_eval vs GT (first N_eval after the reference) ----
    n_eval = min(N_eval, pred_frames.shape[0] - 1)
    eval_frame_paths = frame_paths[1: 1 + n_eval]

    gt_preprocess = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ])

    psnrs, ssims, lpipss = [], [], []
    for i in range(n_eval):
        pred_t = pred_frames[i + 1]               # (3, H, W) [0, 1]
        gt_t   = gt_preprocess(
            Image.open(eval_frame_paths[i]).convert("RGB")
        )                                          # (3, H, W) [0, 1]

        if pred_t.shape[-2:] != gt_t.shape[-2:]:
            pred_t = F.interpolate(
                pred_t.unsqueeze(0), size=gt_t.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        pred_np = pred_t.permute(1, 2, 0).numpy()
        gt_np   = gt_t.permute(1, 2, 0).numpy()
        psnrs.append(compute_psnr(pred_np, gt_np))
        ssims.append(compute_ssim(pred_np, gt_np))

        pred_lpips = (pred_t * 2 - 1).unsqueeze(0).to(device)
        gt_lpips   = (gt_t   * 2 - 1).unsqueeze(0).to(device)
        with torch.no_grad():
            lpipss.append(lpips_fn(pred_lpips, gt_lpips).item())

    # ---- Optionally save renders ----
    if save_renders and out_dir is not None:
        scene_out = out_dir / scene_dir.name
        scene_out.mkdir(parents=True, exist_ok=True)

        pred_for_video = pred_frames[1: 1 + n_eval].clamp(0, 1).unsqueeze(0)
        save_video(pred_for_video, str(scene_out), name="pred", fps=10)

        gt_frames_list = [
            gt_preprocess(Image.open(p).convert("RGB"))
            for p in eval_frame_paths
        ]
        gt_for_video = torch.stack(gt_frames_list).clamp(0, 1).unsqueeze(0)
        save_video(gt_for_video, str(scene_out), name="gt", fps=10)

        if gt_for_video.shape[-2:] != pred_for_video.shape[-2:]:
            gt_for_video = F.interpolate(
                gt_for_video[0], size=pred_for_video.shape[-2:],
                mode="bilinear", align_corners=False,
            ).unsqueeze(0)
        save_video(
            torch.cat([pred_for_video, gt_for_video], dim=-1),
            str(scene_out), name="compare", fps=10,
        )

    del pred_latents_all, images_input_embed_full, plucker_full, rays_os_full, rays_ds_full
    torch.cuda.empty_cache()

    return {
        "psnr":     float(np.mean(psnrs)),
        "ssim":     float(np.mean(ssims)),
        "lpips":    float(np.mean(lpipss)),
        "n_frames": n_eval,
        "coverage": coverage_out.get("mask_coverage"),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",   type=str, default=None,
                   help="Evaluate a single scene directory")
    p.add_argument("--dataset_dir", type=str, default=None,
                   help="Root directory containing multiple scene subdirectories")
    p.add_argument("--num_scenes",  type=int, default=300)
    p.add_argument("--num_eval_frames", type=int, default=14,
                   help="Number of frames after the reference to evaluate")
    p.add_argument("--stride", type=int, default=1,
                   help="Frame sampling stride (default=1)")

    p.add_argument("--config",         type=str, required=True)
    p.add_argument("--config_default", type=str,
                   default="configs/inference/default.yaml")
    p.add_argument("--encoder_ckpt",   type=str, required=True)

    p.add_argument("--output_dir",   type=str, default="outputs/eval/re10k")
    p.add_argument("--save_renders", action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load configs ----
    inf_config  = load_and_merge_configs([args.config_default, args.config])
    main_config = load_and_merge_configs(list(inf_config.config_path))
    main_config.load_latents = False

    img_size              = tuple(main_config.get("img_size",              [704, 1280]))
    patch_size_out_factor = tuple(main_config.get("patch_size_out_factor", [1, 8, 8]))

    # ---- Load models ----
    print("Loading VAE...")
    vae = load_vae(main_config.vae_backbone, main_config.vae_path)
    vae.to(device, dtype=weight_dtype).eval()

    recon_ckpt = inf_config.get("recon_ckpt_path")
    print(f"Loading LatentRecon from {recon_ckpt}...")
    transformer = load_latent_recon(recon_ckpt, main_config, device)

    print(f"Loading residual encoder from {args.encoder_ckpt}...")
    encoder = load_encoder(args.encoder_ckpt, inf_config, device)

    print("Loading MoGe depth model...")
    from moge.model.v1 import MoGeModel
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()

    lpips_fn = lpips_lib.LPIPS(net="alex").to(device)
    lpips_fn.eval()

    # ---- Collect scenes ----
    if args.scene_dir:
        scene_dirs = [Path(args.scene_dir)]
    else:
        assert args.dataset_dir, "Provide --scene_dir or --dataset_dir"
        root = Path(args.dataset_dir)
        scene_dirs = sorted([d for d in root.iterdir() if d.is_dir()])
        if len(scene_dirs) > args.num_scenes:
            random.seed(args.seed)
            scene_dirs = random.sample(scene_dirs, args.num_scenes)
        print(f"Evaluating {len(scene_dirs)} scenes from {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume: load existing results if available ----
    all_psnr, all_ssim, all_lpips = [], [], []
    per_scene = {}
    results_path = out_dir / "results.json"
    if results_path.exists():
        with open(results_path, "r") as f:
            prev = json.load(f)
        if "per_scene" in prev:
            per_scene = prev["per_scene"]
            for name, res in per_scene.items():
                all_psnr.append(res["psnr"])
                all_ssim.append(res["ssim"])
                all_lpips.append(res["lpips"])
            print(f"Resumed {len(per_scene)} scenes from {results_path}")

    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        # Skip already evaluated scenes
        if scene_dir.name in per_scene:
            continue
        try:
            result = eval_scene(
                scene_dir=scene_dir,
                vae=vae, encoder=encoder, transformer=transformer,
                moge_model=moge_model,
                inf_config=inf_config, main_config=main_config,
                lpips_fn=lpips_fn,
                img_size=img_size,
                patch_size_out_factor=patch_size_out_factor,
                num_eval_frames=args.num_eval_frames,
                device=device,
                save_renders=args.save_renders,
                out_dir=out_dir / "renders" if args.save_renders else None,
                stride=args.stride,
            )
            all_psnr.append(result["psnr"])
            all_ssim.append(result["ssim"])
            all_lpips.append(result["lpips"])
            per_scene[scene_dir.name] = result
            tqdm.write(
                f"{scene_dir.name}: "
                f"PSNR={result['psnr']:.2f}  "
                f"SSIM={result['ssim']:.3f}  "
                f"LPIPS={result['lpips']:.3f}  "
                f"({result['n_frames']} frames)"
            )
            # Save after each scene (resume-safe)
            interim = {
                "summary": {"psnr": float(np.mean(all_psnr)), "ssim": float(np.mean(all_ssim)),
                            "lpips": float(np.mean(all_lpips)), "n_scenes": len(all_psnr)},
                "per_scene": per_scene,
            }
            with open(results_path, "w") as f:
                json.dump(interim, f, indent=2)
        except Exception as e:
            import traceback
            tqdm.write(f"SKIP {scene_dir.name}: {e}")
            traceback.print_exc()

    # ---- Aggregate ----
    summary = {
        "psnr":     float(np.mean(all_psnr)),
        "ssim":     float(np.mean(all_ssim)),
        "lpips":    float(np.mean(all_lpips)),
        "n_scenes": len(all_psnr),
    }
    print("\n=== Results ===")
    print(f"  PSNR  : {summary['psnr']:.4f}")
    print(f"  SSIM  : {summary['ssim']:.4f}")
    print(f"  LPIPS : {summary['lpips']:.4f}")
    print(f"  Scenes: {summary['n_scenes']}")
    print(f"  Eval frames/scene: {args.num_eval_frames}")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_scene": per_scene}, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
