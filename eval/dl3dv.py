# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate PRISM on DL3DV.

Same pipeline as eval/re10k.py; DL3DV ships nerfstudio format, so cameras come
from transforms.json and frames from images_4/.
"""

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

import lpips as lpips_lib

from src.models.utils.model import load_vae, encode_video
from src.models.utils.misc import load_and_merge_configs, seed_everything
from src.utils.visu import save_video

from prism.pipeline import (
    load_latent_recon,
    load_encoder,
    run_encoder_all_views,
)

# Reuse MoGe, scale alignment, metric helpers from eval_re10k
from prism.data import (
    run_moge_preloaded,
    compute_psnr,
    compute_ssim,
    compute_scale_via_flow,
)

weight_dtype = torch.bfloat16


# ---------------------------------------------------------------------------
# DL3DV camera/image loading (nerfstudio transforms.json)
# ---------------------------------------------------------------------------

# DL3DV ships transforms.json in different places depending on the variant.
TRANSFORMS_SUBDIRS = ("nerfstudio", "colmap", ".")


def find_transforms_dir(scene_dir: Path):
    """Return the directory holding this scene's transforms.json, or None."""
    for sub in TRANSFORMS_SUBDIRS:
        d = scene_dir if sub == "." else scene_dir / sub
        if (d / "transforms.json").exists():
            return d
    return None


def load_dl3dv_scene(scene_dir: Path, use_960p: bool = True):
    """
    Load cameras and image paths from DL3DV nerfstudio format.

    Args:
        scene_dir: path to scene (contains nerfstudio/ subdirectory)
        use_960p: if True, use images_4/ (960P) instead of images/ (full res)

    Returns:
        c2ws:        (N, 4, 4) float32 - camera-to-world matrices
        K:           (3, 3) float32 - intrinsics matrix (at image resolution)
        frame_paths: list of N Path objects to images
        img_w, img_h: image resolution
    """
    nerfstudio_dir = find_transforms_dir(scene_dir)
    assert nerfstudio_dir is not None, (
        f"No transforms.json under {scene_dir} (looked in nerfstudio/, colmap/ "
        "and the scene root)."
    )
    transforms_path = nerfstudio_dir / "transforms.json"

    with open(transforms_path) as f:
        data = json.load(f)

    # Global intrinsics (at original resolution)
    orig_w, orig_h = data["w"], data["h"]
    fl_x, fl_y = data["fl_x"], data["fl_y"]
    cx, cy = data["cx"], data["cy"]

    # Determine image directory and scale factor. images_4 is nerfstudio's
    # quarter-resolution output; falling back to images/ changes which pixels
    # are scored, so say so rather than switching silently.
    img_dir, scale = nerfstudio_dir / "images", 1.0
    if use_960p:
        if (nerfstudio_dir / "images_4").exists():
            img_dir, scale = nerfstudio_dir / "images_4", 0.25
        else:
            tqdm.write(f"  [dl3dv] {scene_dir.name}: no images_4/, "
                       "scoring full-resolution images/ instead")
    assert img_dir.exists(), f"No image directory found in {nerfstudio_dir}"

    # Scale intrinsics to match image resolution
    fl_x_scaled = fl_x * scale
    fl_y_scaled = fl_y * scale
    cx_scaled = cx * scale
    cy_scaled = cy * scale
    img_w = int(orig_w * scale)
    img_h = int(orig_h * scale)

    K = torch.tensor([
        [fl_x_scaled, 0,           cx_scaled],
        [0,           fl_y_scaled, cy_scaled],
        [0,           0,           1        ],
    ], dtype=torch.float32)

    # Undo applied_transform: nerfstudio convention -> COLMAP convention
    at_raw = data.get("applied_transform", [[1,0,0,0],[0,1,0,0],[0,0,1,0]])
    at_4x4 = torch.eye(4, dtype=torch.float32)
    at_4x4[:3, :] = torch.tensor(at_raw, dtype=torch.float32)
    at_inv = torch.linalg.inv(at_4x4)

    # Load frames
    frames = data["frames"]
    c2w_list = []
    path_list = []

    # OpenGL -> OpenCV: flip Y and Z axes of camera coordinate system
    gl2cv = torch.diag(torch.tensor([1, -1, -1, 1], dtype=torch.float32))

    for frame in frames:
        # c2w in nerfstudio convention -> undo applied_transform -> flip GL->CV
        mat = torch.tensor(frame["transform_matrix"], dtype=torch.float32)  # (4, 4)
        mat = at_inv @ mat  # undo applied_transform (nerfstudio -> COLMAP world)
        mat = mat @ gl2cv   # flip camera axes (OpenGL -> OpenCV)
        c2w_list.append(mat)

        # Image path: transforms.json says "images/frame_xxxxx.png"
        # but we want images_4/ for 960P
        fname = Path(frame["file_path"]).name  # e.g. "frame_00001.png"
        img_path = img_dir / fname
        path_list.append(img_path)

    c2ws = torch.stack(c2w_list, dim=0)  # (N, 4, 4)

    return c2ws, K, path_list, img_w, img_h


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------

def eval_scene(
    scene_dir: Path,
    vae, encoder, transformer, moge_model,
    inf_config, main_config, lpips_fn,
    img_size, patch_size_out_factor,
    num_eval_frames: int,
    device,
    save_renders: bool = False,
    out_dir: Path = None,
    start_frame: int = 0,
):
    """Evaluate a single DL3DV scene."""
    img_h, img_w = img_size

    # ---- Load DL3DV scene ----
    c2ws_all, K_scene, frame_paths_all, scene_w, scene_h = load_dl3dv_scene(scene_dir)

    # Apply start_frame offset
    N_total = min(len(frame_paths_all), c2ws_all.shape[0])
    assert start_frame < N_total, \
        f"start_frame {start_frame} >= {N_total} available frames in {scene_dir.name}"
    frame_paths_all = frame_paths_all[start_frame:]
    c2ws_all = c2ws_all[start_frame:]

    N_available = min(len(frame_paths_all), c2ws_all.shape[0])
    N_eval = min(num_eval_frames, N_available - 1)
    assert N_eval >= 1, f"Scene {scene_dir.name} has too few frames from start_frame={start_frame} ({N_available})"

    N = N_eval + 1
    frame_paths = frame_paths_all[:N]
    c2ws = c2ws_all[:N]  # (N, 4, 4), already c2w

    # Verify first image exists
    if not frame_paths[0].exists():
        raise FileNotFoundError(f"Image not found: {frame_paths[0]}")

    num_model_views = 121

    # ---- MoGe depth + intrinsics on reference frame ----
    moge_image, moge_depth, moge_mask, moge_w2c, moge_intrinsics = run_moge_preloaded(
        str(frame_paths[0]), target_h=img_h, target_w=img_w,
        device=device, moge_model=moge_model,
    )
    ref_depth_hw = (moge_depth if isinstance(moge_depth, torch.Tensor)
                    else torch.tensor(moge_depth, dtype=torch.float32)).float().to(device)
    if ref_depth_hw.dim() == 2:   ref_depth_hw = ref_depth_hw.unsqueeze(0).unsqueeze(0)
    elif ref_depth_hw.dim() == 3: ref_depth_hw = ref_depth_hw.unsqueeze(0)


    # ---- VAE encode reference image ----
    preprocess = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    ref_pil = Image.open(frame_paths[0]).convert("RGB")
    img_tensor = preprocess(ref_pil).unsqueeze(0).unsqueeze(0).to(device, dtype=weight_dtype)
    with torch.no_grad():
        ref_latent = encode_video(vae, img_tensor, main_config.vae_backbone)[:, 0]

    # ---- Generate 6 synthetic trajectories for the encoder ----
    from prism.pipeline import MULTI_TRAJECTORIES
    from cosmos_predict1.diffusion.inference.camera_utils import generate_camera_trajectory
    initial_w2c = moge_w2c[0, 0].to(device)
    initial_intrinsics = moge_intrinsics[0, 0].to(device)

    cameras_list = []
    for traj_name, traj_cfg in MULTI_TRAJECTORIES.items():
        cam_gen_kwargs = traj_cfg.get("camera_gen_kwargs", {})
        syn_w2cs, syn_K = generate_camera_trajectory(
            trajectory_type=traj_name,
            initial_w2c=initial_w2c,
            initial_intrinsics=initial_intrinsics,
            num_frames=121,
            movement_distance=traj_cfg["movement_distance"],
            camera_rotation="center_facing",
            device=device,
            **cam_gen_kwargs,
        )
        cameras_list.append((syn_w2cs, syn_K))

    pred_latents_all, batch_list = run_encoder_all_views(
        cameras_list, ref_latent, ref_depth_hw,
        vae, encoder, main_config, inf_config,
        img_size, patch_size_out_factor, device,
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

    # ---- Build DL3DV cam_view for rendering ----
    # c2ws are in COLMAP/nerfstudio world coordinates.
    # Relativize: make frame 0 the origin.
    from prism.pipeline import build_camera_batch
    c2ws_dev = c2ws.to(device).float()
    c2w_ref_inv = torch.linalg.inv(c2ws_dev[0:1])  # (1, 4, 4)
    # Relative c2w: c2w_rel[i] = c2w_ref_inv @ c2w[i]
    c2ws_rel = c2w_ref_inv @ c2ws_dev  # (N, 4, 4)

    # Scale alignment via optical flow
    align_indices = list(range(1, N))
    align_tgt_paths = [frame_paths[i] for i in align_indices]
    align_c2ws = [c2ws_rel[i] for i in align_indices]

    depth_full = ref_depth_hw.squeeze()
    while depth_full.dim() > 2:
        depth_full = depth_full[0]
    if depth_full.shape != (img_h, img_w):
        depth_full = F.interpolate(depth_full.unsqueeze(0).unsqueeze(0),
                                   size=(img_h, img_w), mode='bilinear',
                                   align_corners=False).squeeze()

    sf = compute_scale_via_flow(frame_paths[0], align_tgt_paths, depth_full,
                                initial_intrinsics, align_c2ws, img_h, img_w)
    c2ws_rel[:, :3, 3] *= sf

    # Pad to 121
    if c2ws_rel.shape[0] < num_model_views:
        pad_len = num_model_views - c2ws_rel.shape[0]
        c2ws_rel = torch.cat([c2ws_rel, c2ws_rel[-1:].expand(pad_len, -1, -1)], 0)

    # Convert to w2c for build_camera_batch
    w2cs = torch.linalg.inv(c2ws_rel).unsqueeze(0)  # (1, 121, 4, 4)
    K_render = initial_intrinsics.unsqueeze(0).unsqueeze(0).expand(1, num_model_views, -1, -1).clone()
    render_batch, _, _ = build_camera_batch(w2cs, K_render, img_size, device)

    b0 = batch_list[0]
    batch = {
        "images_input_embed":     images_input_embed_full,
        "plucker_embedding":      plucker_full,
        "rays_os":                rays_os_full,
        "rays_ds":                rays_ds_full,
        "num_input_multi_views":  V_enc,
        "cam_view":               render_batch["cam_view"],
        "intrinsics":             render_batch["intrinsics"],
        "time_embeddings":        b0["time_embeddings"],
        "time_embeddings_target": b0["time_embeddings_target"],
        "flip_flag":              b0["flip_flag"],
    }

    with torch.no_grad():
        model_output = transformer(batch)

    pred_frames = model_output["images_pred"].float().cpu()
    pred_frames = pred_frames.clamp(0, 1)
    pred_frames = pred_frames[0]  # (T_render, 3, H, W)

    # ---- Evaluate frames 1..N_eval vs GT ----
    n_eval = min(N_eval, pred_frames.shape[0] - 1)
    eval_frame_paths = frame_paths[1: 1 + n_eval]

    gt_preprocess = transforms.Compose([
        transforms.Resize((img_h, img_w)),
        transforms.ToTensor(),
    ])

    psnrs, ssims, lpipss = [], [], []
    for i in range(n_eval):
        pred_t = pred_frames[i + 1]
        gt_t = gt_preprocess(
            Image.open(eval_frame_paths[i]).convert("RGB")
        )

        if pred_t.shape[-2:] != gt_t.shape[-2:]:
            pred_t = F.interpolate(
                pred_t.unsqueeze(0), size=gt_t.shape[-2:],
                mode="bilinear", align_corners=False,
            ).squeeze(0)

        pred_np = pred_t.permute(1, 2, 0).numpy()
        gt_np = gt_t.permute(1, 2, 0).numpy()
        psnrs.append(compute_psnr(pred_np, gt_np))
        ssims.append(compute_ssim(pred_np, gt_np))

        pred_lpips = (pred_t * 2 - 1).unsqueeze(0).to(device)
        gt_lpips = (gt_t * 2 - 1).unsqueeze(0).to(device)
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
    p.add_argument("--num_eval_frames", type=int, default=14)
    p.add_argument("--start_frame", type=int, default=0,
                   help="Starting frame index (ref frame). 0=first frame, 16=frame 16, etc.")

    p.add_argument("--config",         type=str, required=True)
    p.add_argument("--config_default", type=str,
                   default="configs/inference/default.yaml")
    p.add_argument("--encoder_ckpt",   type=str, required=True)

    p.add_argument("--output_dir",   type=str, default="outputs/eval/dl3dv")
    p.add_argument("--save_renders", action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ---- Load configs ----
    inf_config = load_and_merge_configs([args.config_default, args.config])
    main_config = load_and_merge_configs(list(inf_config.config_path))
    main_config.load_latents = False

    img_size = tuple(main_config.get("img_size", [704, 1280]))
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
        scene_dirs = sorted([
            d for d in root.iterdir()
            if d.is_dir() and find_transforms_dir(d) is not None
        ])
        assert scene_dirs, (
            f"No DL3DV scenes found under {root}. Each scene directory needs a "
            "transforms.json in nerfstudio/, colmap/ or the scene root."
        )
        if len(scene_dirs) > args.num_scenes:
            random.seed(args.seed)
            scene_dirs = random.sample(scene_dirs, args.num_scenes)
        print(f"Evaluating {len(scene_dirs)} scenes from {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Resume: load existing results if available ----
    results_path = out_dir / "results.json"
    per_scene = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            prev = json.load(f)
        if "per_scene" in prev:
            per_scene = prev["per_scene"]
            print(f"Resumed {len(per_scene)} clips from {results_path}")

    # ---- Evaluate ----
    skipped = 0
    sf_tag = f"_sf{args.start_frame}" if args.start_frame > 0 else ""

    for scene_dir in tqdm(scene_dirs, desc="Scenes"):
        scene_key = f"{scene_dir.name}{sf_tag}"
        if scene_key in per_scene:
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
                start_frame=args.start_frame,
                out_dir=out_dir / "renders" if args.save_renders else None,
            )
            result["start_frame"] = args.start_frame
            per_scene[scene_key] = result
            tqdm.write(
                f"{scene_key}: "
                f"PSNR={result['psnr']:.2f}  "
                f"SSIM={result['ssim']:.3f}  "
                f"LPIPS={result['lpips']:.3f}  "
                f"({result['n_frames']} frames)"
            )
            # Save after each scene (resume-safe)
            all_p = [r["psnr"] for r in per_scene.values()]
            all_s = [r["ssim"] for r in per_scene.values()]
            all_l = [r["lpips"] for r in per_scene.values()]
            interim = {
                "summary": {"psnr": float(np.mean(all_p)), "ssim": float(np.mean(all_s)),
                            "lpips": float(np.mean(all_l)), "n_clips": len(per_scene)},
                "per_scene": per_scene,
            }
            with open(results_path, "w") as f:
                json.dump(interim, f, indent=2)
        except Exception as e:
            import traceback
            tqdm.write(f"SKIP {scene_dir.name}: {e}")
            traceback.print_exc()
            skipped += 1

    # ---- Aggregate ----
    all_psnr = [r["psnr"] for r in per_scene.values()]
    all_ssim = [r["ssim"] for r in per_scene.values()]
    all_lpips = [r["lpips"] for r in per_scene.values()]
    summary = {
        "psnr":     float(np.mean(all_psnr)) if all_psnr else 0,
        "ssim":     float(np.mean(all_ssim)) if all_ssim else 0,
        "lpips":    float(np.mean(all_lpips)) if all_lpips else 0,
        "n_clips":  len(per_scene),
        "skipped":  skipped,
    }
    print("\n=== Results ===")
    print(f"  PSNR  : {summary['psnr']:.4f}")
    print(f"  SSIM  : {summary['ssim']:.4f}")
    print(f"  LPIPS : {summary['lpips']:.4f}")
    print(f"  Clips: {summary['n_clips']} (skipped: {skipped})")
    print(f"  Eval frames/scene: {args.num_eval_frames}")

    results_path = out_dir / "results.json"
    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_scene": per_scene}, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
