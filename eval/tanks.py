# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Evaluate PRISM on Tanks-and-Temples.

Each scene is cut into non-overlapping clips of --num_eval_frames, so the eight
scenes yield 95 clips. Cameras come from the COLMAP sparse reconstruction
(cameras.bin, images.bin) of the NoPe-NeRF preprocessed release.
"""

import argparse
import json
import struct
from collections import namedtuple
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

from prism.data import (
    run_moge_preloaded,
    compute_psnr,
    compute_ssim,
    compute_scale_via_flow,
)

weight_dtype = torch.bfloat16
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# ---------------------------------------------------------------------------
# COLMAP binary readers
# ---------------------------------------------------------------------------

CameraModel = namedtuple("CameraModel", ["model_id", "model_name", "num_params"])
CAMERA_MODELS = {
    0:  CameraModel(0,  "SIMPLE_PINHOLE", 3),
    1:  CameraModel(1,  "PINHOLE", 4),
    2:  CameraModel(2,  "SIMPLE_RADIAL", 4),
    3:  CameraModel(3,  "RADIAL", 5),
    4:  CameraModel(4,  "OPENCV", 8),
    5:  CameraModel(5,  "OPENCV_FISHEYE", 8),
    6:  CameraModel(6,  "FULL_OPENCV", 12),
    7:  CameraModel(7,  "FOV", 5),
    8:  CameraModel(8,  "SIMPLE_RADIAL_FISHEYE", 4),
    9:  CameraModel(9,  "RADIAL_FISHEYE", 5),
    10: CameraModel(10, "THIN_PRISM_FISHEYE", 12),
}


def read_cameras_binary(path):
    """Read COLMAP cameras.bin -> dict of {cam_id: (model, width, height, params)}."""
    cameras = {}
    with open(path, "rb") as f:
        num_cameras = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_cameras):
            cam_id = struct.unpack("<I", f.read(4))[0]
            model_id = struct.unpack("<i", f.read(4))[0]
            width = struct.unpack("<Q", f.read(8))[0]
            height = struct.unpack("<Q", f.read(8))[0]
            if model_id not in CAMERA_MODELS:
                raise ValueError(f"Unknown COLMAP camera model id {model_id} in {path}")
            num_params = CAMERA_MODELS[model_id].num_params
            params = struct.unpack(f"<{num_params}d", f.read(8 * num_params))
            cameras[cam_id] = {
                "model": CAMERA_MODELS[model_id].model_name,
                "width": width,
                "height": height,
                "params": params,
            }
    return cameras


def read_images_binary(path):
    """Read COLMAP images.bin -> list of (image_name, qvec, tvec, camera_id) sorted by name."""
    images = []
    with open(path, "rb") as f:
        num_images = struct.unpack("<Q", f.read(8))[0]
        for _ in range(num_images):
            struct.unpack("<I", f.read(4))[0]  # image_id, unused but must be consumed
            qvec = struct.unpack("<4d", f.read(32))
            tvec = struct.unpack("<3d", f.read(24))
            camera_id = struct.unpack("<I", f.read(4))[0]
            name = b""
            while True:
                ch = f.read(1)
                if ch == b"\x00":
                    break
                name += ch
            name = name.decode("utf-8")
            # Skip 2D points
            num_points2D = struct.unpack("<Q", f.read(8))[0]
            f.read(num_points2D * 24)  # each point: x, y, point3D_id (8+8+8)
            images.append({
                "name": name,
                "qvec": np.array(qvec),
                "tvec": np.array(tvec),
                "camera_id": camera_id,
            })
    # Sort by image name
    images.sort(key=lambda x: x["name"])
    return images


def qvec2rotmat(qvec):
    """Convert COLMAP quaternion (w, x, y, z) to 3x3 rotation matrix."""
    w, x, y, z = qvec
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*w*z,     2*x*z + 2*w*y],
        [2*x*y + 2*w*z,     1 - 2*x*x - 2*z*z, 2*y*z - 2*w*x],
        [2*x*z - 2*w*y,     2*y*z + 2*w*x,     1 - 2*x*x - 2*y*y],
    ])


def colmap_to_c2w(qvec, tvec):
    """Convert COLMAP (qvec, tvec) to 4x4 c2w matrix."""
    R = qvec2rotmat(qvec)  # w2c rotation
    t = np.array(tvec)      # w2c translation
    # w2c = [R | t], c2w = [R^T | -R^T @ t]
    c2w = np.eye(4)
    c2w[:3, :3] = R.T
    c2w[:3, 3] = -R.T @ t
    return c2w


def get_intrinsics_matrix(camera):
    """Build 3x3 intrinsics matrix from COLMAP camera params."""
    model = camera["model"]
    params = camera["params"]
    if model == "SIMPLE_PINHOLE":
        f, cx, cy = params
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model == "PINHOLE":
        fx, fy, cx, cy = params
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    elif model in ("SIMPLE_RADIAL", "RADIAL"):
        # f, cx, cy, then the distortion coefficients, which are dropped.
        f, cx, cy = params[0], params[1], params[2]
        K = np.array([[f, 0, cx], [0, f, cy], [0, 0, 1]])
    elif model in ("OPENCV", "FULL_OPENCV", "OPENCV_FISHEYE"):
        fx, fy, cx, cy = params[0], params[1], params[2], params[3]
        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    else:
        raise ValueError(f"Unsupported camera model: {model}")
    return K


# ---------------------------------------------------------------------------
# Load T&T scene
# ---------------------------------------------------------------------------

def load_tanks_scene(scene_dir: Path):
    """
    Load Tanks-and-Temples scene from COLMAP sparse reconstruction.

    Returns:
        c2ws:        (N, 4, 4) float32 - camera-to-world matrices
        K:           (3, 3) float32 - intrinsics matrix
        frame_paths: list of N Path objects to images
        img_w, img_h: image resolution
    """
    sparse_dir = scene_dir / "sparse" / "0"
    img_dir = scene_dir / "images"

    assert sparse_dir.exists(), f"sparse/0 not found in {scene_dir}"
    assert img_dir.exists(), f"images/ not found in {scene_dir}"

    cameras = read_cameras_binary(str(sparse_dir / "cameras.bin"))
    images = read_images_binary(str(sparse_dir / "images.bin"))

    # Use first camera's intrinsics
    first_cam_id = images[0]["camera_id"]
    camera = cameras[first_cam_id]
    K = get_intrinsics_matrix(camera)
    img_w, img_h = camera["width"], camera["height"]

    c2w_list = []
    path_list = []

    for img_data in images:
        c2w = colmap_to_c2w(img_data["qvec"], img_data["tvec"])
        c2w_list.append(torch.tensor(c2w, dtype=torch.float32))

        img_path = img_dir / img_data["name"]
        path_list.append(img_path)

    c2ws = torch.stack(c2w_list, dim=0)
    K_tensor = torch.tensor(K, dtype=torch.float32)

    return c2ws, K_tensor, path_list, img_w, img_h


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
    stride: int = 1,
):
    """Evaluate a single Tanks-and-Temples scene (one clip starting at start_frame)."""
    img_h, img_w = img_size

    # ---- Load scene ----
    c2ws_all, K_scene, frame_paths_all, scene_w, scene_h = load_tanks_scene(scene_dir)

    # Apply start_frame offset
    N_total = min(len(frame_paths_all), c2ws_all.shape[0])
    assert start_frame < N_total, \
        f"start_frame {start_frame} >= {N_total} available frames in {scene_dir.name}"
    frame_paths_all = frame_paths_all[start_frame:]
    c2ws_all = c2ws_all[start_frame:]

    # Apply stride (Wonderland protocol: RE10K=3, DL3DV=2, T&T=4) -- same
    # convention as eval/re10k.py: keep the clip's reference frame (index 0
    # post-offset) + every stride-th frame after it.
    if stride > 1:
        stride_indices = [0] + list(range(stride, len(frame_paths_all), stride))
        frame_paths_all = [frame_paths_all[i] for i in stride_indices if i < len(frame_paths_all)]
        c2ws_all = c2ws_all[stride_indices[:len(frame_paths_all)]]

    N_available = min(len(frame_paths_all), c2ws_all.shape[0])
    N_eval = min(num_eval_frames, N_available - 1)
    assert N_eval >= 1, f"Scene {scene_dir.name} has too few frames from start_frame={start_frame} ({N_available})"

    N = N_eval + 1
    frame_paths = frame_paths_all[:N]
    c2ws = c2ws_all[:N]

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

    # ---- Build T&T cam_view for rendering ----
    from prism.pipeline import build_camera_batch
    c2ws_dev = c2ws.to(device).float()
    # Relativize to frame 0
    c2w_ref_inv = torch.linalg.inv(c2ws_dev[0:1])
    c2ws_rel = c2w_ref_inv @ c2ws_dev

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
    w2cs = torch.linalg.inv(c2ws_rel).unsqueeze(0)
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
    pred_frames = pred_frames[0]

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

def get_scene_num_frames(scene_dir: Path):
    """Get total number of frames available for a scene."""
    c2ws, _, frame_paths, _, _ = load_tanks_scene(scene_dir)
    return min(len(frame_paths), c2ws.shape[0])


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--scene_dir",   type=str, default=None)
    p.add_argument("--dataset_dir", type=str, default=None)
    p.add_argument("--num_eval_frames", type=int, default=14,
                   help="Frames per clip (including reference). Clips start at 0, N, 2N, ...")
    p.add_argument("--stride", type=int, default=1,
                   help="Frame sampling stride within each clip (Wonderland protocol: "
                        "RE10K=3, DL3DV=2, T&T=4). default=1 (consecutive frames).")

    p.add_argument("--config",         type=str, required=True)
    p.add_argument("--config_default", type=str, default="configs/inference/default.yaml")
    p.add_argument("--encoder_ckpt",   type=str, required=True)

    p.add_argument("--output_dir",   type=str, default="outputs/eval/tanks")
    p.add_argument("--save_renders", action="store_true")
    p.add_argument("--seed",         type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    inf_config = load_and_merge_configs([args.config_default, args.config])
    main_config = load_and_merge_configs(list(inf_config.config_path))
    main_config.load_latents = False

    img_size = tuple(main_config.get("img_size", [704, 1280]))
    patch_size_out_factor = tuple(main_config.get("patch_size_out_factor", [1, 8, 8]))

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
            if d.is_dir() and (d / "sparse" / "0" / "cameras.bin").exists()
        ])
        print(f"Found {len(scene_dirs)} scenes from {root}")

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Build clip list: (scene_dir, start_frame) ----
    # At stride>1, one clip spans (clip_size-1)*stride+1 raw frames, so
    # non-overlapping clips must be spaced by that same span in raw-frame terms.
    clip_size = args.num_eval_frames  # 14 frames per clip (1 ref + 13 eval)
    clip_span = (clip_size - 1) * args.stride + 1
    clip_list = []
    for sd in scene_dirs:
        n_frames = get_scene_num_frames(sd)
        n_clips = max(1, (n_frames - 1) // clip_span)
        for ci in range(n_clips):
            sf = ci * clip_span
            if sf + clip_span <= n_frames:
                clip_list.append((sd, sf))
        print(f"  {sd.name}: {n_frames} frames -> {n_clips} clips (stride={args.stride})")
    print(f"Total clips: {len(clip_list)}")

    # ---- Resume support ----
    results_path = out_dir / "results.json"
    per_clip = {}
    if results_path.exists():
        with open(results_path, "r") as f:
            saved = json.load(f)
        per_clip = saved.get("per_clip", {})
        print(f"Resumed {len(per_clip)} clips from {results_path}")

    # ---- Evaluate ----
    all_psnr, all_ssim, all_lpips = [], [], []
    # Add already-completed clips to running averages
    for key, val in per_clip.items():
        all_psnr.append(val["psnr"])
        all_ssim.append(val["ssim"])
        all_lpips.append(val["lpips"])

    skipped = 0

    for scene_dir, start_frame in tqdm(clip_list, desc="Clips"):
        clip_key = f"{scene_dir.name}_sf{start_frame}"
        if clip_key in per_clip:
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
                start_frame=start_frame,
                stride=args.stride,
            )
            result["start_frame"] = start_frame
            all_psnr.append(result["psnr"])
            all_ssim.append(result["ssim"])
            all_lpips.append(result["lpips"])
            per_clip[clip_key] = result
            tqdm.write(
                f"{clip_key}: "
                f"PSNR={result['psnr']:.2f}  "
                f"SSIM={result['ssim']:.3f}  "
                f"LPIPS={result['lpips']:.3f}  "
                f"({result['n_frames']} frames)"
            )

            # Save after each clip for resume
            interim = {
                "summary": {"psnr": float(np.mean(all_psnr)), "ssim": float(np.mean(all_ssim)),
                            "lpips": float(np.mean(all_lpips)), "n_clips": len(per_clip)},
                "per_clip": per_clip,
            }
            with open(results_path, "w") as f:
                json.dump(interim, f, indent=2)

        except Exception as e:
            import traceback
            tqdm.write(f"SKIP {clip_key}: {e}")
            traceback.print_exc()
            skipped += 1

    summary = {
        "psnr":     float(np.mean(all_psnr)) if all_psnr else 0,
        "ssim":     float(np.mean(all_ssim)) if all_ssim else 0,
        "lpips":    float(np.mean(all_lpips)) if all_lpips else 0,
        "n_clips":  len(per_clip),
        "n_scenes": len(scene_dirs),
        "skipped":  skipped,
    }
    print("\n=== Results ===")
    print(f"  PSNR  : {summary['psnr']:.4f}")
    print(f"  SSIM  : {summary['ssim']:.4f}")
    print(f"  LPIPS : {summary['lpips']:.4f}")
    print(f"  Clips : {summary['n_clips']} from {summary['n_scenes']} scenes (skipped: {skipped})")

    with open(results_path, "w") as f:
        json.dump({"summary": summary, "per_clip": per_clip}, f, indent=2)
    print(f"\nSaved to {results_path}")


if __name__ == "__main__":
    main()
