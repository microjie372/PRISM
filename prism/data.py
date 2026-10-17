# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Dataset loading, depth estimation and metrics shared by the benchmark scripts.

Every benchmark reads cameras the same way, aligns monocular depth to the
dataset's camera scale the same way, and scores renders with the same metric
implementations, so numbers stay comparable across RE10K, DL3DV and
Tanks-and-Temples.

Contents:
    load_cameras_raw        raw cameras.pt to 4x4 extrinsics + normalised intrinsics
    load_images             frame paths in sorted order
    run_moge_preloaded      MoGe depth with a model loaded once per run
    compute_psnr/ssim       pixel metrics on [0, 1] float arrays
    compute_scale_via_flow  per-scene depth/camera scale from sparse optical flow
"""

from pathlib import Path

import numpy as np
import torch
from skimage.metrics import peak_signal_noise_ratio as psnr_fn
from skimage.metrics import structural_similarity as ssim_fn

weight_dtype = torch.bfloat16
IMG_EXTENSIONS = {".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"}


# ---------------------------------------------------------------------------
# Camera loading
# ---------------------------------------------------------------------------

def load_cameras_raw(scene_dir: Path):
    """
    Load raw cameras.pt from a scene directory.

    Returns:
        raw_matrices: (N, 4, 4) float32 - the raw 3x4 matrices padded to 4x4
        intr_norm:    (N, 4) float32 - normalised [fx/W, fy/H, cx/W, cy/H]
    """
    cam_path = scene_dir / "cameras.pt"
    assert cam_path.exists(), f"cameras.pt not found in {scene_dir}"
    data = torch.load(cam_path, map_location="cpu").float()  # (N, 18)

    assert data.ndim == 2 and data.shape[1] == 18, \
        f"Expected cameras.pt shape (N, 18), got {data.shape}"

    N = data.shape[0]
    intr_norm = data[:, :4]                       # (N, 4)

    raw_34 = data[:, 6:18].reshape(N, 3, 4)
    raw = torch.eye(4).unsqueeze(0).repeat(N, 1, 1)
    raw[:, :3, :4] = raw_34

    return raw, intr_norm


def load_images(scene_dir: Path):
    """Load all image frames sorted by filename."""
    frames = sorted(
        p for p in scene_dir.iterdir()
        if p.suffix in IMG_EXTENSIONS
    )
    assert len(frames) >= 2, f"Need at least 2 frames in {scene_dir}, found {len(frames)}"
    return frames


# ---------------------------------------------------------------------------
# MoGe depth (scene-level, with preloaded model)
# ---------------------------------------------------------------------------

def run_moge_preloaded(image_path: str, target_h: int, target_w: int,
                       device, moge_model):
    """Run MoGe depth estimation using a preloaded model (avoids reload per scene)."""
    from cosmos_predict1.diffusion.inference.gen3c_single_image_sdg import _predict_moge_depth
    return _predict_moge_depth(
        image_path, target_h=target_h, target_w=target_w,
        device=device, moge_model=moge_model,
    )


# ---------------------------------------------------------------------------
# Metric helpers
# ---------------------------------------------------------------------------

def compute_psnr(pred: np.ndarray, gt: np.ndarray) -> float:
    return psnr_fn(gt, pred, data_range=1.0)


def compute_ssim(pred: np.ndarray, gt: np.ndarray) -> float:
    return ssim_fn(gt, pred, data_range=1.0, channel_axis=-1)


# ---------------------------------------------------------------------------
# Per-scene depth-camera scale alignment via sparse optical flow
# ---------------------------------------------------------------------------

def compute_scale_via_flow(ref_path, tgt_paths, depth_hw, moge_K, c2ws_re10k,
                           img_h, img_w):
    """
    Compute per-scene scale factor that aligns MoGe depth with RE10K cameras.

    Uses sparse Lucas-Kanade optical flow between ref and multiple target frames.
    For each matched point: s = measured_displacement / predicted_displacement(s=1).
    The median over all points gives a robust scale estimate.

    Args:
        ref_path:     path to reference image
        tgt_paths:    list of paths to target images (frames 1, 5, 10, ...)
        depth_hw:     (H, W) MoGe depth at full resolution
        moge_K:       (3, 3) MoGe intrinsics
        c2ws_re10k:   (N, 4, 4) RE10K c2ws (ref = identity)
        img_h, img_w: target resolution
    Returns:
        scale: float - multiply RE10K c2w translations by this value
    """
    import cv2

    # Load ref image as grayscale
    ref_bgr = cv2.imread(str(ref_path))
    ref_bgr = cv2.resize(ref_bgr, (img_w, img_h))
    ref_gray = cv2.cvtColor(ref_bgr, cv2.COLOR_BGR2GRAY)

    K = moge_K.cpu().float().numpy()
    K_inv = np.linalg.inv(K)
    depth = depth_hw.cpu().float().numpy()  # (H, W)

    # Detect good features in ref
    corners = cv2.goodFeaturesToTrack(ref_gray, maxCorners=500, qualityLevel=0.01,
                                     minDistance=10)
    if corners is None or len(corners) < 20:
        return 1.0  # fallback

    all_scales = []

    for tgt_path, c2w_tgt in zip(tgt_paths, c2ws_re10k):
        tgt_bgr = cv2.imread(str(tgt_path))
        tgt_bgr = cv2.resize(tgt_bgr, (img_w, img_h))
        tgt_gray = cv2.cvtColor(tgt_bgr, cv2.COLOR_BGR2GRAY)

        # Sparse LK optical flow
        tgt_pts, status, _ = cv2.calcOpticalFlowPyrLK(ref_gray, tgt_gray, corners, None)
        if tgt_pts is None:
            continue
        valid = status.flatten() == 1
        if valid.sum() < 10:
            continue

        src_pts = corners[valid].reshape(-1, 2)   # (M, 2) - pixel coords in ref
        dst_pts = tgt_pts[valid].reshape(-1, 2)   # (M, 2) - pixel coords in target

        # Reproject each match at scale=1, then take
        # scale = observed displacement / predicted displacement.

        R = c2w_tgt[:3, :3].cpu().float().numpy()
        t = c2w_tgt[:3, 3].cpu().float().numpy()
        R_T = R.T
        neg_R_T_t = -R_T @ t  # w2c translation

        for k in range(len(src_pts)):
            u0, v0 = src_pts[k]
            u0i, v0i = int(round(u0)), int(round(v0))
            if u0i < 0 or u0i >= img_w or v0i < 0 or v0i >= img_h:
                continue
            d_val = depth[v0i, u0i]
            if d_val < 1e-3:
                continue

            p3d = d_val * (K_inv @ np.array([u0, v0, 1.0]))

            # Rotation-only displacement (s=0): P_tgt = R^T @ P (no translation)
            p_rot = R_T @ p3d
            if p_rot[2] < 1e-4:
                continue
            u_rot = K[0, 0] * p_rot[0] / p_rot[2] + K[0, 2]
            v_rot = K[1, 1] * p_rot[1] / p_rot[2] + K[1, 2]

            # Full displacement at s=1: P_tgt = R^T @ (P - t)
            p_full = R_T @ p3d + neg_R_T_t
            if p_full[2] < 1e-4:
                continue
            u_full = K[0, 0] * p_full[0] / p_full[2] + K[0, 2]
            v_full = K[1, 1] * p_full[1] / p_full[2] + K[1, 2]

            # Translation-only contribution at s=1
            du_trans = u_full - u_rot
            dv_trans = v_full - v_rot
            trans_mag = np.sqrt(du_trans ** 2 + dv_trans ** 2)
            if trans_mag < 0.5:  # too small to be reliable
                continue

            # Actual flow minus rotation = actual translation contribution
            u1_flow, v1_flow = dst_pts[k]
            du_actual_trans = (u1_flow - u0) - (u_rot - u0)
            dv_actual_trans = (v1_flow - v0) - (v_rot - v0)

            # Project actual translation onto predicted translation direction
            # to get signed scale (handles direction correctly)
            direction = np.array([du_trans, dv_trans])
            actual_vec = np.array([du_actual_trans, dv_actual_trans])
            s_point = np.dot(actual_vec, direction) / np.dot(direction, direction)

            if s_point > 0.001:  # only positive scale makes sense
                all_scales.append(s_point)

    if len(all_scales) < 10:
        return 1.0  # fallback

    return float(np.median(all_scales))


# ---------------------------------------------------------------------------
# Per-scene evaluation
# ---------------------------------------------------------------------------
