# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Portions of this file are derived from NVIDIA Lyra
# (https://github.com/nv-tlabs/lyra), Copyright (c) 2025 NVIDIA CORPORATION,
# licensed under the Apache License, Version 2.0.

"""
Shared building blocks for the PRISM pipeline.

Model loading, camera/Plucker construction and the encoder forward pass, used
by every training and evaluation entry point so that the geometric prior is
built identically in all of them.

Contents:
    fill_warp_holes          hole filling for visualisation only, never in the blend
    MULTI_TRAJECTORIES       the six demo camera trajectories
    load_latent_recon        frozen 3DGS decoder (LatentRecon) from a Lyra checkpoint
    load_encoder             the trainable residual encoder from a PRISM checkpoint
    run_moge                 monocular depth and intrinsics
    build_camera_batch       camera batch in the layout LatentRecon expects
    run_encoder_all_views    geometric prior + residual for every (view, frame)
"""

import torch
import torch.nn.functional as F

from prism.encoder import WarpResidualEncoder
from prism.warp import forward_warp_latent_zbuffer
from src.models.recon.model_latent_recon import LatentRecon
from src.models.utils.model import encode_multi_view_video, encode_plucker_vae
from src.models.utils.render import get_plucker_embedding_and_rays

weight_dtype = torch.bfloat16


def fill_warp_holes(warp, mask, kernel_size=15):
    """Fill mask=0 holes with mask-weighted average of neighboring valid pixels.
    warp: (B, V, T, C, H, W),  mask: (B, V, T, 1, H, W)
    Returns:
        filled:     (B, V, T, C, H, W)
        fill_valid: (B, V, T, 1, H, W)  smooth weight [0,1]; 0=no neighbors, 1=fully covered
    """
    B, V, T, C, H, W = warp.shape
    w_flat = warp.reshape(B * V * T, C, H, W)
    m_flat = mask.reshape(B * V * T, 1, H, W)
    pad    = kernel_size // 2
    w_sum  = F.avg_pool2d(w_flat * m_flat, kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
    m_sum  = F.avg_pool2d(m_flat,          kernel_size, stride=1, padding=pad) * (kernel_size ** 2)
    filled     = (w_sum / (m_sum + 1e-6)).reshape(B, V, T, C, H, W)
    fill_valid = (m_sum / (kernel_size ** 2)).clamp(0, 1).reshape(B, V, T, 1, H, W)
    return filled, fill_valid

MULTI_TRAJECTORIES = {
    "left":       {"movement_distance": 0.25},
    "right":      {"movement_distance": 0.25},
    "up":         {"movement_distance": 0.15},
    "zoom_out":   {"movement_distance": 0.35},
    "zoom_in":    {"movement_distance": 0.35},
    "clockwise":  {"movement_distance": 0.5,
                   "camera_gen_kwargs": {"radius_x_factor": 0.15,
                                         "radius_y_factor": 0.10,
                                         "num_circles": 2}},
}


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def load_latent_recon(ckpt_path, config, device):
    transformer = LatentRecon(config)
    data = torch.load(ckpt_path, map_location="cpu")
    if isinstance(data, dict) and "module" in data:
        state_dict = data["module"]
    elif isinstance(data, dict) and "model" in data:
        state_dict = data["model"]
    else:
        state_dict = data
    transformer.load_state_dict(state_dict, strict=False)
    transformer.to(device=device, dtype=weight_dtype).eval()
    return transformer


def load_encoder(ckpt_path, config, device) -> WarpResidualEncoder:
    encoder = WarpResidualEncoder(
        latent_channels     = config.get("prism_latent_channels",    16),
        plucker_channels    = config.get("prism_plucker_channels",   32),
        embed_dim           = config.get("prism_embed_dim",          512),
        depth               = config.get("prism_depth",              8),
        num_heads           = config.get("prism_num_heads",          8),
        mlp_ratio           = config.get("prism_mlp_ratio",          4.0),
        patch_size          = config.get("prism_patch_size",         4),
        use_flow_correction = config.get("use_flow_correction",    True),
        flow_hidden         = config.get("flow_hidden",            64),
        flow_max_flow       = config.get("flow_max_flow",          4.0),


    )
    data  = torch.load(ckpt_path, map_location="cpu")
    state = data["module"] if isinstance(data, dict) and "module" in data else data
    encoder.load_state_dict(state)
    encoder.to(device=device, dtype=weight_dtype).eval()
    print(f"Residual encoder loaded from {ckpt_path}  ({encoder.count_parameters():,} params)")
    return encoder


# ---------------------------------------------------------------------------
# Camera / depth helpers
# ---------------------------------------------------------------------------

def run_moge(image_path, target_h, target_w, device):
    from moge.model.v1 import MoGeModel
    from cosmos_predict1.diffusion.inference.gen3c_single_image_sdg import _predict_moge_depth
    moge_model = MoGeModel.from_pretrained("Ruicheng/moge-vitl").to(device).eval()
    return _predict_moge_depth(image_path, target_h=target_h, target_w=target_w,
                               device=device, moge_model=moge_model)


def build_camera_batch(w2cs, intrinsics_3x3, img_size, device):
    """
    Convert generate_camera_trajectory outputs to the format expected by the pipeline.

    Returns:
        batch:      dict with c2ws_input, intrinsics_input, cam_view, etc.
        K_full:     (1, T, 3, 3) - full-res K matrices for depth_forward_warp
        c2ws_float: (1, T, 4, 4) - float32 c2ws
    """
    c2ws = torch.linalg.inv(w2cs[0])                               # (T, 4, 4)
    c2ws = torch.linalg.inv(c2ws[0]).unsqueeze(0) @ c2ws           # normalise to ref cam
    K    = intrinsics_3x3[0]                                        # (T, 3, 3)
    intr = torch.stack([K[:, 0, 0], K[:, 1, 1], K[:, 0, 2], K[:, 1, 2]], dim=-1)  # (T, 4)

    c2ws = c2ws.unsqueeze(0)   # (1, T, 4, 4)
    intr = intr.unsqueeze(0)   # (1, T, 4)
    cam_view = torch.linalg.inv(c2ws[0]).transpose(-1, -2).unsqueeze(0)

    batch = {
        "c2ws_input":       c2ws.to(device, dtype=weight_dtype),
        "intrinsics_input": intr.to(device, dtype=weight_dtype),
        "cam_view":         cam_view.to(device, dtype=weight_dtype),
        "intrinsics":       intr.to(device, dtype=weight_dtype),
        "flip_flag":        torch.zeros(1, 1, device=device, dtype=weight_dtype),
        "num_input_multi_views": 1,
        "time_embeddings": False,
        "time_embeddings_target": [False],
    }

    # Full-res K (3x3) for depth_forward_warp
    B, T, _ = intr.shape
    K_full = torch.zeros(B, T, 3, 3, device=device, dtype=torch.float32)
    K_full[:, :, 0, 0] = intr[:, :, 0].float()
    K_full[:, :, 1, 1] = intr[:, :, 1].float()
    K_full[:, :, 0, 2] = intr[:, :, 2].float()
    K_full[:, :, 1, 2] = intr[:, :, 3].float()
    K_full[:, :, 2, 2] = 1.0

    return batch, K_full, c2ws.to(device, dtype=torch.float32)


# ---------------------------------------------------------------------------
# Per-trajectory inference
# ---------------------------------------------------------------------------

def _build_view_warp_plucker(w2cs, intrinsics_3x3, ref_latent, ref_depth_hw,
                              vae, main_config, inf_config, img_size,
                              patch_size_out_factor, device):
    """Build warp (1,T,C,H,W), mask (1,T,1,H,W), plucker (1,T,Cp,H,W) for one trajectory."""
    latent_scale = main_config.get("latent_spat_compression", 8)
    T_lat    = inf_config.get("prism_latent_T", 16)
    T_traj   = w2cs.shape[1]
    t_stride = max(1, (T_traj - 1) // (T_lat - 1))

    batch, K_full, c2ws_float = build_camera_batch(w2cs, intrinsics_3x3, img_size, device)

    batch["plucker_embedding"], batch["rays_os"], batch["rays_ds"] = \
        get_plucker_embedding_and_rays(
            batch["intrinsics_input"], batch["c2ws_input"],
            img_size, patch_size_out_factor, batch["flip_flag"],
            get_batch_index=False, out_dtype=weight_dtype,
        )
    if main_config.get("plucker_embedding_vae", True):
        batch = encode_plucker_vae(
            batch,
            lambda x: encode_multi_view_video(vae, x, 1, main_config.vae_backbone),
        )
    plucker_lat = batch["plucker_embedding"]   # (1, T_lat, Cp, H_lat, W_lat)

    # Warp: exactly mirrors infer_warp_residual.py online path and precompute_warp.py
    # Use forward_warp_latent_zbuffer with latent-res K + latent-res depth + use_splat=True
    H_lat = ref_latent.shape[-2]
    W_lat = ref_latent.shape[-1]
    depth_4d = ref_depth_hw.float()
    while depth_4d.dim() > 4:
        depth_4d = depth_4d[0]       # strip leading dims: (1,1,1,H,W) -> (1,1,H,W)
    while depth_4d.dim() < 4:
        depth_4d = depth_4d.unsqueeze(0)
    ref_depth_lat = F.interpolate(
        depth_4d, size=(H_lat, W_lat),
        mode='bilinear', align_corners=False,
    )  # (1, 1, H_lat, W_lat)

    K_lat = K_full[:, 0].clone()           # (B, 3, 3) at full-res
    K_lat[:, :2] = K_lat[:, :2] / latent_scale   # scale down to latent-res

    src_c2w = c2ws_float[:, 0]             # (B, 4, 4)
    tgt_c2ws = torch.stack(
        [c2ws_float[:, t * t_stride] for t in range(T_lat)], dim=1
    )  # (B, T_lat, 4, 4)

    warped_v, mask_v = forward_warp_latent_zbuffer(
        ref_latent.float(), ref_depth_lat, src_c2w, tgt_c2ws, K_lat,
        use_splat=True,
    )
    # warped_v: (B, T_lat, C, H, W) - same as forward_warp_latent_zbuffer output
    # mask_v:   (B, T_lat, 1, H, W)
    warp = warped_v.to(weight_dtype)   # (1, T, C, H, W)
    mask = mask_v.to(weight_dtype)     # (1, T, 1, H, W)

    return warp, mask, plucker_lat, batch


@torch.no_grad()
def run_encoder_all_views(
    cameras_list, ref_latent, ref_depth_hw, vae, encoder,
    main_config, inf_config, img_size, patch_size_out_factor, device,
    coverage_out: dict = None,
):
    """
    Build V warps from a list of (w2cs, intrinsics_3x3), run encoder once.

    Args:
        coverage_out: optional dict; if provided, populated in-place with
            {"mask_coverage": float} (mean warp validity mask across V*T*H*W)
            for callers that want to log per-scene warp coverage (e.g. for
            coverage-stratified reporting) without changing the return
            signature for existing callers.

    Returns:
        pred_latents_all: (1, V, T, C, H, W)
        batches:          list of V batch dicts (for LatentRecon rendering)
    """
    warp_list, mask_list, plucker_list, batch_list = [], [], [], []
    for w2cs, intrinsics_3x3 in cameras_list:
        warp, mask, plucker_lat, batch = _build_view_warp_plucker(
            w2cs, intrinsics_3x3, ref_latent, ref_depth_hw,
            vae, main_config, inf_config, img_size, patch_size_out_factor, device,
        )
        warp_list.append(warp)
        mask_list.append(mask)
        plucker_list.append(plucker_lat)
        batch_list.append(batch)

    # Stack V views: each (1,T,C,H,W) -> (1,V,T,C,H,W)
    warp_vt  = torch.stack(warp_list,    dim=1)
    mask_vt  = torch.stack(mask_list,    dim=1)
    pluckers = torch.stack(plucker_list, dim=1)

    mask_coverage = mask_vt.float().mean().item()
    if coverage_out is not None:
        coverage_out["mask_coverage"] = mask_coverage
    delta        = encoder(warp_vt, mask_vt, ref_latent, pluckers)
    ref_exp          = ref_latent[:, None, None].expand_as(warp_vt)
    blended_base     = warp_vt * mask_vt + ref_exp * (1 - mask_vt)
    pred_latents_all = blended_base + delta                     # (1, V, T, C, H, W)

    del warp_vt, mask_vt, pluckers, delta, blended_base
    torch.cuda.empty_cache()

    return pred_latents_all, batch_list


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
