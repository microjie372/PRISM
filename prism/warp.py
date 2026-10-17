# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The geometric warp prior: reprojects the reference latent to the target
viewpoints from monocular depth and camera geometry, leaving the encoder only
the residual to predict.

Training and inference call the same functions here, so the prior the encoder
learns against is the prior it sees at test time.
"""

import torch


def forward_warp_latent(
    ref_latent: torch.Tensor,   # (B, C, H, W)
    ref_depth:  torch.Tensor,   # (B, 1, H, W) - source depth at latent resolution
    ref_c2w:    torch.Tensor,   # (B, 4, 4)
    tgt_c2ws:   torch.Tensor,   # (B, T, 4, 4)
    K_lat:      torch.Tensor,   # (B, 3, 3) already at latent resolution
) -> tuple:
    """
    Forward warp: unproject source pixels with source depth, splat to target views.

    Consistent between training (precompute_warp.py) and inference (run MoGe on ref image).
    Returns:
        warped  (B, T, C, H, W)
        mask    (B, T, 1, H, W)  - 1 where target pixel received a valid splat
    """
    B, C, H, W = ref_latent.shape
    T = tgt_c2ws.shape[1]
    device = ref_latent.device
    N = H * W

    # Build source pixel grid and unproject to 3D
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    pix = torch.stack([gx.flatten(), gy.flatten(),
                       torch.ones(N, device=device)], dim=0)           # (3, N)
    pix = pix.unsqueeze(0).expand(B, -1, -1)                           # (B, 3, N)

    K      = K_lat.float()
    K_inv  = torch.linalg.inv(K)                                       # (B, 3, 3)
    d      = ref_depth.view(B, 1, N).clamp(min=1e-4)                   # (B, 1, N)
    pts    = (K_inv @ pix) * d                                         # (B, 3, N)
    pts_h  = torch.cat([pts, torch.ones(B, 1, N, device=device)], 1)  # (B, 4, N)
    pts_world = ref_c2w.float() @ pts_h                                # (B, 4, N)

    val = ref_latent.float().reshape(B, C, N)   # source latent values

    fx = K[:, 0, 0].unsqueeze(1)   # (B, 1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    warped_list, mask_list = [], []
    for t in range(T):
        tgt_w2c = torch.linalg.inv(tgt_c2ws[:, t].float())   # (B, 4, 4)
        pts_tgt = tgt_w2c @ pts_world                         # (B, 4, N)
        z = pts_tgt[:, 2].clamp(min=1e-4)                    # (B, N)
        u = fx * pts_tgt[:, 0] / z + cx                      # (B, N)
        v = fy * pts_tgt[:, 1] / z + cy                      # (B, N)

        warped_t = torch.zeros(B, C, H, W, device=device, dtype=torch.float32)
        weight_t = torch.zeros(B, 1, H, W, device=device, dtype=torch.float32)

        u0  = u.floor().long()
        v0  = v.floor().long()
        wu1 = (u - u0.float()).clamp(0, 1)
        wv1 = (v - v0.float()).clamp(0, 1)
        wu0 = 1 - wu1
        wv0 = 1 - wv1

        in_front = pts_tgt[:, 2] > 1e-4   # (B, N)
        for (ui, vi, wu, wv) in [(u0,   v0,   wu0, wv0),
                                  (u0+1, v0,   wu1, wv0),
                                  (u0,   v0+1, wu0, wv1),
                                  (u0+1, v0+1, wu1, wv1)]:
            in_bounds = in_front & (ui >= 0) & (ui < W) & (vi >= 0) & (vi < H)
            w_val = wu * wv   # (B, N)
            for b in range(B):
                m = in_bounds[b]
                if not m.any():
                    continue
                idx = (vi[b][m] * W + ui[b][m]).long()   # (M,)
                wb  = w_val[b][m]                         # (M,)
                vb  = val[b][:, m]                        # (C, M)
                warped_t[b].reshape(C, -1).scatter_add_(
                    1, idx.unsqueeze(0).expand(C, -1), vb * wb.unsqueeze(0))
                weight_t[b, 0].flatten().scatter_add_(0, idx, wb)

        valid = weight_t > 0.5
        warped_t = warped_t / weight_t.clamp(min=1e-8) * valid.float()

        warped_list.append(warped_t.to(ref_latent.dtype))
        mask_list.append(valid.float().to(ref_latent.dtype))

    return torch.stack(warped_list, dim=1), torch.stack(mask_list, dim=1)


def forward_warp_latent_zbuffer(
    ref_latent: torch.Tensor,   # (B, C, H, W)
    ref_depth:  torch.Tensor,   # (B, 1, H, W)
    ref_c2w:    torch.Tensor,   # (B, 4, 4)
    tgt_c2ws:   torch.Tensor,   # (B, T, 4, 4)
    K_lat:      torch.Tensor,   # (B, 3, 3)
    use_splat:  bool = False,
) -> tuple:
    """
    Forward warp with z-buffer: nearest depth wins at each target pixel.

    Replaces scatter_add_ averaging with depth-based occlusion handling -
    only the closest splatted value is kept per pixel, avoiding blur at
    occlusion boundaries.

    Returns:
        warped  (B, T, C, H, W)
        mask    (B, T, 1, H, W)  - 1 where target pixel received a valid splat
    """
    B, C, H, W = ref_latent.shape
    T = tgt_c2ws.shape[1]
    device = ref_latent.device
    N = H * W

    # Unproject source pixels to 3D world points
    ys = torch.arange(H, device=device, dtype=torch.float32)
    xs = torch.arange(W, device=device, dtype=torch.float32)
    gy, gx = torch.meshgrid(ys, xs, indexing='ij')
    pix = torch.stack([gx.flatten(), gy.flatten(),
                       torch.ones(N, device=device)], dim=0)           # (3, N)
    pix = pix.unsqueeze(0).expand(B, -1, -1)                           # (B, 3, N)

    K     = K_lat.float()
    K_inv = torch.linalg.inv(K)
    d     = ref_depth.view(B, 1, N).clamp(min=1e-4)
    pts   = (K_inv @ pix) * d
    pts_h = torch.cat([pts, torch.ones(B, 1, N, device=device)], 1)
    pts_world = ref_c2w.float() @ pts_h                                # (B, 4, N)

    val = ref_latent.float().reshape(B, C, N)

    fx = K[:, 0, 0].unsqueeze(1)
    fy = K[:, 1, 1].unsqueeze(1)
    cx = K[:, 0, 2].unsqueeze(1)
    cy = K[:, 1, 2].unsqueeze(1)

    warped_list, mask_list = [], []
    INF = 1e9

    for t in range(T):
        tgt_w2c = torch.linalg.inv(tgt_c2ws[:, t].float())
        pts_tgt = tgt_w2c @ pts_world
        z = pts_tgt[:, 2].clamp(min=1e-4)                             # (B, N)
        u = fx * pts_tgt[:, 0] / z + cx                               # (B, N)
        v = fy * pts_tgt[:, 1] / z + cy

        warped_t  = torch.zeros(B, C, H * W, device=device, dtype=torch.float32)
        zbuf      = torch.full((B, H * W), INF, device=device, dtype=torch.float32)
        in_front  = pts_tgt[:, 2] > 1e-4

        if not use_splat:
            u0 = u.round().long()
            v0 = v.round().long()
            for b in range(B):
                in_bounds = in_front[b] & (u0[b] >= 0) & (u0[b] < W) & \
                            (v0[b] >= 0) & (v0[b] < H)
                if not in_bounds.any():
                    continue
                idx   = (v0[b][in_bounds] * W + u0[b][in_bounds]).long()
                z_b   = z[b][in_bounds]
                val_b = val[b][:, in_bounds]
                order_rev = z_b.argsort().flip(0)
                idx_r = idx[order_rev]
                z_r   = z_b[order_rev]
                val_r = val_b[:, order_rev]
                zbuf[b].scatter_(0, idx_r, z_r)
                warped_t[b].scatter_(1, idx_r.unsqueeze(0).expand(C, -1), val_r)
        else:
            # Bilinear splatting: distribute each source pixel to 4 neighbors
            weight_t = torch.zeros(B, 1, H * W, device=device, dtype=torch.float32)
            u0f = u.floor().long()
            v0f = v.floor().long()
            wu1 = (u - u0f.float()).clamp(0, 1)
            wv1 = (v - v0f.float()).clamp(0, 1)
            wu0, wv0 = 1 - wu1, 1 - wv1

            for b in range(B):
                for (du, dv, wgt) in [(0,0,wu0*wv0),(1,0,wu1*wv0),(0,1,wu0*wv1),(1,1,wu1*wv1)]:
                    uc = u0f[b] + du
                    vc = v0f[b] + dv
                    in_b = in_front[b] & (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
                    if not in_b.any():
                        continue
                    idx_b = (vc[in_b] * W + uc[in_b]).long()
                    w_b   = wgt[b][in_b].unsqueeze(0)         # (1, M)
                    weight_t[b].scatter_add_(1, idx_b.unsqueeze(0), w_b)
                    warped_t[b].scatter_add_(1, idx_b.unsqueeze(0).expand(C, -1),
                                             val[b][:, in_b] * w_b)

            has_w = (weight_t > 0).expand(B, C, H * W)
            denom = weight_t.clamp(min=1e-6).expand(B, C, H * W)
            warped_t = torch.where(has_w, warped_t / denom, warped_t)
            zbuf = torch.where(weight_t.squeeze(1) > 0,
                               torch.zeros_like(zbuf), zbuf)           # mark valid pixels

        valid_mask = zbuf < INF                                        # (B, H*W)
        warped_t = warped_t * valid_mask.unsqueeze(1).float()
        warped_t = warped_t.reshape(B, C, H, W)
        valid_2d = valid_mask.reshape(B, 1, H, W).float()

        warped_list.append(warped_t.to(ref_latent.dtype))
        mask_list.append(valid_2d.to(ref_latent.dtype))

    return torch.stack(warped_list, dim=1), torch.stack(mask_list, dim=1)
