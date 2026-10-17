# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
The residual encoder: a ~26M ViT predicting a correction on top of the
geometric warp prior, for every (view, frame) latent.

Factored attention, repeating every four blocks:
PerView over space, CrossView over the 6 views, PerView, Temporal over the
16 frames.
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


# ---------------------------------------------------------------------------
# Forward warp utilities
# ---------------------------------------------------------------------------

def depth_forward_warp(
    src_feat: torch.Tensor,
    src_depth: torch.Tensor,
    src_c2w: torch.Tensor,
    src_K: torch.Tensor,
    tgt_c2w: torch.Tensor,
    tgt_K: torch.Tensor,
    latent_scale: int = 8,
    use_splat: bool = False,
) -> tuple:
    """Forward-warp src_feat to a single target view. Returns (warped, mask).

    use_splat=False (default): nearest-neighbor scatter with z-sort occlusion.
    use_splat=True:            bilinear splatting - sub-pixel accuracy, smoother
                               output but no explicit occlusion handling.
    """
    while src_depth.dim() > 3:
        src_depth = src_depth[:, 0]
    if src_feat.dim()  == 3: src_feat  = src_feat.unsqueeze(0)
    if src_depth.dim() == 2: src_depth = src_depth.unsqueeze(0)
    if src_c2w.dim()   == 2: src_c2w   = src_c2w.unsqueeze(0)
    if src_K.dim()     == 2: src_K     = src_K.unsqueeze(0)
    if tgt_c2w.dim()   == 2: tgt_c2w   = tgt_c2w.unsqueeze(0)
    if tgt_K.dim()     == 2: tgt_K     = tgt_K.unsqueeze(0)

    B, C, H_lat, W_lat = src_feat.shape
    H, W = src_depth.shape[-2:]
    device = src_feat.device

    ys = torch.arange(H_lat, device=device, dtype=torch.float32)
    xs = torch.arange(W_lat, device=device, dtype=torch.float32)
    py = (ys + 0.5) * latent_scale - 0.5
    px = (xs + 0.5) * latent_scale - 0.5
    py_grid, px_grid = torch.meshgrid(py, px, indexing='ij')
    py_grid = py_grid.unsqueeze(0).expand(B, -1, -1)
    px_grid = px_grid.unsqueeze(0).expand(B, -1, -1)

    norm_x = (px_grid / (W - 1)) * 2 - 1
    norm_y = (py_grid / (H - 1)) * 2 - 1
    depth_grid = torch.stack([norm_x, norm_y], dim=-1)
    depth_lat  = F.grid_sample(src_depth.float().unsqueeze(1), depth_grid,
                               mode='bilinear', align_corners=True,
                               padding_mode='border')[:, 0]

    fx = src_K[:, 0, 0].view(B, 1, 1); fy = src_K[:, 1, 1].view(B, 1, 1)
    cx = src_K[:, 0, 2].view(B, 1, 1); cy = src_K[:, 1, 2].view(B, 1, 1)
    x_cam = (px_grid - cx) / fx * depth_lat
    y_cam = (py_grid - cy) / fy * depth_lat
    z_cam = depth_lat

    N = H_lat * W_lat
    ones  = torch.ones_like(z_cam)
    pts   = torch.stack([x_cam, y_cam, z_cam, ones], dim=1).view(B, 4, N)
    pts_w = torch.bmm(src_c2w.float(), pts)
    pts_t = torch.bmm(torch.linalg.inv(tgt_c2w.float()), pts_w)

    fx_t = tgt_K[:, 0, 0].view(B, 1); fy_t = tgt_K[:, 1, 1].view(B, 1)
    cx_t = tgt_K[:, 0, 2].view(B, 1); cy_t = tgt_K[:, 1, 2].view(B, 1)
    z_t  = pts_t[:, 2].clamp(min=1e-4)
    u_lat = ((fx_t * pts_t[:, 0] / z_t + cx_t) / latent_scale).round().long()
    v_lat = ((fy_t * pts_t[:, 1] / z_t + cy_t) / latent_scale).round().long()

    valid = (pts_t[:, 2] > 1e-4) & (u_lat >= 0) & (u_lat < W_lat) & \
            (v_lat >= 0) & (v_lat < H_lat)

    warped = torch.zeros(B, C, H_lat, W_lat, device=device, dtype=src_feat.dtype)
    mask   = torch.zeros(B, 1, H_lat, W_lat, device=device, dtype=src_feat.dtype)

    batch_offset = (torch.arange(B, device=device) * (H_lat * W_lat)).unsqueeze(1)
    flat_global  = (v_lat * W_lat + u_lat) + batch_offset
    flat_global  = flat_global.clamp(0, B * H_lat * W_lat - 1)

    z_all   = z_t.reshape(B * N)
    val_all = valid.reshape(B * N)
    fg_all  = flat_global.reshape(B * N)
    src_all = rearrange(src_feat, 'b c h w -> (b h w) c')

    valid_idx = val_all.nonzero(as_tuple=False).squeeze(1)
    if valid_idx.numel() > 0:
        order      = torch.argsort(z_all[valid_idx], descending=True)
        sorted_idx = valid_idx[order]
        warped.view(B * H_lat * W_lat, C).index_put_(
            (fg_all[sorted_idx],), src_all[sorted_idx].to(src_feat.dtype), accumulate=False)
        mask.view(B * H_lat * W_lat, 1).index_put_(
            (fg_all[sorted_idx],),
            torch.ones(sorted_idx.shape[0], 1, device=device, dtype=src_feat.dtype),
            accumulate=False)

    if not use_splat:
        return warped, mask

    # ---- Bilinear splatting (replaces nearest-neighbor result above) ----
    u_f = (fx_t * pts_t[:, 0] / z_t + cx_t) / latent_scale  # (B, N) float
    v_f = (fy_t * pts_t[:, 1] / z_t + cy_t) / latent_scale

    valid_s  = (pts_t[:, 2] > 1e-4) & (u_f >= -1) & (u_f < W_lat + 1) & \
               (v_f >= -1) & (v_f < H_lat + 1)
    val_flat = valid_s.reshape(B * N)

    u0   = u_f.floor().long()
    v0   = v_f.floor().long()
    wu1  = (u_f - u0.float()).clamp(0, 1)
    wv1  = (v_f - v0.float()).clamp(0, 1)
    wu0, wv0 = 1 - wu1, 1 - wv1

    warped_acc = torch.zeros(B * H_lat * W_lat, C, device=device, dtype=torch.float32)
    weight_acc = torch.zeros(B * H_lat * W_lat, 1, device=device, dtype=torch.float32)
    src_f      = rearrange(src_feat.float(), 'b c h w -> (b h w) c')  # (B*N, C)
    batch_off  = (torch.arange(B, device=device) * (H_lat * W_lat)).unsqueeze(1)

    for (du, dv, wgt) in [(0,0,wu0*wv0),(1,0,wu1*wv0),(0,1,wu0*wv1),(1,1,wu1*wv1)]:
        uc      = (u0 + du).clamp(0, W_lat - 1)
        vc      = (v0 + dv).clamp(0, H_lat - 1)
        in_b    = (u0+du >= 0) & (u0+du < W_lat) & (v0+dv >= 0) & (v0+dv < H_lat)
        fg_c    = (vc * W_lat + uc + batch_off).reshape(B * N)
        w_c     = wgt.reshape(B * N)
        idx     = (val_flat & in_b.reshape(B * N)).nonzero(as_tuple=False).squeeze(1)
        if idx.numel() > 0:
            weight_acc.scatter_add_(0, fg_c[idx].unsqueeze(1),
                                    w_c[idx].unsqueeze(1).float())
            warped_acc.scatter_add_(0, fg_c[idx].unsqueeze(1).expand(-1, C),
                                    src_f[idx] * w_c[idx].unsqueeze(1))

    has_w  = weight_acc.squeeze(1) > 0
    warped_acc[has_w] = warped_acc[has_w] / weight_acc[has_w]
    warped = warped_acc.reshape(B, H_lat, W_lat, C).permute(0, 3, 1, 2).to(src_feat.dtype)
    mask   = has_w.reshape(B, H_lat, W_lat).unsqueeze(1).to(src_feat.dtype)
    return warped, mask


# ---------------------------------------------------------------------------
# Transformer building blocks
# ---------------------------------------------------------------------------

def _make_ff(d_model: int, dim_feedforward: int, drop_rate: float = 0.0):
    """Build FF block, omitting Dropout layers when drop_rate == 0 for checkpoint compat."""
    if drop_rate > 0:
        return nn.Sequential(
            nn.Linear(d_model, dim_feedforward), nn.GELU(),
            nn.Dropout(drop_rate),
            nn.Linear(dim_feedforward, d_model),
            nn.Dropout(drop_rate),
        )
    return nn.Sequential(
        nn.Linear(d_model, dim_feedforward), nn.GELU(),
        nn.Linear(dim_feedforward, d_model),
    )


class PerViewBlock(nn.Module):
    """Spatial self-attention over N tokens, independently per (B*V*T) position."""

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=drop_rate)
        self.ff    = _make_ff(d_model, dim_feedforward, drop_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*V*T, N, D)"""
        h = self.norm1(x)
        x = x + self.attn(h, h, h, need_weights=False)[0]
        x = x + self.ff(self.norm2(x))
        return x


class CrossViewBlock(nn.Module):
    """
    Cross-view attention: at each (B*T*N) position, attend over V views.
    Couples views at the same temporal step for spatial consistency.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=drop_rate)
        self.ff    = _make_ff(d_model, dim_feedforward, drop_rate)

    def forward(self, x: torch.Tensor, V: int, T: int) -> torch.Tensor:
        """x: (B*V*T, N, D) -> attend over V at each (B*T*N) -> (B*V*T, N, D)"""
        BVT, N, D = x.shape
        B = BVT // (V * T)
        x_cv = rearrange(x, '(b v t) n d -> (b t n) v d', b=B, v=V, t=T)
        h    = self.norm1(x_cv)
        x_cv = x_cv + self.attn(h, h, h, need_weights=False)[0]
        x_cv = x_cv + self.ff(self.norm2(x_cv))
        return rearrange(x_cv, '(b t n) v d -> (b v t) n d', b=B, t=T, n=N)


class TemporalBlock(nn.Module):
    """
    Temporal self-attention: at each (B*V*N) position, attend over T frames.
    Couples temporal frames within each view for temporal coherence.
    """

    def __init__(self, d_model: int, nhead: int, dim_feedforward: int, drop_rate: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.attn  = nn.MultiheadAttention(d_model, nhead, batch_first=True, dropout=drop_rate)
        self.ff    = _make_ff(d_model, dim_feedforward, drop_rate)

    def forward(self, x: torch.Tensor, V: int, T: int) -> torch.Tensor:
        """x: (B*V*T, N, D) -> attend over T at each (B*V*N) -> (B*V*T, N, D)"""
        BVT, N, D = x.shape
        B = BVT // (V * T)
        x_t = rearrange(x, '(b v t) n d -> (b v n) t d', b=B, v=V, t=T)
        h   = self.norm1(x_t)
        x_t = x_t + self.attn(h, h, h, need_weights=False)[0]
        x_t = x_t + self.ff(self.norm2(x_t))
        return rearrange(x_t, '(b v n) t d -> (b v t) n d', b=B, v=V, n=N)


# ---------------------------------------------------------------------------
# FlowHead
# ---------------------------------------------------------------------------

class FlowHead(nn.Module):
    """
    Predicts a 2D flow field that realigns the blended base before the ViT
    reads it, absorbing the misalignment left by imperfect depth.

    (B*V*T, in_ch, H, W) -> (B*V*T, 2, H, W), displacement bounded to
    +/-max_flow latent pixels.
    """

    def __init__(self, in_ch: int, hidden: int = 64, max_flow: float = 4.0):
        super().__init__()
        self.max_flow = max_flow
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, hidden, 3, padding=1), nn.GELU(),
            nn.Conv2d(hidden, 2, 1),
        )
        # Zero-init the last conv so the flow starts at identity.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B*V*T, in_ch, H, W) -> flow: (B*V*T, 2, H, W)"""
        return self.net(x).tanh() * self.max_flow


# ---------------------------------------------------------------------------
# WarpResidualEncoder
# ---------------------------------------------------------------------------

class WarpResidualEncoder(nn.Module):
    """
    Lightweight ViT (~26M params) predicting delta latents for all (V, T) positions.

    Input:
        warp_latents  (B, V, T, C, H, W)  - geometric warp prior from precompute
        warp_mask     (B, V, T, 1, H, W)  - 1=valid warp coverage, 0=hole
        ref_latent    (B, C, H, W)         - anchor: view0 T=0
        pluckers      (B, V, T, C_p, H, W) - camera pose per position

    Output:
        delta  (B, V, T, C, H, W), added to the blended base by the caller.
    """

    def __init__(
        self,
        latent_channels:    int   = 16,
        plucker_channels:   int   = 32,
        embed_dim:          int   = 512,
        depth:              int   = 8,
        num_heads:          int   = 8,
        mlp_ratio:          float = 4.0,
        patch_size:         int   = 4,
        use_flow_correction: bool  = True,
        flow_hidden:         int   = 64,
        flow_max_flow:       float = 4.0,
        drop_rate:           float = 0.0,
    ):
        super().__init__()
        self.latent_channels  = latent_channels
        self.plucker_channels = plucker_channels
        self.embed_dim        = embed_dim
        self.patch_size       = patch_size

        # in_channels: warp(C) + mask(1) + ref(C) + plucker(C_p)
        in_ch = latent_channels + 1 + latent_channels + plucker_channels  # = 65
        self.patch_embed = nn.Conv2d(
            in_ch, embed_dim, kernel_size=patch_size, stride=patch_size, bias=True,
        )

        # Positional embedding: 22x40 = 880 tokens for 88x160 latent with patch=4
        self.pos_embed = nn.Parameter(torch.zeros(1, 22 * 40, embed_dim))

        # Block schedule: PerView -> CrossView -> PerView -> Temporal (x2 for depth=8)
        dim_ff = int(embed_dim * mlp_ratio)
        self.blocks = nn.ModuleList()
        for i in range(depth):
            mod = i % 4
            if mod == 0 or mod == 2:
                self.blocks.append(PerViewBlock(embed_dim, num_heads, dim_ff, drop_rate))
            elif mod == 1:
                self.blocks.append(CrossViewBlock(embed_dim, num_heads, dim_ff, drop_rate))
            else:  # mod == 3
                self.blocks.append(TemporalBlock(embed_dim, num_heads, dim_ff, drop_rate))

        self.norm = nn.LayerNorm(embed_dim)

        # Head: trunc_normal(std=1e-3) -> near-zero delta at init, gradient flows from step 1
        self.head = nn.Linear(embed_dim, patch_size * patch_size * latent_channels)
        nn.init.trunc_normal_(self.head.weight, std=1e-3)
        nn.init.zeros_(self.head.bias)

        self.use_flow_correction = use_flow_correction
        if use_flow_correction:
            self.flow_head = FlowHead(in_ch, hidden=flow_hidden, max_flow=flow_max_flow)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.patch_embed.weight, std=0.02)
        nn.init.zeros_(self.patch_embed.bias)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def _get_pos_embed(self, h_tok: int, w_tok: int) -> torch.Tensor:
        N = h_tok * w_tok
        if N == self.pos_embed.shape[1]:
            return self.pos_embed
        h0, w0 = 22, 40
        pos = self.pos_embed.reshape(1, h0, w0, self.embed_dim).permute(0, 3, 1, 2)
        pos = F.interpolate(pos, size=(h_tok, w_tok), mode='bilinear', align_corners=False)
        return pos.permute(0, 2, 3, 1).reshape(1, N, self.embed_dim)

    @staticmethod
    def _apply_flow(feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        """
        Resample feat by a 2D flow field.

        Args:
            feat: (N, C, H, W)  - tensor to be spatially resampled
            flow: (N, 2, H, W)  - (dx, dy) displacement in latent pixel units
                                   flow[:,0] = horizontal shift (-> positive)
                                   flow[:,1] = vertical   shift (down positive)
        Returns:
            warped: (N, C, H, W)  - same dtype as feat

        The sampling location for output pixel (x, y) is (x + dx, y + dy) in feat.
        Boundary pixels are replicated (padding_mode='border') to avoid
        black borders at image edges common with zero-padding.
        """
        N, C, H, W = feat.shape
        # Build base identity grid in pixel coordinates
        ys = torch.arange(H, device=flow.device, dtype=torch.float32)
        xs = torch.arange(W, device=flow.device, dtype=torch.float32)
        grid_y, grid_x = torch.meshgrid(ys, xs, indexing='ij')  # (H, W) each

        # Add flow displacement: where to sample FROM in the source (blended_base)
        sample_x = grid_x.unsqueeze(0) + flow[:, 0].float()  # (N, H, W)
        sample_y = grid_y.unsqueeze(0) + flow[:, 1].float()  # (N, H, W)

        # Normalize pixel coordinates to [-1, 1] for grid_sample
        sample_x = sample_x / (W - 1) * 2 - 1
        sample_y = sample_y / (H - 1) * 2 - 1

        grid = torch.stack([sample_x, sample_y], dim=-1)  # (N, H, W, 2)
        return F.grid_sample(
            feat.float(), grid,
            mode='bilinear', align_corners=True, padding_mode='border',
        ).to(feat.dtype)

    def forward(
        self,
        warp_latents: torch.Tensor,
        warp_mask:    torch.Tensor,
        ref_latent:   torch.Tensor,
        pluckers:     torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            warp_latents: (B, V, T, C, H, W)
            warp_mask:    (B, V, T, 1, H, W)
            ref_latent:   (B, C, H, W)
            pluckers:     (B, V, T, C_p, H, W)
        Returns:
            delta: (B, V, T, C, H, W). The caller always adds this to the
            blended base, whether or not flow correction is enabled.
        """
        B, V, T, C, H, W = warp_latents.shape
        p     = self.patch_size
        h_tok = H // p
        w_tok = W // p

        # Expand ref_latent to all (V, T) positions
        ref_exp = ref_latent[:, None, None].expand(B, V, T, C, H, W)

        # Concatenate input modalities: (B, V, T, in_ch, H, W)
        # warp + mask + ref + plucker - same channel layout as v1
        inp = torch.cat([warp_latents, warp_mask, ref_exp, pluckers], dim=3)
        inp = inp.reshape(B * V * T, -1, H, W)           # (B*V*T, in_ch, H, W)

        # Realign the blended base before the ViT reads it.
        if self.use_flow_correction:
            blended_base_5d   = warp_latents * warp_mask + ref_exp * (1 - warp_mask)
            blended_base_flat = blended_base_5d.reshape(B * V * T, C, H, W)

            flow = self.flow_head(inp)                    # (B*V*T, 2, H, W)
            corrected_base_flat = self._apply_flow(blended_base_flat, flow)

            inp_vit = torch.cat([
                corrected_base_flat,                                # (B*V*T, C,   H, W)
                warp_mask.reshape(B * V * T, 1, H, W),             # (B*V*T, 1,   H, W)
                ref_exp.reshape(B * V * T, C, H, W),               # (B*V*T, C,   H, W)
                pluckers.reshape(B * V * T, -1, H, W),             # (B*V*T, C_p, H, W)
            ], dim=1)
        else:
            inp_vit = inp

        # Patch embedding + positional encoding
        tokens = self.patch_embed(inp_vit)                # (B*V*T, D, h_tok, w_tok)
        tokens = rearrange(tokens, 'bvt d h w -> bvt (h w) d')
        pos    = self._get_pos_embed(h_tok, w_tok).to(tokens.dtype)
        tokens = tokens + pos                             # (B*V*T, N, D)

        # Transformer blocks: PerView / CrossView / Temporal
        for block in self.blocks:
            if isinstance(block, PerViewBlock):
                tokens = block(tokens)
            elif isinstance(block, CrossViewBlock):
                tokens = block(tokens, V, T)
            else:  # TemporalBlock
                tokens = block(tokens, V, T)

        tokens = self.norm(tokens)

        # Decode to residual delta latent patches
        delta_flat = self.head(tokens)                    # (B*V*T, N, p*p*C)
        delta_residual = rearrange(
            delta_flat,
            '(b v t) (h w) (p1 p2 c) -> b v t c (h p1) (w p2)',
            b=B, v=V, t=T, h=h_tok, w=w_tok, p1=p, p2=p, c=self.latent_channels,
        )

        # Fold the realignment into the returned delta, so the caller's
        # blended_base + delta still lands on the corrected base.
        if self.use_flow_correction:
            flow_correction = corrected_base_flat.reshape(B, V, T, C, H, W) - blended_base_5d
            return flow_correction + delta_residual       # (B, V, T, C, H, W)

        return delta_residual                              # (B, V, T, C, H, W)

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
