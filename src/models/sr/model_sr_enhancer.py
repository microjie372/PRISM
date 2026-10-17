# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Lightweight residual image enhancement network (SR Enhancer).

Input:  blurry RGB rendered by LatentRecon (B, 3, H, W), range [0, 1]
Output: perceptually enhanced RGB (B, 3, H, W), range [0, 1]

Architecture: residual CNN that learns a delta on top of the blurry input.
No upscaling — same resolution, perceptual quality enhancement only.
"""

import torch
import torch.nn as nn


class ResBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1, bias=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.block(x)


class SREnhancer(nn.Module):
    """
    Residual perceptual enhancement network.

    Args:
        channels:   intermediate feature width (default 64, ~6M params)
        num_blocks: number of residual blocks (default 8)

    At init, tail weights are zeroed so the network is an identity map —
    training starts from the blurry input baseline.
    """

    def __init__(self, channels: int = 64, num_blocks: int = 8):
        super().__init__()
        self.head = nn.Conv2d(3, channels, 3, 1, 1, bias=True)
        self.body = nn.Sequential(*[ResBlock(channels) for _ in range(num_blocks)])
        self.tail = nn.Conv2d(channels, 3, 3, 1, 1, bias=True)

        nn.init.zeros_(self.tail.weight)
        nn.init.zeros_(self.tail.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, 3, H, W) in [0, 1]  →  (B, 3, H, W) in [0, 1]"""
        feat = self.head(x)
        feat = self.body(feat)
        residual = self.tail(feat)
        return (x + residual).clamp(0.0, 1.0)
