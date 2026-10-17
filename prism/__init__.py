# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
PRISM: feed-forward single-image 3D reconstruction via geometric warp-residual
modelling.

The package holds the method's own code. The frozen 3DGS decoder it predicts
latents for, and the data stack it reads, come from Lyra under `src/`.
"""

from prism.encoder import WarpResidualEncoder

__all__ = ["WarpResidualEncoder"]
