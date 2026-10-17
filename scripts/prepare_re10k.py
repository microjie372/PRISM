# SPDX-FileCopyrightText: Copyright (c) 2026 The PRISM Authors. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Expand pixelSplat's RE10K chunks into the per-scene layout the evaluation
reads: one directory per scene holding the frames as PNG plus cameras.pt.

Usage:
    python scripts/prepare_re10k.py \
        --chunks_dir /path/to/re10k/test \
        --output_dir ./datasets/re10k/test
"""

import argparse
import io
from pathlib import Path

import torch
from PIL import Image
from tqdm import tqdm


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--chunks_dir", type=str, required=True,
                   help="Directory of pixelSplat .torch chunks (a split directory).")
    p.add_argument("--output_dir", type=str, required=True)
    p.add_argument("--num_scenes", type=int, default=None,
                   help="Stop after this many scenes. Default: all.")
    p.add_argument("--overwrite", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    chunks = sorted(Path(args.chunks_dir).glob("*.torch"))
    assert chunks, f"No .torch chunks found in {args.chunks_dir}"
    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    written = 0
    for chunk_path in tqdm(chunks, desc="chunks"):
        for example in torch.load(chunk_path, map_location="cpu", weights_only=True):
            if args.num_scenes is not None and written >= args.num_scenes:
                print(f"Wrote {written} scenes to {out_root}")
                return

            scene_dir = out_root / str(example["key"])
            # cameras.pt is written last, so its presence means the scene is
            # complete; a directory alone may be a half-finished run.
            if (scene_dir / "cameras.pt").exists() and not args.overwrite:
                written += 1
                continue
            scene_dir.mkdir(parents=True, exist_ok=True)
            for stale in scene_dir.glob("frame_*.png"):
                stale.unlink()

            # Frames are stored as encoded bytes; name them so a plain
            # filename sort reproduces the capture order.
            for i, raw in enumerate(example["images"]):
                img = Image.open(io.BytesIO(raw.numpy().tobytes())).convert("RGB")
                img.save(scene_dir / f"frame_{i:05d}.png")

            # (N, 18): normalised fx, fy, cx, cy, two unused columns, then a
            # row-major 3x4 world-to-camera matrix.
            cameras = torch.stack(list(example["cameras"])).float()
            assert cameras.shape[1] == 18, f"Expected 18 camera columns, got {cameras.shape}"
            assert cameras.shape[0] == len(example["images"]), (
                f"{scene_dir.name}: {len(example['images'])} frames but "
                f"{cameras.shape[0]} cameras in the chunk."
            )
            torch.save(cameras, scene_dir / "cameras.pt")

            written += 1

    print(f"Wrote {written} scenes to {out_root}")


if __name__ == "__main__":
    main()
