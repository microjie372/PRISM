# Setup

## Environment and weights

PRISM runs inside Lyra's environment and uses its frozen decoder. Set up both
the environment and the Lyra and Cosmos weights from Lyra's own repository:

> https://github.com/nv-tlabs/lyra (see `Lyra-1/INSTALL.md`)

The Lyra decoder is released under the NVIDIA Open Model License, not Apache
2.0. See [licence](licence.md).

```bash
git clone https://github.com/microjie372/PRISM.git
cd PRISM
export PYTHONPATH=$(pwd)
```

The configs read the weights from `checkpoints/`:

```
checkpoints/
  Lyra/lyra_static.pt                             from Lyra
  cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p/  from Lyra
  prism/prism_stage2.pt                           PRISM's encoder
```

PRISM's encoder weights: [huggingface.co/microjet378/PRISM](https://huggingface.co/microjet378/PRISM).

Our results were produced with Python 3.10, torch 2.6.0+cu124 and
flash-attn 2.7.4.post1 on a single A6000.

## Training data

Lyra's training data: synthetic scenes from text prompts through FLUX and
Gen3C, with no real images or multi-view captures at any stage.

```bash
huggingface-cli download nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG \
    --repo-type dataset --local-dir lyra_dataset/tar
```

Untar it, then point `dataset_registry['lyra_static']` in
[`src/models/data/registry.py`](../src/models/data/registry.py) at the result.

## Evaluation data

All three benchmarks are out-of-distribution; none is seen during training.

### RealEstate10K

Start from pixelSplat's processed release, which is also what pixelSplat,
MVSplat and NAS3R evaluate on:

> https://github.com/dcharatan/pixelsplat#acquiring-datasets

Then expand its chunks into one directory per scene:

```bash
python scripts/prepare_re10k.py \
    --chunks_dir /path/to/pixelsplat/re10k/test \
    --output_dir ./datasets/re10k/test
```

This writes `frame_00000.png ...` plus a `cameras.pt` per scene, an `(N, 18)`
tensor in pixelSplat's column layout: normalised `fx, fy, cx, cy`, two unused
columns, then a row-major 3x4 world-to-camera matrix, in any world frame.

### DL3DV

Download [DL3DV-10K](https://github.com/DL3DV-10K/Dataset). Each scene needs a
`transforms.json` with the frames beside it; the evaluation looks for it under
`nerfstudio/`, then `colmap/`, then the scene root, and prefers a quarter
resolution `images_4/` when one is present.

### Tanks-and-Temples

The evaluation reads a COLMAP sparse reconstruction: each scene needs
`images/` beside `sparse/0/cameras.bin` and `sparse/0/images.bin`. The
[NoPe-NeRF](https://github.com/ActiveVisionLab/nope-nerf) release provides the
eight scenes and their frames; if your copy ships poses in another form, run
COLMAP over `images/` to produce `sparse/0/`.

### Structure

```
PRISM/
  checkpoints/                      weights, from Lyra plus PRISM's encoder
    Lyra/lyra_static.pt
    cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p/
    prism/prism_stage2.pt

  datasets/                         you create this

    re10k/                          --dataset_dir datasets/re10k/test
      test/
        000c3ab189999a83/           one directory per scene
          frame_00000.png           frames, any name that sorts correctly
          frame_00001.png
          ...
          cameras.pt                (N, 18) tensor, N == number of frames
        00262dd0d9b66d4d/
        ...

    dl3dv/                          --dataset_dir datasets/dl3dv
      1K1_fd80f388.../
        nerfstudio/                 or colmap/, or the scene root
          transforms.json           w, h, fl_x, fl_y, cx, cy, frames
          images_4/                 preferred; images/ also works
            frame_00001.png
            ...
      ...

    tanks/                          --dataset_dir datasets/tanks
      Ballroom/
        images/                     names must match those in images.bin
          00000.jpg
          ...
        sparse/0/
          cameras.bin               COLMAP sparse reconstruction
          images.bin
      Barn/
      ...                           8 scenes

    lyra_static/                    training data; --dataset_root datasets/lyra_static
      0/                            one directory per trajectory, 0 to 5
        rgb/<scene_id>.mp4
        latent/<scene_id>.pkl       Gen3C latents, the training supervision
        pose/<scene_id>.npz
        intrinsics/<scene_id>.npz
        caption/<scene_id>.txt
      1/
      ...                           through 5
```

Either directory may live elsewhere: pass a different `--dataset_dir`, or
symlink.
