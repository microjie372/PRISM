# Evaluation

The reference frame is each clip's first frame, and the following 14 frames are
scored against the ground truth at 704x1280.

These datasets give camera poses only up to an unknown scale, so each scene's
monocular depth is aligned to its cameras from sparse optical flow before
warping.

## The three benchmarks

```bash
CKPT=checkpoints/prism/prism_stage2.pt

# RealEstate10K, 300 scenes
CUDA_VISIBLE_DEVICES=0 python eval/re10k.py \
    --dataset_dir datasets/re10k/test \
    --config configs/inference/prism.yaml --encoder_ckpt $CKPT \
    --output_dir outputs/eval/re10k --num_scenes 300

# DL3DV, 140 scenes
CUDA_VISIBLE_DEVICES=0 python eval/dl3dv.py \
    --dataset_dir datasets/dl3dv \
    --config configs/inference/prism.yaml --encoder_ckpt $CKPT \
    --output_dir outputs/eval/dl3dv --num_scenes 140 --start_frame 0

# Tanks-and-Temples, 8 scenes cut into 95 clips
CUDA_VISIBLE_DEVICES=0 python eval/tanks.py \
    --dataset_dir datasets/tanks \
    --config configs/inference/prism.yaml --encoder_ckpt $CKPT \
    --output_dir outputs/eval/tanks
```

Each run writes per-scene and aggregate PSNR, SSIM and LPIPS as JSON into
`--output_dir`, and resumes from it if interrupted. Add `--save_renders` to
keep the rendered frames.

Note: set `--stride n` on RE10K or T&T for a different frame sampling interval.

## A single image

```bash
CUDA_VISIBLE_DEVICES=0 python eval/single_image.py \
    --image path/to/image.png \
    --config configs/inference/prism.yaml \
    --multi_trajectory
```

`--multi_trajectory` renders all six trajectories; `--trajectory left` picks
one of `left`, `right`, `up`, `zoom_out`, `zoom_in`, `clockwise`.
