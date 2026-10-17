# Training

PRISM trains the residual encoder against Lyra's 3DGS decoder, which stays
frozen throughout. One A6000 is enough. Training needs Lyra's training data;
see [Setup](setup.md).

## 1. Precompute the warp prior

The prior depends only on depth and camera geometry, so it is computed once and
reused by every run.

```bash
CUDA_VISIBLE_DEVICES=0 python train/precompute_warp.py \
    --config configs/inference/prism.yaml
```

Note: set `dataset_root=/your/lyra_static` to read the dataset from outside the repo.

## 2. The three stages

Run them in order. Each starts from the previous stage's checkpoint through
`init_ckpt` in its config.

```bash
ACC="--config_file configs/accelerate/accelerate_config_single.yaml"

# Warm-up: latent regression, to start stage 1 from a stable point
CUDA_VISIBLE_DEVICES=0 accelerate launch $ACC train/stage0.py \
    --config configs/training/prism_init.yaml

# Stage 1: cosine feature distillation against the frozen decoder
CUDA_VISIBLE_DEVICES=0 accelerate launch $ACC train/stage1.py \
    --config configs/training/prism_stage1.yaml

# Stage 2: LPIPS with a dynamically weighted MSE term
CUDA_VISIBLE_DEVICES=0 accelerate launch $ACC train/stage2.py \
    --config configs/training/prism_stage2.yaml
```

|  | warm-up | stage 1 | stage 2 |
|---|---|---|---|
| config | `prism_init.yaml` | `prism_stage1.yaml` | `prism_stage2.yaml` |
| steps | 20000 | 5000 | 10000 |
| learning rate | 5e-4 | 1e-4 | 1e-4 |
| loss | latent MSE | cosine on decoder blocks 7 and 15 | LPIPS + dynamic MSE |

Checkpoints land in `<output_dir>/checkpoint-<step>/encoder.pt`, which is what
`--encoder_ckpt` takes at [evaluation](eval.md). Any config field can be
overridden on the command line; if you shorten a stage with `max_train_steps`,
point the next stage's `init_ckpt` at a step that stage actually wrote.
