<div align="center">

# PRISM: Feed-Forward Single-Image 3D Reconstruction via Geometric Warp-Residual Modeling

[Paper](https://arxiv.org/abs/2606.25430)

</div>

Reconstructing 3D scenes from a single image is a fundamental challenge in
computer vision, with broad applications in virtual reality, robotics, and
content creation. Recent methods achieve outstanding performance by leveraging
camera-controlled video diffusion models, but rely on iterative diffusion
sampling, which greatly limits their practical deployment. We observe that
geometric forward warping alone can cover the majority of a target view directly
from the input image, with only a compact residual left for the encoder to
correct. Motivated by this observation, we propose PRISM, a feed-forward
framework that decomposes multi-view latent prediction into a parameter-free
geometric prior and a learned residual correction, with no diffusion sampling
required at inference. To enable generalization from purely synthetic training
data, we devise a two-stage training strategy combining latents supervised
distillation for geometric generalization and perceptual fine-tuning for
appearance quality optimization. Extensive experiments on three benchmarks
demonstrate that PRISM achieves competitive reconstruction quality compared with
diffusion-based methods, while reducing inference time dramatically to only 36
seconds per scene.

---

## Documentation

| | |
|---|---|
| [Setup](docs/setup.md) | environment, weights, dataset structure |
| [Training](docs/train.md) | the three stages and what each one optimises |
| [Evaluation](docs/eval.md) | the three benchmarks and the protocol |
| [Licence](docs/licence.md) | terms and attribution |

---

## Model Zoo

| Checkpoint | Trained on | Stages | Download |
|---|---|---|---|
| `prism_stage2.pt` | Lyra's static training data, 401 scenes | 20k warm-up, 5k distillation, 10k perceptual | TBD |

The checkpoint holds the residual encoder alone. The frozen 3DGS decoder is a
separate download from NVIDIA under its own model licence; see
[Setup](docs/setup.md).

---

## Quick start

### 1. Clone, then set up Lyra

```bash
git clone https://github.com/microjie372/PRISM.git
cd PRISM
```

PRISM runs inside Lyra's environment and uses its frozen decoder, so follow
Lyra's own instructions for both the environment and the weight downloads:

> https://github.com/nv-tlabs/lyra (see `Lyra-1/INSTALL.md`)

### 2. `Checkpoints Structure`

```
checkpoints/
  Lyra/lyra_static.pt                             from Lyra
  cosmos_predict1/Cosmos-Tokenize1-CV8x8x8-720p/  from Lyra
  prism/prism_stage2.pt                           PRISM's encoder
```

```bash
ln -s /path/to/lyra/checkpoints checkpoints
mkdir -p checkpoints/prism
ln -s /path/to/prism_stage2.pt checkpoints/prism/prism_stage2.pt
```

### 3. `Datasets Structure`

Download and convert each benchmark as [Setup](docs/setup.md) describes, which
also gives the full directory structure.

```bash
mkdir -p datasets
ln -s /path/to/re10k/test datasets/re10k/test
```

### 4. Environment

```bash
conda activate prism          # whatever you named Lyra's environment
export PYTHONPATH=$(pwd)
```

### 5. Run

```bash
CUDA_VISIBLE_DEVICES=0 python eval/re10k.py \
    --dataset_dir datasets/re10k/test \
    --config configs/inference/prism.yaml \
    --encoder_ckpt checkpoints/prism/prism_stage2.pt \
    --output_dir outputs/eval/re10k \
    --num_scenes 20
```

| | |
|---|---|
| [Evaluation](docs/eval.md) | all three benchmarks at full size, and a single image |
| [Training](docs/train.md) | the warp prior, then the three stages |

---

## Licence

PRISM's code is released under the Apache License 2.0, the same licence as the
Lyra code it builds on. Read [`LICENSE`](LICENSE), [`NOTICE`](NOTICE) and
[docs/licence.md](docs/licence.md) before use.

The frozen 3DGS decoder is the Lyra checkpoint, released under the NVIDIA Open
Model License rather than Apache 2.0, and is required at inference. It is not
redistributed here.

NVIDIA is not affiliated with, and does not endorse, this project.

---

## Citation

If you find PRISM useful for your research or applications, please consider
citing our paper.

```bibtex
@article{zheng2026prism,
  title={PRISM: Feed-Forward Single-Image 3D Reconstruction via Geometric Warp-Residual Modeling},
  author={Zheng, Zhijie and Xiang, Xinhao and Zhang, Jiawei},
  journal={arXiv preprint arXiv:2606.25430},
  year={2026}
}
```

---

## Acknowledgements

Built on [Lyra](https://github.com/nv-tlabs/lyra), whose 3DGS decoder PRISM
predicts latents for, and which supplies the training data and the data stack.
Lyra in turn builds on [NVIDIA Cosmos](https://github.com/NVIDIA/Cosmos) and
[GEN3C](https://github.com/nv-tlabs/GEN3C), with input images from
[FLUX](https://github.com/black-forest-labs/flux). Depth comes from
[MoGe](https://github.com/microsoft/MoGe). The evaluation protocol follows
[Wonderland](https://arxiv.org/abs/2412.12091) and
[Bolt3D](https://arxiv.org/abs/2503.14445).
