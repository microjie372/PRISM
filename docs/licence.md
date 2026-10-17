# Licence and attribution

PRISM's own code is released under the Apache License, Version 2.0, the same
licence as the Lyra code it builds on. Read [`LICENSE`](../LICENSE) and
[`NOTICE`](../NOTICE) before use.

## Code

PRISM's contribution is the residual encoder and geometric warp prior under
`prism/`, the training entry points under `train/`, the benchmark scripts under
`eval/`, `scripts/prepare_re10k.py` and the documentation under `docs/`. These
are Copyright (c) 2026 The PRISM Authors.

The rest is carried over from [Lyra](https://github.com/nv-tlabs/lyra)
(Lyra-1), Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES, under Apache 2.0:
the 3DGS decoder, data stack and rendering code under `src/`, the vendored
NVIDIA Cosmos under `cosmos_predict1/`, the model configs under `configs/`
other than PRISM's own. `NOTICE` lists which of those files have been
modified; each modified file also carries the notice Apache 2.0 requires.

Lyra's environment files, checkpoint download scripts and own entry points are
not redistributed here. Obtain them from Lyra, as
[setup](setup.md) describes.

Third-party attributions for the vendored Cosmos code are in
[`ATTRIBUTIONS.md`](../ATTRIBUTIONS.md), carried over from Lyra unchanged.

## Weights

The weights do not all share the code's licence.

The frozen 3DGS decoder is the Lyra checkpoint, released under the
[NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license)
rather than Apache 2.0. It is not redistributed here; obtain it from
[NVIDIA](https://huggingface.co/nvidia/Lyra) under that licence. Running PRISM
means accepting those terms, since the decoder is required at inference.

PRISM's released encoder weights hold the residual encoder alone and carry no
NVIDIA weights.

## Data

Training uses the
[Lyra training data](https://huggingface.co/datasets/nvidia/PhysicalAI-SpatialIntelligence-Lyra-SDG),
released by NVIDIA under its own terms. The evaluation benchmarks
(RealEstate10K, DL3DV, Tanks-and-Temples) are not redistributed here and each
carries its own licence.

## Citing Lyra

Using PRISM means using Lyra's decoder, so cite Lyra alongside PRISM. PRISM's
own citation is in the [README](../README.md).

```bibtex
@inproceedings{bahmani2026lyra,
  title={Lyra: Generative 3D Scene Reconstruction via Video Diffusion Model Self-Distillation},
  author={Bahmani, Sherwin and Shen, Tianchang and Ren, Jiawei and Huang, Jiahui and Jiang, Yifeng and
          Turki, Haithem and Tagliasacchi, Andrea and Lindell, David B. and Gojcic, Zan and Fidler, Sanja and
          Ling, Huan and Gao, Jun and Ren, Xuanchi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026}
}
```

NVIDIA is not affiliated with, and does not endorse, this project.
