# ARCQ-ViT

Official implementation of **ARCQ: Adaptive Residual Codebook Quantization for Vision Transformers**.

ARCQ quantizes normalized group residuals instead of raw tensor values and adapts non-uniform codebooks at scheduled training stages.`r`n`r`nThis repository contains the ARCQ implementation only; reference methods and baseline implementations are not included.

## Features

- Group-residual quantization for weights and activations.
- Distribution-fitted non-uniform codebooks.
- Scheduled phase compilation with cached codebooks and thresholds.
- Soft-logit knowledge distillation for ARCQ students.
- Packed On-The-Fly inference with low-bit storage and Triton kernels.

The main experiments use W4A4 and W3A3.

## Structure

```text
configs/                                  # ARCQ example configurations
dcarcq.py                                 # Core ARCQ quantizer
methods/fair_qat_framework/               # Training and model integration
methods/fair_qat_framework/quant/         # ARCQ quantization modules
methods/fair_qat_framework/fair_qat/      # ARCQ backends and inference
scripts/                                  # Conversion, benchmarking, launchers
requirements.txt
```

## Installation

```bash
pip install torch torchvision
pip install timm==0.9.16 pyyaml numpy pillow triton
```

Use a PyTorch build compatible with your CUDA version.

## Data and Checkpoints

Official dataset sources:

- CIFAR-100: https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
- FGVC Aircraft: https://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/
- Stanford Cars: https://ai.stanford.edu/~jkrause/cars/car_dataset.html
- Flowers102: https://www.robots.ox.ac.uk/~vgg/data/flowers/102/
- ImageNet-1K: https://image-net.org/download.php

Datasets and checkpoints are not included. An ImageNet layout is:

```text
data/imagenet2012/train
data/imagenet2012/val
pretrained/deit/deit_tiny_patch16_224.pth
```

## Quick Start

Set `data_dir`, `pretrained_checkpoint`, `initial_checkpoint`, `teacher`, and `output_dir` in an example configuration.

```yaml
method: pcarcq_nodc
w_bits: 4
a_bits: 4
group_size: 64
distillation: true
kd_type: logit_soft
distill_alpha: 0.5
distill_tau: 2.0
pc_compile_updates_per_epoch: 4
ddp_broadcast_buffers: true
ddp_init_sync: true
```

Run CIFAR-100 or transfer training:

```bash
cd methods/fair_qat_framework
python fair_qat/train_cifar100_deit_qat.py --config <cifar_config.yaml> --device cuda:0
python fair_qat/train_transfer_deit_qat.py --config <transfer_config.yaml> --device cuda:0
```

Run ImageNet with multiple GPUs:

```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun --standalone --nnodes=1 --nproc_per_node=4 \
  methods/fair_qat_framework/fair_qat/train_imagenet_qat.py \
  --config configs/example_imagenet_deit_tiny_arcq_w4a4.yaml --device cuda:0
```

## Packed On-The-Fly Inference

```bash
python scripts/convert_arcq_to_packed.py \
  --model deit_small --bits 4 \
  --config configs/example_imagenet_deit_small_arcq_w4a4.yaml \
  --checkpoint /path/to/arcq/best.pt \
  --output-dir /path/to/packed_output

python scripts/benchmark_arcq_on_the_fly.py \
  --model deit_small --bits 4 --batch-sizes 1,64
```

The packed representation stores low-bit weight indices, weight codebooks, per-group means and scales, code sums, activation codebooks, thresholds, bias, and reconstruction metadata. Triton kernels perform activation quantization, index unpacking, codebook lookup, reconstruction, and half-precision linear computation.

## Reproducibility

- ARCQ students use frozen FP32 teachers and soft-logit distillation.
- The main reported precisions are W4A4 and W3A3.
- The main quantized components are Transformer linear layers and their input activations.
- Patch embedding, classification head, LayerNorm, Softmax, GELU, and residual connections remain floating point unless explicitly changed in a configuration.
- Training and benchmarking require a CUDA environment.

## License

Please check the licenses of this repository and its dependencies before redistribution.
