# DDFZ for Vision Transformer Quantization

这是当前 DDFZ 视觉 Transformer 实现的可复现代码包。代码包含：

- 基于组残差的 DDFZ 权重和激活量化；
- 阶段式动态非均匀码本编译；
- DeiT 的 CIFAR-100、迁移数据集和 ImageNet-1K 训练入口；
- DDFZ packed on-the-fly 推理转换与测速工具；
- 四卡 `torchrun` ImageNet 启动脚本。

当前主实验使用 `pcddfz_nodc`，主要位宽为 W4A4 和 W3A3。DDFZ 的知识蒸馏使用软标签对数值蒸馏：`kd_type: logit_soft`。

## 目录结构

```text
.
├── methods/fair_qat_framework/
│   ├── fair_qat/                  # ViT 后端、训练入口、packed 推理
│   ├── quant/dcddfz.py            # DDFZ 核心量化器
│   ├── data.py                    # CIFAR-100 数据接口
│   ├── data_transfer.py           # Aircraft/Cars/Flowers102 数据接口
│   └── data_imagenet.py           # ImageNet-1K 数据接口
├── scripts/
│   ├── convert_ddfz_to_packed.py  # QAT checkpoint 转 packed 模型
│   ├── benchmark_ddfz_on_the_fly.py
│   └── run_ddfz_imagenet_4gpu.sh
└── configs/
    └── example_*.yaml             # ImageNet 配置示例
```

## 环境安装

建议使用 Python 3.10 或更高版本，并安装与本机 CUDA 匹配的 PyTorch：

```bash
pip install torch torchvision
pip install timm==0.9.16 pyyaml numpy pillow
```

如果使用 packed on-the-fly 推理，还需要安装与 PyTorch/CUDA 匹配的 Triton：

```bash
pip install triton
```

## 快速开始：ImageNet-1K 四卡训练

准备以下目录：

```text
data/imagenet2012/train
data/imagenet2012/val
pretrained/deit/deit_tiny_patch16_224.pth
```

修改 `configs/example_imagenet_deit_tiny_ddfz_w4a4.yaml` 中的 `data_dir`、`pretrained_checkpoint`、`initial_checkpoint`、`teacher` 和 `output_dir`。关键 DDFZ 配置如下：

```yaml
method: pcddfz_nodc
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

从 `code` 目录执行四卡训练：

```bash
cd DDFZ-ViT
CUDA_VISIBLE_DEVICES=0,1,2,3 torchrun \
  --standalone --nnodes=1 --nproc_per_node=4 \
  methods/fair_qat_framework/fair_qat/train_imagenet_qat.py \
  --config configs/example_imagenet_deit_tiny_ddfz_w4a4.yaml \
  --device cuda:0
```

四卡 DDP 使用 `DistributedSampler` 切分数据，DDFZ 编译码本通过 DDP 缓冲区同步。每个训练阶段最多执行四次码本更新，由 `pc_compile_updates_per_epoch` 控制。

## CIFAR-100 和迁移数据集

训练入口分别为：

```bash
cd methods/fair_qat_framework
python fair_qat/train_cifar100_deit_qat.py --config <cifar_config.yaml> --device cuda:0
python fair_qat/train_transfer_deit_qat.py --config <transfer_config.yaml> --device cuda:0
```

配置中的 `initial_checkpoint` 用于加载目标数据集上的 FP32 DeiT 模型；启用 DDFZ 蒸馏时，将 `teacher` 指向同一个 FP32 teacher，并设置：

```yaml
distillation: true
kd_type: logit_soft
distill_alpha: 0.5
distill_tau: 2.0
```

## Packed On-The-Fly 推理

训练完成后，先将 DDFZ QAT checkpoint 转换为 packed 模型。转换结果保存低比特权重索引和必要的 DDFZ 元数据，包括：

- packed 权重索引；
- 权重码本；
- 每组权重均值、尺度和码点和；
- 激活码本和量化阈值；
- bias。

转换命令示例：

```bash
cd DDFZ-ViT
python scripts/convert_ddfz_to_packed.py \
  --model deit_small \
  --bits 4 \
  --config configs/example_imagenet_deit_small_ddfz_w4a4.yaml \
  --checkpoint /path/to/ddfz/best.pt \
  --output-dir /path/to/packed_output
```

运行 on-the-fly 推理测速：

```bash
python scripts/benchmark_ddfz_on_the_fly.py \
  --model deit_small \
  --bits 4 \
  --batch-sizes 1,64
```

该推理路径在前向过程中使用 Triton 完成激活量化、权重索引解包、码本查找和组统计量恢复，再进行半精度线性计算；固定输入形状时可以配合 CUDA Graph 降低内核启动开销。

## 重要说明

- 本代码包不包含 ImageNet、CIFAR-100 或其他数据集；
- 本代码包不包含模型权重；
- 训练和推理应在 GPU 环境中执行；
- DDFZ 的核心实现位于 `methods/fair_qat_framework/quant/dcddfz.py`；
- DeiT 线性层和卷积层适配位于 `methods/fair_qat_framework/fair_qat/quant_backends.py`；
- packed 推理实现位于 `methods/fair_qat_framework/fair_qat/ddfz_packed_on_the_fly.py` 和 `ddfz_packed_triton.py`。
