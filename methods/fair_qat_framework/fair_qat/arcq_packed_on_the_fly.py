"""
ARCQ Packed Linear with on-the-fly weight unpack + standard cuBLAS matmul.

Stores ONLY packed codes + metadata (~16 MB for DeiT-Small W4A4).
Forward: vectorized GPU unpack weight to fp16 → F.linear (cuBLAS).
No pre-computed weight_dequant buffer → memory stays minimal.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def _unpack_weight_gpu(
    packed: torch.Tensor,         # uint8 [out_f, packed_cols]
    codebook: torch.Tensor,        # fp32 [levels]
    center: torch.Tensor,          # fp32 [out_f, groups]
    scale: torch.Tensor,           # fp32 [out_f, groups]
    code_sum: torch.Tensor,        # fp32 [out_f, groups]
    bits: int,
    group_size: int,
    out_f: int,
    in_f: int,
) -> torch.Tensor:
    """Vectorized GPU unpack: packed bits → int codes → codebook lookup → center/scale → fp16."""
    device = packed.device
    groups = in_f // group_size
    mask = (1 << bits) - 1

    # Pre-compute byte index and shift for each column
    bit_offsets = torch.arange(in_f, device=device, dtype=torch.int64) * bits
    byte_idx = bit_offsets // 8
    shift = bit_offsets % 8
    cross_byte = (shift + bits) > 8

    # Gather packed bytes at byte_idx positions → [out_f, in_f]
    lo_bytes = packed[:, byte_idx]                                   # [out_f, in_f]  
    codes = ((lo_bytes.to(torch.int32) >> shift) & mask).to(torch.int32)

    # Handle codes that cross byte boundaries
    if cross_byte.any():
        hi_bytes = packed[:, byte_idx + 1]                           # [out_f, in_f]
        hi_bits = mask & ~((1 << (8 - shift[0].item())) - 1)  # simplified
        # Per-column handling for cross-byte: vectorized
        codes[..., cross_byte] = codes[..., cross_byte] | (
            (hi_bytes[..., cross_byte].to(torch.int32) << (8 - shift[cross_byte])) & mask
        )

    # Reshape to groups, codebook lookup, apply center/scale
    codes_g = codes.reshape(out_f, groups, group_size)               # [out_f, groups, gs]
    values = codebook.to(device=device)[codes_g.long()]              # [out_f, groups, gs] fp32

    # Mean-preserve correction
    cs = code_sum.unsqueeze(-1)                                      # [out_f, groups, 1]
    values_c = values - cs / float(group_size)

    # Apply scale and center
    s = scale.unsqueeze(-1)                                          # [out_f, groups, 1]
    c = center.unsqueeze(-1)                                         # [out_f, groups, 1]
    w_g = c + s * values_c                                           # [out_f, groups, gs]

    return w_g.reshape(out_f, in_f).to(dtype=torch.float16)


def _dequant_activation_torch(
    x: torch.Tensor,              # [rows, in_features] fp32
    thresholds: torch.Tensor,      # [levels-1] fp32
    codebook: torch.Tensor,        # [levels] fp32
    group_size: int,
) -> torch.Tensor:
    """Pure PyTorch activation quantize + dequant (matches CUDA kernel)."""
    rows, cols = x.shape
    groups = cols // group_size
    eps = 1e-6

    x3 = x.reshape(rows, groups, group_size)                         # [rows, groups, gs]
    center = x3.mean(dim=-1)                                          # [rows, groups]
    residual = x3 - center.unsqueeze(-1)
    scale = (residual.square().mean(dim=-1) + eps).sqrt()             # [rows, groups]
    t = residual / scale.unsqueeze(-1)                                # [rows, groups, gs]

    codes = torch.bucketize(t, thresholds.to(device=x.device))        # [rows, groups, gs]
    code_sum = codebook.to(device=x.device)[codes].sum(dim=-1)        # [rows, groups]
    code_mean = code_sum / float(group_size)

    x_hat_g = center.unsqueeze(-1) + scale.unsqueeze(-1) * (
        codebook.to(device=x.device)[codes] - code_mean.unsqueeze(-1)
    )
    return x_hat_g.reshape(rows, cols)


class ARCQPackedLinearOnTheFly(nn.Module):
    """Inference-only packed ARCQ linear. Unpacks weight on the fly.

    Stores: packed_weight_codes (int4) + center/scale/codebook + bias.
    No weight_dequant buffer → ~16 MB GPU memory for DeiT-Small W4A4.
    Inherits from nn.Module (NOT nn.Linear) to avoid dummy FP32 weight allocation.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool,
        packed_weight_codes: torch.Tensor,   # uint8 [out_f, packed_cols]
        weight_codebook: torch.Tensor,        # [levels] fp32
        weight_center: torch.Tensor,          # [out_f, groups] fp32
        weight_scale: torch.Tensor,           # [out_f, groups] fp32
        weight_code_sum: torch.Tensor,        # [out_f, groups] fp32
        activation_codebook: torch.Tensor,    # [levels] fp32
        activation_thresholds: torch.Tensor,  # [levels-1] fp32
        bits: int,
        group_size: int,
        bias_tensor: torch.Tensor | None,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bits = bits
        self.group_size = group_size
        self.groups = in_features // group_size

        # Register as buffers
        self.register_buffer("packed_weight_codes", packed_weight_codes)
        self.register_buffer("weight_codebook", weight_codebook)
        self.register_buffer("weight_center", weight_center)
        self.register_buffer("weight_scale", weight_scale)
        self.register_buffer("weight_code_sum", weight_code_sum)
        self.register_buffer("activation_codebook", activation_codebook)
        self.register_buffer("activation_thresholds", activation_thresholds)
        if bias_tensor is not None and bias_tensor.numel() > 0:
            self.register_buffer("bias_tensor", bias_tensor)
        else:
            self.register_buffer("bias_tensor", torch.empty(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        x2 = x.reshape(-1, self.in_features).to(dtype=torch.float16).contiguous()

        # Dequantize activation: Triton fused (fp16 in → fp16 out, zero intermediates)
        from fair_qat.arcq_packed_triton import dequant_activation_triton
        x_hat = dequant_activation_triton(
            x2, self.activation_thresholds, self.activation_codebook, self.group_size, out=x2
        )

        # Unpack weight: Triton fused
        from fair_qat.arcq_packed_triton import unpack_weight_triton
        shared_buf = getattr(self, '_weight_buffer', None)
        w_fp16 = unpack_weight_triton(
            self.packed_weight_codes, self.weight_codebook,
            self.weight_center, self.weight_scale, self.weight_code_sum,
            self.bits, self.group_size,
            out=shared_buf,
        )
        if shared_buf is not None:
            w_fp16 = w_fp16[:self.out_features * self.in_features].view(self.out_features, self.in_features)

        bias = self.bias_tensor.to(dtype=torch.float16) if self.bias_tensor.numel() > 0 else None
        y = F.linear(x_hat, w_fp16, bias)
        return y.reshape(*original_shape[:-1], self.out_features)

    def estimated_persistent_mb(self) -> float:
        """Estimated GPU memory in MB for this layer (no runtime cache)."""
        return self.packed_weight_codes.numel() / 1024 / 1024 + sum(
            buf.numel() * buf.element_size() / 1024 / 1024
            for buf in self.buffers()
            if buf is not self.packed_weight_codes
        )
