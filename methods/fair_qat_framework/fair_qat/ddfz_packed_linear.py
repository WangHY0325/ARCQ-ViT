from __future__ import annotations

import math
from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from fair_qat.ddfz_infer_cuda import (
    ddfz_linear_forward,
    ddfz_linear_forward_u8,
    pack_codes,
    quantize_activation,
    quantize_activation_dequant,
    quantize_activation_u8,
    unpack_codes_u8,
)


def _require_vector(name: str, value: torch.Tensor) -> torch.Tensor:
    if not isinstance(value, torch.Tensor) or value.numel() == 0:
        raise ValueError(f"{name} is missing; load a checkpoint with compiled DDFZ codebooks first")
    return value.detach().float().flatten().contiguous()


def _quantize_weight_reference(
    weight: torch.Tensor,
    thresholds: torch.Tensor,
    codebook: torch.Tensor,
    group_size: int,
    eps: float,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if weight.dim() != 2:
        raise ValueError(f"DDFZPackedLinear expects 2D linear weight, got shape={tuple(weight.shape)}")
    out_features, in_features = weight.shape
    if in_features % group_size != 0:
        raise ValueError(
            f"in_features={in_features} must be divisible by group_size={group_size} "
            "for the first packed kernel version"
        )

    w = weight.detach().float().contiguous()
    groups = in_features // group_size
    w3 = w.reshape(out_features, groups, group_size)
    center = w3.mean(dim=-1)
    residual = w3 - center.unsqueeze(-1)
    scale = (residual.square().mean(dim=-1) + float(eps)).sqrt()
    t = residual / scale.unsqueeze(-1)
    codes = torch.bucketize(t, thresholds.to(device=t.device, dtype=t.dtype))
    values = codebook.to(device=t.device, dtype=t.dtype)[codes]
    code_sum = values.sum(dim=-1)
    return codes.reshape(out_features, in_features).to(torch.int64), center, scale, code_sum


class DDFZPackedLinear(nn.Module):
    """Inference-only packed DDFZ linear layer.

    This module replaces QuantLinearPCDDFZ after training. It stores weight
    low-bit indices plus per-group metadata and never keeps the full floating
    weight matrix as a parameter.
    """

    def __init__(self, source: nn.Module, runtime_code_format: str = "packed"):
        super().__init__()
        if runtime_code_format not in ("packed", "u8", "dequant_gemm"):
            raise ValueError(
                "runtime_code_format must be 'packed', 'u8', or 'dequant_gemm', "
                f"got {runtime_code_format}"
            )
        self.runtime_code_format = runtime_code_format
        linear = getattr(source, "linear", None)
        if linear is None or not hasattr(linear, "weight"):
            raise TypeError("DDFZPackedLinear source must expose source.linear.weight")
        act_quant = getattr(source, "act_quant", None)
        weight_quant = getattr(source, "weight_quant", None)
        if act_quant is None or weight_quant is None:
            raise TypeError("DDFZPackedLinear source must expose act_quant and weight_quant")

        self.in_features = int(linear.weight.shape[1])
        self.out_features = int(linear.weight.shape[0])
        self.group_size = int(getattr(weight_quant, "group_size", 64))
        self.bits = int(getattr(weight_quant, "bits", 4))
        self.act_bits = int(getattr(act_quant, "bits", self.bits))
        if self.bits != self.act_bits:
            raise ValueError(f"packed kernel currently requires W bits == A bits, got W{self.bits} A{self.act_bits}")
        if self.bits not in (2, 3, 4):
            raise ValueError(f"packed kernel supports bits 2/3/4, got {self.bits}")
        if self.in_features % self.group_size != 0:
            raise ValueError(
                f"in_features={self.in_features} must be divisible by group_size={self.group_size}"
            )

        eps = float(getattr(weight_quant, "eps", getattr(act_quant, "eps", 1.0e-6)))
        activation_codebook = _require_vector("act_quant._pc_cb", getattr(act_quant, "_pc_cb", None))
        activation_thresholds = _require_vector("act_quant._pc_thresholds", getattr(act_quant, "_pc_thresholds", None))
        weight_codebook = _require_vector("weight_quant._pc_cb", getattr(weight_quant, "_pc_cb", None))
        weight_thresholds = _require_vector("weight_quant._pc_thresholds", getattr(weight_quant, "_pc_thresholds", None))
        if activation_codebook.numel() != 2 ** self.bits or weight_codebook.numel() != 2 ** self.bits:
            raise ValueError(
                f"codebook size must be {2 ** self.bits}, got "
                f"A={activation_codebook.numel()} W={weight_codebook.numel()}"
            )

        device = linear.weight.device
        weight_codes, weight_center, weight_scale, weight_code_sum = _quantize_weight_reference(
            linear.weight,
            weight_thresholds.to(device=device),
            weight_codebook.to(device=device),
            self.group_size,
            eps,
        )
        packed_weight_codes = pack_codes(weight_codes.to(device=device), self.bits)
        product_table = (
            activation_codebook.to(device=device).view(-1, 1)
            * weight_codebook.to(device=device).view(1, -1)
        ).float().contiguous()

        self.register_buffer("packed_weight_codes", packed_weight_codes)
        if self.runtime_code_format == "u8":
            weight_codes_u8 = unpack_codes_u8(packed_weight_codes, self.bits, self.in_features).contiguous()
        else:
            weight_codes_u8 = torch.empty(0, device=device, dtype=torch.uint8)
        self.register_buffer("weight_codes_u8", weight_codes_u8)

        if self.runtime_code_format == "dequant_gemm":
            weight_values = weight_codebook.to(device=device, dtype=torch.float32)[weight_codes]
            weight_values = weight_values.reshape(self.out_features, self.in_features // self.group_size, self.group_size)
            weight_offset = weight_values - (weight_code_sum.unsqueeze(-1) / float(self.group_size))
            weight_dequant = weight_center.unsqueeze(-1) + weight_scale.unsqueeze(-1) * weight_offset
            weight_dequant = weight_dequant.reshape(self.out_features, self.in_features).to(torch.float16).contiguous()
        else:
            weight_dequant = torch.empty(0, device=device, dtype=torch.float16)
        self.register_buffer("weight_dequant", weight_dequant)

        self.register_buffer("weight_center", weight_center.float().contiguous())
        self.register_buffer("weight_scale", weight_scale.float().contiguous())
        self.register_buffer("weight_code_sum", weight_code_sum.float().contiguous())
        self.register_buffer("weight_codebook", weight_codebook.to(device=device).float().contiguous())
        self.register_buffer("activation_codebook", activation_codebook.to(device=device).float().contiguous())
        self.register_buffer("activation_thresholds", activation_thresholds.to(device=device).float().contiguous())
        self.register_buffer("product_table", product_table)
        if linear.bias is None:
            self.register_buffer("bias", torch.empty(0, device=device, dtype=torch.float32))
        else:
            self.register_buffer("bias", linear.bias.detach().to(device=device, dtype=torch.float32).contiguous())

        self.packed_weight_bytes = int(self.packed_weight_codes.numel())
        self.weight_meta_floats = int(
            self.weight_center.numel() + self.weight_scale.numel() + self.weight_code_sum.numel()
        )

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bits={self.bits}, group_size={self.group_size}, "
            f"runtime_code_format={self.runtime_code_format}, "
            f"packed_weight_bytes={self.packed_weight_bytes}"
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if original_shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {original_shape[-1]}")
        x2 = x.reshape(-1, self.in_features).to(device=self.packed_weight_codes.device, dtype=torch.float32).contiguous()
        bias = self.bias if self.bias.numel() > 0 else None
        if self.runtime_code_format == "dequant_gemm":
            x_hat = quantize_activation_dequant(
                x2,
                self.activation_thresholds,
                self.activation_codebook,
                self.bits,
                self.group_size,
            )
            gemm_bias = self.bias.to(torch.float16) if self.bias.numel() > 0 else None
            y2 = F.linear(x_hat.to(torch.float16), self.weight_dequant, gemm_bias).to(torch.float32)
        elif self.runtime_code_format == "u8":
            x_codes, x_center, x_scale, x_code_sum = quantize_activation_u8(
                x2,
                self.activation_thresholds,
                self.activation_codebook,
                self.bits,
                self.group_size,
            )
            y2 = ddfz_linear_forward_u8(
                x_codes,
                x_center,
                x_scale,
                x_code_sum,
                self.weight_codes_u8,
                self.weight_center,
                self.weight_scale,
                self.weight_code_sum,
                self.product_table,
                bias,
                self.group_size,
            )
        else:
            packed_x, x_center, x_scale, x_code_sum = quantize_activation(
                x2,
                self.activation_thresholds,
                self.activation_codebook,
                self.bits,
                self.group_size,
            )
            y2 = ddfz_linear_forward(
                packed_x,
                x_center,
                x_scale,
                x_code_sum,
                self.packed_weight_codes,
                self.weight_center,
                self.weight_scale,
                self.weight_code_sum,
                self.product_table,
                bias,
                self.bits,
                self.group_size,
                self.in_features,
                self.out_features,
            )
        return y2.reshape(*original_shape[:-1], self.out_features)

    def estimated_storage_bytes(self) -> int:
        meta = (
            self.weight_center.numel()
            + self.weight_scale.numel()
            + self.weight_code_sum.numel()
            + self.weight_codebook.numel()
            + self.activation_codebook.numel()
            + self.activation_thresholds.numel()
            + self.product_table.numel()
            + self.bias.numel()
        ) * 4
        return int(self.packed_weight_codes.numel() + meta)

    def runtime_cache_bytes(self) -> int:
        return int(
            self.weight_codes_u8.numel() * self.weight_codes_u8.element_size()
            + self.weight_dequant.numel() * self.weight_dequant.element_size()
        )
