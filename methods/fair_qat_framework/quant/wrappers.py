"""
Quantized Linear wrapper modules for ViT.

Provides QuantLinearLSQ and QuantLinearDDFZ that replace nn.Linear
in DeiT-Tiny's attention and MLP blocks. The classification head
is excluded from quantization.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .lsq import LSQQuantizer
from .dcddfz import DDFZActQuantizer, DDFZWeightQuantizer


# ============================================================================
#  QuantLinearLSQ
# ============================================================================

class QuantLinearLSQ(nn.Module):
    """Linear layer with LSQ activation and weight quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 4,
        a_bits: int = 4,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.act_quant = LSQQuantizer(
            bits=a_bits,
            signed=True,
            per_channel=False,
        )
        self.weight_quant = LSQQuantizer(
            bits=w_bits,
            signed=True,
            per_channel=True,
            ch_axis=0,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)


# ============================================================================
#  QuantLinearDDFZ
# ============================================================================

class QuantLinearDDFZ(nn.Module):
    """Linear layer with DDFZ activation and weight quantization."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 4,
        a_bits: int = 4,
        group_size: int = 64,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        self.act_quant = DDFZActQuantizer(
            bits=a_bits,
            group_size=group_size,
        )
        self.weight_quant = DDFZWeightQuantizer(
            bits=w_bits,
            group_size=group_size,
        )

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        # DDFZ needs to group along the last dim of activation and in_features of weight
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.weight)
        return F.linear(x_q, w_q, self.bias)


# ============================================================================
#  Model-level replacement
# ============================================================================

def _replace_linear_recursive(module, quant_cls, **quant_kwargs):
    """Recursively replace all nn.Linear with quantized versions.

    Skips modules whose name contains 'head' (classification head).
    """
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if "head" in name:
                continue
            new_linear = quant_cls(
                in_features=child.in_features,
                out_features=child.out_features,
                bias=child.bias is not None,
                **quant_kwargs,
            )
            # Copy original weights if available
            with torch.no_grad():
                new_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_linear.bias.data.copy_(child.bias.data)
            setattr(module, name, new_linear)
        else:
            _replace_linear_recursive(child, quant_cls, **quant_kwargs)


def apply_lsq_quant(model: nn.Module, config: dict) -> nn.Module:
    """Apply LSQ quantization to all Linear layers in a ViT model.

    Args:
        model: DeiT-Tiny model.
        config: dict with w_bits, a_bits.

    Returns:
        Modified model (in-place).
    """
    w_bits = config.get("w_bits", 4)
    a_bits = config.get("a_bits", 4)
    _replace_linear_recursive(model, QuantLinearLSQ, w_bits=w_bits, a_bits=a_bits)
    return model


def apply_dcddfz_quant(model: nn.Module, config: dict) -> nn.Module:
    """Apply DC-DDFZ quantization to all Linear layers in a ViT model.

    Args:
        model: DeiT-Tiny model.
        config: dict with w_bits, a_bits, group_size.

    Returns:
        Modified model (in-place).
    """
    w_bits = config.get("w_bits", 4)
    a_bits = config.get("a_bits", 4)
    group_size = config.get("group_size", 64)
    _replace_linear_recursive(model, QuantLinearDDFZ, w_bits=w_bits, a_bits=a_bits, group_size=group_size)
    return model
