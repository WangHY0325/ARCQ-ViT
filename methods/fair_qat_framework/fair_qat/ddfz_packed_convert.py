from __future__ import annotations

import torch.nn as nn

from fair_qat.ddfz_packed_linear import DDFZPackedLinear


def _is_pcddfz_linear(module: nn.Module) -> bool:
    return (
        module.__class__.__name__ == "QuantLinearPCDDFZ"
        and hasattr(module, "linear")
        and hasattr(module, "act_quant")
        and hasattr(module, "weight_quant")
    )


def convert_ddfz_linear_to_packed(model: nn.Module, runtime_code_format: str = "packed") -> nn.Module:
    """Recursively replace QuantLinearPCDDFZ modules with DDFZPackedLinear."""
    for child_name, child in list(model.named_children()):
        if _is_pcddfz_linear(child):
            setattr(model, child_name, DDFZPackedLinear(child, runtime_code_format=runtime_code_format))
        else:
            convert_ddfz_linear_to_packed(child, runtime_code_format=runtime_code_format)
    return model


def count_packed_linear(model: nn.Module) -> int:
    return sum(1 for module in model.modules() if isinstance(module, DDFZPackedLinear))
