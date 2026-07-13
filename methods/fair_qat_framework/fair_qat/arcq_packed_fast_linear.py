"""
ARCQ Packed Linear with Python-based weight unpack + standard matmul.
Replaces the slow CUDA fused kernel with a simpler approach.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

from fair_qat.arcq_infer_cuda import unpack_codes
from fair_qat.arcq_packed_linear import ARCQPackedLinear, _require_vector


class ARCQPackedLinearFast(ARCQPackedLinear):
    """Same storage as ARCQPackedLinear, but forward uses standard matmul.
    
    Replaces the slow CUDA fused kernel with:
    1. Unpack weight codes → fp16 weight (GPU)
    2. Standard F.linear(x, w_fp16, bias) (cuBLAS)
    """
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        original_shape = x.shape
        if original_shape[-1] != self.in_features:
            raise ValueError(f"expected last dim {self.in_features}, got {original_shape[-1]}")
        
        device = self.packed_weight_codes.device
        x_fp32 = x.reshape(-1, self.in_features).to(device=device, dtype=torch.float32)
        
        # Unpack weights: packed codes → fp16 weight matrix
        w_fp16 = _unpack_arcq_weight(
            self.packed_weight_codes,
            self.weight_codebook,
            self.weight_center,
            self.weight_scale,
            self.weight_code_sum,
            self.bits,
            self.group_size,
            self.in_features,
            self.out_features,
        )
        
        bias = self.bias if self.bias.numel() > 0 else None
        y = F.linear(x_fp32, w_fp16.to(dtype=torch.float32), bias)
        return y.reshape(*original_shape[:-1], self.out_features)


def _unpack_arcq_weight(
    packed_codes: torch.Tensor,       # uint8 [out_f, packed_cols]
    codebook: torch.Tensor,            # fp32 [levels]
    center: torch.Tensor,              # fp32 [out_f, groups]
    scale: torch.Tensor,               # fp32 [out_f, groups]
    code_sum: torch.Tensor,            # fp32 [out_f, groups]
    bits: int,
    group_size: int,
    in_features: int,
    out_features: int,
) -> torch.Tensor:
    """Unpack weight codes and reconstruct fp16 weight matrix.
    
    Returns [out_f, in_f] fp16 tensor.
    """
    groups = in_features // group_size
    levels = 2 ** bits
    
    # Step 1: unpack bit-packed codes → int64 indices
    codes = unpack_codes(packed_codes, bits, in_features)  # [out_f, in_f] int64
    
    # Step 2: codebook lookup
    cb = codebook.to(dtype=torch.float16)  # [levels] fp16
    values = cb[codes.to(dtype=torch.long)]  # [out_f, in_f] fp16
    
    # Step 3: reshape and apply mean-preserve
    # values: [out_f, in_f] → [out_f, groups, group_size]
    values_g = values.reshape(out_features, groups, group_size)  # fp16
    
    # code_sum: [out_f, groups] → [out_f, groups, 1]
    cs = code_sum.to(dtype=torch.float16).unsqueeze(-1)
    
    # t_hat = values - code_sum / group_size
    t_hat = values_g - cs / float(group_size)
    
    # Step 4: apply scale and center
    # scale: [out_f, groups] → [out_f, groups, 1]  
    s = scale.to(dtype=torch.float16).unsqueeze(-1)
    c = center.to(dtype=torch.float16).unsqueeze(-1)
    
    w_g = t_hat * s + c  # [out_f, groups, group_size] fp16
    
    # Step 5: reshape back
    w = w_g.reshape(out_features, in_features)  # fp16
    
    return w.contiguous()
