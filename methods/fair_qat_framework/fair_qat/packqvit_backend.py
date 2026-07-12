"""
PackQViT backend for fair DeiT QAT experiments.

This backend directly reuses the official PackQViT quantized linear and
activation modules from PackQViT-main/Quant.py. The fair matrix replaces only
the DeiT linear projections used by attention and MLP blocks. Patch embedding,
classifier head, LayerNorm, Softmax, and GELU remain floating point.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import torch
import torch.nn as nn


def _packqvit_root() -> Path:
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "duibi_method_vit" / "third_party" / "PackQvit" / "PackQViT-main"


def _import_packqvit_quant():
    root = _packqvit_root()
    if not root.is_dir():
        raise FileNotFoundError(f"PackQViT official code not found: {root}")

    root_s = str(root)
    if root_s in sys.path:
        sys.path.remove(root_s)
    sys.path.insert(0, root_s)

    # Quant.py and _quan_func.py are generic module names. Drop stale modules
    # imported from other third-party quantizers before loading PackQViT.
    for name in ("Quant", "_quan_func"):
        mod = sys.modules.get(name)
        mod_file = str(getattr(mod, "__file__", "")) if mod is not None else ""
        if mod is not None and root_s not in mod_file:
            del sys.modules[name]

    quant_mod = importlib.import_module("Quant")
    return quant_mod.LinearQuant, quant_mod.ActQ


class PackQViTLinearLastDim(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, w_bits: int):
        super().__init__()
        LinearQuant, _ = _import_packqvit_quant()
        self.linear = LinearQuant(in_features, out_features, bias=bias, nbits_w=w_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() <= 2:
            return self.linear(x)
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = self.linear(x_flat)
        return y_flat.reshape(*leading, y_flat.shape[-1])


class PackQViTBackend:
    name = "packqvit"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return PackQViTLinearLastDim(in_f, out_f, bias=bias, w_bits=w_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        _, ActQ = _import_packqvit_quant()
        in_features = int(shape_hint) if shape_hint is not None else 1
        return ActQ(in_features=in_features, nbits_a=a_bits)

    @staticmethod
    def make_head(in_f, out_f, bias):
        return nn.Linear(in_f, out_f, bias=bias)
