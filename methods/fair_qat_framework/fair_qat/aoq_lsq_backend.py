"""
AOQ + LSQ baseline backend for ViT QAT.

Implements AOQ (Alternating Optimization of Quantization) weight quantizer
with LSQ-style activation quantization, as a baseline for AAAI experiments.

Reference:
  - AOQ paper: Alternating Optimization for Learned Step Size Quantization
  - Official code: AO_QAT/quan/quantizer.py (AOQ class, hardcoded 2-bit)
  - AO_QAT/quan/func.py (QuanLinear wrapper pattern)

NOTE: AOQ official code hardcodes 2-bit weight quantizer only.
      This implementation generalizes to W2/W3/W4 using the paper formula.
      If strict official-code AOQ is desired, only W2A2 should be used.
      Experiment name: aoq_lsq (LSQ activation + AOQ-style weight)
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

# ============================================================================
#  AOQ Global State (replaces official AOQ globalVal)
# ============================================================================

AOQ_GLOBAL_STATE = {
    "epoch": 0,
    "aux_loss": None,
}


def set_aoq_epoch(epoch: int) -> None:
    """Set current epoch for AOQ quantizer schedule."""
    AOQ_GLOBAL_STATE["epoch"] = epoch


def reset_aoq_aux_loss(device: torch.device) -> None:
    """Reset AOQ auxiliary loss accumulator to zero."""
    AOQ_GLOBAL_STATE["aux_loss"] = torch.zeros((), device=device)


def add_aoq_aux_loss(loss_tensor: torch.Tensor) -> None:
    """Accumulate AOQ auxiliary loss as a 0-d scalar tensor."""
    if not torch.is_tensor(loss_tensor):
        raise TypeError(f"AOQ aux loss must be a torch.Tensor, got {type(loss_tensor)}")
    loss_scalar = loss_tensor.mean()
    if loss_scalar.dim() != 0:
        loss_scalar = loss_scalar.reshape(())
    if AOQ_GLOBAL_STATE["aux_loss"] is None:
        AOQ_GLOBAL_STATE["aux_loss"] = torch.zeros((), device=loss_scalar.device)
    AOQ_GLOBAL_STATE["aux_loss"] = AOQ_GLOBAL_STATE["aux_loss"] + loss_scalar


def get_aoq_aux_loss() -> Optional[torch.Tensor]:
    """Return accumulated AOQ auxiliary loss, or None if not set."""
    return AOQ_GLOBAL_STATE["aux_loss"]


# ============================================================================
#  STE helpers
# ============================================================================

def round_pass(x: torch.Tensor) -> torch.Tensor:
    """Straight-through estimator for rounding (official AOQ style)."""
    y = x.round()
    return (y - x).detach() + x


# ============================================================================
#  LSQ Activation Quantizer
# ============================================================================

class AOQLSQActivationQuantizer(nn.Module):
    """
    LSQ-style activation quantizer for AOQ baseline.
    
    NOTE: AOQ official LTQ activation is positive-biased (designed for
    ReLU outputs). DeiT Linear inputs can be signed. For fairness and
    stability, this baseline uses LSQ activation quantization instead.
    """

    def __init__(self, bits: int = 4, signed: bool = False, eps: float = 1e-8):
        super().__init__()
        self.bits = int(bits)
        self.signed = bool(signed)
        self.eps = float(eps)
        self._initialized = False
        self.step_size = nn.Parameter(torch.tensor(1.0))

    @property
    def qrange(self):
        if self.signed:
            return -2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1
        else:
            return 0, 2 ** self.bits - 1

    @torch.no_grad()
    def init_from(self, x: torch.Tensor) -> None:
        if self._initialized:
            return
        qn, qp = self.qrange
        qp_val = max(qp, 1)
        init_val = 2.0 * x.detach().abs().mean() / math.sqrt(qp_val)
        self.step_size.data.fill_(float(init_val.clamp_min(self.eps)))
        self._initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self._initialized:
            self.init_from(x)
        qn, qp = self.qrange
        x_scaled = x / self.step_size
        x_q = round_pass(x_scaled).clamp(qn, qp)
        return x_q * self.step_size


# ============================================================================
#  AOQ Weight Quantizer
# ============================================================================

def _make_level_codes(bits: int) -> torch.Tensor:
    """Symmetric zero-free mid-rise level codes.
    
    2-bit: [-1.5, -0.5, 0.5, 1.5]
    3-bit: [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    4-bit: [-7.5, ..., 7.5]
    """
    L = 2 ** bits
    return torch.arange(L).float() - (L - 1) / 2.0


def _make_threshold_codes(bits: int) -> torch.Tensor:
    """Thresholds (midpoints between level codes).
    
    2-bit: [-1.0, 0.0, 1.0]
    3-bit: [-3, -2, -1, 0, 1, 2, 3]
    4-bit: [-7, ..., 7]
    """
    L = 2 ** bits
    return torch.arange(1, L).float() - L / 2.0


class AOQWeightQuantizer(nn.Module):
    """
    AOQ-style weight quantizer with per-tensor granularity.
    
    Three-stage training implicitly handled by alpha schedule:
      - Warmup (epoch <= warmup): alpha < 1.0 (cosine decay), levels use init_interval * alpha
      - After warmup (epoch > warmup): alpha = 1.0, levels use learnable step_size * alpha
    
    Args:
        bits: Quantization bitwidth (2, 3, or 4).
        aoq_warmup_epochs: Epochs where alpha < 1.0 (default 50).
        aoq_alpha_base: Base value for cosine alpha (default 0.65).
        aoq_alpha_amp: Amplitude of cosine alpha oscillation (default 0.35).
        aoq_alpha_period: Period of cosine alpha in epochs (default 100.0).
        aoq_aux_lambda: Weight for auxiliary clustering loss (default 1e-5).
        aoq_aux_type: Type of auxiliary loss: 'cluster_l2' or 'cluster_l1'.
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        bits: int = 4,
        aoq_warmup_epochs: int = 50,
        aoq_alpha_base: float = 0.65,
        aoq_alpha_amp: float = 0.35,
        aoq_alpha_period: float = 100.0,
        aoq_aux_lambda: float = 1e-5,
        aoq_aux_type: str = "cluster_l2",
        eps: float = 1e-8,
    ):
        super().__init__()
        assert bits in (2, 3, 4), f"AOQ supports bits 2/3/4, got {bits}"
        self.bits = int(bits)
        self.eps = float(eps)
        self.aoq_warmup_epochs = int(aoq_warmup_epochs)
        self.aoq_alpha_base = float(aoq_alpha_base)
        self.aoq_alpha_amp = float(aoq_alpha_amp)
        self.aoq_alpha_period = float(aoq_alpha_period)
        self.aoq_aux_lambda = float(aoq_aux_lambda)
        self.aoq_aux_type = str(aoq_aux_type)

        self.num_levels = 2 ** bits

        # Scalar step_size (learnable)
        self.step_size = nn.Parameter(torch.tensor(1.0))

        # Initialization interval buffer (non-learnable reference)
        self.register_buffer("init_interval", torch.tensor(1.0))

        # Level and threshold codes (non-learnable, fixed patterns)
        self.register_buffer("raw_level_codes", _make_level_codes(bits))
        self.register_buffer("raw_threshold_codes", _make_threshold_codes(bits))

        self._initialized = False

        # Persistent alpha buffer (updated every 5 epochs during warmup)
        self.register_buffer("alpha", torch.tensor(1.0, dtype=torch.float32))

    def _current_alpha(self) -> float:
        epoch = int(AOQ_GLOBAL_STATE.get("epoch", 0))
        if epoch <= self.aoq_warmup_epochs and epoch % 5 == 0:
            alpha = self.aoq_alpha_amp * math.cos(2.0 * math.pi * epoch / self.aoq_alpha_period) + self.aoq_alpha_base
            self.alpha.fill_(float(alpha))
        return float(self.alpha.item())

    @torch.no_grad()
    def init_from(self, w: torch.Tensor) -> None:
        """Initialize AOQ parameters from weight statistics (per-tensor)."""
        if self._initialized:
            return
        sigma = torch.std(w.detach()).clamp(min=self.eps)
        init_interval = sigma / (2 ** (self.bits - 2))
        self.step_size.data.fill_(init_interval.item())
        self.init_interval.data.fill_(init_interval.item())
        self._initialized = True

    def forward(self, w: torch.Tensor) -> torch.Tensor:
        """
        AOQ weight quantization forward.
        
        Returns quantized weight with STE gradient.
        Accumulates auxiliary clustering loss via add_aoq_aux_loss.
        """
        if not self._initialized:
            self.init_from(w)

        alpha = self._current_alpha()
        epoch = AOQ_GLOBAL_STATE["epoch"]
        device = w.device
        dtype = w.dtype

        # Compute level interval and thresholds
        if epoch <= self.aoq_warmup_epochs:
            level_interval = self.init_interval * alpha
        else:
            level_interval = self.step_size * alpha

        threshold_interval = self.init_interval * alpha

        # Build levels and thresholds from raw codes
        levels = self.raw_level_codes.to(device=device, dtype=dtype) * level_interval
        thresholds = self.raw_threshold_codes.to(device=device, dtype=dtype) * threshold_interval

        # Quantize via bucketize
        w_flat = w.reshape(-1)
        codes = torch.bucketize(w_flat, thresholds)
        q_flat = levels[codes]
        q = q_flat.reshape_as(w)

        # STE forward (official AOQ style: backward sees q + w)
        x_backward = q + w
        out = x_backward + (q - x_backward).detach()

        # AOQ cluster target (for auxiliary loss)
        cluster_levels = self.raw_level_codes.to(device=device, dtype=dtype) * (self.init_interval * alpha)
        cluster_flat = cluster_levels[codes]
        cluster = cluster_flat.reshape_as(w)

        # Accumulate auxiliary loss (official AOQ norm style, no lambda scaling here)
        if self.aoq_aux_type == "official_norm":
            aux = torch.norm(w - cluster.detach())
        elif self.aoq_aux_type == "cluster_l2":
            aux = torch.mean((w - cluster.detach()) ** 2)
        elif self.aoq_aux_type == "cluster_l1":
            aux = torch.mean(torch.abs(w - cluster.detach()))
        else:
            raise ValueError(f"Unknown aoq_aux_type: {self.aoq_aux_type}")
        add_aoq_aux_loss(aux)

        return out


# ============================================================================
#  AOQ Linear Wrapper
# ============================================================================

class AOQLinear(nn.Linear):
    """Linear layer with AOQ weight quantization + LSQ activation quantization.
    
    Follows backend interface: constructed from (in_f, out_f, bias).
    Weight/bias are copied later by _copy_linear.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 4,
        a_bits: int = 4,
        a_signed: bool = True,
        aoq_warmup_epochs: int = 50,
        aoq_alpha_base: float = 0.65,
        aoq_alpha_amp: float = 0.35,
        aoq_alpha_period: float = 100.0,
        aoq_aux_lambda: float = 1e-5,
        aoq_aux_type: str = "cluster_l2",
    ):
        super().__init__(in_features, out_features, bias=bias)

        self.weight_quantizer = AOQWeightQuantizer(
            bits=w_bits,
            aoq_warmup_epochs=aoq_warmup_epochs,
            aoq_alpha_base=aoq_alpha_base,
            aoq_alpha_amp=aoq_alpha_amp,
            aoq_alpha_period=aoq_alpha_period,
            aoq_aux_lambda=aoq_aux_lambda,
            aoq_aux_type=aoq_aux_type,
        )
        self.activation_quantizer = AOQLSQActivationQuantizer(
            bits=a_bits, signed=a_signed
        )

    def _init_weight_quantizer(self) -> None:
        """Call after weight is loaded (e.g., by _copy_linear)."""
        self.weight_quantizer.init_from(self.weight.data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.activation_quantizer._initialized:
            self.activation_quantizer.init_from(x.detach())
        x_q = self.activation_quantizer(x)
        w_q = self.weight_quantizer(self.weight)
        return F.linear(x_q, w_q, self.bias)


# ============================================================================
#  Model-level replacement
# ============================================================================

def _replace_linear_aoq(module: nn.Module, w_bits: int, a_bits: int, **kwargs) -> None:
    """Recursively replace nn.Linear with AOQLinear. Skips 'head'."""
    for name, child in list(module.named_children()):
        if isinstance(child, nn.Linear):
            if "head" in name:
                continue
            new_linear = AOQLinear(
                child.in_features, child.out_features,
                bias=child.bias is not None,
                w_bits=w_bits, a_bits=a_bits, **kwargs
            )
            # Copy weights and init quantizer
            with torch.no_grad():
                new_linear.weight.data.copy_(child.weight.data)
                if child.bias is not None:
                    new_linear.bias.data.copy_(child.bias.data)
            new_linear._init_weight_quantizer()
            setattr(module, name, new_linear)
        else:
            _replace_linear_aoq(child, w_bits=w_bits, a_bits=a_bits, **kwargs)


def apply_aoq_lsq_quant(model: nn.Module, config: dict) -> nn.Module:
    """Apply AOQ+LSQ quantization to all Linear layers."""
    w_bits = int(config.get("w_bits", 4))
    a_bits = int(config.get("a_bits", 4))
    a_signed = bool(config.get("a_signed", True))
    aoq_kwargs = {
        "a_signed": a_signed,
        "aoq_warmup_epochs": int(config.get("aoq_warmup_epochs", 50)),
        "aoq_alpha_base": float(config.get("aoq_alpha_base", 0.65)),
        "aoq_alpha_amp": float(config.get("aoq_alpha_amp", 0.35)),
        "aoq_alpha_period": float(config.get("aoq_alpha_period", 100.0)),
        "aoq_aux_lambda": float(config.get("aoq_aux_lambda", 0.01)),
        "aoq_aux_type": str(config.get("aoq_aux_type", "official_norm")),
    }
    _replace_linear_aoq(model, w_bits=w_bits, a_bits=a_bits, **aoq_kwargs)
    return model


# ============================================================================
#  AOQ + LSQ Backend
# ============================================================================

class AOQLSQBackend:
    """
    Backend for AOQ weight + LSQ activation quantization.
    
    Implements the standard backend interface (make_linear, make_act, etc.)
    for compatibility with build_timm_quant_model and _replace_children.
    """
    name = "aoq_lsq"
    _config = {}

    @classmethod
    def set_config(cls, config):
        cls._config = dict(config)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        cfg = AOQLSQBackend._config
        return AOQLinear(
            in_f, out_f, bias,
            w_bits=w_bits,
            a_bits=a_bits,
            a_signed=bool(cfg.get("a_signed", True)),
            aoq_warmup_epochs=int(cfg.get("aoq_warmup_epochs", 50)),
            aoq_alpha_base=float(cfg.get("aoq_alpha_base", 0.65)),
            aoq_alpha_amp=float(cfg.get("aoq_alpha_amp", 0.35)),
            aoq_alpha_period=float(cfg.get("aoq_alpha_period", 100.0)),
            aoq_aux_lambda=float(cfg.get("aoq_aux_lambda", 0.01)),
            aoq_aux_type=str(cfg.get("aoq_aux_type", "official_norm")),
        )

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        # AOQ baseline: use simple LSQ conv (AOQ focuses on linear/weight)
        from fair_qat.quant_backends import QuantConv2dLSQ
        return QuantConv2dLSQ(in_ch, out_ch, k, s, p, bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        return AOQLSQActivationQuantizer(bits=a_bits, signed=True)

    @staticmethod
    def make_head(in_f, out_f, bias):
        from fair_qat.quant_backends import QuantLinearLSQ
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)
