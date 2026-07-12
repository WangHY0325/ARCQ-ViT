"""
AOQ (Alternating Optimization of Quantization) for ViT QAT.

Implementation based on official AOQ paper and AO_QAT/quan/quantizer.py.

Granularity: per_tensor (scalar step_size per layer), aligned with official AOQ.
Per-channel is NOT used in the default configuration.

Reference:
  - Paper: Alternating Optimization for Learned Step Size Quantization
  - Official: https://github.com/ModelTC/AOQ
  - AO_QAT/quan/quantizer.py: AOQ class with per-tensor level/threshold
"""

from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _make_level_codes(bits: int) -> torch.Tensor:
    """
    Symmetric mid-rise level codes for AOQ.

    For 2-bit: [-1.5, -0.5, 0.5, 1.5]
    For 3-bit: [-3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.5, 3.5]
    For 4-bit: [-7.5, -6.5, ..., -0.5, 0.5, ..., 6.5, 7.5]

    Formula: arange(L) - (L/2 - 0.5)  where L = 2**bits
    """
    L = 2 ** bits
    return torch.arange(L).float() - (L / 2.0 - 0.5)


def _make_threshold_codes(bits: int) -> torch.Tensor:
    """
    Symmetric threshold codes (midpoints between levels).

    For 2-bit: [-1.0, 0.0, 1.0]
    For 3-bit: [-3, -2, -1, 0, 1, 2, 3]
    For 4-bit: [-7, -6, ..., -1, 0, 1, ..., 6]

    Formula: arange(1, L) - L/2  where L = 2**bits
    """
    L = 2 ** bits
    return torch.arange(1, L).float() - L / 2.0


class AOQWeightQuantizer(nn.Module):
    """
    AOQ weight quantizer with per-tensor granularity.

    Three-stage training schedule:
      Stage 1 (0% to 20%): levels frozen at init * factor, thresholds scale with factor.
      Stage 2 (20% to 60%): levels learnable via STE, thresholds scale with alpha.
      Stage 3 (60% to 100%): levels learnable, thresholds frozen, dampening loss active.

    Args:
        bits: Quantization bitwidth (2, 3, or 4).
        aoq_granularity: Must be 'per_tensor'. 'per_channel' is reserved for future ablation.
        total_epochs: Total training epochs for schedule computation.
        stage1_ratio: Fraction of epochs for stage 1 (default 0.2).
        stage2_ratio: Fraction of epochs for stage 2 (default 0.4).
        dampen_lambda: Weight for dampening regularization loss (default 0.01).
        eps: Small constant for numerical stability.
    """

    def __init__(
        self,
        bits: int = 4,
        aoq_granularity: str = "per_tensor",
        total_epochs: int = 200,
        stage1_ratio: float = 0.2,
        stage2_ratio: float = 0.4,
        dampen_lambda: float = 0.01,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bits = int(bits)
        self.eps = float(eps)
        self.dampen_lambda = float(dampen_lambda)
        self.total_epochs = int(total_epochs)
        self.stage1_epochs = int(total_epochs * stage1_ratio)
        self.stage2_epochs = int(total_epochs * stage2_ratio)
        self.initialized = False

        L = 2 ** bits

        # Base codes (non-learnable, fixed)
        self.register_buffer("base_level_codes", _make_level_codes(bits))
        self.register_buffer("base_threshold_codes", _make_threshold_codes(bits))

        # Learnable parameters (scalar per-tensor)
        self.sle0 = nn.Parameter(torch.ones(1), requires_grad=False)  # level init interval
        self.sth0 = nn.Parameter(torch.ones(1), requires_grad=False)  # threshold init interval

        # Threshold buffer (computed from sth0 * base_threshold_codes * alpha)
        self.register_buffer("thresholds", torch.zeros(L - 1))

        # Level parameter (learnable in stage 2+)
        self.levels = nn.Parameter(torch.zeros(L))

        # Alpha for threshold scaling (cosine schedule)
        self.alpha = nn.Parameter(torch.ones(1), requires_grad=False)

        # Current stage tracker
        self.register_buffer("_current_epoch", torch.zeros(1, dtype=torch.long))
        self.register_buffer("_dampen_loss", torch.zeros(1))

    def set_epoch(self, epoch: int) -> None:
        """Update current epoch for stage scheduling."""
        self._current_epoch.fill_(epoch)
        self._update_stage()

    def _update_stage(self) -> None:
        """Update stage-dependent parameters based on current epoch."""
        epoch = int(self._current_epoch.item())
        if epoch > self.total_epochs:
            epoch = self.total_epochs

        # Update alpha with cosine schedule
        if epoch <= self.stage1_epochs:
            # Stage 1: alpha follows cosine
            alpha = 0.35 * math.cos(2 * math.pi * epoch / (self.stage1_epochs * 2 if self.stage1_epochs > 0 else 1)) + 0.65
            self.alpha.data.fill_(alpha)
        elif epoch <= self.stage1_epochs + self.stage2_epochs:
            # Stage 2: alpha continues cosine
            alpha = 0.35 * math.cos(2 * math.pi * epoch / 100.0) + 0.65
            self.alpha.data.fill_(alpha)
        else:
            # Stage 3: alpha fixed
            pass

    @torch.no_grad()
    def _initialize(self, weight: torch.Tensor) -> None:
        """Initialize AOQ parameters from weight statistics.

        Uses torch.std(weight) for per-tensor initialization, matching official AOQ.
        """
        if self.bits >= 16:
            self.initialized = True
            return

        std = torch.std(weight, unbiased=False)
        init_interval = std / (2 ** (self.bits - 2))
        init_interval = init_interval.clamp_min(self.eps).item()

        self.sle0.data.fill_(init_interval)
        self.sth0.data.fill_(init_interval)

        # Initialize levels and thresholds
        self.levels.data.copy_(self.base_level_codes * init_interval)
        self.thresholds.data.copy_(self.base_threshold_codes * init_interval)

        self.initialized = True

    def _get_current_factor(self) -> float:
        """Get scaling factor for current epoch schedule."""
        epoch = int(self._current_epoch.item())
        if epoch <= self.stage1_epochs:
            # Stage 1: factor grows from small to 1.0
            progress = epoch / max(self.stage1_epochs, 1)
            return max(0.01, progress)
        else:
            return 1.0

    def get_regularization_loss(self) -> torch.Tensor:
        """Return accumulated dampening loss for this quantizer."""
        return self._dampen_loss.clone()

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        """
        AOQ quantization forward pass.

        Args:
            weight: Full weight tensor (per-tensor statistics used).

        Returns:
            Quantized weight with STE gradient.
        """
        if not self.initialized or self.bits >= 16:
            return weight

        epoch = int(self._current_epoch.item())

        # ---- Update level/threshold based on stage ----
        if epoch <= self.stage1_epochs:
            # Stage 1: levels frozen at sle0 * base_codes * factor
            factor = self._get_current_factor()
            with torch.no_grad():
                self.levels.data.copy_(self.base_level_codes * self.sle0 * factor)
            self.thresholds.data.copy_(self.base_threshold_codes * self.sth0 * factor)

        elif epoch <= self.stage1_epochs + self.stage2_epochs:
            # Stage 2: levels learnable, thresholds scale with alpha
            self.thresholds.data.copy_(self.base_threshold_codes * self.sth0 * self.alpha)

        else:
            # Stage 3: levels learnable, thresholds frozen, dampening active
            self.thresholds.data.copy_(self.base_threshold_codes * self.sth0 * self.alpha)

        # ---- Quantization ----
        w_flat = weight.reshape(-1)
        codes = torch.bucketize(w_flat, self.thresholds, right=True)
        w_q_flat = self.levels[codes]
        w_q = w_q_flat.reshape_as(weight)

        # ---- Dampening loss (stage 3 only) ----
        if epoch > self.stage1_epochs + self.stage2_epochs:
            # Compute cluster assignment using current levels as cluster centers
            cluster_flat = self.levels[codes]
            raw_dampen = F.mse_loss(w_flat, cluster_flat.detach(), reduction="mean")
            self._dampen_loss = self._dampen_loss + self.dampen_lambda * raw_dampen.detach()

        # ---- STE ----
        return weight + (w_q - weight).detach()


class AOQActQuantizer(nn.Module):
    """
    AOQ activation quantizer (placeholder / simple STE).

    For the first baseline round, we use a simple per-tensor min-max quantizer
    for activations, keeping the AOQ-specific optimizations for weights only.
    This can be upgraded to a learned activation quantizer in future rounds.
    """

    def __init__(self, bits: int = 4, aoq_granularity: str = "per_tensor", eps: float = 1e-8):
        super().__init__()
        self.bits = int(bits)
        self.eps = float(eps)
        self.initialized = False

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.initialized or self.bits >= 16:
            return x
        # Simple per-tensor uniform symmetric quantization for activations
        max_val = x.detach().abs().max() + self.eps
        scale = max_val / (2 ** (self.bits - 1))
        q = (x / scale).round().clamp(-2 ** (self.bits - 1), 2 ** (self.bits - 1) - 1)
        return q * scale


def get_aoq_regularization_loss(model: nn.Module) -> torch.Tensor:
    """
    Aggregate AOQ dampening loss across all AOQWeightQuantizers in the model.

    Returns scalar tensor (0 if no AOQ quantizers found).
    """
    total = torch.tensor(0.0, device=next(model.parameters()).device)
    count = 0
    for module in model.modules():
        if isinstance(module, AOQWeightQuantizer):
            total = total + module.get_regularization_loss()
            count += 1
    return total / max(count, 1)
