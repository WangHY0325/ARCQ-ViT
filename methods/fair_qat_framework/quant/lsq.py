"""
LSQ (Learned Step Size Quantization) for ViT QAT.

Implements LSQQuantizer with STE gradient scaling for weight and activation
quantization. Supports per-channel (weight) and per-tensor (activation) modes.

Note: First round uses simple STE scale; custom LSQ grad scaling is deferred.
"""

import math
import torch
import torch.nn as nn


class RoundSTE(torch.autograd.Function):
    """Straight-through estimator for rounding."""

    @staticmethod
    def forward(ctx, x):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def grad_scale(x: torch.Tensor, scale: float) -> torch.Tensor:
    """Gradient scaling: x with gradient multiplied by scale."""
    return (x - x * scale).detach() + x * scale


class LSQQuantizer(nn.Module):
    """Learned Step Size Quantization.

    Args:
        bits: Quantization bitwidth.
        signed: Whether quantization range is signed.
        per_channel: If True, learn per-channel step sizes (for weights).
        ch_axis: Axis for per-channel quantization (default -1 for last dim).
        eps: Small constant to prevent division by zero.
    """

    def __init__(
        self,
        bits: int = 4,
        signed: bool = True,
        per_channel: bool = False,
        ch_axis: int = -1,
        eps: float = 1e-8,
    ):
        super().__init__()
        self.bits = int(bits)
        self.signed = bool(signed)
        self.per_channel = bool(per_channel)
        self.ch_axis = int(ch_axis)
        self.eps = float(eps)
        self.initialized = False
        self.step = nn.Parameter(torch.ones(1))

    @property
    def qrange(self):
        """Return (qmin, qmax) for the quantization range."""
        if self.signed:
            qn = -2 ** (self.bits - 1)
            qp = 2 ** (self.bits - 1) - 1
        else:
            qn = 0
            qp = 2 ** self.bits - 1
        return qn, qp

    @torch.no_grad()
    def _initialize(self, x: torch.Tensor) -> None:
        """Initialize step size based on input statistics.

        For per_channel mode, computes per-channel mean(|x|).
        For scalar mode, uses global mean(|x|).
        """
        if self.bits >= 16:
            self.initialized = True
            return

        qn, qp = self.qrange
        qp_val = max(qp, 1)

        if self.per_channel:
            # Reduce all dims except the channel axis
            dims = [d for d in range(x.ndim) if d != self.ch_axis]
            init = 2.0 * x.detach().abs().mean(dim=tuple(dims), keepdim=True)
            init = init / math.sqrt(qp_val)
            init = init.clamp_min(self.eps)
            # Reshape step to broadcast correctly
            shape = [1] * x.ndim
            shape[self.ch_axis] = init.shape[self.ch_axis]
            self.step = nn.Parameter(init.view(*shape))
        else:
            init = 2.0 * x.detach().abs().mean() / math.sqrt(qp_val)
            self.step.data.fill_(float(init.clamp_min(self.eps)))

        self.initialized = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Quantize input tensor x.

        Returns x_hat approximating x with STE gradient.
        """
        if self.bits >= 16:
            return x

        if not self.initialized:
            self._initialize(x)

        qn, qp = self.qrange
        qp_val = max(qp, 1)

        # Gradient scaling factor (LSQ paper)
        grad_scale_factor = 1.0 / math.sqrt(max(x.numel() * qp_val, 1))
        step = grad_scale(self.step.clamp_min(self.eps), grad_scale_factor)

        x_scaled = x / step
        x_clamped = torch.clamp(x_scaled, qn, qp)
        q = RoundSTE.apply(x_clamped)
        x_hat = q * step

        return x_hat
