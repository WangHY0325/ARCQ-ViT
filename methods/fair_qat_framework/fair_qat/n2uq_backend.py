"""
N2UQ backend for fair DeiT QAT experiments.

The implementation follows the official N2UQ quantizers:
  - LTQ for activation quantization.
  - HardQuantizeConv-style learnable clipping and G-STE for weights.

The original N2UQ release targets CNNs. This backend adapts the same
quantizers to DeiT linear layers while leaving patch embedding, classifier
head, LayerNorm, Softmax, and GELU in floating point for component parity with
the existing fair DeiT matrix.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class N2UQActivationQuantizer(nn.Module):
    """Official LTQ-style activation quantizer."""

    def __init__(self, bits: int, interval: float | None = None):
        super().__init__()
        self.bits = int(bits)
        self.n_val = 2 ** self.bits - 1
        init_interval = float(interval) if interval is not None else 2.0 / float(self.n_val)
        self.interval = init_interval

        self.start = nn.Parameter(torch.zeros(1))
        self.a = nn.Parameter(torch.full((self.n_val,), init_interval))
        self.scale1 = nn.Parameter(torch.ones(1))
        self.scale2 = nn.Parameter(torch.ones(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.bits >= 32:
            return x

        x_scaled = x * self.scale1.to(device=x.device, dtype=x.dtype)
        start = self.start.to(device=x.device, dtype=x.dtype)
        intervals = self.a.to(device=x.device, dtype=x.dtype)
        eps = torch.as_tensor(1e-3, device=x.device, dtype=x.dtype)
        a_pos = torch.where(intervals > eps, intervals, eps)
        interval = torch.as_tensor(self.interval, device=x.device, dtype=x.dtype)
        two = torch.as_tensor(2.0, device=x.device, dtype=x.dtype)
        zero = torch.as_tensor(0.0, device=x.device, dtype=x.dtype)

        x_forward = x_scaled
        x_backward = x_scaled
        step_right = zero
        thre_forward = start
        thre_backward = start
        for level in range(self.n_val):
            step_right = step_right + interval
            if level == 0:
                thre_forward = start + a_pos[0] / 2.0
                thre_backward = start + 0.0
                x_forward = torch.where(x_scaled > thre_forward, step_right, zero)
                x_backward = torch.where(
                    x_scaled > thre_backward,
                    interval / a_pos[level] * (x_scaled - thre_backward) + step_right - interval,
                    zero,
                )
            else:
                thre_forward = thre_forward + a_pos[level - 1] / 2.0 + a_pos[level] / 2.0
                thre_backward = thre_backward + a_pos[level - 1]
                x_forward = torch.where(x_scaled > thre_forward, step_right, x_forward)
                x_backward = torch.where(
                    x_scaled > thre_backward,
                    interval / a_pos[level] * (x_scaled - thre_backward) + step_right - interval,
                    x_backward,
                )

        thre_backward = thre_backward + a_pos[self.n_val - 1]
        x_backward = torch.where(x_scaled > thre_backward, two, x_backward)
        out = x_forward.detach() + x_backward - x_backward.detach()
        return out * self.scale2.to(device=x.device, dtype=x.dtype)


class N2UQWeightQuantizer(nn.Module):
    """Official HardQuantizeConv weight rule adapted to nn.Linear."""

    def __init__(self, bits: int, clip_val: float = 2.0):
        super().__init__()
        self.bits = int(bits)
        self.clip_val = nn.Parameter(torch.tensor([float(clip_val)]))

    def forward(self, weight: torch.Tensor) -> torch.Tensor:
        if self.bits >= 32:
            return weight

        clip_val = torch.clamp(self.clip_val.to(device=weight.device, dtype=weight.dtype), min=1e-6)
        # Official N2UQ HardQuantizeConv scaling:
        # gamma = (2^b - 1) / 2^(b - 1), independent of layer width.
        gamma = (2.0 ** self.bits - 1.0) / (2.0 ** (self.bits - 1))
        scaling_factor = gamma * weight.detach().abs().mean(dim=1, keepdim=True).clamp_min(1e-8)
        scaled_weight = weight / scaling_factor

        clipped = torch.where(scaled_weight < clip_val / 2.0, scaled_weight, clip_val / 2.0)
        clipped = torch.where(clipped > -clip_val / 2.0, clipped, -clip_val / 2.0)
        n = (2 ** self.bits - 1) / clip_val
        q_no_grad = scaling_factor * (
            torch.round((clipped + clip_val / 2.0) * n) / n - clip_val / 2.0
        )

        return q_no_grad.detach() - scaled_weight.detach() + scaled_weight


class N2UQLinear(nn.Module):
    def __init__(self, in_features: int, out_features: int, bias: bool, w_bits: int, a_bits: int):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.act_quant = N2UQActivationQuantizer(a_bits)
        self.weight_quant = N2UQWeightQuantizer(w_bits)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)


class N2UQBackend:
    name = "n2uq"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return N2UQLinear(in_f, out_f, bias=bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        return N2UQActivationQuantizer(a_bits)

    @staticmethod
    def make_head(in_f, out_f, bias):
        return nn.Linear(in_f, out_f, bias=bias)
