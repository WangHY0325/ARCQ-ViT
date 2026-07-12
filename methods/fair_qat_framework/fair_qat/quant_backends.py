"""
Fair QAT quantization backends.

Three backends exposing the same interface, differing only in quantizer algorithm:
  - QViTBackend  (third_party/qvit)
  - LSQBackend   (quant.lsq)
  - DDFZBackend  (quant.dcddfz)

Interface per backend:
  make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits) -> nn.Module
  make_linear(in_f, out_f, bias, w_bits, a_bits) -> nn.Module
  make_act(a_bits, shape_hint=None) -> nn.Module
  make_head(in_f, out_f, bias) -> nn.Module (fixed W8A8)
"""

import sys, os

# --- project root ---
_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.abspath(os.path.join(_PROJ, "..", ".."))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

import torch
import torch.nn as nn
import torch.nn.functional as F


# ======================================================================
#  Q-ViT Backend
# ======================================================================

def _first_existing_dir(*paths):
    for path in paths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            return path
    return os.path.abspath(paths[0])


_QVIT_ROOT = _first_existing_dir(
    os.path.join(_REPO_ROOT, "duibi_method_vit", "third_party", "qvit"),
    os.path.join(_PROJ, "..", "third_party", "qvit"),
)
if _QVIT_ROOT not in sys.path:
    sys.path.insert(0, _QVIT_ROOT)


class QViTLinearLastDim(nn.Module):
    """
    Adapter around official Q-ViT LinearQ for timm tensors.

    Q-ViT's ActQ handles 2D inputs and NCHW 4D inputs. Swin patch merging uses
    NHWC-like tensors `[B, H, W, C]`, so the official ActQ broadcasts alpha on
    the wrong dimension. Flattening all leading dimensions preserves LinearQ's
    math while making every timm Linear operate on the semantic last dimension.
    """

    def __init__(self, in_f, out_f, bias, w_bits):
        super().__init__()
        from Quant import LinearQ
        self.linear = LinearQ(in_f, out_f, bias=bias, nbits_w=w_bits)

    def forward(self, x):
        if x.dim() <= 2:
            return self.linear(x)
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        y_flat = self.linear(x_flat)
        return y_flat.reshape(*leading, y_flat.shape[-1])


class QViTBackend:
    name = "qvit"

    @staticmethod
    def _import():
        from _quan_base import Qmodes
        from Quant import Conv2dQ, LinearQ, ActQ
        return Qmodes, Conv2dQ, LinearQ, ActQ

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        _, Conv2dQ, _, _ = QViTBackend._import()
        # QViT Conv2dQ uses w_bits for both weight and act
        return Conv2dQ(in_ch, out_ch, k, stride=s, padding=p, bias=bias, nbits_w=w_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return QViTLinearLastDim(in_f, out_f, bias=bias, w_bits=w_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        _, _, _, ActQ = QViTBackend._import()
        # Q-ViT ActQ alpha is sized by in_features (num_heads for Q/K/V/attn)
        in_features = int(shape_hint) if shape_hint is not None else 1
        return ActQ(in_features=in_features, nbits_a=a_bits)

    @staticmethod
    def make_head(in_f, out_f, bias):
        """Head: shared LSQ W8A8 across all backends for fair comparison."""
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)


# ======================================================================
#  LSQ Backend
# ======================================================================

class QuantConv2dLSQ(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        super().__init__()
        from quant.lsq import LSQQuantizer
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)
        self.act_quant = LSQQuantizer(bits=a_bits, signed=True, per_channel=False)
        self.weight_quant = LSQQuantizer(bits=w_bits, signed=True, per_channel=True, ch_axis=0)

    def forward(self, x):
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.conv.weight)
        return F.conv2d(x_q, w_q, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


class QuantLinearLSQ(nn.Module):
    def __init__(self, in_f, out_f, bias, w_bits, a_bits):
        super().__init__()
        from quant.lsq import LSQQuantizer
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.act_quant = LSQQuantizer(bits=a_bits, signed=True, per_channel=False)
        self.weight_quant = LSQQuantizer(bits=w_bits, signed=True, per_channel=True, ch_axis=0)

    def forward(self, x):
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)


class LSQBackend:
    name = "lsq"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return QuantConv2dLSQ(in_ch, out_ch, k, s, p, bias, w_bits, a_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return QuantLinearLSQ(in_f, out_f, bias, w_bits, a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from quant.lsq import LSQQuantizer
        return LSQQuantizer(bits=a_bits, signed=True, per_channel=False)

    @staticmethod
    def make_head(in_f, out_f, bias):
        """Head: shared LSQ W8A8 across all backends for fair comparison."""
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)  # already LSQ8


# ======================================================================
#  Diagonal DC Conditioner (SmoothQuant-style channel balancing)
# ======================================================================

class DiagonalDCConditioner(nn.Module):
    """
    Lightweight diagonal distribution conditioning.

    Estimates per-input-channel scale d and applies:
      Linear: x*d, W/d
      Conv2d: x_c*d_c, W_c/d_c

    Keeps FP computation approximately equivalent before quantization,
    while making activation/weight residual distributions easier for DDFZ.
    """

    def __init__(
        self,
        num_features: int,
        momentum: float = 0.95,
        eps: float = 1e-6,
        clamp_min: float = 0.25,
        clamp_max: float = 4.0,
        update_interval: int = 100,
        freeze_after: int = 1500,
    ):
        super().__init__()
        self.num_features = int(num_features)
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.clamp_min = float(clamp_min)
        self.clamp_max = float(clamp_max)
        self.update_interval = int(update_interval)
        self.freeze_after = int(freeze_after)
        self.register_buffer("d", torch.ones(num_features))
        self.step = 0
        self.ready = False

    @torch.no_grad()
    def maybe_update_linear(self, x, w):
        if (not self.training) or (self.freeze_after >= 0 and self.step > self.freeze_after):
            self.step += 1
            return
        if self.ready and self.step % max(1, self.update_interval) != 0:
            self.step += 1
            return

        xf = x.detach().float().reshape(-1, x.shape[-1])
        wf = w.detach().float()

        act_rms = (xf.square().mean(dim=0) + self.eps).sqrt()
        w_rms = (wf.square().mean(dim=0) + self.eps).sqrt()

        new_d = (w_rms / act_rms).sqrt().clamp(self.clamp_min, self.clamp_max)

        if self.ready:
            self.d.mul_(self.momentum).add_(new_d.to(self.d.device), alpha=1.0 - self.momentum)
        else:
            self.d.copy_(new_d.to(self.d.device))
            self.ready = True

        if self.step <= 3 or self.step % 500 == 0:
            print(
                f"[VIT_DDFZ_DC] linear step={self.step} "
                f"d_min={float(self.d.min()):.4f} d_max={float(self.d.max()):.4f} "
                f"d_mean={float(self.d.mean()):.4f}"
            )
        self.step += 1

    @torch.no_grad()
    def maybe_update_conv(self, x, w):
        if (not self.training) or (self.freeze_after >= 0 and self.step > self.freeze_after):
            self.step += 1
            return
        if self.ready and self.step % max(1, self.update_interval) != 0:
            self.step += 1
            return

        xf = x.detach().float()
        wf = w.detach().float()

        act_rms = (xf.square().mean(dim=(0, 2, 3)) + self.eps).sqrt()
        w_rms = (wf.square().mean(dim=(0, 2, 3)) + self.eps).sqrt()

        new_d = (w_rms / act_rms).sqrt().clamp(self.clamp_min, self.clamp_max)

        if self.ready:
            self.d.mul_(self.momentum).add_(new_d.to(self.d.device), alpha=1.0 - self.momentum)
        else:
            self.d.copy_(new_d.to(self.d.device))
            self.ready = True

        if self.step <= 3 or self.step % 500 == 0:
            print(
                f"[VIT_DDFZ_DC] conv step={self.step} "
                f"d_min={float(self.d.min()):.4f} d_max={float(self.d.max()):.4f} "
                f"d_mean={float(self.d.mean()):.4f}"
            )
        self.step += 1

    def linear_x(self, x):
        d = self.d.to(device=x.device, dtype=x.dtype)
        return x * d.view(*([1] * (x.dim() - 1)), -1)

    def linear_w(self, w):
        d = self.d.to(device=w.device, dtype=w.dtype)
        return w / d.view(1, -1).clamp_min(self.eps)

    def conv_x(self, x):
        d = self.d.to(device=x.device, dtype=x.dtype)
        return x * d.view(1, -1, 1, 1)

    def conv_w(self, w):
        d = self.d.to(device=w.device, dtype=w.dtype)
        return w / d.view(1, -1, 1, 1).clamp_min(self.eps)


# ======================================================================
#  PC-DDFZ Linear wrapper
# ======================================================================

class QuantLinearPCDDFZ(nn.Module):
    def __init__(
        self,
        in_f,
        out_f,
        bias,
        w_bits,
        a_bits,
        group_size=64,
        use_dc=False,
        compile_steps="auto",
    ):
        super().__init__()
        from quant.dcddfz import DDFZPCActQuantizer, DDFZPCWeightQuantizer
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.use_dc = bool(use_dc)
        self.dc = DiagonalDCConditioner(in_f) if self.use_dc else None

        self.act_quant = DDFZPCActQuantizer(
            bits=a_bits,
            group_size=group_size,
            compile_steps=compile_steps,
        )
        self.weight_quant = DDFZPCWeightQuantizer(
            bits=w_bits,
            group_size=group_size,
            compile_steps=compile_steps,
            freeze_codebook=True,
        )

    def forward(self, x):
        w = self.linear.weight

        if self.use_dc and self.dc is not None:
            self.dc.maybe_update_linear(x, w)
            x = self.dc.linear_x(x)
            w = self.dc.linear_w(w)

        x_q = self.act_quant(x)
        w_q = self.weight_quant(w)
        return F.linear(x_q, w_q, self.linear.bias)


# ======================================================================
#  PC-DDFZ Conv2d wrapper
# ======================================================================

class QuantConv2dPCDDFZ(nn.Module):
    def __init__(
        self,
        in_ch,
        out_ch,
        k,
        s,
        p,
        bias,
        w_bits,
        a_bits,
        group_size=64,
        use_dc=False,
        compile_steps="auto",
    ):
        super().__init__()
        from quant.dcddfz import DDFZPCActQuantizer, DDFZPCWeightQuantizer
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)
        self.use_dc = bool(use_dc)
        self.dc = DiagonalDCConditioner(in_ch) if self.use_dc else None

        act_gs = min(group_size, in_ch) if in_ch > 0 else group_size
        self.act_quant = DDFZPCActQuantizer(
            bits=a_bits,
            group_size=act_gs,
            compile_steps=compile_steps,
        )
        self.weight_quant = DDFZPCWeightQuantizer(
            bits=w_bits,
            group_size=group_size,
            compile_steps=compile_steps,
            freeze_codebook=True,
        )

    def forward(self, x):
        w = self.conv.weight

        if self.use_dc and self.dc is not None:
            self.dc.maybe_update_conv(x, w)
            x = self.dc.conv_x(x)
            w = self.dc.conv_w(w)

        # DDFZ activation groups channels, so use NHWC
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()
        x_q = self.act_quant(x_nhwc)
        x_q = x_q.permute(0, 3, 1, 2).contiguous()

        w_2d = w.reshape(w.shape[0], -1)
        w_q_2d = self.weight_quant(w_2d)
        w_q = w_q_2d.reshape_as(w)

        return F.conv2d(
            x_q, w_q, self.conv.bias,
            self.conv.stride, self.conv.padding,
            self.conv.dilation, self.conv.groups,
        )


# ======================================================================
#  PC-DDFZ NoDC Backend
# ======================================================================

class PCDDFZNoDCBackend:
    name = "pcddfz_nodc"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return QuantConv2dPCDDFZ(
            in_ch, out_ch, k, s, p, bias,
            w_bits=w_bits, a_bits=a_bits,
            use_dc=False,
        )

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return QuantLinearPCDDFZ(
            in_f, out_f, bias,
            w_bits=w_bits, a_bits=a_bits,
            use_dc=False,
        )

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from quant.dcddfz import DDFZPCActQuantizer
        return DDFZPCActQuantizer(
            bits=a_bits,
            group_size=64,
            compile_steps="auto",
        )

    @staticmethod
    def make_head(in_f, out_f, bias):
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)


# ======================================================================
#  PC-DDFZ DC Backend
# ======================================================================

class PCDDFZDCBackend:
    name = "pcddfz_dc"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return QuantConv2dPCDDFZ(
            in_ch, out_ch, k, s, p, bias,
            w_bits=w_bits, a_bits=a_bits,
            use_dc=True,
        )

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return QuantLinearPCDDFZ(
            in_f, out_f, bias,
            w_bits=w_bits, a_bits=a_bits,
            use_dc=True,
        )

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from quant.dcddfz import DDFZPCActQuantizer
        return DDFZPCActQuantizer(
            bits=a_bits,
            group_size=64,
            compile_steps="auto",
        )

    @staticmethod
    def make_head(in_f, out_f, bias):
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)

class QuantConv2dDDFZ(nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, bias, w_bits, a_bits, group_size=64):
        super().__init__()
        from quant.dcddfz import DDFZActQuantizer, DDFZWeightQuantizer
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)
        # Activation: group along channel dim (NHWC), so group_size capped by in_ch
        act_gs = min(group_size, in_ch) if in_ch > 0 else group_size
        self.act_quant = DDFZActQuantizer(bits=a_bits, group_size=act_gs)
        self.weight_quant = DDFZWeightQuantizer(bits=w_bits, group_size=group_size)

    def forward(self, x):
        # DDFZ groups along last dim → permute to NHWC for channel grouping
        x_nhwc = x.permute(0, 2, 3, 1).contiguous()  # (B, H, W, C)
        x_q = self.act_quant(x_nhwc)
        x_q = x_q.permute(0, 3, 1, 2).contiguous()     # back to NCHW

        w = self.conv.weight  # (out_ch, in_ch, k, k)
        w_2d = w.reshape(w.shape[0], -1)  # (out_ch, in_ch*k*k)
        w_q_2d = self.weight_quant(w_2d)
        w_q = w_q_2d.reshape_as(w)
        return F.conv2d(x_q, w_q, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


class QuantLinearDDFZ(nn.Module):
    def __init__(self, in_f, out_f, bias, w_bits, a_bits, group_size=64):
        super().__init__()
        from quant.dcddfz import DDFZActQuantizer, DDFZWeightQuantizer
        self.linear = nn.Linear(in_f, out_f, bias=bias)
        self.act_quant = DDFZActQuantizer(bits=a_bits, group_size=group_size)
        self.weight_quant = DDFZWeightQuantizer(bits=w_bits, group_size=group_size)

    def forward(self, x):
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)


class DDFZBackend:
    name = "dcddfz"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return QuantConv2dDDFZ(in_ch, out_ch, k, s, p, bias, w_bits, a_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return QuantLinearDDFZ(in_f, out_f, bias, w_bits, a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from quant.dcddfz import DDFZActQuantizer
        return DDFZActQuantizer(bits=a_bits, group_size=64)

    @staticmethod
    def make_head(in_f, out_f, bias):
        """Head: shared LSQ W8A8 across all backends for fair comparison."""
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)


# ======================================================================
#  FIMA-Q Backend (PTQ only — not used in QAT training)
# ======================================================================

class FIMAQBackend:
    name = "fimaq"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        from fair_qat.fimaq_adapter import FIMAQuantConv2d
        return FIMAQuantConv2d(in_ch, out_ch, k, stride=s, padding=p,
                               bias=bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        from fair_qat.fimaq_adapter import FIMAQuantLinear
        return FIMAQuantLinear(in_f, out_f, bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from fair_qat.fimaq_adapter import FIMAActQuantizer
        return FIMAActQuantizer(bits=a_bits)

    @staticmethod
    def make_head(in_f, out_f, bias):
        """Head: W8A8 via FIMA-Q PTQ wrappers."""
        from fair_qat.fimaq_adapter import FIMAQuantLinear
        return FIMAQuantLinear(in_f, out_f, bias, w_bits=8, a_bits=8)


# ======================================================================
#  GPLQ Backend (activation QAT + weight PTQ hybrid)
# ======================================================================

class GPLQBackend:
    name = "gplq"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        from fair_qat.gplq_adapter import GPLQQuantConv2d
        return GPLQQuantConv2d(in_ch, out_ch, k, stride=s, padding=p,
                               bias=bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        from fair_qat.gplq_adapter import GPLQQuantLinear
        return GPLQQuantLinear(in_f, out_f, bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from fair_qat.gplq_adapter import GPLQActQuantizer
        return GPLQActQuantizer(bits=a_bits)

    @staticmethod
    def make_head(in_f, out_f, bias):
        """Head: shared LSQ W8A8 across all backends for fair comparison."""
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)


# ======================================================================
#  FP32 Backend (real native Layers for FIMA-Q PTQ conversion)
# ======================================================================

class FP32Backend:
    name = "fp32"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return nn.Linear(in_f, out_f, bias=bias)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        return nn.Identity()

    @staticmethod
    def make_head(in_f, out_f, bias):
        return nn.Linear(in_f, out_f, bias=bias)


# ======================================================================
#  AOQ Backend
# ======================================================================

class QuantLinearAOQ(nn.Module):
    """Linear layer with AOQ weight quantization + simple act quant."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        w_bits: int = 4,
        a_bits: int = 4,
        aoq_granularity: str = "per_tensor",
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

        from quant.aoq import AOQWeightQuantizer, AOQActQuantizer
        self.act_quant = AOQActQuantizer(bits=a_bits, aoq_granularity=aoq_granularity)
        self.weight_quant = AOQWeightQuantizer(bits=w_bits, aoq_granularity=aoq_granularity)

        self._init_weights()

    def _init_weights(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x):
        w_q = self.weight_quant(self.weight)
        return F.linear(x, w_q, self.bias)


class AOQBackend:
    name = "aoq"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        # Use simple LSQ conv for AOQ baseline (AOQ focuses on linear/weight)
        return QuantConv2dLSQ(in_ch, out_ch, k, s, p, bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return QuantLinearAOQ(in_f, out_f, bias, w_bits=w_bits, a_bits=a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        from quant.aoq import AOQActQuantizer
        return AOQActQuantizer(bits=a_bits)

    @staticmethod
    def make_head(in_f, out_f, bias):
        return QuantLinearLSQ(in_f, out_f, bias, w_bits=8, a_bits=8)


# ======================================================================
#  Registry
# ======================================================================

BACKENDS = {
    "fp32": FP32Backend,
    "dcddfz": DDFZBackend,
    "pcddfz_nodc": PCDDFZNoDCBackend,
    "pcddfz_dc": PCDDFZDCBackend,
}


def get_backend(method: str):
    if method not in BACKENDS:
        raise ValueError(f"Unknown method '{method}'. Choose from {list(BACKENDS)}")
    return BACKENDS[method]
