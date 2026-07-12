"""
FIMA-Q adapter for Fair QAT framework.

Wraps third_party/fima_q PTQ SLBatchingQuant layers + UniformQuantizer for
standalone activation quantization. FIMA-Q is a PTQ method — no gradient training.
"""

import sys, os

_FIMA_ROOT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                          "..", "third_party", "fima_q")
if _FIMA_ROOT not in sys.path:
    sys.path.insert(0, _FIMA_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from quantizers.uniform import UniformQuantizer
from quant_layers.linear import PTQSLBatchingQuantLinear


# ---------------------------------------------------------------------------
#  FIMA-Q Activation Quantizer (standalone, for Q/K/V/attn-map positions)
# ---------------------------------------------------------------------------

class FIMAActQuantizer(nn.Module):
    """
    Standalone activation quantizer using FIMA-Q's UniformQuantizer.
    Supports three modes:
      - "raw"            : pass-through, collect calibration samples
      - "quant_forward"  : quantize input
    """

    def __init__(self, bits: int):
        super().__init__()
        self.mode = "raw"
        self.calibrated = False
        self.samples = []  # collected calibration samples (List[torch.Tensor])
        self.quantizer = UniformQuantizer(n_bits=bits, symmetric=True, channel_wise=False)
        self.max_calib_elems = int(os.environ.get("FIMAQ_MAX_ACT_ELEMS", 200000))

    def collect_sample(self, x: torch.Tensor):
        """Store a detached CPU sample of x for later calibration (capped)."""
        flat = x.detach().float().reshape(-1).cpu()
        if self.max_calib_elems > 0 and flat.numel() > self.max_calib_elems:
            idx = torch.randperm(flat.numel())[:self.max_calib_elems]
            flat = flat[idx]
        self.samples.append(flat.contiguous())

    def calibrate(self):
        """Compute scale from collected samples and mark as calibrated."""
        if not self.samples:
            raise RuntimeError("FIMAActQuantizer: no calibration samples collected.")
        all_samples = torch.cat([s.flatten() for s in self.samples], dim=0)
        self.samples.clear()
        n_levels = self.quantizer.n_levels
        scale_val = all_samples.abs().max() / (n_levels - 0.5)
        self.quantizer.scale = nn.Parameter(scale_val.reshape(1), requires_grad=False)
        self.quantizer.inited = True
        self.calibrated = True
        self.mode = "quant_forward"

    def forward(self, x):
        if self.mode == "raw":
            return x
        return self.quantizer(x)


# ---------------------------------------------------------------------------
#  FIMA-Q Quantized Linear (wraps PTQSLBatchingQuantLinear)
# ---------------------------------------------------------------------------

class FIMAQuantLinear(nn.Module):
    """
    Wraps FIMA-Q's PTQSLBatchingQuantLinear for the Fair QAT model.
    Builds in 'raw' mode; calibration and hyperparameter_searching happen externally.
    """

    def __init__(self, in_features, out_features, bias, w_bits, a_bits):
        super().__init__()
        self.layer = PTQSLBatchingQuantLinear(
            in_features=in_features,
            out_features=out_features,
            bias=bias,
            mode="raw",
            w_bit=w_bits,
            a_bit=a_bits,
            metric="mse",
            calib_batch_size=32,
            search_round=1,
            eq_n=50,
            n_V=1,
        )
        # hook handles for calibration
        self._in_handle = None
        self._out_handle = None
        self._collected_inputs = []
        self._collected_outputs = []

    def forward(self, x):
        return self.layer(x)

    def register_calib_hooks(self):
        """Register hooks to collect per-module input/output during calibration."""
        self._collected_inputs = []
        self._collected_outputs = []

        def in_hook(m, inp):
            self._collected_inputs.append(inp[0].detach().cpu())

        def out_hook(m, inp, out):
            self._collected_outputs.append(out.detach().cpu())

        self._in_handle = self.layer.register_forward_pre_hook(in_hook)
        self._out_handle = self.layer.register_forward_hook(out_hook)

    def remove_calib_hooks(self):
        if self._in_handle is not None:
            self._in_handle.remove()
            self._in_handle = None
        if self._out_handle is not None:
            self._out_handle.remove()
            self._out_handle = None

    def calibrate(self):
        """Assemble collected i/o, assign to layer, run hyperparameter_searching."""
        if not self._collected_inputs or not self._collected_outputs:
            raise RuntimeError("FIMAQuantLinear: no calibration data collected.")
        self.layer.raw_input = torch.cat(self._collected_inputs, dim=0)
        self.layer.raw_out = torch.cat(self._collected_outputs, dim=0)
        self.layer.hyperparameter_searching()
        self.layer.mode = "quant_forward"
        self._collected_inputs.clear()
        self._collected_outputs.clear()


# ---------------------------------------------------------------------------
#  FIMA-Q Quantized Conv2d (standalone, no PTQSLBatchingQuantConv2d dependency)
# ---------------------------------------------------------------------------

class FIMAQuantConv2d(nn.Module):
    """
    FIMA-Q PTQ Conv2d — built from nn.Conv2d + UniformQuantizer (act + weight).
    Does NOT rely on PTQSLBatchingQuantConv2d (which has hyperparameter_searching
    that fails at runtime due to internal dependency issues).
    Calibration uses min-max init from collected samples, same as FIMAActQuantizer.
    """

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, w_bits=4, a_bits=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.act_quant = UniformQuantizer(n_bits=a_bits, symmetric=True, channel_wise=False)
        self.weight_quant = UniformQuantizer(n_bits=w_bits, symmetric=True, channel_wise=True)

        # Manually create scale params (UniformQuantizer expects .scale to be set)
        self.act_quant.scale = nn.Parameter(torch.zeros((1, 1, 1, 1)))
        self.weight_quant.scale = nn.Parameter(torch.zeros((out_channels, 1)))

        self._in_handle = None
        self._collected_inputs = []

    def forward(self, x):
        x_q = self.act_quant(x) if self.act_quant.inited else x
        w = self.conv.weight
        w_2d = w.reshape(w.shape[0], -1)
        w_q_2d = self.weight_quant(w_2d) if self.weight_quant.inited else w_2d
        w_q = w_q_2d.reshape_as(w)
        return F.conv2d(x_q, w_q, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)

    def register_calib_hooks(self):
        self._collected_inputs = []

        def in_hook(m, inp):
            self._collected_inputs.append(inp[0].detach().cpu())

        self._in_handle = self.register_forward_pre_hook(in_hook)

    def remove_calib_hooks(self):
        if self._in_handle is not None:
            self._in_handle.remove()
            self._in_handle = None

    def calibrate(self):
        if not self._collected_inputs:
            raise RuntimeError("FIMAQuantConv2d: no calibration data collected.")

        # Act quant: min-max init from collected inputs
        all_inputs = torch.cat(self._collected_inputs, dim=0)
        act_scale = (all_inputs.abs().max() / (self.act_quant.n_levels - 0.5)).detach()
        self.act_quant.scale.data.fill_(act_scale.item())
        self.act_quant.inited = True

        # Weight quant: per-channel min-max init
        w = self.conv.weight.data
        w_2d = w.reshape(w.shape[0], -1)
        w_max = w_2d.abs().amax(dim=-1, keepdim=True)
        w_scale = w_max / (self.weight_quant.n_levels - 0.5)
        self.weight_quant.scale.data.copy_(w_scale)
        self.weight_quant.inited = True

        self._collected_inputs.clear()


# ---------------------------------------------------------------------------
#  Model conversion helper
# ---------------------------------------------------------------------------

def _unwrap_conv_module(m):
    """Get the underlying nn.Conv2d from a wrapper."""
    if isinstance(m, nn.Conv2d):
        return m
    if hasattr(m, "conv") and isinstance(m.conv, nn.Conv2d):
        return m.conv
    raise TypeError(f"Expected Conv2d or wrapper with .conv, got {type(m)}")


def _unwrap_linear_module(m):
    """Get the underlying nn.Linear from a wrapper."""
    if isinstance(m, nn.Linear):
        return m
    if hasattr(m, "linear") and isinstance(m.linear, nn.Linear):
        return m.linear
    if hasattr(m, "layer") and isinstance(m.layer, nn.Linear):
        return m.layer
    raise TypeError(f"Expected Linear or wrapper with .linear/.layer, got {type(m)}")


def convert_fair_model_to_fimaq(model, w_bits, a_bits):
    """
    Replaces quantized modules in a FP32 FairLowBitVisionTransformer with
    FIMA-Q PTQ wrappers. The model must have been built with FP32Backend
    so that patch_embed.proj / attn.qkv / etc. are real nn.Conv2d / nn.Linear.
    """
    # patch_embed.proj
    old_conv = _unwrap_conv_module(model.patch_embed.proj)
    new_conv = FIMAQuantConv2d(
        old_conv.in_channels, old_conv.out_channels,
        old_conv.kernel_size[0],
        stride=old_conv.stride[0], padding=old_conv.padding[0],
        bias=(old_conv.bias is not None),
        w_bits=w_bits, a_bits=a_bits,
    )
    with torch.no_grad():
        new_conv.conv.weight.copy_(old_conv.weight)
        if old_conv.bias is not None:
            new_conv.conv.bias.copy_(old_conv.bias)
    model.patch_embed.proj = new_conv

    for block in model.blocks:
        attn = block.attn
        mlp = block.mlp

        _replace_linear(attn, "qkv", w_bits, a_bits)
        _replace_linear(attn, "proj", w_bits, a_bits)

        for aname in ("q_act", "k_act", "v_act", "attn_act"):
            setattr(attn, aname, FIMAActQuantizer(bits=a_bits))

        _replace_linear(mlp, "fc1", w_bits, a_bits)
        _replace_linear(mlp, "fc2", w_bits, a_bits)

    # head (W8A8)
    old_head = _unwrap_linear_module(model.head)
    new_head = FIMAQuantLinear(
        old_head.in_features, old_head.out_features,
        bias=(old_head.bias is not None),
        w_bits=8, a_bits=8,
    )
    with torch.no_grad():
        new_head.layer.weight.copy_(old_head.weight)
        if old_head.bias is not None:
            new_head.layer.bias.copy_(old_head.bias)
    model.head = new_head

    return model


def _replace_linear(parent, attr_name, w_bits, a_bits):
    old_raw = getattr(parent, attr_name)
    old = _unwrap_linear_module(old_raw)

    new = FIMAQuantLinear(
        old.in_features, old.out_features,
        bias=(old.bias is not None),
        w_bits=w_bits, a_bits=a_bits,
    )
    with torch.no_grad():
        new.layer.weight.copy_(old.weight)
        if old.bias is not None:
            new.layer.bias.copy_(old.bias)
    setattr(parent, attr_name, new)


# ---------------------------------------------------------------------------
#  Calibration helpers
# ---------------------------------------------------------------------------

def register_all_fimaq_hooks(model):
    """Register calibration hooks on all FIMAQuantLinear / FIMAQuantConv2d modules."""
    for m in model.modules():
        if isinstance(m, (FIMAQuantLinear, FIMAQuantConv2d)):
            m.register_calib_hooks()


def remove_all_fimaq_hooks(model):
    for m in model.modules():
        if isinstance(m, (FIMAQuantLinear, FIMAQuantConv2d)):
            m.remove_calib_hooks()


def calibrate_all_fimaq_modules(model):
    """After collecting calibration data, calibrate all FIMA-Q modules."""
    # First calibrate all act quantizers
    for m in model.modules():
        if isinstance(m, FIMAActQuantizer):
            if m.samples:
                m.calibrate()
    # Then calibrate all Linear/Conv2d modules
    for m in model.modules():
        if isinstance(m, (FIMAQuantLinear, FIMAQuantConv2d)):
            if m._collected_inputs:
                m.calibrate()
