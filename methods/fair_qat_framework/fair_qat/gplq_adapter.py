"""
GPLQ adapter for Fair QAT framework.

GPLQ combines activation QAT (QuantAct from stage1_qat) with weight PTQ
(UniformQuantizer from stage2_ptq). We provide wrappers matching the
backend.make_conv2d / make_linear / make_act interface.

Key fix: calibration samples are capped to avoid OOM on ViT activations.
Instead of storing full tensors, we store a small 1D CPU sample per QuantAct.
"""

import sys, os

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_REPO_ROOT = os.path.abspath(os.path.join(_PROJ, "..", ".."))


def _first_existing_dir(*paths):
    for path in paths:
        path = os.path.abspath(path)
        if os.path.isdir(path):
            return path
    return os.path.abspath(paths[0])


_GPLQ_ROOT = _first_existing_dir(
    os.path.join(_REPO_ROOT, "duibi_method_vit", "third_party", "gplq"),
    os.path.join(_PROJ, "..", "third_party", "gplq"),
)
if _GPLQ_ROOT not in sys.path:
    sys.path.insert(0, _GPLQ_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from gplq.stage1_qat.layers import QuantAct
from gplq.stage2_ptq.quantizer import UniformQuantizer


# ---------------------------------------------------------------------------
#  Calibration helpers — native QuantAct sample collection with between-batch clear
# ---------------------------------------------------------------------------

@torch.no_grad()
def _clear_gplq_act_samples(model):
    """Clear collected samples from all QuantAct instances (including in wrappers)."""
    for m in model.modules():
        if isinstance(m, QuantAct):
            if hasattr(m, 'act_samples') and m.act_samples is not None:
                m.act_samples.clear()
        if isinstance(m, GPLQActQuantizer):
            if hasattr(m.quant_act, 'act_samples') and m.quant_act.act_samples is not None:
                m.quant_act.act_samples.clear()
        if isinstance(m, (GPLQQuantLinear, GPLQQuantConv2d)):
            act = getattr(m, 'act_quant', None)
            if act is not None and hasattr(act, 'act_samples') and act.act_samples is not None:
                act.act_samples.clear()


# ---------------------------------------------------------------------------
#  GPLQ Activation Quantizer (standalone)
# ---------------------------------------------------------------------------

class GPLQActQuantizer(nn.Module):
    """
    Wraps GPLQ's QuantAct for standalone activation positions (Q/K/V/attn-map).
    Uses native QuantAct.forward() for sample collection (full tensors, capped by
    clearing samples between calibration batches to avoid OOM).
    """

    def __init__(self, bits: int):
        super().__init__()
        self.quant_act = QuantAct(
            nbits=bits,
            signed=True,
            offset=False,
            learned=True,
            mixpre=False,
            channel_wise=False,
        )
        self.calibrated = False

    @property
    def init_state(self):
        return self.quant_act.init_state

    @init_state.setter
    def init_state(self, val):
        self.quant_act.init_state.fill_(val)

    def forward(self, x):
        return self.quant_act(x)

    def initialize(self, device):
        self.quant_act.initialize_scale_offset(device)
        self.calibrated = True


# ---------------------------------------------------------------------------
#  GPLQ Weight Quantizer (standalone)
# ---------------------------------------------------------------------------

class GPLQWeightQuantizer(nn.Module):
    def __init__(self, bits: int, channel_wise: bool = True):
        super().__init__()
        self.quantizer = UniformQuantizer(n_bits=bits, channel_wise=channel_wise)

    def forward(self, x):
        return self.quantizer(x)


# ---------------------------------------------------------------------------
#  GPLQ Quantized Linear
# ---------------------------------------------------------------------------

class GPLQQuantLinear(nn.Module):
    def __init__(self, in_features, out_features, bias, w_bits, a_bits):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.weight_quant = UniformQuantizer(n_bits=w_bits, channel_wise=True)
        self.act_quant = QuantAct(
            nbits=a_bits, signed=True, offset=False,
            learned=True, mixpre=False, channel_wise=False,
        )

    def forward(self, x):
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.linear.weight)
        return F.linear(x_q, w_q, self.linear.bias)


# ---------------------------------------------------------------------------
#  GPLQ Quantized Conv2d
# ---------------------------------------------------------------------------

class GPLQQuantConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, bias=True, w_bits=4, a_bits=4):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.weight_quant = UniformQuantizer(n_bits=w_bits, channel_wise=True)
        self.act_quant = QuantAct(
            nbits=a_bits, signed=True, offset=False,
            learned=True, mixpre=False, channel_wise=False,
        )

    def forward(self, x):
        x_q = self.act_quant(x)
        w_q = self.weight_quant(self.conv.weight)
        return F.conv2d(x_q, w_q, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


# ---------------------------------------------------------------------------
#  GPLQ standalone activation helpers
# ---------------------------------------------------------------------------

def collect_gplq_activation_stats(model, loader, device, num_batches=4):
    """
    Run model in 'collect' mode (init_state=0) over num_batches.
    Clears samples between batches so only the last batch is kept —
    this avoids OOM while keeping full-tensor samples for accurate init.
    """
    print(f"[GPLQ] collect calib batches={num_batches} (native QuantAct, no capped sampling)")

    model.train()
    _set_all_gplq_state(model, 0)

    with torch.no_grad():
        for i, (images, _) in enumerate(loader):
            if i >= num_batches:
                break
            # Clear samples from previous batch to bound memory
            _clear_gplq_act_samples(model)
            images = images.to(device)
            model(images)

    # Verify collection
    for m in model.modules():
        if isinstance(m, (GPLQActQuantizer, GPLQQuantLinear, GPLQQuantConv2d)):
            _check_samples(m)


def initialize_gplq_acts(model, device):
    """
    After collecting calibration samples, initialize all QuantAct alpha/beta
    parameters and switch to quantize mode (init_state=1).
    """
    # First initialize standalone act quantizers
    for m in model.modules():
        if isinstance(m, GPLQActQuantizer):
            m.initialize(device)

    # Then initialize QuantAct inside Linear/Conv2d wrappers
    for m in model.modules():
        if isinstance(m, (GPLQQuantLinear, GPLQQuantConv2d)):
            if hasattr(m, 'act_quant') and isinstance(m.act_quant, QuantAct):
                m.act_quant.initialize_scale_offset(device)

    # Switch all to quantize mode
    _set_all_gplq_state(model, 1)


def _set_all_gplq_state(model, state: int):
    """Set init_state on all QuantAct instances in the model."""
    for m in model.modules():
        if isinstance(m, GPLQActQuantizer):
            m.init_state = state
        elif isinstance(m, (GPLQQuantLinear, GPLQQuantConv2d)):
            if hasattr(m, 'act_quant') and hasattr(m.act_quant, 'init_state'):
                m.act_quant.init_state.fill_(state)
        if isinstance(m, QuantAct):
            m.init_state.fill_(state)


def _check_samples(m):
    """Helper: print sample count for debugging."""
    target = None
    if isinstance(m, GPLQActQuantizer):
        target = m.quant_act
    elif isinstance(m, (GPLQQuantLinear, GPLQQuantConv2d)):
        target = getattr(m, 'act_quant', None)
    if target is not None and hasattr(target, 'act_samples') and target.act_samples is not None:
        n = len(target.act_samples)
        if n > 0:
            total_elems = sum(s.numel() for s in target.act_samples)
            print(f"  [GPLQ] {target.extra_repr()}  collected={n} batches, total_elems={total_elems}")
