"""Official nuLSQ-WA quantizers adapted to the fair DeiT backend.

The quantizer classes are loaded from the official nuLSQ release at runtime.
The DeiT adapter keeps the official Positive_nuLSQ activation rule,
Symmetric_nuLSQ weight rule, NMSE initialization, and gradient scaling.
"""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F


def _official_nulsq_path() -> Path:
    root = Path(__file__).resolve().parents[3]
    return root / "duibi_method_vit" / "third_party" / "nuLSQ" / "nuLSQ-master" / "src" / "quantizer" / "nonuniform" / "nulsq.py"


@lru_cache(maxsize=1)
def _official_quantizers():
    path = _official_nulsq_path()
    if not path.is_file():
        raise FileNotFoundError(f"official nuLSQ quantizer not found: {path}")
    spec = importlib.util.spec_from_file_location("official_nulsq_quantizer", path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official nuLSQ quantizer: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Positive_nuLSQ_quantizer, module.Symmetric_nuLSQ_quantizer


_SYMMETRIC_DELTA = {
    1: 0.9956866859435065,
    3: 0.5860194414434872,
    7: 0.33520061219993685,
    15: 0.18813879027991698,
    31: 0.10406300944201481,
    63: 0.05686767238235839,
    127: 0.03076238758025524,
}
_ASYMMETRIC_DELTA = {
    3: 0.65076985,
    7: 0.35340955,
    15: 0.19324868,
    31: 0.10548752,
    63: 0.0572659,
    127: 0.03087133,
    255: 0.01652923,
}


def _nmse_scale(x: torch.Tensor, qmax: int, mode: str) -> torch.Tensor:
    """Exact scalar rule from the official NMSE_initializer."""
    table = _SYMMETRIC_DELTA if mode == "symmetric" else _ASYMMETRIC_DELTA
    if qmax not in table:
        raise ValueError(f"official NMSE initializer has no qmax={qmax}")
    if mode == "symmetric":
        x_stat = x.detach().std()
    else:
        x_stat = torch.sqrt(2.0 * (x.detach() ** 2).mean())
    return x_stat * table[qmax]


class _NuLSQStateMixin:
    def _initialize_official_scales(self, x: torch.Tensor) -> None:
        if bool(self.init_state.item()):
            return

        x_scale = _nmse_scale(x, self.x_Qp, "asymmetric")
        w_scale = _nmse_scale(self._weight_for_quant(), self.w_Qp, "symmetric")
        x_scale = x_scale.detach().clamp_min(1e-8)
        w_scale = w_scale.detach().clamp_min(1e-8)
        self.x_Qparms["init_scale"] = x_scale
        self.w_Qparms["init_scale"] = w_scale
        self.x_quantizer.scale_to_Qparms(self.x_Qparms, self.x_Qn, self.x_Qp)
        self.w_quantizer.scale_to_Qparms(self.w_Qparms, self.w_Qn, self.w_Qp)
        self.init_state.fill_(True)


class NuLSQLinear(_NuLSQStateMixin, nn.Module):
    def __init__(self, in_features, out_features, bias, w_bits, a_bits):
        super().__init__()
        positive_cls, symmetric_cls = _official_quantizers()
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.num_bits = int(w_bits)
        self.x_grad_scale_mode = "LSQ_grad_scale"
        self.w_grad_scale_mode = "LSQ_grad_scale"
        self.weight_norm = False
        self.first_layer = False
        self.x_Qparms = {}
        self.w_Qparms = {}
        self.x_quantizer = positive_cls(self, int(a_bits), "activation")
        self.w_quantizer = symmetric_cls(self, int(w_bits), "weight")
        self.register_buffer("init_state", torch.tensor(False))

    def _weight_for_quant(self):
        return self.linear.weight

    def forward(self, x):
        leading = x.shape[:-1]
        x_flat = x.reshape(-1, x.shape[-1])
        self._initialize_official_scales(x_flat)
        x_q = self.x_quantizer(
            x_flat,
            self.x_Qparms,
            self.x_Qn,
            self.x_Qp,
            x_flat.shape[1],
            self.x_grad_scale_mode,
        )
        w_q = self.w_quantizer(
            self.linear.weight,
            self.w_Qparms,
            self.w_Qn,
            self.w_Qp,
            self.linear.weight.numel(),
            self.w_grad_scale_mode,
        )
        y = F.linear(x_q, w_q, self.linear.bias)
        return y.reshape(*leading, y.shape[-1])


class NuLSQConv2d(_NuLSQStateMixin, nn.Module):
    def __init__(self, in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        super().__init__()
        positive_cls, symmetric_cls = _official_quantizers()
        self.conv = nn.Conv2d(in_ch, out_ch, k, stride=s, padding=p, bias=bias)
        self.num_bits = int(w_bits)
        self.x_grad_scale_mode = "LSQ_grad_scale"
        self.w_grad_scale_mode = "LSQ_grad_scale"
        self.weight_norm = False
        self.first_layer = False
        self.x_Qparms = {}
        self.w_Qparms = {}
        self.x_quantizer = positive_cls(self, int(a_bits), "activation")
        self.w_quantizer = symmetric_cls(self, int(w_bits), "weight")
        self.register_buffer("init_state", torch.tensor(False))

    def _weight_for_quant(self):
        return self.conv.weight

    def forward(self, x):
        self._initialize_official_scales(x)
        x_q = self.x_quantizer(
            x,
            self.x_Qparms,
            self.x_Qn,
            self.x_Qp,
            x.shape[1],
            self.x_grad_scale_mode,
        )
        w_q = self.w_quantizer(
            self.conv.weight,
            self.w_Qparms,
            self.w_Qn,
            self.w_Qp,
            self.conv.weight.numel(),
            self.w_grad_scale_mode,
        )
        return F.conv2d(
            x_q,
            w_q,
            self.conv.bias,
            self.conv.stride,
            self.conv.padding,
            self.conv.dilation,
            self.conv.groups,
        )


def initialize_nulsq_model(model, train_loader, device, distributed=False, rank=0):
    """Run the official one-pass initialization before DDP wrapping."""
    was_training = model.training
    model.eval()
    images, _ = next(iter(train_loader))
    images = images.to(device, non_blocking=True)
    with torch.no_grad():
        model(images)

    modules = [m for m in model.modules() if isinstance(m, (NuLSQLinear, NuLSQConv2d))]
    if distributed:
        for module in modules:
            dist.broadcast(module.x_scale.data, src=0)
            dist.broadcast(module.w_scale.data, src=0)
            dist.broadcast(module.init_state.data, src=0)
        dist.barrier(device_ids=[device.index])
    if was_training:
        model.train()
    print(f"[nuLSQ-WA] initialized_modules={len(modules)} rank={rank}", flush=(rank == 0))


def nulsq_parameter_groups(model, lr, weight_decay, coeff_qparm_lr=0.01, qparm_wd=1e-4):
    x_params, w_params, other_params = [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if "x_scale" in name:
            x_params.append(param)
        elif "w_scale" in name:
            w_params.append(param)
        else:
            other_params.append(param)
    qlr = float(lr) * float(coeff_qparm_lr)
    return [
        {"params": x_params, "lr": qlr, "weight_decay": float(qparm_wd)},
        {"params": w_params, "lr": qlr, "weight_decay": float(qparm_wd)},
        {"params": other_params, "lr": float(lr), "weight_decay": float(weight_decay)},
    ]


class NuLSQBackend:
    name = "nulsq_wa"

    @staticmethod
    def make_conv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits):
        return NuLSQConv2d(in_ch, out_ch, k, s, p, bias, w_bits, a_bits)

    @staticmethod
    def make_linear(in_f, out_f, bias, w_bits, a_bits):
        return NuLSQLinear(in_f, out_f, bias, w_bits, a_bits)

    @staticmethod
    def make_act(a_bits, shape_hint=None):
        return nn.Identity()

    @staticmethod
    def make_head(in_f, out_f, bias):
        return nn.Linear(in_f, out_f, bias=bias)
