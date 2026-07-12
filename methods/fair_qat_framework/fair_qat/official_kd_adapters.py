"""Adapters for the official KD losses shipped with the baseline methods."""

from __future__ import annotations

import importlib.util
from functools import lru_cache
from pathlib import Path

import torch
import torch.nn as nn


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[3]


@lru_cache(maxsize=None)
def _load_loss_module(method: str):
    if method == "n2uq":
        path = _repo_root() / "duibi_method_vit" / "third_party" / "N2UQ" / "utils" / "KD_loss.py"
        module_name = "official_n2uq_kd_loss"
    elif method == "packqvit":
        path = _repo_root() / "duibi_method_vit" / "third_party" / "PackQvit" / "PackQViT-main" / "losses.py"
        module_name = "official_packqvit_losses"
    else:
        raise ValueError(f"unsupported official KD method: {method}")
    if not path.is_file():
        raise FileNotFoundError(f"official KD implementation not found: {path}")
    spec = importlib.util.spec_from_file_location(module_name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load official KD implementation: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _logits(outputs):
    if isinstance(outputs, (tuple, list)):
        return outputs[0]
    return outputs


class OfficialN2UQDistributionLoss(nn.Module):
    """Official N2UQ DistributionLoss with a frozen target-data teacher."""

    def __init__(self, teacher_model: nn.Module):
        super().__init__()
        self.teacher_model = teacher_model
        self.criterion = _load_loss_module("n2uq").DistributionLoss()

    def forward(self, images, outputs, targets):
        del targets
        student_logits = _logits(outputs)
        with torch.no_grad():
            teacher_logits = _logits(self.teacher_model(images))
        return self.criterion(student_logits, teacher_logits)


class OfficialPackQViTDistillationLoss(nn.Module):
    """Official PackQViT/DeiT DistillationLoss.

    The fair timm student has one classifier output rather than a separate
    distillation head, so the same student logits are passed as both official
    branches. The loss implementation itself is loaded directly from the
    PackQViT release.
    """

    def __init__(self, base_criterion, teacher_model, distillation_type, alpha, tau):
        super().__init__()
        self.criterion = _load_loss_module("packqvit").DistillationLoss(
            base_criterion,
            teacher_model,
            distillation_type,
            float(alpha),
            float(tau),
        )

    def forward(self, images, outputs, targets):
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            official_outputs = outputs
        else:
            logits = _logits(outputs)
            official_outputs = (logits, logits)
        return self.criterion(images, official_outputs, targets)


def build_official_n2uq_distribution_loss(teacher_model):
    return OfficialN2UQDistributionLoss(teacher_model)


def build_official_packqvit_distillation_loss(
    base_criterion, teacher_model, distillation_type, alpha, tau
):
    return OfficialPackQViTDistillationLoss(
        base_criterion, teacher_model, distillation_type, alpha, tau
    )
