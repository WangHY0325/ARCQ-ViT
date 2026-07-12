"""
timm model quantization wrappers for AAAI ImageNet experiments.

This module converts DeiT/Swin timm models by replacing Conv2d/Linear modules
with the selected fair quantization backend. It is intentionally conservative:
LayerNorm, residual paths, position embeddings, and softmax operators are left
unquantized.

The first migrated version guarantees identical module-level quantization scope
across backends. Architecture-specific q/k/v and attention-map hooks should be
added only after `inspect_quant_scope_imagenet.py` confirms the baseline scope.
"""

from __future__ import annotations

import os
from typing import Dict, Iterable, Tuple

import torch
import torch.nn as nn
import timm


HEAD_NAMES = {"head", "head_dist", "classifier", "fc"}


def _copy_linear(dst: nn.Module, src: nn.Linear) -> None:
    target = getattr(dst, "linear", None)
    if target is None:
        target = getattr(dst, "layer", None)
    if target is None and isinstance(dst, nn.Linear):
        target = dst
    if target is None and hasattr(dst, "weight"):
        target = dst
    if target is None:
        raise TypeError(f"Cannot locate Linear weight target in {type(dst)}")
    with torch.no_grad():
        target.weight.copy_(src.weight)
        if src.bias is not None and getattr(target, "bias", None) is not None:
            target.bias.copy_(src.bias)


def _copy_conv(dst: nn.Module, src: nn.Conv2d) -> None:
    target = getattr(dst, "conv", None)
    if target is None:
        target = getattr(dst, "layer", None)
    if target is None and isinstance(dst, nn.Conv2d):
        target = dst
    if target is None and hasattr(dst, "weight"):
        target = dst
    if target is None:
        raise TypeError(f"Cannot locate Conv2d weight target in {type(dst)}")
    with torch.no_grad():
        target.weight.copy_(src.weight)
        if src.bias is not None and getattr(target, "bias", None) is not None:
            target.bias.copy_(src.bias)


def _is_quantizable_linear_name(name: str) -> bool:
    return True


def _replace_children(module: nn.Module, backend, w_bits: int, a_bits: int,
                      quantize_head: str, prefix: str = "") -> Dict[str, int]:
    stats = {
        "linear": 0,
        "conv2d": 0,
        "head": 0,
        "skipped_linear": 0,
    }

    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name

        if isinstance(child, nn.Linear):
            if child_name in HEAD_NAMES or full_name.split(".")[-1] in HEAD_NAMES:
                if quantize_head == "fp32":
                    continue
                new_child = backend.make_head(child.in_features, child.out_features, child.bias is not None)
                _copy_linear(new_child, child)
                setattr(module, child_name, new_child)
                stats["head"] += 1
                continue

            if not _is_quantizable_linear_name(child_name):
                stats["skipped_linear"] += 1
                continue

            new_child = backend.make_linear(
                child.in_features,
                child.out_features,
                bias=child.bias is not None,
                w_bits=w_bits,
                a_bits=a_bits,
            )
            _copy_linear(new_child, child)
            setattr(module, child_name, new_child)
            stats["linear"] += 1
            continue

        if isinstance(child, nn.Conv2d):
            new_child = backend.make_conv2d(
                child.in_channels,
                child.out_channels,
                child.kernel_size,
                child.stride,
                child.padding,
                bias=child.bias is not None,
                w_bits=w_bits,
                a_bits=a_bits,
            )
            _copy_conv(new_child, child)
            setattr(module, child_name, new_child)
            stats["conv2d"] += 1
            continue

        child_stats = _replace_children(child, backend, w_bits, a_bits, quantize_head, full_name)
        for key, value in child_stats.items():
            stats[key] += value

    return stats


def _load_checkpoint_if_needed(model: nn.Module, config: Dict) -> None:
    ckpt_path = config.get("pretrained_checkpoint") or config.get("finetune") or ""
    if not ckpt_path:
        return
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f"pretrained checkpoint not found: {ckpt_path}")

    checkpoint = torch.load(ckpt_path, map_location="cpu")
    state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint))
    cleaned = {}
    for key, value in state.items():
        key = key.replace("module.", "")
        cleaned[key] = value
    model_state = model.state_dict()
    shape_mismatch = {}
    compatible = {}
    for key, value in cleaned.items():
        if key in model_state and tuple(value.shape) != tuple(model_state[key].shape):
            shape_mismatch[key] = (tuple(value.shape), tuple(model_state[key].shape))
            continue
        compatible[key] = value
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    print(f"[LOAD] checkpoint={ckpt_path}")
    print(
        f"[LOAD] missing={len(missing)} unexpected={len(unexpected)} "
        f"shape_mismatch={len(shape_mismatch)}"
    )
    if shape_mismatch:
        print(f"[LOAD] skipped_shape_mismatch={sorted(shape_mismatch)}")


def build_timm_quant_model(config: Dict, backend):
    model_name = config["model_name"]
    num_classes = int(config.get("num_classes", 1000))
    pretrained = bool(config.get("timm_pretrained", False))
    w_bits = int(config.get("w_bits", config.get("bits", 4)))
    a_bits = int(config.get("a_bits", config.get("bits", 4)))
    quantize_head = config.get("head_quant", "w8a8")

    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
    )
    _load_checkpoint_if_needed(model, config)

    if hasattr(backend, "set_config"):
        backend.set_config(config)

    stats = _replace_children(
        model,
        backend=backend,
        w_bits=w_bits,
        a_bits=a_bits,
        quantize_head=quantize_head,
    )
    model.quant_scope_stats = stats
    print(f"[QUANT_SCOPE] {stats}")
    return model
