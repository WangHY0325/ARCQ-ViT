#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import re
import sys
from pathlib import Path

import torch
import torch.nn as nn


ROOT = Path(__file__).resolve().parents[1]
FAIR_ROOT = ROOT / "methods" / "fair_qat_framework"
if not FAIR_ROOT.exists():
    FAIR_ROOT = ROOT.parent / "methods" / "fair_qat_framework"
if str(FAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIR_ROOT))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

from benchmark_ddfz_packed_inference import (  # noqa: E402
    build_ddfz_model,
    default_paths,
    synchronize,
)
from fair_qat.ddfz_packed_convert import convert_ddfz_linear_to_packed, count_packed_linear  # noqa: E402
from fair_qat.ddfz_packed_linear import DDFZPackedLinear  # noqa: E402
from fair_qat.quant_backends import QuantConv2dPCDDFZ, QuantLinearPCDDFZ, QuantLinearLSQ  # noqa: E402


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit_small")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--runtime-code-format", choices=["packed", "u8"], default="packed")
    parser.add_argument("--out-dir", default=str(ROOT / "figures" / "ddfz_packed_debug"))
    return parser.parse_args()


def write_csv(path: Path, rows, fields=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def freeze_ddfz_compile(model: nn.Module):
    ready = 0
    missing = 0
    for module in model.modules():
        if hasattr(module, "_pc_cb") and hasattr(module, "_pc_thresholds"):
            cb = getattr(module, "_pc_cb")
            thresholds = getattr(module, "_pc_thresholds")
            if isinstance(cb, torch.Tensor) and cb.numel() > 0 and isinstance(thresholds, torch.Tensor) and thresholds.numel() > 0:
                setattr(module, "_pc_ready", True)
                ready += 1
            else:
                missing += 1
        if hasattr(module, "phase_compile"):
            setattr(module, "phase_compile", True)
        if hasattr(module, "compile_steps"):
            try:
                setattr(module, "compile_steps", set())
            except Exception:
                pass
    model.eval()
    return ready, missing


def is_transformer_block_name(name: str) -> bool:
    return re.fullmatch(r"blocks\.\d+", name) is not None


def should_hook(name: str, module: nn.Module) -> bool:
    if isinstance(module, (QuantConv2dPCDDFZ, QuantLinearPCDDFZ, DDFZPackedLinear, QuantLinearLSQ)):
        return True
    if is_transformer_block_name(name):
        return True
    return False


def collect_outputs(model: nn.Module, x: torch.Tensor):
    rows = []
    outputs = {}
    hooks = []

    def make_hook(name, module):
        def hook(_module, _inputs, output):
            value = output[0] if isinstance(output, (tuple, list)) else output
            if isinstance(value, torch.Tensor):
                outputs[name] = value.detach().float().cpu()
                rows.append({
                    "order": len(rows),
                    "name": name,
                    "class": module.__class__.__name__,
                    "shape": "x".join(str(v) for v in value.shape),
                })
        return hook

    for name, module in model.named_modules():
        if should_hook(name, module):
            hooks.append(module.register_forward_hook(make_hook(name, module)))

    with torch.no_grad():
        logits = model(x).detach().float().cpu()
    for handle in hooks:
        handle.remove()
    return logits, outputs, rows


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DDFZ packed correctness debug must run on a Slurm GPU node")
    torch.manual_seed(20260630)
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    default_config, default_checkpoint = default_paths(args.model, args.bits)
    config_path = Path(args.config) if args.config else default_config
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint

    print(f"[BUILD] pseudo DDFZ model={args.model} bits={args.bits}", flush=True)
    pseudo_model, pseudo_info = build_ddfz_model(config_path, checkpoint_path, device)
    pseudo_ready, pseudo_missing = freeze_ddfz_compile(pseudo_model)

    print(f"[BUILD] packed DDFZ format={args.runtime_code_format}", flush=True)
    packed_source, packed_info = build_ddfz_model(config_path, checkpoint_path, device)
    source_ready, source_missing = freeze_ddfz_compile(packed_source)
    packed_model = convert_ddfz_linear_to_packed(
        packed_source,
        runtime_code_format=args.runtime_code_format,
    ).eval()
    packed_ready, packed_missing = freeze_ddfz_compile(packed_model)
    packed_count = count_packed_linear(packed_model)
    print(
        "[READY] "
        f"pseudo={pseudo_ready}/{pseudo_missing} source={source_ready}/{source_missing} "
        f"packed={packed_ready}/{packed_missing} packed_linear={packed_count}",
        flush=True,
    )

    x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    print("[FORWARD] pseudo", flush=True)
    pseudo_logits, pseudo_outputs, pseudo_order = collect_outputs(pseudo_model, x)
    synchronize(device)
    print("[FORWARD] packed", flush=True)
    packed_logits, packed_outputs, packed_order = collect_outputs(packed_model, x)
    synchronize(device)

    order_rows = []
    pseudo_classes = {row["name"]: row["class"] for row in pseudo_order}
    packed_classes = {row["name"]: row["class"] for row in packed_order}
    pseudo_shapes = {row["name"]: row["shape"] for row in pseudo_order}
    packed_shapes = {row["name"]: row["shape"] for row in packed_order}
    names = []
    seen = set()
    for row in pseudo_order:
        name = row["name"]
        if name not in seen:
            names.append(name)
            seen.add(name)
    for row in packed_order:
        name = row["name"]
        if name not in seen:
            names.append(name)
            seen.add(name)
    for idx, name in enumerate(names):
        order_rows.append({
            "order": idx,
            "name": name,
            "pseudo_class": pseudo_classes.get(name, ""),
            "packed_class": packed_classes.get(name, ""),
            "pseudo_shape": pseudo_shapes.get(name, ""),
            "packed_shape": packed_shapes.get(name, ""),
        })

    layer_rows = []
    first_large_seen = False
    for idx, name in enumerate(names):
        left = pseudo_outputs.get(name)
        right = packed_outputs.get(name)
        row = {
            "order": idx,
            "name": name,
            "pseudo_class": pseudo_classes.get(name, ""),
            "packed_class": packed_classes.get(name, ""),
            "shape": pseudo_shapes.get(name, packed_shapes.get(name, "")),
            "present_in_pseudo": left is not None,
            "present_in_packed": right is not None,
            "max_abs_error": "",
            "mean_abs_error": "",
            "first_large_error": False,
        }
        if left is not None and right is not None and tuple(left.shape) == tuple(right.shape):
            err = (left - right).abs()
            max_err = float(err.max().item())
            mean_err = float(err.mean().item())
            row["max_abs_error"] = max_err
            row["mean_abs_error"] = mean_err
            large = max_err > 1.0e-4 or mean_err > 1.0e-5
            if large and not first_large_seen:
                row["first_large_error"] = True
                first_large_seen = True
        layer_rows.append(row)

    logit_err = (pseudo_logits - packed_logits).abs()
    top_pseudo = pseudo_logits.argmax(dim=-1)
    top_packed = packed_logits.argmax(dim=-1)
    logit_rows = [{
        "model": args.model,
        "bits": args.bits,
        "runtime_code_format": args.runtime_code_format,
        "batch_size": args.batch_size,
        "max_abs_error": float(logit_err.max().item()),
        "mean_abs_error": float(logit_err.mean().item()),
        "top1_agreement": float((top_pseudo == top_packed).float().mean().item()),
        "pseudo_top1": " ".join(str(int(v)) for v in top_pseudo.tolist()),
        "packed_top1": " ".join(str(int(v)) for v in top_packed.tolist()),
        **{f"pseudo_{k}": v for k, v in pseudo_info.items()},
        **{f"packed_{k}": v for k, v in packed_info.items()},
    }]

    write_csv(out_dir / "module_order.csv", order_rows)
    write_csv(out_dir / "layerwise_diff.csv", layer_rows)
    write_csv(out_dir / "logit_diff.csv", logit_rows)
    print(
        f"[DONE] logits max={logit_rows[0]['max_abs_error']:.8g} "
        f"mean={logit_rows[0]['mean_abs_error']:.8g} "
        f"agree={logit_rows[0]['top1_agreement']:.6f}",
        flush=True,
    )
    print(f"[DONE] out_dir={out_dir}", flush=True)


if __name__ == "__main__":
    main()
