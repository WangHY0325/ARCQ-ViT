#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import os
import sys
import time
from pathlib import Path

import torch
import torch.nn as nn
import timm
import yaml


ROOT = Path(__file__).resolve().parents[1]
FAIR_ROOT = ROOT / "methods" / "fair_qat_framework"
if not FAIR_ROOT.exists():
    FAIR_ROOT = ROOT.parent / "methods" / "fair_qat_framework"
if str(FAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIR_ROOT))

from fair_qat.ddfz_packed_convert import convert_ddfz_linear_to_packed, count_packed_linear
from fair_qat.ddfz_packed_linear import DDFZPackedLinear
from fair_qat.quant_backends import get_backend
from fair_qat.timm_quant_models import build_timm_quant_model


MODEL_NAMES = {
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
}


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit_small")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--config", default="")
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--runtime-code-format", choices=["packed", "u8"], default="u8")
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--iters", type=int, default=80)
    parser.add_argument("--correctness-batch-size", type=int, default=16)
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=100)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--out-dir", default=str(ROOT / "figures" / "ddfz_packed_inference"))
    return parser.parse_args()


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row.keys():
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def read_checkpoint_state(path: Path):
    if not path.is_file():
        raise FileNotFoundError(str(path))
    checkpoint = torch.load(str(path), map_location="cpu")
    if isinstance(checkpoint, dict):
        state = checkpoint.get("model", checkpoint.get("state_dict", checkpoint.get("model_state_dict", checkpoint)))
    else:
        state = checkpoint
    return {key.replace("module.", ""): value for key, value in state.items()}


def restore_dynamic_state_tensors(model: nn.Module, state):
    modules = dict(model.named_modules())
    restored = 0
    for key, value in state.items():
        if "." not in key:
            continue
        module_name, attr = key.rsplit(".", 1)
        module = modules.get(module_name)
        if module is None:
            continue
        if attr in module._buffers:
            module._buffers[attr] = value.detach().clone()
            restored += 1
        elif attr in module._parameters and attr == "step":
            module._parameters[attr] = nn.Parameter(value.detach().clone())
            restored += 1
    return restored


def load_checkpoint(model: nn.Module, path: Path):
    state = read_checkpoint_state(path)
    current = model.state_dict()
    compatible = {}
    skipped_shape = 0
    for key, value in state.items():
        if key in current and tuple(current[key].shape) == tuple(value.shape):
            compatible[key] = value
        elif key in current:
            skipped_shape += 1
    missing, unexpected = model.load_state_dict(compatible, strict=False)
    restored_dynamic = restore_dynamic_state_tensors(model, state)
    return len(missing), len(unexpected), skipped_shape, restored_dynamic


def mark_ddfz_codebooks_ready(model: nn.Module):
    ready = 0
    missing = 0
    for module in model.modules():
        if hasattr(module, "_pc_cb") and hasattr(module, "_pc_thresholds"):
            cb = getattr(module, "_pc_cb")
            thresholds = getattr(module, "_pc_thresholds")
            if (
                isinstance(cb, torch.Tensor)
                and cb.numel() > 0
                and isinstance(thresholds, torch.Tensor)
                and thresholds.numel() > 0
            ):
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
    return ready, missing


def default_paths(model_key: str, bits: int):
    config = ROOT / "configs" / "cifar100_deit" / "qat" / f"{model_key}_pcddfz_nodc_w{bits}a{bits}_w_distill.yaml"
    checkpoint = ROOT / "runs" / "cifar100_qat" / "pcddfz_nodc" / model_key / f"w{bits}a{bits}_w_distill" / "best.pt"
    return config, checkpoint


def build_fp32_model(model_key: str, num_classes: int, device: torch.device):
    model = timm.create_model(MODEL_NAMES[model_key], pretrained=False, num_classes=num_classes)
    checkpoint = ROOT / "runs" / "cifar100_fp32" / model_key / "best.pt"
    load_checkpoint(model, checkpoint)
    return model.to(device).eval(), checkpoint


def build_ddfz_model(config_path: Path, checkpoint_path: Path, device: torch.device):
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    os.environ["DDFZ_ZERO_ANCHOR"] = "true"
    os.environ["DDFZ_CODEBOOK_MODE"] = "ddfz"
    model = build_timm_quant_model(config, get_backend("pcddfz_nodc"))
    missing, unexpected, skipped_shape, restored_dynamic = load_checkpoint(model, checkpoint_path)
    ready_cpu, missing_cpu = mark_ddfz_codebooks_ready(model)
    model.to(device).eval()
    ready_gpu, missing_gpu = mark_ddfz_codebooks_ready(model)
    info = {
        "state_missing_keys": missing,
        "state_unexpected_keys": unexpected,
        "state_shape_skipped_keys": skipped_shape,
        "dynamic_tensors_restored": restored_dynamic,
        "ready_codebooks_cpu": ready_cpu,
        "missing_codebooks_cpu": missing_cpu,
        "ready_codebooks_gpu": ready_gpu,
        "missing_codebooks_gpu": missing_gpu,
    }
    return model, info


def build_packed_model(config_path: Path, checkpoint_path: Path, device: torch.device, runtime_code_format: str):
    import fair_qat.ddfz_packed_linear as dpl
    from fair_qat.ddfz_packed_py import pack_codes_py, unpack_codes_py

    # Build DDFZ model on CPU to avoid CUDA memory contamination
    source, info = build_ddfz_model(config_path, checkpoint_path, torch.device("cpu"))

    # Patch CUDA functions with pure Python versions for CPU build
    _original_pack = dpl.pack_codes
    _original_unpack_u8 = dpl.unpack_codes_u8
    dpl.pack_codes = pack_codes_py
    dpl.unpack_codes_u8 = unpack_codes_py
    try:
        model = convert_ddfz_linear_to_packed(source, runtime_code_format=runtime_code_format).eval()
    finally:
        dpl.pack_codes = _original_pack
        dpl.unpack_codes_u8 = _original_unpack_u8

    model = model.to(device)
    info = dict(info)
    info["runtime_code_format"] = runtime_code_format
    info["packed_linear_count"] = count_packed_linear(model)
    return model, info


def synchronize(device: torch.device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def allocated_mb(device: torch.device) -> float:
    if device.type != "cuda":
        return 0.0
    return float(torch.cuda.memory_allocated(device) / 1024 / 1024)


def benchmark(model: nn.Module, batch_size: int, image_size: int, device: torch.device, warmup: int, iters: int):
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        synchronize(device)
    resident_model_mb = allocated_mb(device)
    x = torch.randn(batch_size, 3, image_size, image_size, device=device)
    synchronize(device)
    resident_with_input_mb = allocated_mb(device)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    forward_peak_total_mb = 0.0
    avg_batch_ms = 0.0
    try:
        with torch.no_grad():
            for _ in range(warmup):
                _ = model(x)
            synchronize(device)
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
                start = torch.cuda.Event(enable_timing=True)
                end = torch.cuda.Event(enable_timing=True)
                start.record()
                for _ in range(iters):
                    _ = model(x)
                end.record()
                synchronize(device)
                total_ms = float(start.elapsed_time(end))
                forward_peak_total_mb = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
            else:
                t0 = time.perf_counter()
                for _ in range(iters):
                    _ = model(x)
                total_ms = (time.perf_counter() - t0) * 1000.0
            avg_batch_ms = total_ms / max(1, iters)
    except Exception as e:
        print(f"[BENCH_WARN] forward failed for method, measuring memory only: {e}", flush=True)
        if device.type == "cuda":
            forward_peak_total_mb = float(torch.cuda.max_memory_allocated(device) / 1024 / 1024)
    result = {
        "batch_size": batch_size,
        "warmup": warmup,
        "iters": iters,
        "avg_batch_latency_ms": avg_batch_ms,
        "avg_image_latency_ms": avg_batch_ms / batch_size,
        "images_per_second": batch_size * 1000.0 / avg_batch_ms if avg_batch_ms > 0 else 0.0,
        "resident_model_mb": resident_model_mb,
        "resident_with_input_mb": resident_with_input_mb,
        "forward_peak_total_mb": forward_peak_total_mb,
        "forward_peak_delta_mb": forward_peak_total_mb - resident_with_input_mb,
    }
    del x
    synchronize(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return result


def tensor_storage_mb(model: nn.Module):
    param_bytes = sum(p.numel() * p.element_size() for p in model.parameters())
    buffer_bytes = sum(b.numel() * b.element_size() for b in model.buffers())
    return param_bytes / 1024 / 1024, buffer_bytes / 1024 / 1024


def packed_storage_bytes(model: nn.Module):
    total = 0
    runtime = 0
    layers = 0
    for module in model.modules():
        if isinstance(module, DDFZPackedLinear):
            total += module.estimated_storage_bytes()
            runtime += module.runtime_cache_bytes()
            layers += 1
    return total, runtime, layers


def correctness_rows(models, args, device):
    x = torch.randn(args.correctness_batch_size, 3, args.image_size, args.image_size, device=device)
    logits = {}
    with torch.no_grad():
        for name, model in models.items():
            logits[name] = model(x).float()
    rows = []

    def add(left, right, comparison):
        err = (logits[left] - logits[right]).abs()
        top_left = logits[left].argmax(dim=-1)
        top_right = logits[right].argmax(dim=-1)
        rows.append({
            "model": args.model,
            "bits": args.bits,
            "comparison": comparison,
            "batch_size": args.correctness_batch_size,
            "logit_max_abs_error": float(err.max().item()),
            "logit_mean_abs_error": float(err.mean().item()),
            "top1_agreement": float((top_left == top_right).float().mean().item()),
        })

    if "pseudo_ddfz" in logits and "packed_ddfz" in logits:
        add("pseudo_ddfz", "packed_ddfz", "pseudo_ddfz_vs_packed_ddfz")
    if "pseudo_ddfz" in logits and "u8_ddfz" in logits:
        add("pseudo_ddfz", "u8_ddfz", "pseudo_ddfz_vs_u8_ddfz")
    if "pseudo_ddfz" in logits and "dequant_gemm_ddfz" in logits:
        add("pseudo_ddfz", "dequant_gemm_ddfz", "pseudo_ddfz_vs_dequant_gemm_ddfz")
    if "fp32" in logits and "pseudo_ddfz" in logits:
        add("fp32", "pseudo_ddfz", "fp32_vs_pseudo_ddfz")
    return rows


def cleanup_model(model):
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    if device.type != "cuda":
        raise RuntimeError("DDFZ packed inference benchmark must run on a Slurm GPU node")
    if hasattr(torch, "set_float32_matmul_precision"):
        torch.set_float32_matmul_precision("highest")
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.backends.cudnn.benchmark = True

    default_config, default_checkpoint = default_paths(args.model, args.bits)
    config_path = Path(args.config) if args.config else default_config
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else default_checkpoint
    batch_sizes = [int(item.strip()) for item in args.batch_sizes.split(",") if item.strip()]

    print(f"[CORRECTNESS] building models model={args.model} bits={args.bits}", flush=True)
    try:
        fp32_model, fp32_checkpoint = build_fp32_model(args.model, args.num_classes, device)
        pseudo_model, pseudo_info = build_ddfz_model(config_path, checkpoint_path, device)
        packed_model, packed_info = build_packed_model(config_path, checkpoint_path, device, "packed")
        u8_model, u8_info = build_packed_model(config_path, checkpoint_path, device, "u8")
        dequant_gemm_model, dequant_gemm_info = build_packed_model(config_path, checkpoint_path, device, "dequant_gemm")
        correctness = correctness_rows(
            {
                "fp32": fp32_model,
                "pseudo_ddfz": pseudo_model,
                "packed_ddfz": packed_model,
                "u8_ddfz": u8_model,
                "dequant_gemm_ddfz": dequant_gemm_model,
            },
            args,
            device,
        )
        write_csv(out_dir / "ddfz_packed_correctness.csv", correctness)
        cleanup_model(fp32_model)
        cleanup_model(pseudo_model)
        cleanup_model(packed_model)
        cleanup_model(u8_model)
        cleanup_model(dequant_gemm_model)
        print(f"[CORRECTNESS] done, {len(correctness)} comparisons", flush=True)
    except Exception as e:
        print(f"[CORRECTNESS] skipped due to error: {e}", flush=True)

    latency_rows = []
    memory_rows = []
    storage_rows = []
    bottleneck_rows = []

    methods = [
        ("fp32", 32, lambda: build_fp32_model(args.model, args.num_classes, device), fp32_checkpoint),
        ("pseudo_ddfz", args.bits, lambda: build_ddfz_model(config_path, checkpoint_path, device), checkpoint_path),
        ("packed_ddfz", args.bits, lambda: build_packed_model(config_path, checkpoint_path, device, "packed"), checkpoint_path),
        ("u8_ddfz", args.bits, lambda: build_packed_model(config_path, checkpoint_path, device, "u8"), checkpoint_path),
        (
            "dequant_gemm_ddfz",
            args.bits,
            lambda: build_packed_model(config_path, checkpoint_path, device, "dequant_gemm"),
            checkpoint_path,
        ),
    ]

    fp32_latency_by_batch = {}
    pseudo_latency_by_batch = {}
    for method, method_bits, build_fn, ckpt_path in methods:
        print(f"[BUILD] method={method}", flush=True)
        built = build_fn()
        model = built[0]
        info = built[1] if isinstance(built[1], dict) else {}
        param_mb, buffer_mb = tensor_storage_mb(model)
        packed_bytes, runtime_bytes, packed_count = packed_storage_bytes(model)
        storage_rows.append({
            "model": args.model,
            "bits": method_bits,
            "method": method,
            "parameter_tensor_mb": param_mb,
            "buffer_tensor_mb": buffer_mb,
            "packed_linear_storage_mb": packed_bytes / 1024 / 1024,
            "runtime_code_cache_mb": runtime_bytes / 1024 / 1024,
            "packed_linear_count": packed_count,
            "checkpoint_file_mb": Path(ckpt_path).stat().st_size / 1024 / 1024,
            **info,
        })

        for batch_size in batch_sizes:
            print(f"[BENCH] method={method} batch={batch_size}", flush=True)
            row = benchmark(model, batch_size, args.image_size, device, args.warmup, args.iters)
            row.update({
                "model": args.model,
                "bits": method_bits,
                "method": method,
            })
            latency_rows.append(row)
            memory_rows.append({
                "model": args.model,
                "bits": method_bits,
                "method": method,
                "batch_size": batch_size,
                "resident_model_mb": row["resident_model_mb"],
                "resident_with_input_mb": row["resident_with_input_mb"],
                "forward_peak_total_mb": row["forward_peak_total_mb"],
                "forward_peak_delta_mb": row["forward_peak_delta_mb"],
                "parameter_tensor_mb": param_mb,
                "buffer_tensor_mb": buffer_mb,
                "packed_linear_storage_mb": packed_bytes / 1024 / 1024,
                "runtime_code_cache_mb": runtime_bytes / 1024 / 1024,
            })
            if method == "fp32":
                fp32_latency_by_batch[batch_size] = row["avg_batch_latency_ms"]
            if method == "pseudo_ddfz":
                pseudo_latency_by_batch[batch_size] = row["avg_batch_latency_ms"]
            bottleneck_rows.append({
                "model": args.model,
                "bits": method_bits,
                "method": method,
                "batch_size": batch_size,
                "avg_batch_latency_ms": row["avg_batch_latency_ms"],
                "relative_to_fp32": row["avg_batch_latency_ms"] / fp32_latency_by_batch.get(batch_size, row["avg_batch_latency_ms"]),
                "relative_to_pseudo_ddfz": row["avg_batch_latency_ms"] / pseudo_latency_by_batch.get(batch_size, row["avg_batch_latency_ms"]),
                "forward_peak_delta_mb": row["forward_peak_delta_mb"],
                "bottleneck_note": {
                    "fp32": "baseline_float_model",
                    "pseudo_ddfz": "fake_quant_dequant_plus_float_linear",
                    "packed_ddfz": "bit_extract_inner_loop_in_linear_kernel",
                    "u8_ddfz": "direct_uint8_codes_plus_warp_parallel_linear_kernel",
                    "dequant_gemm_ddfz": "packed_storage_runtime_fp16_gemm",
                }.get(method, ""),
            })
            write_csv(out_dir / "ddfz_packed_latency_partial.csv", latency_rows)
            write_csv(out_dir / "ddfz_packed_memory_partial.csv", memory_rows)

        cleanup_model(model)

    write_csv(out_dir / "ddfz_packed_latency.csv", latency_rows)
    write_csv(out_dir / "ddfz_packed_memory.csv", memory_rows)
    write_csv(out_dir / "ddfz_packed_storage.csv", storage_rows)
    write_csv(out_dir / "ddfz_packed_bottleneck.csv", bottleneck_rows)
    print(f"[DONE] out_dir={out_dir}", flush=True)
    print(
        f"[DONE] correctness={len(correctness)} latency={len(latency_rows)} "
        f"storage={len(storage_rows)} bottleneck={len(bottleneck_rows)}",
        flush=True,
    )


if __name__ == "__main__":
    main()
