#!/usr/bin/env python3
"""
Phase 3: Benchmark on-the-fly packed DDFZ inference.
Measures: storage (.pt file size), GPU resident memory, inference latency.
Compares: fp32, pseudo_ddfz, packed_on_the_fly_ddfz.
"""
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
if str(FAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIR_ROOT))

import fair_qat.ddfz_packed_linear as dpl
from fair_qat.ddfz_packed_convert import convert_ddfz_linear_to_packed
from fair_qat.ddfz_packed_on_the_fly import _unpack_weight_gpu, _dequant_activation_torch
from fair_qat.ddfz_packed_py import pack_codes_py
from fair_qat.quant_backends import get_backend
from fair_qat.timm_quant_models import build_timm_quant_model

MODEL_NAMES = {
    "deit_tiny": "deit_tiny_patch16_224",
    "deit_small": "deit_small_patch16_224",
    "deit_base": "deit_base_patch16_224",
}


def check_ready(quantizer):
    cb = getattr(quantizer, "_pc_cb", None)
    return isinstance(cb, torch.Tensor) and cb.numel() > 0


def build_fp32(model_key, device):
    m = timm.create_model(MODEL_NAMES[model_key], pretrained=False, num_classes=100)
    ckpt = torch.load(str(ROOT / "runs" / "cifar100_fp32" / model_key / "best.pt"), map_location="cpu")
    m.load_state_dict(ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt)), strict=False)
    return m.to(device).eval()


def _model_input_dtype(model):
    for p in model.parameters():
        if p.is_floating_point():
            return p.dtype
    for b in model.buffers():
        if b.is_floating_point():
            return b.dtype
    return torch.float32


def _restore_otf_metadata_fp32(model):
    from fair_qat.ddfz_packed_on_the_fly import DDFZPackedLinearOnTheFly

    fp32_buffer_names = (
        "weight_codebook",
        "weight_center",
        "weight_scale",
        "weight_code_sum",
        "activation_codebook",
        "activation_thresholds",
    )
    for module in model.modules():
        if isinstance(module, DDFZPackedLinearOnTheFly):
            for name in fp32_buffer_names:
                value = module._buffers.get(name)
                if value is not None and value.is_floating_point():
                    module._buffers[name] = value.float().contiguous()


def _prepare_half_chain(model):
    model.half()
    _restore_otf_metadata_fp32(model)
    return model


def build_packed_on_the_fly(device, model_key, bits, half_chain=True):
    """Build a model with DDFZPackedLinearOnTheFly layers. All CPU work, GPU only at the end."""
    bit_name = f"w{bits}a{bits}"
    config_path = ROOT / "configs" / "cifar100_deit" / "qat" / f"{model_key}_pcddfz_nodc_{bit_name}_w_distill.yaml"
    ckpt_path = ROOT / "runs" / "cifar100_qat" / "pcddfz_nodc" / model_key / f"{bit_name}_w_distill" / "best.pt"

    if not config_path.exists():
        raise FileNotFoundError(f"Missing config: {config_path}")
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Missing checkpoint: {ckpt_path}")

    with open(config_path) as f:
        config = yaml.safe_load(f)
    os.environ["DDFZ_ZERO_ANCHOR"] = "true"
    os.environ["DDFZ_CODEBOOK_MODE"] = "ddfz"

    model = build_timm_quant_model(config, get_backend("pcddfz_nodc"))  # CPU

    # Load checkpoint
    sd = torch.load(str(ckpt_path), map_location="cpu")
    sd = sd.get("model_state_dict", sd.get("state_dict", sd))
    sd = {k.replace("module.", ""): v for k, v in sd.items()}

    # Load state & mark codebooks ready
    current = model.state_dict()
    compat = {k: v for k, v in sd.items() if k in current and current[k].shape == v.shape}
    model.load_state_dict(compat, strict=False)
    
    # Force codebook buffers: directly copy from checkpoint
    restored = 0
    for key, value in sd.items():
        if "_pc_cb" in key or "_pc_thresholds" in key:
            if key in current and current[key].shape == value.shape:
                continue  # already loaded via compat
            # Find the model buffer and set directly
            module_path = key.rsplit(".", 1)
            if len(module_path) == 2:
                mod_name, attr = module_path
                mod = dict(model.named_modules()).get(mod_name)
                if mod is not None and attr in mod._buffers:
                    mod._buffers[attr] = value.detach().clone()
                    restored += 1
    print(f"  Restored codebook buffers: {restored}")
    
    for module in model.modules():
        if hasattr(module, "_pc_cb") and check_ready(module):
            module._pc_ready = True
            module.phase_compile = True

    model.eval()

    # Patch pack_codes to Python version for CPU
    orig_pack = dpl.pack_codes
    dpl.pack_codes = pack_codes_py
    try:
        # First convert to packed to get packed codes
        packed_model = convert_ddfz_linear_to_packed(model, runtime_code_format="packed")
    finally:
        dpl.pack_codes = orig_pack

    # Now rebuild with OnTheFly linears using packed data
    from fair_qat.ddfz_packed_on_the_fly import DDFZPackedLinearOnTheFly

    def _replace_with_on_the_fly(module):
        for name, child in list(module.named_children()):
            if isinstance(child, dpl.DDFZPackedLinear):
                new = DDFZPackedLinearOnTheFly(
                    in_features=child.in_features,
                    out_features=child.out_features,
                    bias=child.bias.numel() > 0,
                    packed_weight_codes=child.packed_weight_codes,
                    weight_codebook=child.weight_codebook,
                    weight_center=child.weight_center,
                    weight_scale=child.weight_scale,
                    weight_code_sum=child.weight_code_sum,
                    activation_codebook=child.activation_codebook,
                    activation_thresholds=child.activation_thresholds,
                    bits=child.bits,
                    group_size=child.group_size,
                    bias_tensor=child.bias if child.bias.numel() > 0 else None,
                )
                setattr(module, name, new)
            else:
                _replace_with_on_the_fly(child)

    _replace_with_on_the_fly(packed_model)
    del model  # drop original DDFZ model reference
    gc.collect()
    torch.cuda.empty_cache()
    packed_model.to(device)
    if half_chain:
        _prepare_half_chain(packed_model)
    packed_model.eval()
    
    # Set up shared weight buffer: all 48 layers reuse the same fp16 tensor
    max_w = 0
    otf_layers = []
    for m in packed_model.modules():
        if isinstance(m, DDFZPackedLinearOnTheFly):
            max_w = max(max_w, m.out_features * m.in_features)
            otf_layers.append(m)
    shared_buf = torch.zeros(max_w, dtype=torch.float16, device=device)
    for m in otf_layers:
        m._weight_buffer = shared_buf
    
    params_mb = sum(p.numel() * p.element_size() for p in packed_model.parameters()) / 1e6
    bufs_mb = sum(b.numel() * b.element_size() for b in packed_model.buffers()) / 1e6
    print(f"  GPU: params={params_mb:.1f}MB buf={bufs_mb:.1f}MB sum={params_mb+bufs_mb:.1f}MB "
          f"max_weight={max_w*2/1e6:.1f}MB layers={len(otf_layers)} "
          f"half_chain={half_chain} shared_buf_ok")
    
    return packed_model.eval()


def measure(model, name, batch_sizes, device, model_key, bits, warmup=10, iters=50):
    results = []
    for bs in batch_sizes:
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        resident = torch.cuda.memory_allocated() / 1024 / 1024

        input_dtype = _model_input_dtype(model)
        x = torch.randn(bs, 3, 224, 224, device=device, dtype=input_dtype)
        resident_with_input = torch.cuda.memory_allocated() / 1024 / 1024

        lat_ms = 0.0
        peak_mb = resident_with_input
        graph = None
        try:
            with torch.no_grad():
                for _ in range(warmup):
                    _ = model(x)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                
                # CUDA Graph for bs≤1: eliminates 96 kernel launches
                if bs <= 1:
                    try:
                        graph = torch.cuda.CUDAGraph()
                        with torch.cuda.graph(graph):
                            _ = model(x)
                        torch.cuda.synchronize()
                    except Exception:
                        graph = None
                
                t0 = time.perf_counter()
                for _ in range(iters):
                    if graph is not None:
                        graph.replay()
                    else:
                        _ = model(x)
                torch.cuda.synchronize()
                t1 = time.perf_counter()
                lat_ms = (t1 - t0) / iters * 1000
                peak_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

            param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
            buf_mb = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024

            results.append({
                "model": model_key,
                "bits": bits,
                "method": name, "batch_size": bs,
                "resident_model_mb": resident,
                "resident_with_input_mb": resident_with_input,
                "forward_peak_mb": peak_mb,
                "forward_delta_mb": peak_mb - resident,
                "avg_batch_latency_ms": lat_ms,
                "avg_image_latency_ms": lat_ms / bs,
                "img_per_sec": bs * 1000 / lat_ms if lat_ms > 0 else 0,
                "param_mb": param_mb,
                "buffer_mb": buf_mb,
                "persistent_mb": param_mb + buf_mb,
            })
            print(f"  bs={bs}: resident={resident:.1f}MB peak={peak_mb:.1f}MB "
                  f"img_latency={lat_ms/bs:.3f}ms img/s={bs*1000/lat_ms:.0f}")
        except Exception as e:
            results.append({
                "model": model_key,
                "bits": bits,
                "method": name, "batch_size": bs,
                "resident_model_mb": resident,
                "resident_with_input_mb": resident_with_input,
                "error": str(e),
            })
            print(f"  bs={bs}: forward FAILED: {e}")
            break

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", choices=sorted(MODEL_NAMES), default="deit_small")
    parser.add_argument("--bits", type=int, choices=[3, 4], default=4)
    parser.add_argument("--batch-sizes", default="1,64")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--no-half-chain", action="store_true")
    parser.add_argument("--skip-val", action="store_true")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    batch_sizes = [int(x.strip()) for x in args.batch_sizes.split(",")]
    half_chain = not args.no_half_chain
    otf_name = "packed_on_the_fly_v2" if half_chain else "packed_on_the_fly"
    model_key = args.model
    bits = int(args.bits)
    bit_name = f"w{bits}a{bits}"
    all_results = []

    # 1. FP32 baseline
    print(f"[FP32] Building model={model_key} bits={bits}...")
    m = build_fp32(model_key, device)
    print("[FP32] Benchmarking...")
    all_results += measure(m, "fp32", batch_sizes, device, model_key, bits)
    del m; gc.collect(); torch.cuda.empty_cache()

    # 2. Packed on-the-fly
    print(f"[OTF] Building model={model_key} bits={bits}...")
    m_otf = build_packed_on_the_fly(device, model_key, bits, half_chain=half_chain)
    print("[OTF] Benchmarking...")
    all_results += measure(m_otf, otf_name, batch_sizes, device, model_key, bits)
    del m_otf; gc.collect(); torch.cuda.empty_cache()

    # --- Correctness: validation accuracy vs training log ---
    acc_rows = []
    if not args.skip_val:
        print("[VAL] Computing validation accuracy on CIFAR100...")
        from data import build_cifar100_loaders
        cfg = {
            "data_dir": "/gpool/home/wanghongyang/WangHY/QuEST/duibi_method_vit/datasets/cifar-100-python",
            "batch_size": 64, "val_batch_size": 64,
            "num_workers": 4, "image_size": 224, "normalization": "imagenet",
        }
        _, val_loader = build_cifar100_loaders(cfg)

        otf_model = build_packed_on_the_fly(device, model_key, bits, half_chain=half_chain)
        otf_model.eval()
        input_dtype = _model_input_dtype(otf_model)

        correct = 0
        total = 0
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device=device, dtype=input_dtype)
                targets = targets.to(device)
                outputs = otf_model(images)
                if torch.is_tensor(outputs):
                    _, pred = outputs.max(1)
                else:
                    _, pred = outputs[0].max(1) if isinstance(outputs, (tuple, list)) else (None, None)
                correct += pred.eq(targets).sum().item()
                total += targets.size(0)
        del otf_model; gc.collect(); torch.cuda.empty_cache()

        val_acc = 100.0 * correct / total
        acc_rows.append({
            "model": model_key,
            "bits": bits,
            "method": otf_name,
            "val_top1": val_acc,
            "total": total,
            "correct": correct,
        })
        print(f"  OTF val_top1={val_acc:.2f}%")

    # Save results
    out_dir = ROOT / "figures" / "ddfz_packed_inference"
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = f"{model_key}_{bit_name}"
    path = out_dir / f"bench_phase3_{tag}.csv"
    write_csv(path, all_results)
    print(f"[SAVE] {path}")
    if model_key == "deit_small" and bits == 4:
        legacy_path = out_dir / "bench_phase3.csv"
        write_csv(legacy_path, all_results)
        print(f"[SAVE] {legacy_path}")
    if acc_rows:
        acc_path = out_dir / f"bench_phase3_accuracy_{tag}.csv"
        write_csv(acc_path, acc_rows)
        print(f"[SAVE] {acc_path}")

    print("[DONE]")


def write_csv(path, rows):
    if not rows:
        return
    fields = list(rows[0].keys())
    for r in rows[1:]:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
