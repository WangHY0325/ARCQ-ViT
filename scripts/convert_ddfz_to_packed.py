#!/usr/bin/env python3
"""
Convert a trained DDFZ checkpoint into a packed checkpoint.
The packed checkpoint contains ONLY packed weight codes + metadata,
not the original FP32 weights. Expected file size: ~18 MB for DeiT-Small W4A4.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import torch
import yaml

ROOT = Path(__file__).resolve().parents[1]
FAIR_ROOT = ROOT / "methods" / "fair_qat_framework"
if str(FAIR_ROOT) not in sys.path:
    sys.path.insert(0, str(FAIR_ROOT))

import fair_qat.ddfz_packed_linear as dpl
from fair_qat.ddfz_packed_convert import convert_ddfz_linear_to_packed, count_packed_linear
from fair_qat.ddfz_packed_py import pack_codes_py, unpack_codes_py
from fair_qat.quant_backends import get_backend
from fair_qat.timm_quant_models import build_timm_quant_model, _load_checkpoint_if_needed


def convert(args):
    device = torch.device("cpu")  # always on CPU to get clean storage numbers
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    os.environ["DDFZ_ZERO_ANCHOR"] = "true"
    os.environ["DDFZ_CODEBOOK_MODE"] = "ddfz"
    
    # Build DDFZ model on CPU
    print(f"[BUILD] Creating DDFZ model on CPU...")
    model = build_timm_quant_model(config, get_backend("pcddfz_nodc"))
    
    # Load checkpoint
    print(f"[LOAD] Loading checkpoint: {args.checkpoint}")
    from benchmark_ddfz_packed_inference import load_checkpoint, mark_ddfz_codebooks_ready
    missing, unexpected, skipped, restored = load_checkpoint(model, Path(args.checkpoint))
    ready, _ = mark_ddfz_codebooks_ready(model)
    print(f"  missing={missing} unexpected={unexpected} skipped_shape={skipped} restored={restored}")
    print(f"  ready_codebooks={ready}")
    
    # Count original params
    orig_params = sum(p.numel() * p.element_size() for p in model.parameters()) / 1024 / 1024
    orig_buffers = sum(b.numel() * b.element_size() for b in model.buffers()) / 1024 / 1024
    print(f"\n[ORIG] parameters={orig_params:.1f}MB buffers={orig_buffers:.1f}MB")
    
    # Patch CUDA functions for CPU
    _pack = dpl.pack_codes
    _unpack_u8 = dpl.unpack_codes_u8
    dpl.pack_codes = pack_codes_py
    dpl.unpack_codes_u8 = unpack_codes_py
    
    try:
        # Convert to packed with all three formats
        formats = {}
        for fmt in ["packed", "dequant_gemm"]:
            print(f"\n[CONVERT] Format: {fmt}")
            model_copy = convert_ddfz_linear_to_packed(model, runtime_code_format=fmt)
            packed_count = count_packed_linear(model_copy)
            
            params = sum(p.numel() * p.element_size() for p in model_copy.parameters()) / 1024 / 1024
            buffers = sum(b.numel() * b.element_size() for b in model_copy.buffers()) / 1024 / 1024
            print(f"  packed_layers={packed_count} params={params:.1f}MB buffers={buffers:.1f}MB total={(params+buffers):.1f}MB")
            
            formats[fmt] = {
                "model": model_copy,
                "params_mb": params,
                "buffers_mb": buffers,
            }
    finally:
        dpl.pack_codes = _pack
        dpl.unpack_codes_u8 = _unpack_u8
    
    # Save each format
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for fmt_name, fmt_data in formats.items():
        model_copy = fmt_data["model"]
        out_path = out_dir / f"deit_small_w4a4_packed_{fmt_name}.pt"
        
        torch.save(model_copy.state_dict(), out_path)
        file_size_mb = out_path.stat().st_size / 1024 / 1024
        
        print(f"\n[SAVE] {out_path}")
        print(f"  File size: {file_size_mb:.1f} MB")
        print(f"  GPU memory (params+buf): {fmt_data['params_mb'] + fmt_data['buffers_mb']:.1f} MB")
    
    print("\n[DONE]")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="deit_small")
    parser.add_argument("--bits", type=int, default=4)
    parser.add_argument("--config", default=str(ROOT / "configs" / "cifar100_deit" / "qat" / "deit_small_pcddfz_nodc_w4a4_w_distill.yaml"))
    parser.add_argument("--checkpoint", default=str(ROOT / "runs" / "cifar100_qat" / "pcddfz_nodc" / "deit_small" / "w4a4_w_distill" / "best.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "runs" / "cifar100_qat" / "ddfz_packed"))
    args = parser.parse_args()
    
    convert(args)
