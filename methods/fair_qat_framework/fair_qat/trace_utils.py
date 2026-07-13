"""
Trace utilities for Figure 4: codebook evolution, quant error, residual stats.

This module is standalone — no imports from training scripts.
"""

from __future__ import annotations

import csv
import math
import os
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# 1. ensure_trace_dir
# ---------------------------------------------------------------------------

def ensure_trace_dir(config: dict) -> Path:
    out = Path(config["trace_output_dir"])
    out.mkdir(parents=True, exist_ok=True)
    print(f"[TRACE] output_dir={out}")
    return out


# ---------------------------------------------------------------------------
# 2. select_trace_layers
# ---------------------------------------------------------------------------

def select_trace_layers(model: nn.Module, max_layers: int = 3) -> List[Tuple[str, nn.Module]]:
    """Select up to 3 representative linear layers for tracing."""
    candidates = []
    for name, module in model.named_modules():
        cls_name = type(module).__name__
        if "Linear" in cls_name or "QuantizedLinear" in cls_name:
            if "head" in name.lower() or "classifier" in name.lower():
                continue
            candidates.append((name, module))

    # Prefer DeiT block attention qkv layers
    preferred = []
    for name, module in candidates:
        if "attn.qkv" in name or "attn.proj" in name:
            preferred.append((name, module))

    if len(preferred) >= max_layers:
        # Pick first, middle, last
        idxs = [0, len(preferred) // 2, len(preferred) - 1]
        selected = [preferred[i] for i in idxs[:max_layers]]
    elif len(candidates) >= max_layers:
        idxs = [0, len(candidates) // 2, len(candidates) - 1]
        selected = [candidates[i] for i in idxs[:max_layers]]
    else:
        selected = candidates

    for name, _ in selected:
        print(f"  [TRACE] selected layer: {name}")
    return selected


# ---------------------------------------------------------------------------
# 3–4. Quantizer accessors
# ---------------------------------------------------------------------------

def get_activation_quantizer(module: nn.Module) -> Optional[nn.Module]:
    for attr in ["activation_quantizer", "act_quantizer", "a_quantizer", "input_quantizer", "act_quant"]:
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


def get_weight_quantizer(module: nn.Module) -> Optional[nn.Module]:
    for attr in ["weight_quantizer", "w_quantizer", "weight_quant"]:
        if hasattr(module, attr):
            return getattr(module, attr)
    return None


# ---------------------------------------------------------------------------
# 5. extract_codebook
# ---------------------------------------------------------------------------

def extract_codebook(quantizer: nn.Module) -> Optional[torch.Tensor]:
    if quantizer is None:
        return None
    for attr in ["codebook", "codes", "cb", "centers", "_codebook", "_codebook_values", "_pc_cb", "_cached_cb"]:
        if hasattr(quantizer, attr):
            t = getattr(quantizer, attr)
            if isinstance(t, torch.Tensor) and t.numel() > 0:
                t = t.detach().float().flatten().unique().sort()[0]
                return t.cpu()
    return None


# ---------------------------------------------------------------------------
# 6. compute_group_residual_stats
# ---------------------------------------------------------------------------

def compute_group_residual_stats(
    x: torch.Tensor,
    group_size: int = 64,
    hist_bins: int = 160,
    hist_min: float = -4.0,
    hist_max: float = 4.0,
) -> Tuple[Dict[str, float], List[Dict[str, Any]]]:
    # Flatten non-last dimensions, keep last dim
    x = x.float()
    orig_shape = x.shape
    D = x.shape[-1]

    # Truncate last dim to multiple of group_size
    if D < group_size:
        group_size = D
    n_groups = D // group_size
    if n_groups == 0:
        return {}, []

    usable = n_groups * group_size
    x_trunc = x[..., :usable]

    x_grp = x_trunc.reshape(-1, n_groups, group_size)
    group_mean = x_grp.mean(dim=-1, keepdim=True)
    residual = x_grp - group_mean
    rms = residual.square().mean(dim=-1, keepdim=True).sqrt().clamp(min=1e-8)
    t = (residual / rms).flatten()

    numel = t.numel()
    mean = t.mean().item()
    std = t.std().item()

    # Skewness & kurtosis
    if numel > 1:
        t_centered = t - mean
        var = t_centered.square().mean()
        if var > 1e-12:
            skew = (t_centered.pow(3).mean() / (var ** 1.5)).item()
            kurt = (t_centered.pow(4).mean() / (var ** 2)).item()
        else:
            skew = 0.0
            kurt = 3.0
    else:
        skew = 0.0
        kurt = 3.0

    # Percentiles
    if numel > 0:
        sorted_t, _ = t.sort()
        p01 = sorted_t[int(0.01 * numel)].item() if int(0.01 * numel) < numel else sorted_t[-1].item()
        p05 = sorted_t[int(0.05 * numel)].item() if int(0.05 * numel) < numel else sorted_t[-1].item()
        p50 = sorted_t[int(0.50 * numel)].item() if int(0.50 * numel) < numel else sorted_t[-1].item()
        p95 = sorted_t[min(int(0.95 * numel), numel - 1)].item()
        p99 = sorted_t[min(int(0.99 * numel), numel - 1)].item()
    else:
        p01 = p05 = p50 = p95 = p99 = 0.0

    # Histogram
    bins = torch.linspace(hist_min, hist_max, hist_bins + 1, device=t.device)
    bin_centers = (bins[:-1] + bins[1:]) / 2
    hist_count = torch.histc(t, bins=hist_bins, min=hist_min, max=hist_max)
    total = hist_count.sum()
    hist_density = hist_count / total if total > 0 else hist_count

    stats = {
        "numel": numel,
        "mean": round(mean, 6),
        "std": round(std, 6),
        "skewness": round(skew, 6),
        "kurtosis": round(kurt, 6),
        "p01": round(p01, 6),
        "p05": round(p05, 6),
        "p50": round(p50, 6),
        "p95": round(p95, 6),
        "p99": round(p99, 6),
    }

    hist_rows = []
    for i in range(hist_bins):
        hist_rows.append({
            "bin_left": round(float(bins[i].item()), 6),
            "bin_right": round(float(bins[i + 1].item()), 6),
            "bin_center": round(float(bin_centers[i].item()), 6),
            "density": round(float(hist_density[i].item()), 10),
        })

    return stats, hist_rows


# ---------------------------------------------------------------------------
# 7. compute_quant_error_from_quantizer
# ---------------------------------------------------------------------------

def compute_quant_error_from_quantizer(
    quantizer: nn.Module,
    x: torch.Tensor,
) -> Optional[Dict[str, Any]]:
    try:
        with torch.no_grad():
            qx = quantizer(x)
    except Exception as e:
        print(f"  [WARNING] quantizer forward failed: {e}")
        return None

    x_f = x.float().flatten()
    qx_f = qx.float().flatten()
    diff = qx_f - x_f
    numel = x_f.numel()

    mse = diff.square().mean().item()
    mae = diff.abs().mean().item()
    cosine = F.cosine_similarity(x_f.unsqueeze(0), qx_f.unsqueeze(0)).item()
    fp32_norm = (x_f.norm() / math.sqrt(numel)).item()
    quant_norm = (qx_f.norm() / math.sqrt(numel)).item()

    return {
        "numel": numel,
        "mse": round(mse, 12),
        "mae": round(mae, 10),
        "cosine": round(cosine, 8),
        "fp32_norm": round(fp32_norm, 8),
        "quant_norm": round(quant_norm, 8),
    }


# ---------------------------------------------------------------------------
# 8. TraceCollector
# ---------------------------------------------------------------------------

class TraceCollector:
    """Collects codebook, quant error, residual stats, and train/val metrics."""

    def __init__(self, model: nn.Module, config: dict):
        self.enabled = bool(config.get("trace_enabled", False))
        if not self.enabled:
            return

        self.output_dir = ensure_trace_dir(config)
        self.selected_layers = select_trace_layers(model, max_layers=3)
        self.trace_num_batches = int(config.get("trace_num_batches", 4))
        self.hist_bins = int(config.get("trace_hist_bins", 160))
        self.hist_min = float(config.get("trace_hist_min", -4.0))
        self.hist_max = float(config.get("trace_hist_max", 4.0))

        self.save_codebook = bool(config.get("trace_save_codebook", True))
        self.save_quant_error = bool(config.get("trace_save_quant_error", True))
        self.save_residual_stats = bool(config.get("trace_save_residual_stats", True))
        self.save_train_val = bool(config.get("trace_save_train_val", True))

        # Per-epoch activation cache: {layer_name: [tensor, ...]}
        self._act_cache: Dict[str, List[torch.Tensor]] = {}
        self._current_epoch = 0
        self._batch_count = 0

        # Accumulated rows
        self.codebook_rows: List[Dict] = []
        self.quant_error_rows: List[Dict] = []
        self.residual_stats_rows: List[Dict] = []
        self.hist_rows: List[Dict] = []
        self.train_val_rows: List[Dict] = []

        # Register hooks
        self._hooks = []
        for name, module in self.selected_layers:
            hook = module.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

        self._base_row = {
            "dataset": config.get("dataset", "cifar100"),
            "backbone": config.get("model_key", "deit_small"),
            "method": config.get("method", "pcarcq_nodc"),
            "bit": config.get("w_bits", config.get("bits", 3)),
        }

        print(f"[TRACE] Collector initialized: layers={len(self.selected_layers)} "
              f"num_batches={self.trace_num_batches}")

    def _make_hook(self, layer_name: str):
        def hook_fn(module, input_, output_):
            if not self.enabled:
                return
            if self._batch_count >= self.trace_num_batches:
                return
            inp = input_[0] if isinstance(input_, (tuple, list)) else input_
            self._act_cache.setdefault(layer_name, []).append(inp.detach().cpu())
        return hook_fn

    def reset_epoch(self, epoch: int):
        self._current_epoch = epoch
        self._batch_count = 0
        self._act_cache.clear()

    def record_batch(self):
        self._batch_count += 1

    def record_train_val(
        self,
        epoch: int,
        train_loss: float,
        train_top1: float,
        val_loss: float,
        val_top1: float,
        lr: float,
        compile_index: int = 0,
    ):
        if not self.enabled:
            return
        self.train_val_rows.append({
            **self._base_row,
            "epoch": epoch,
            "train_loss": round(train_loss, 6),
            "train_top1": round(train_top1, 4),
            "val_loss": round(val_loss, 6),
            "val_top1": round(val_top1, 4),
            "lr": f"{lr:.8e}",
            "compile_index": compile_index,
        })

    def on_epoch_end(self, epoch: int, compile_index: int = 0):
        if not self.enabled:
            return

        for layer_name, module in self.selected_layers:
            acts = self._act_cache.get(layer_name, [])
            if not acts:
                continue
            x = torch.cat(acts, dim=0).float()

            # Get group_size from activation quantizer
            act_q = get_activation_quantizer(module)
            group_size = 64
            if act_q is not None and hasattr(act_q, "group_size"):
                group_size = act_q.group_size
            elif hasattr(module, "act_quant") and hasattr(module.act_quant, "group_size"):
                group_size = module.act_quant.group_size

            # --- Codebook ---
            if self.save_codebook and act_q is not None:
                cb = extract_codebook(act_q)
                if cb is not None:
                    for rank, val in enumerate(cb):
                        self.codebook_rows.append({
                            **self._base_row,
                            "epoch": epoch,
                            "compile_index": compile_index,
                            "layer_name": layer_name,
                            "quantizer_type": "activation",
                            "codebook_rank": rank,
                            "codebook_value": round(float(val), 8),
                        })

            # --- Quant error ---
            if self.save_quant_error and act_q is not None:
                qe = compute_quant_error_from_quantizer(act_q, x)
                if qe is not None:
                    self.quant_error_rows.append({
                        **self._base_row,
                        "epoch": epoch,
                        "compile_index": compile_index,
                        "layer_name": layer_name,
                        "quantizer_type": "activation",
                        **qe,
                    })

            # --- Residual stats ---
            if self.save_residual_stats:
                stats, hist_rows = compute_group_residual_stats(
                    x, group_size=group_size,
                    hist_bins=self.hist_bins,
                    hist_min=self.hist_min,
                    hist_max=self.hist_max,
                )
                if stats:
                    self.residual_stats_rows.append({
                        **self._base_row,
                        "epoch": epoch,
                        "compile_index": compile_index,
                        "layer_name": layer_name,
                        **stats,
                    })
                    for hr in hist_rows:
                        self.hist_rows.append({
                            **self._base_row,
                            "epoch": epoch,
                            "compile_index": compile_index,
                            "layer_name": layer_name,
                            **hr,
                        })

    def flush(self):
        if not self.enabled:
            return

        base = self.output_dir

        if self.save_codebook and self.codebook_rows:
            path = base / "codebook_trace.csv"
            self._write_csv(path, self.codebook_rows, [
                "dataset", "backbone", "method", "bit",
                "epoch", "compile_index", "layer_name", "quantizer_type",
                "codebook_rank", "codebook_value",
            ])

        if self.save_quant_error and self.quant_error_rows:
            path = base / "quant_error_trace.csv"
            self._write_csv(path, self.quant_error_rows, [
                "dataset", "backbone", "method", "bit",
                "epoch", "compile_index", "layer_name", "quantizer_type",
                "numel", "mse", "mae", "cosine", "fp32_norm", "quant_norm",
            ])

        if self.save_residual_stats and self.residual_stats_rows:
            path = base / "residual_stats_trace.csv"
            self._write_csv(path, self.residual_stats_rows, [
                "dataset", "backbone", "method", "bit",
                "epoch", "compile_index", "layer_name",
                "numel", "mean", "std", "skewness", "kurtosis",
                "p01", "p05", "p50", "p95", "p99",
            ])

            path = base / "residual_hist_trace.csv"
            self._write_csv(path, self.hist_rows, [
                "dataset", "backbone", "method", "bit",
                "epoch", "compile_index", "layer_name",
                "bin_left", "bin_right", "bin_center", "density",
            ])

        if self.save_train_val and self.train_val_rows:
            path = base / "train_val_trace.csv"
            self._write_csv(path, self.train_val_rows, [
                "dataset", "backbone", "method", "bit",
                "epoch", "train_loss", "train_top1", "val_loss", "val_top1",
                "lr", "compile_index",
            ])

    def _write_csv(self, path: Path, rows: List[Dict], fieldnames: List[str]):
        if not rows:
            return
        with open(path, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fieldnames)
            w.writeheader()
            w.writerows(rows)
        print(f"[TRACE] wrote {path} rows={len(rows)}")
