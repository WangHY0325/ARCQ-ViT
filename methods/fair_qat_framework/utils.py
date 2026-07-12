from __future__ import annotations

import json
import random
import time
from pathlib import Path

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def format_time(seconds: float) -> str:
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def save_checkpoint(model, optimizer, epoch: int, best_top1: float, improved: bool, output_dir) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    state = {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer is not None else None,
        "best_top1": float(best_top1),
    }
    torch.save(state, output / "latest.pt")
    if improved:
        torch.save(state, output / "best.pt")


def save_metrics(output_dir, metrics: dict) -> None:
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "metrics.json").open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
