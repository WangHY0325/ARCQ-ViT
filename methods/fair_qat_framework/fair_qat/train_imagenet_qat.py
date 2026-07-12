"""
Unified ImageNet-1K QAT entry for AAAI ViT experiments.

This script is intentionally separate from the old CIFAR entry.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from contextlib import nullcontext

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.distributed as dist
import yaml
from torch.cuda.amp import autocast, GradScaler
from torch.nn.parallel import DistributedDataParallel as DDP

_PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJ not in sys.path:
    sys.path.insert(0, _PROJ)

from utils import set_seed, format_time, save_checkpoint, save_metrics
from data_imagenet import build_imagenet_loaders
from fair_qat.quant_backends import get_backend
from fair_qat.timm_quant_models import build_timm_quant_model


def init_distributed():
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    distributed = world_size > 1
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP requires CUDA in this training script.")
        torch.cuda.set_device(local_rank)
        dist.init_process_group(backend="nccl", init_method="env://")
    return distributed, rank, local_rank, world_size


def is_main_process(rank: int) -> bool:
    return rank == 0


def should_save_checkpoint(method: str, config: dict) -> bool:
    """Only keep model checkpoints for our DDFZ-family runs by default."""
    if "save_checkpoints" in config:
        return bool(config["save_checkpoints"])
    return "ddfz" in str(method).lower()


def ddp_barrier(distributed: bool, local_rank: int):
    if not distributed:
        return
    if torch.cuda.is_available():
        dist.barrier(device_ids=[local_rank])
    else:
        dist.barrier()


def ddp_sum(values, device, distributed: bool):
    tensor = torch.tensor(values, dtype=torch.float64, device=device)
    if distributed:
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor.cpu().tolist()


def configure_pcddfz_schedule(config, steps_per_epoch: int, rank: int):
    epochs = int(config.get("epochs", 200))
    explicit_steps = config.get("pc_compile_steps", None)
    updates_per_epoch = config.get("pc_compile_updates_per_epoch", None)
    default_epochs = [0, 0.5, 1, 2, 5, 10, 20, 40, 80, 120, 160]

    if explicit_steps is not None:
        if isinstance(explicit_steps, str):
            steps = sorted({
                max(0, int(x.strip()))
                for x in explicit_steps.split(",")
                if x.strip()
            })
        else:
            steps = sorted({max(0, int(x)) for x in explicit_steps})
        schedule_desc = f"explicit_steps={steps}"
    elif updates_per_epoch is not None:
        updates_per_epoch = max(1, int(updates_per_epoch))
        steps = []
        for epoch_idx in range(epochs):
            epoch_base = epoch_idx * steps_per_epoch
            for update_idx in range(updates_per_epoch):
                offset = int(round(update_idx * steps_per_epoch / updates_per_epoch))
                steps.append(epoch_base + offset)
        steps = sorted(set(steps))
        schedule_desc = (
            f"updates_per_epoch={updates_per_epoch} "
            f"total_updates={len(steps)}"
        )
    else:
        compile_epochs = config.get("pc_compile_epochs", default_epochs)
        if isinstance(compile_epochs, str):
            compile_epochs = [float(x.strip()) for x in compile_epochs.split(",") if x.strip()]
        else:
            compile_epochs = [float(x) for x in compile_epochs]
        steps = sorted({
            max(0, int(round(e * steps_per_epoch)))
            for e in compile_epochs
            if e <= epochs
        })
        schedule_desc = f"compile_epochs={compile_epochs}"

    if 0 not in steps:
        steps.insert(0, 0)
    os.environ["DDFZ_PC_COMPILE_STEPS"] = ",".join(str(s) for s in steps)

    if is_main_process(rank):
        print(
            "[PCDDFZ_SCHEDULE] "
            f"steps_per_epoch={steps_per_epoch} "
            f"{schedule_desc} "
            f"compile_steps={os.environ['DDFZ_PC_COMPILE_STEPS']}"
        )
    return steps


def amp_context(device):
    return autocast() if device.type == "cuda" else nullcontext()


def accuracy(output, target, topk=(1, 5)):
    maxk = max(topk)
    _, pred = output.topk(maxk, dim=1, largest=True, sorted=True)
    pred = pred.t()
    correct = pred.eq(target.reshape(1, -1).expand_as(pred))
    result = []
    for k in topk:
        correct_k = correct[:k].reshape(-1).float().sum(0)
        result.append(correct_k)
    return result


def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    scaler,
    device,
    grad_accum_steps,
    distributed=False,
    epoch=0,
    rank=0,
    world_size=1,
    log_interval=200,
    max_steps=None,
    kd_criterion=None,
):
    model.train()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_sum = 0
    optimizer.zero_grad(set_to_none=True)
    t_epoch = time.time()
    last_loss = 0.0

    step_count = 0
    for step, (images, targets) in enumerate(loader, start=1):
        if max_steps is not None and step > int(max_steps):
            break
        step_count = step
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        with amp_context(device):
            outputs = model(images)
            logits = extract_logits(outputs)
            if kd_criterion is not None:
                raw_loss = kd_criterion(images, outputs, targets)
            else:
                raw_loss = criterion(logits, targets)
            loss = raw_loss / grad_accum_steps

        scaler.scale(loss).backward()
        if step % grad_accum_steps == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        batch_n = targets.size(0)
        top1, top5 = accuracy(logits.detach(), targets, topk=(1, 5))
        last_loss = float(raw_loss.detach())
        loss_sum += float(raw_loss.detach()) * batch_n
        top1_sum += float(top1)
        top5_sum += float(top5)
        n_sum += batch_n

        if rank == 0 and log_interval > 0 and (step == 1 or step % log_interval == 0):
            elapsed = max(time.time() - t_epoch, 1e-6)
            seen = n_sum * max(1, world_size)
            img_s = seen / elapsed
            print(
                f"[TRAIN_STEP] epoch={epoch} step={step}/{len(loader)} "
                f"loss={last_loss:.4f} img_s={img_s:.1f}",
                flush=True,
            )

    if step_count > 0 and step_count % grad_accum_steps != 0:
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad(set_to_none=True)

    loss_sum, top1_sum, top5_sum, n_sum = ddp_sum(
        [loss_sum, top1_sum, top5_sum, n_sum], device, distributed
    )
    return {
        "loss": loss_sum / max(n_sum, 1.0),
        "top1": 100.0 * top1_sum / max(n_sum, 1.0),
        "top5": 100.0 * top5_sum / max(n_sum, 1.0),
    }


@torch.no_grad()
def validate(model, loader, criterion, device, distributed=False, max_steps=None):
    model.eval()
    loss_sum = 0.0
    top1_sum = 0.0
    top5_sum = 0.0
    n_sum = 0

    for step, (images, targets) in enumerate(loader, start=1):
        if max_steps is not None and step > int(max_steps):
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        outputs = model(images)
        loss = criterion(outputs, targets)
        batch_n = targets.size(0)
        top1, top5 = accuracy(outputs, targets, topk=(1, 5))
        loss_sum += float(loss) * batch_n
        top1_sum += float(top1)
        top5_sum += float(top5)
        n_sum += batch_n

    loss_sum, top1_sum, top5_sum, n_sum = ddp_sum(
        [loss_sum, top1_sum, top5_sum, n_sum], device, distributed
    )
    return {
        "loss": loss_sum / max(n_sum, 1.0),
        "top1": 100.0 * top1_sum / max(n_sum, 1.0),
        "top5": 100.0 * top5_sum / max(n_sum, 1.0),
    }


class FeatureHook:
    """Capture penultimate-layer features via head forward pre-hook."""

    def __init__(self, model):
        self.features = None
        core = model.module if hasattr(model, "module") else model
        head = None
        for name in ("head", "classifier", "fc"):
            if hasattr(core, name):
                head = getattr(core, name)
                break
        if head is None:
            raise RuntimeError("Cannot find classifier head for feature hook")
        self.handle = head.register_forward_pre_hook(self._hook)

    def _hook(self, module, inputs):
        self.features = inputs[0]

    def get(self):
        if self.features is None:
            raise RuntimeError("FeatureHook captured no features")
        x = self.features
        if x.dim() > 2:
            x = x.flatten(1)
        return x

    def remove(self):
        self.handle.remove()


def extract_logits(outputs):
    if torch.is_tensor(outputs):
        return outputs
    if isinstance(outputs, (tuple, list)):
        for x in reversed(outputs):
            if torch.is_tensor(x) and x.dim() == 2:
                return x
    raise TypeError("Cannot extract logits from output type: " + str(type(outputs)))


class PCAProjector(nn.Module):
    def __init__(self, mean, components):
        super().__init__()
        self.register_buffer("mean", mean.float())
        self.register_buffer("components", components.float())

    def transform(self, x):
        if x.dim() > 2:
            x = x.flatten(1)
        x = x.float() - self.mean.to(x.device)
        return x @ self.components.to(x.device).t()


@torch.no_grad()
def fit_or_load_pca_projector(
    teacher_model,
    teacher_hook,
    train_loader,
    device,
    config,
    distributed=False,
    rank=0,
    local_rank=0,
):
    pca_path = config.get("kd_pca_path", "")
    pca_components = int(config.get("kd_pca_components", 64))
    pca_samples = int(config.get("kd_pca_samples", 2048))

    if distributed and not pca_path:
        raise RuntimeError("Distributed GPLQ TCS KD requires kd_pca_path")

    if distributed and rank != 0:
        ddp_barrier(distributed, local_rank)
        obj = torch.load(pca_path, map_location="cpu")
        print("[KD][GPLQ_TCS] load PCA: " + pca_path, flush=True)
        return PCAProjector(obj["mean"], obj["components"]).to(device)

    if pca_path and os.path.isfile(pca_path):
        obj = torch.load(pca_path, map_location="cpu")
        print("[KD][GPLQ_TCS] load PCA: " + pca_path, flush=True)
        projector = PCAProjector(obj["mean"], obj["components"]).to(device)
        if distributed:
            ddp_barrier(distributed, local_rank)
        return projector

    feats = []
    seen = 0
    teacher_model.eval()
    for images, _ in train_loader:
        images = images.to(device, non_blocking=True)
        teacher_model(images)
        f = teacher_hook.get().detach().float().cpu()
        feats.append(f)
        seen += f.shape[0]
        if seen >= pca_samples:
            break
    x = torch.cat(feats, dim=0)[:pca_samples]
    mean = x.mean(dim=0)
    centered = x - mean
    q = min(pca_components, centered.shape[0] - 1, centered.shape[1])
    if q <= 0:
        raise RuntimeError("Invalid PCA shape: " + str(tuple(centered.shape)))
    _, _, v = torch.pca_lowrank(centered, q=q, center=False)
    components = v[:, :q].t().contiguous()
    if pca_path:
        os.makedirs(os.path.dirname(pca_path), exist_ok=True)
        torch.save({"mean": mean, "components": components}, pca_path)
        print("[KD][GPLQ_TCS] save PCA: " + pca_path, flush=True)
    if distributed:
        ddp_barrier(distributed, local_rank)
    return PCAProjector(mean, components).to(device)


def build_fp32_teacher(config, device):
    teacher_ckpt = config.get("teacher_checkpoint") or config.get("teacher") or config.get("initial_checkpoint")
    if not teacher_ckpt or not os.path.isfile(teacher_ckpt):
        raise FileNotFoundError("KD teacher checkpoint not found: " + str(teacher_ckpt))
    teacher_config = dict(config)
    teacher_config["method"] = "fp32"
    teacher_config["w_bits"] = 32
    teacher_config["a_bits"] = 32
    teacher_config["head_quant"] = "fp32"
    teacher_config["initial_checkpoint"] = teacher_ckpt
    teacher_config["pretrained_checkpoint"] = teacher_ckpt
    teacher_config["timm_pretrained"] = False
    teacher = build_timm_quant_model(teacher_config, get_backend("fp32")).to(device)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    print("[KD] teacher_checkpoint=" + str(teacher_ckpt), flush=True)
    return teacher


class SoftLogitKDLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, alpha, tau):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = float(alpha)
        self.tau = float(tau)

    def forward(self, images, outputs, targets):
        student_logits = extract_logits(outputs)
        base_loss = self.base_criterion(student_logits, targets)
        with torch.no_grad():
            teacher_logits = extract_logits(self.teacher_model(images))
        t = self.tau
        kd_loss = F.kl_div(
            F.log_softmax(student_logits / t, dim=1),
            F.log_softmax(teacher_logits / t, dim=1),
            reduction="batchmean",
            log_target=True,
        ) * (t * t)
        return base_loss * (1.0 - self.alpha) + kd_loss * self.alpha


class HardLogitKDLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, alpha):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = float(alpha)

    def forward(self, images, outputs, targets):
        student_logits = extract_logits(outputs)
        base_loss = self.base_criterion(student_logits, targets)
        with torch.no_grad():
            teacher_logits = extract_logits(self.teacher_model(images))
        kd_loss = F.cross_entropy(student_logits, teacher_logits.argmax(dim=1))
        return base_loss * (1.0 - self.alpha) + kd_loss * self.alpha


class OfficialPackQViTLoss(nn.Module):
    """PackQViT official DeiT-style hard/soft distillation.

    The official PackQViT loss applies the task loss on the primary student
    logits and distillation on the auxiliary branch when present. For our fair
    timm wrapper, most models return one logits tensor, so the same logits are
    used for both branches.
    """

    def __init__(self, base_criterion, teacher_model, alpha, tau, distillation_type):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.alpha = float(alpha)
        self.tau = float(tau)
        self.distillation_type = str(distillation_type).lower()

    @staticmethod
    def _split_outputs(outputs):
        if isinstance(outputs, (tuple, list)) and len(outputs) >= 2:
            return outputs[0], outputs[1]
        logits = extract_logits(outputs)
        return logits, logits

    def forward(self, images, outputs, targets):
        student_logits, student_kd_logits = self._split_outputs(outputs)
        base_loss = self.base_criterion(student_logits, targets)
        with torch.no_grad():
            teacher_logits = extract_logits(self.teacher_model(images))

        if self.distillation_type == "soft":
            t = self.tau
            distill_loss = F.kl_div(
                F.log_softmax(student_kd_logits / t, dim=1),
                F.log_softmax(teacher_logits / t, dim=1),
                reduction="sum",
                log_target=True,
            ) * (t * t) / student_kd_logits.numel()
        elif self.distillation_type == "hard":
            distill_loss = F.cross_entropy(student_kd_logits, teacher_logits.argmax(dim=1))
        else:
            raise ValueError(
                "PackQViT distillation_type must be hard or soft, got "
                + str(self.distillation_type)
            )

        return base_loss * (1.0 - self.alpha) + distill_loss * self.alpha


class GPLQTCSLoss(nn.Module):
    def __init__(self, base_criterion, teacher_model, student_hook, teacher_hook, pca_projector, alpha):
        super().__init__()
        self.base_criterion = base_criterion
        self.teacher_model = teacher_model
        self.student_hook = student_hook
        self.teacher_hook = teacher_hook
        self.pca_projector = pca_projector
        self.alpha = float(alpha)

    def forward(self, images, outputs, targets):
        student_logits = extract_logits(outputs)
        base_loss = self.base_criterion(student_logits, targets)
        student_features = self.student_hook.get()
        with torch.no_grad():
            self.teacher_model(images)
            teacher_features = self.teacher_hook.get()
        student_pca = self.pca_projector.transform(student_features)
        teacher_pca = self.pca_projector.transform(teacher_features)
        tcs_loss = F.mse_loss(student_pca, teacher_pca)
        return base_loss + self.alpha * tcs_loss


def build_kd_criterion(
    config,
    model,
    train_loader,
    base_criterion,
    device,
    distributed=False,
    rank=0,
    local_rank=0,
):
    if not bool(config.get("distillation", False)):
        return None
    method = str(config.get("method", "")).lower()
    if method == "lsq":
        return None
    distillation_mode = str(config.get("distillation", "")).lower()
    if method == "packqvit" or distillation_mode == "official_packqvit":
        teacher = build_fp32_teacher(config, device)
        alpha = float(config.get("distill_alpha", 0.5))
        tau = float(config.get("distill_tau", 1.0))
        distillation_type = str(config.get("distillation_type", "hard")).lower()
        print(
            "[PACKQVIT] official_distillation="
            + distillation_type
            + " alpha="
            + str(alpha)
            + " tau="
            + str(tau)
            + " teacher="
            + str(config.get("teacher") or config.get("teacher_checkpoint")),
            flush=True,
        )
        return OfficialPackQViTLoss(base_criterion, teacher, alpha, tau, distillation_type)
    kd_type = str(config.get("kd_type", "")).lower()
    teacher = build_fp32_teacher(config, device)
    if kd_type == "logit_soft":
        alpha = float(config.get("distill_alpha", 0.5))
        tau = float(config.get("distill_tau", 2.0))
        print("[KD] type=logit_soft alpha=" + str(alpha) + " tau=" + str(tau), flush=True)
        return SoftLogitKDLoss(base_criterion, teacher, alpha, tau)
    if kd_type == "qvit_hard":
        alpha = float(config.get("distill_alpha", 0.5))
        print("[KD] type=qvit_hard alpha=" + str(alpha), flush=True)
        return HardLogitKDLoss(base_criterion, teacher, alpha)
    if kd_type == "gplq_tcs":
        alpha = float(config.get("distill_alpha", 0.1))
        student_hook = FeatureHook(model)
        teacher_hook = FeatureHook(teacher)
        pca_projector = fit_or_load_pca_projector(
            teacher,
            teacher_hook,
            train_loader,
            device,
            config,
            distributed=distributed,
            rank=rank,
            local_rank=local_rank,
        )
        print(
            "[KD] type=gplq_tcs alpha="
            + str(alpha)
            + " pca_dim="
            + str(pca_projector.components.shape[0]),
            flush=True,
        )
        return GPLQTCSLoss(base_criterion, teacher, student_hook, teacher_hook, pca_projector, alpha)
    raise ValueError("distillation=true but invalid kd_type=" + str(kd_type))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    print(
        f"[BOOT] pid={os.getpid()} rank_env={os.environ.get('RANK', '0')} "
        f"local_rank_env={os.environ.get('LOCAL_RANK', '0')} world_size_env={os.environ.get('WORLD_SIZE', '1')}",
        flush=True,
    )
    distributed, rank, local_rank, world_size = init_distributed()
    if is_main_process(rank):
        print("[BOOT] distributed init done", flush=True)

    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    if distributed:
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    # Keep every rank's module initialization identical. This lets us disable
    # DDP's expensive initial state sync safely; DistributedSampler still gives
    # each rank different samples.
    set_seed(int(config.get("seed", 42)))

    output_dir = config["output_dir"]
    if is_main_process(rank):
        os.makedirs(output_dir, exist_ok=True)

    method = config["method"]
    backend = get_backend(method)
    save_ckpt = should_save_checkpoint(method, config)
    if is_main_process(rank):
        print(f"[INFO] method={method} backend={backend.name} model={config['model_name']}")
        print(f"[INFO] output_dir={output_dir}")
        print(f"[DDP] distributed={distributed} world_size={world_size}")
        print(f"[CHECKPOINT] save_checkpoints={save_ckpt}")
        if str(method).lower() == "n2uq":
            print("[N2UQ] learnable_threshold=True output_levels=uniform", flush=True)

    run_config = dict(config)
    per_gpu_batch = int(config.get("batch_size", 128))
    per_gpu_val_batch = int(config.get("val_batch_size", per_gpu_batch))
    run_config["batch_size"] = per_gpu_batch
    run_config["val_batch_size"] = per_gpu_val_batch
    global_batch = per_gpu_batch * max(1, world_size)
    global_val_batch = per_gpu_val_batch * max(1, world_size)

    train_loader, val_loader = build_imagenet_loaders(
        run_config,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )
    if is_main_process(rank):
        print(
            f"[DATA] train={len(train_loader.dataset)} val={len(val_loader.dataset)} "
            f"per_gpu_batch={per_gpu_batch} global_batch={global_batch} "
            f"per_gpu_val_batch={per_gpu_val_batch} global_val_batch={global_val_batch} "
            f"grad_accum={config.get('grad_accum_steps', 1)} "
            f"effective_batch={global_batch * int(config.get('grad_accum_steps', 1))}",
            flush=True,
        )
    configure_pcddfz_schedule(config, len(train_loader), rank)

    model = build_timm_quant_model(config, backend).to(device)

    if method == "gplq":
        calib_batches = int(config.get("calib_batches", 32))
        if is_main_process(rank):
            print(f"[GPLQ] collecting activation stats calib_batches={calib_batches}", flush=True)
        from fair_qat.gplq_adapter import collect_gplq_activation_stats, initialize_gplq_acts
        collect_gplq_activation_stats(model, train_loader, device, num_batches=calib_batches)
        initialize_gplq_acts(model, device)
        if is_main_process(rank):
            print("[GPLQ] activation initialization complete", flush=True)

    criterion = nn.CrossEntropyLoss()
    kd_criterion = build_kd_criterion(
        config=config,
        model=model,
        train_loader=train_loader,
        base_criterion=criterion,
        device=device,
        distributed=distributed,
        rank=rank,
        local_rank=local_rank,
    )

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if distributed:
        find_unused = bool(config.get("ddp_find_unused_parameters", True))
        broadcast_buffers = bool(config.get("ddp_broadcast_buffers", False))
        init_sync = bool(config.get("ddp_init_sync", False))
        if is_main_process(rank):
            print(f"[DDP] find_unused_parameters={find_unused}")
            print(f"[DDP] broadcast_buffers={broadcast_buffers}")
            print(f"[DDP] init_sync={init_sync}")
        ddp_kwargs = dict(
            device_ids=[local_rank],
            output_device=local_rank,
            find_unused_parameters=find_unused,
            broadcast_buffers=broadcast_buffers,
        )
        try:
            model = DDP(model, init_sync=init_sync, **ddp_kwargs)
        except TypeError:
            if is_main_process(rank):
                print("[DDP] init_sync argument unsupported; falling back to default init sync", flush=True)
            model = DDP(model, **ddp_kwargs)
    if is_main_process(rank):
        print(f"[MODEL] trainable_params={n_params:,}")

    optimizer = optim.AdamW(
        model.parameters(),
        lr=float(config.get("lr", 5e-5)),
        weight_decay=float(config.get("weight_decay", 0.05)),
    )
    epochs = int(config.get("epochs", 200))
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    scaler = GradScaler(enabled=(device.type == "cuda"))
    grad_accum_steps = int(config.get("grad_accum_steps", 1))
    log_interval = int(config.get("log_interval", 200))
    max_train_steps = config.get("max_train_steps_per_epoch", None)
    max_val_steps = config.get("max_val_steps", None)

    min_epochs = int(config.get("early_stop_min_epochs", 80))
    patience = int(config.get("early_stop_patience", 20))
    min_delta = float(config.get("early_stop_min_delta", 0.05))
    best_top1 = -1.0
    best_epoch = 0
    stale_epochs = 0
    logs = []
    t_start = time.time()

    if is_main_process(rank):
        with open(os.path.join(output_dir, "config.yaml"), "w") as f:
            yaml.safe_dump(config, f, sort_keys=False)
        print(
            f"{'Epoch':>5} {'Train Loss':>10} {'Train Top1':>10} "
            f"{'Val Loss':>10} {'Val Top1':>9} {'LR':>12} {'Time':>10}",
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        if hasattr(train_loader.sampler, "set_epoch"):
            train_loader.sampler.set_epoch(epoch)
        t0 = time.time()
        train_stats = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            scaler,
            device,
            grad_accum_steps,
            distributed,
            epoch,
            rank,
            world_size,
            log_interval,
            max_steps=max_train_steps,
            kd_criterion=kd_criterion,
        )
        val_stats = validate(model, val_loader, criterion, device, distributed, max_steps=max_val_steps)
        scheduler.step()

        improved = val_stats["top1"] > best_top1 + min_delta
        if improved:
            best_top1 = val_stats["top1"]
            best_epoch = epoch
            stale_epochs = 0
        else:
            stale_epochs += 1

        model_to_save = model.module if distributed else model
        if save_ckpt and is_main_process(rank):
            save_checkpoint(model_to_save, optimizer, epoch, best_top1, improved, output_dir)
        rec = {
            "epoch": epoch,
            "lr": optimizer.param_groups[0]["lr"],
            "train": train_stats,
            "val": val_stats,
            "time": format_time(time.time() - t0),
            "best_top1": best_top1,
            "best_epoch": best_epoch,
        }
        if is_main_process(rank):
            logs.append(rec)
            print(
                f"{epoch:5d} {train_stats['loss']:10.4f} {train_stats['top1']:10.2f} "
                f"{val_stats['loss']:10.4f} {val_stats['top1']:9.2f} "
                f"{optimizer.param_groups[0]['lr']:12.6e} {rec['time']:>10}",
                flush=True,
            )
            print(json.dumps(rec, ensure_ascii=False), flush=True)

        if epoch >= min_epochs and stale_epochs >= patience:
            if is_main_process(rank):
                print(f"[EARLY_STOP] epoch={epoch} best_epoch={best_epoch} best_top1={best_top1:.3f}")
            break

    if is_main_process(rank):
        metrics = {
            "config": config,
            "method": method,
            "model_name": config["model_name"],
            "best_top1": best_top1,
            "best_epoch": best_epoch,
            "total_time": format_time(time.time() - t_start),
            "epochs": logs,
            "distributed": distributed,
            "world_size": world_size,
        }
        save_metrics(output_dir, metrics)
        print(f"[DONE] best_top1={best_top1:.3f} best_epoch={best_epoch}")
    if distributed:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
