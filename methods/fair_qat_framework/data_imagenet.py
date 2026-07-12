"""
ImageNet-1K dataloaders for AAAI ViT quantization experiments.

Expected layout:
    data_dir/
      train/<synset>/*.JPEG
      val/<synset>/*.JPEG
"""

from __future__ import annotations

import os
from typing import Dict, Optional, Tuple

import torch
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision import datasets, transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.data import create_transform


def _require_imagenet_layout(data_dir: str) -> Tuple[str, str]:
    train_dir = os.path.join(data_dir, "train")
    val_dir = os.path.join(data_dir, "val")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(f"ImageNet train directory not found: {train_dir}")
    if not os.path.isdir(val_dir):
        raise FileNotFoundError(f"ImageNet val directory not found: {val_dir}")
    return train_dir, val_dir


def build_imagenet_transform(config: Dict, train: bool):
    image_size = int(config.get("image_size", 224))
    interpolation = config.get("interpolation", "bicubic")
    mean = tuple(config.get("mean", IMAGENET_DEFAULT_MEAN))
    std = tuple(config.get("std", IMAGENET_DEFAULT_STD))

    if train:
        return create_transform(
            input_size=image_size,
            is_training=True,
            color_jitter=float(config.get("color_jitter", 0.4)),
            auto_augment=config.get("aa", "rand-m9-mstd0.5-inc1"),
            interpolation=interpolation,
            re_prob=float(config.get("reprob", 0.25)),
            re_mode=config.get("remode", "pixel"),
            re_count=int(config.get("recount", 1)),
            mean=mean,
            std=std,
        )

    resize_size = int(config.get("val_resize", int(image_size / float(config.get("crop_pct", 0.875)))))
    return transforms.Compose([
        transforms.Resize(resize_size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ])


def build_imagenet_loaders(
    config: Dict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
):
    data_dir = config.get("data_dir", "/gpool/home/wanghongyang/WangHY/QuEST/AAAI/data/imagenet1k")
    train_dir, val_dir = _require_imagenet_layout(data_dir)

    batch_size = int(config.get("batch_size", 128))
    val_batch_size = int(config.get("val_batch_size", batch_size))
    num_workers = int(config.get("num_workers", 8))

    train_dataset = datasets.ImageFolder(train_dir, build_imagenet_transform(config, train=True))
    val_dataset = datasets.ImageFolder(val_dir, build_imagenet_transform(config, train=False))

    if int(config.get("num_classes", 1000)) != len(train_dataset.classes):
        raise ValueError(
            f"num_classes={config.get('num_classes')} but train folder has "
            f"{len(train_dataset.classes)} classes"
        )
    if len(val_dataset.classes) != len(train_dataset.classes):
        raise ValueError(
            f"train class count {len(train_dataset.classes)} != "
            f"val class count {len(val_dataset.classes)}"
        )

    train_sampler: Optional[DistributedSampler] = None
    val_sampler: Optional[DistributedSampler] = None
    if distributed:
        train_sampler = DistributedSampler(
            train_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        val_sampler = DistributedSampler(
            val_dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=False,
            drop_last=False,
        )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
        persistent_workers=num_workers > 0,
    )
    return train_loader, val_loader
