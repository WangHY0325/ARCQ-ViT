"""
ImageFolder loader for transfer-learning ViT experiments.

Expected layout:
    dataset_root/
      train/class_x/*.jpg
      test/class_x/*.jpg

The training scripts use this for Aircraft, Cars and Flowers102 after exporting
the datasets to ImageFolder format.
"""

from __future__ import annotations

import os
from pathlib import Path

import torch.utils.data as data
import torchvision.datasets as datasets
import torchvision.transforms as transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _clean_path(path: str) -> str:
    return str(path).replace("\\", "/")


def _split_dirs(dataset_root: str):
    root = Path(_clean_path(dataset_root))
    train_dir = root / "train"
    val_dir = root / "test"
    if not val_dir.exists():
        val_dir = root / "val"
    if not train_dir.exists():
        raise FileNotFoundError(f"Missing transfer train directory: {train_dir}")
    if not val_dir.exists():
        raise FileNotFoundError(f"Missing transfer val/test directory: {val_dir}")
    return str(train_dir), str(val_dir)


def build_transfer_transform(train: bool, image_size: int = 224):
    if train:
        return transforms.Compose(
            [
                transforms.RandomResizedCrop(image_size, scale=(0.08, 1.0)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            ]
        )
    return transforms.Compose(
        [
            transforms.Resize(int(image_size * 256 / 224)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def build_transfer_loaders(config: dict):
    dataset_root = config.get("dataset_root", config.get("data_dir", ""))
    if not dataset_root:
        raise ValueError("Transfer config must provide dataset_root or data_dir")

    batch_size = int(config.get("batch_size", 32))
    val_batch_size = int(config.get("val_batch_size", batch_size))
    num_workers = int(config.get("num_workers", 4))
    image_size = int(config.get("image_size", 224))

    train_dir, val_dir = _split_dirs(dataset_root)
    train_dataset = datasets.ImageFolder(
        train_dir, transform=build_transfer_transform(True, image_size)
    )
    val_dataset = datasets.ImageFolder(
        val_dir, transform=build_transfer_transform(False, image_size)
    )

    expected_classes = int(config.get("num_classes", len(train_dataset.classes)))
    if len(train_dataset.classes) != expected_classes:
        raise ValueError(
            f"Transfer train class count mismatch: got {len(train_dataset.classes)}, "
            f"expected {expected_classes}, root={train_dir}"
        )
    if len(val_dataset.classes) != len(train_dataset.classes):
        raise ValueError(
            f"Transfer val class count mismatch: train={len(train_dataset.classes)} "
            f"val={len(val_dataset.classes)}"
        )

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader
