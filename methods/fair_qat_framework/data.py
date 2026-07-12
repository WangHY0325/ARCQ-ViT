"""
CIFAR-100 Python-format dataset loader for ViT experiments.

Reads the CIFAR-100 dataset stored in the original Python pickle format:
    datasets/cifar-100-python/train
    datasets/cifar-100-python/test
    datasets/cifar-100-python/meta

Provides build_cifar100_loaders(config) returning (train_loader, val_loader).
"""

import os
import pickle
import numpy as np
from PIL import Image

import torch
import torch.utils.data as data
from torch.utils.data.distributed import DistributedSampler
import torchvision.transforms as transforms


CIFAR100_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR100_STD  = (0.2675, 0.2565, 0.2761)
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD  = (0.229, 0.224, 0.225)


def _unpickle(filepath: str):
    with open(filepath, "rb") as f:
        d = pickle.load(f, encoding="latin1")
    return d


class CIFAR100PythonDataset(data.Dataset):
    """CIFAR-100 dataset from the original Python pickle files."""

    def __init__(self, root: str, train: bool = True, transform=None):
        self.root = root
        self.train = train
        self.transform = transform

        filename = "train" if train else "test"
        filepath = os.path.join(root, filename)

        entry = _unpickle(filepath)
        self.data = entry["data"]
        self.labels = entry["fine_labels"]
        self.coarse_labels = entry.get("coarse_labels", None)

    def __getitem__(self, index):
        img_flat = self.data[index]  # shape [3072]
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)  # HWC
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        label = self.labels[index]
        return img, label

    def __len__(self):
        return len(self.data)


def _normalization_stats(name: str):
    name = str(name or "cifar100").lower()
    if name in {"imagenet", "imagenet1k", "in1k"}:
        return IMAGENET_MEAN, IMAGENET_STD
    if name in {"cifar", "cifar100"}:
        return CIFAR100_MEAN, CIFAR100_STD
    raise ValueError(f"Unsupported CIFAR-100 normalization: {name}")


def build_cifar100_transform(train: bool, image_size: int = 224, normalization: str = "cifar100"):
    """Build transform pipeline for CIFAR-100 ViT experiments."""
    mean, std = _normalization_stats(normalization)
    if train:
        return transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])
    else:
        return transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ])


def build_cifar100_loaders(config: dict):
    """Build train/val dataloaders for CIFAR-100.

    Args:
        config: dict with keys:
            data_dir: str — path to cifar-100-python/
            batch_size: int
            val_batch_size: int (default batch_size)
            num_workers: int (default 4)
            image_size: int (default 224)
            normalization: cifar100 | imagenet

    Returns:
        (train_loader, val_loader)
    """
    data_dir = config.get("data_dir", "")
    batch_size = config.get("batch_size", 128)
    val_batch_size = config.get("val_batch_size", batch_size)
    num_workers = config.get("num_workers", 4)
    image_size = config.get("image_size", 224)
    normalization = config.get("normalization", config.get("norm", "cifar100"))

    train_transform = build_cifar100_transform(
        train=True, image_size=image_size, normalization=normalization,
    )
    val_transform = build_cifar100_transform(
        train=False, image_size=image_size, normalization=normalization,
    )

    train_dataset = CIFAR100PythonDataset(data_dir, train=True,  transform=train_transform)
    val_dataset   = CIFAR100PythonDataset(data_dir, train=False, transform=val_transform)

    distributed = bool(config.get("distributed", False))
    rank = int(config.get("rank", 0))
    world_size = int(config.get("world_size", 1))
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
    else:
        train_sampler = None
        val_sampler = None

    train_loader = data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(train_sampler is None),
    )
    val_loader = data.DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        shuffle=False,
        sampler=val_sampler,
        num_workers=num_workers,
        pin_memory=True,
    )
    return train_loader, val_loader
