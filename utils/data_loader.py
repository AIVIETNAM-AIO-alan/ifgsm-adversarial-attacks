"""
utils/data_loader.py
─────────────────────────────────────────────────────────────
Load MNIST / CIFAR-10 với train / val / test split chuẩn.
"""

import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from torchvision.datasets import ImageFolder
from torchvision.datasets.utils import download_and_extract_archive
from typing import Tuple

# ── ImageNette constants ──────────────────────────────────────
_IMAGENETTE_URL  = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-320.tgz"
_IMAGENETTE_DIR  = "imagenette2-320"
_IMAGENETTE_MEAN = (0.485, 0.456, 0.406)
_IMAGENETTE_STD  = (0.229, 0.224, 0.225)


def _download_imagenette(root: str) -> str:
    """Download và giải nén ImageNette nếu chưa có. Trả về đường dẫn thư mục."""
    extract_dir = os.path.join(root, _IMAGENETTE_DIR)
    if not os.path.isdir(extract_dir):
        print(f"[ImageNette] Downloading (~1.5 GB) → {root}")
        download_and_extract_archive(_IMAGENETTE_URL, root)
        print(f"[ImageNette] Extracted → {extract_dir}")
    return extract_dir


def get_transforms(dataset_name: str) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Trả về (train_transform, test_transform) phù hợp với dataset.
    """
    if dataset_name.upper() == "MNIST":
        train_tf = transforms.Compose([
            transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
            transforms.ToTensor(),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
        ])

    elif dataset_name.upper() == "CIFAR10":
        mean = (0.4914, 0.4822, 0.4465)
        std  = (0.2470, 0.2435, 0.2616)
        train_tf = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
        test_tf = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])
    elif dataset_name.upper() == "IMAGENETTE":
        train_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.RandomCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENETTE_MEAN, _IMAGENETTE_STD),
        ])
        test_tf = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(_IMAGENETTE_MEAN, _IMAGENETTE_STD),
        ])
    else:
        raise ValueError(f"Dataset không được hỗ trợ: {dataset_name}")

    return train_tf, test_tf


def get_dataloaders(
    dataset_name : str  = "MNIST",
    root         : str  = "./data",
    batch_size   : int  = 64,
    val_split    : float = 0.1,
    num_workers  : int  = 2,
    seed         : int  = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Tạo train / val / test DataLoader.

    Args:
        dataset_name : "MNIST" | "CIFAR10"
        root         : thư mục lưu data
        batch_size   : batch size
        val_split    : tỷ lệ validation trong tập train
        num_workers  : số worker cho DataLoader
        seed         : random seed

    Returns:
        (train_loader, val_loader, test_loader)
    """
    train_tf, test_tf = get_transforms(dataset_name)

    # ── ImageNette: dùng ImageFolder (cấu trúc train/ val/) ───
    if dataset_name.upper() == "IMAGENETTE":
        data_dir  = _download_imagenette(root)
        full_train = ImageFolder(os.path.join(data_dir, "train"), transform=train_tf)
        test_set   = ImageFolder(os.path.join(data_dir, "val"),   transform=test_tf)

        n_val   = int(len(full_train) * val_split)
        n_train = len(full_train) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

        print(f"[DataLoader] ImageNette: "
              f"train={n_train:,} | val={n_val:,} | test={len(test_set):,}")

    # ── MNIST / CIFAR-10: không thay đổi ──────────────────────
    else:
        DS = datasets.MNIST if dataset_name.upper() == "MNIST" else datasets.CIFAR10

        full_train = DS(root=root, train=True,  download=True, transform=train_tf)
        test_set   = DS(root=root, train=False, download=True, transform=test_tf)

        n_val   = int(len(full_train) * val_split)
        n_train = len(full_train) - n_val
        generator = torch.Generator().manual_seed(seed)
        train_set, val_set = random_split(full_train, [n_train, n_val], generator=generator)

        print(f"[DataLoader] {dataset_name}: "
              f"train={n_train:,} | val={n_val:,} | test={len(test_set):,}")

    # ── DataLoaders ───────────────────────────────────────────
    train_loader = DataLoader(
        train_set, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        val_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        test_set, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, val_loader, test_loader


def get_in_channels(dataset_name: str) -> int:
    """Trả về số kênh ảnh của dataset."""
    return 1 if dataset_name.upper() == "MNIST" else 3


def get_input_size(dataset_name: str) -> int:
    """Trả về kích thước ảnh H=W của dataset."""
    if dataset_name.upper() == "IMAGENETTE":
        return 224
    return 28 if dataset_name.upper() == "MNIST" else 32


def get_clip_values(dataset_name: str):
    """
    Trả về (clip_min, clip_max) phù hợp với normalization của dataset.

    - MNIST / CIFAR-10 : pixel trong [0, 1]  → scalar float
    - ImageNette        : ImageNet-normalized  → tensor per-channel [3]
    """
    if dataset_name.upper() == "IMAGENETTE":
        mean = torch.tensor(_IMAGENETTE_MEAN)
        std  = torch.tensor(_IMAGENETTE_STD)
        return (0 - mean) / std, (1 - mean) / std
    return 0.0, 1.0
