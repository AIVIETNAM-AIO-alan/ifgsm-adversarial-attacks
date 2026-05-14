"""
models/resnet.py
─────────────────────────────────────────────────────────────
Wrapper cho ResNet-18 từ torchvision, hỗ trợ MNIST & CIFAR-10.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import ResNet18_Weights


def get_resnet18(in_channels: int = 1, num_classes: int = 10) -> nn.Module:
    """
    Trả về ResNet-18 đã được điều chỉnh cho dataset nhỏ.

    - in_channels=1  : thay conv1 để nhận ảnh grayscale (MNIST)
    - in_channels=3  : giữ nguyên conv1 gốc (CIFAR-10 / ImageNet)
    - Thay fc cuối   : num_classes đầu ra

    Args:
        in_channels : số kênh đầu vào (1 hoặc 3)
        num_classes : số lớp phân loại

    Returns:
        model: nn.Module
    """
    model = models.resnet18(weights=None)

    # Thay conv đầu tiên nếu ảnh grayscale
    if in_channels != 3:
        model.conv1 = nn.Conv2d(
            in_channels, 64,
            kernel_size=3, stride=1, padding=1, bias=False
        )
        # Bỏ maxpool đầu để không giảm kích thước quá nhiều với ảnh 28x28
        model.maxpool = nn.Identity()

    # Thay lớp fully-connected cuối
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    return model


def get_resnet18_imagenette(num_classes: int = 10) -> nn.Module:
    """
    ResNet-18 pretrained trên ImageNet, fine-tune cho ImageNette (10 lớp).

    Dùng kiến trúc gốc (conv1 7×7, stride=2, maxpool) phù hợp với ảnh 224×224.
    Chỉ thay lớp fc cuối cho 10 lớp ImageNette.

    Args:
        num_classes : số lớp đầu ra (mặc định 10)

    Returns:
        model: nn.Module đã load pretrained weights
    """
    model = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model
