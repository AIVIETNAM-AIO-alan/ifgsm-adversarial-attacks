"""
models/mobilenet.py
─────────────────────────────────────────────────────────────
MobileNetV2 pretrained trên ImageNet, fine-tune cho ImageNette.
"""

import torch.nn as nn
from torchvision import models
from torchvision.models import MobileNet_V2_Weights


def get_mobilenetv2_imagenette(num_classes: int = 10) -> nn.Module:
    """
    MobileNetV2 pretrained ImageNet, thay classifier head → num_classes.

    Dùng cho ImageNette (224×224, ImageNet normalization).
    Params: ~3.4M   ImageNet Top-1: ~71.9%
    """
    model = models.mobilenet_v2(weights=MobileNet_V2_Weights.IMAGENET1K_V1)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_mobilenetv2_cifar10(num_classes: int = 10) -> nn.Module:
    """
    MobileNetV2 cho CIFAR-10 (32×32, không pretrained).

    Điều chỉnh: stride conv đầu 2→1 để giữ spatial resolution cho ảnh nhỏ.
    Không dùng pretrained ImageNet vì ảnh 32×32 khác xa 224×224.
    Params: ~3.4M
    """
    model = models.mobilenet_v2(weights=None)
    # stride 2→1: 32×32 giữ nguyên thay vì bị thu nhỏ xuống 16×16 ngay bước đầu
    model.features[0][0] = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model
