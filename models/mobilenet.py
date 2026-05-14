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
