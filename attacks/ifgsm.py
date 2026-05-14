"""
attacks/ifgsm.py
─────────────────────────────────────────────────────────────
I-FGSM — Iterative Fast Gradient Sign Method
  (Kurakin, Goodfellow & Bengio, 2016 — arXiv:1607.02533)

Còn được gọi là BIM (Basic Iterative Method).

Công thức:
    x₀     = x
    xₜ₊₁  = Clip_{x,ε}[ xₜ + α · sign(∇ₓ J(θ, xₜ, y)) ]

  trong đó:
    α    = bước nhỏ mỗi iteration (thường = ε / num_steps)
    Clip : giữ nhiễu trong [-ε, +ε] so với ảnh gốc
           và giữ pixel trong [clip_min, clip_max]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def _clip(tensor: torch.Tensor, clip_min, clip_max) -> torch.Tensor:
    """Clip — hỗ trợ cả scalar lẫn tensor per-channel (ImageNette)."""
    if isinstance(clip_min, torch.Tensor):
        lo = clip_min.view(1, -1, 1, 1).to(tensor.device)
        hi = clip_max.view(1, -1, 1, 1).to(tensor.device)
        return torch.max(torch.min(tensor, hi), lo)
    return torch.clamp(tensor, clip_min, clip_max)


# ─────────────────────────────────────────────────────────────
# Class-based API (khuyến nghị dùng trong experiment)
# ─────────────────────────────────────────────────────────────

class IFGSMAttack:
    """
    I-FGSM Attack (class interface).

    Ví dụ sử dụng:
        attacker = IFGSMAttack(model, epsilon=0.3, num_steps=40)
        adv_images = attacker(images, labels)
        stats = attacker.last_stats      # dict chứa thông tin sau attack

    Args:
        model      : nn.Module — mô hình phân lớp
        epsilon    : float     — biên độ nhiễu L∞ tối đa (mặc định 0.3)
        alpha      : float|None — bước mỗi iteration;
                                  None → tự tính = epsilon / num_steps
        num_steps  : int       — số bước lặp (mặc định 40)
        targeted   : bool      — targeted attack (mặc định False)
        clip_min   : float     — giá trị pixel min (mặc định 0.0)
        clip_max   : float     — giá trị pixel max (mặc định 1.0)
        random_start: bool     — khởi tạo nhiễu ngẫu nhiên (PGD-style)
    """

    def __init__(
        self,
        model        : nn.Module,
        epsilon      : float          = 0.3,
        alpha        : Optional[float] = None,
        num_steps    : int            = 40,
        targeted     : bool           = False,
        clip_min     : float          = 0.0,
        clip_max     : float          = 1.0,
        random_start : bool           = False,
    ):
        self.model        = model
        self.epsilon      = epsilon
        self.alpha        = alpha if alpha is not None else epsilon / num_steps
        self.num_steps    = num_steps
        self.targeted     = targeted
        self.clip_min     = clip_min
        self.clip_max     = clip_max
        self.random_start = random_start
        self.last_stats   = {}

    # ── Main API ──────────────────────────────────────────────
    def __call__(
        self,
        images : torch.Tensor,
        labels : torch.Tensor,
    ) -> torch.Tensor:
        return self.attack(images, labels)

    def attack(
        self,
        images : torch.Tensor,
        labels : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Thực hiện I-FGSM attack.

        Args:
            images : ảnh gốc [B, C, H, W]
            labels : nhãn thực (untargeted) hoặc nhãn mục tiêu (targeted)

        Returns:
            adv_images  : ảnh đối kháng [B, C, H, W]
        """
        self.model.eval()
        images = images.clone().detach()
        device = images.device

        # ── Khởi tạo ảnh đối kháng ────────────────────────────
        if self.random_start:
            # PGD-style: nhiễu ngẫu nhiên trong [-ε, +ε]
            delta = torch.empty_like(images).uniform_(-self.epsilon, self.epsilon)
            adv_images = _clip(images + delta, self.clip_min, self.clip_max)
        else:
            adv_images = images.clone()

        # ── Lặp I-FGSM ────────────────────────────────────────
        loss_history = []

        for step in range(self.num_steps):
            adv_images = adv_images.detach().requires_grad_(True)

            outputs = self.model(adv_images)
            loss    = F.cross_entropy(outputs, labels)

            self.model.zero_grad()
            loss.backward()

            grad_sign = adv_images.grad.data.sign()

            # Targeted: đi ngược chiều gradient
            direction = -1 if self.targeted else 1
            adv_images = adv_images.detach() + direction * self.alpha * grad_sign

            # ── Clip 1: giữ nhiễu trong ε-ball quanh ảnh gốc ──
            perturbation = torch.clamp(
                adv_images - images,
                -self.epsilon, self.epsilon
            )
            # ── Clip 2: giữ pixel trong [clip_min, clip_max] ───
            adv_images = _clip(images + perturbation, self.clip_min, self.clip_max)

            loss_history.append(loss.item())

        # ── Lưu thống kê ──────────────────────────────────────
        final_perturbation = adv_images - images
        self.last_stats = {
            "epsilon"          : self.epsilon,
            "alpha"            : self.alpha,
            "num_steps"        : self.num_steps,
            "loss_history"     : loss_history,
            "final_loss"       : loss_history[-1],
            "perturbation_l2"  : final_perturbation.norm(p=2, dim=(1,2,3)).mean().item(),
            "perturbation_linf": final_perturbation.abs().max().item(),
        }

        return adv_images.detach()

    def get_perturbation(
        self,
        images : torch.Tensor,
        labels : torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Trả về cả ảnh đối kháng lẫn nhiễu.

        Returns:
            (adv_images, perturbation)
        """
        adv_images  = self.attack(images, labels)
        perturbation = adv_images - images
        return adv_images, perturbation

    def __repr__(self) -> str:
        return (
            f"IFGSMAttack("
            f"ε={self.epsilon}, α={self.alpha:.4f}, "
            f"steps={self.num_steps}, targeted={self.targeted})"
        )


# ─────────────────────────────────────────────────────────────
# Functional API (dùng nhanh)
# ─────────────────────────────────────────────────────────────

def ifgsm_attack(
    model        : nn.Module,
    images       : torch.Tensor,
    labels       : torch.Tensor,
    epsilon      : float          = 0.3,
    alpha        : Optional[float] = None,
    num_steps    : int            = 40,
    targeted     : bool           = False,
    clip_min     : float          = 0.0,
    clip_max     : float          = 1.0,
    random_start : bool           = False,
) -> torch.Tensor:
    """
    Functional wrapper của IFGSMAttack.

    Args:
        model      : mô hình phân lớp
        images     : ảnh gốc [B, C, H, W]
        labels     : nhãn [B]
        epsilon    : biên độ nhiễu L∞ tối đa
        alpha      : bước mỗi iteration (None → epsilon/num_steps)
        num_steps  : số bước lặp
        targeted   : targeted attack
        clip_min/max: khoảng giá trị pixel hợp lệ
        random_start: khởi tạo nhiễu ngẫu nhiên

    Returns:
        adv_images: ảnh đối kháng [B, C, H, W]
    """
    attacker = IFGSMAttack(
        model        = model,
        epsilon      = epsilon,
        alpha        = alpha,
        num_steps    = num_steps,
        targeted     = targeted,
        clip_min     = clip_min,
        clip_max     = clip_max,
        random_start = random_start,
    )
    return attacker(images, labels)


# ─────────────────────────────────────────────────────────────
# Quick test
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, "..")
    from models import SimpleCNN

    device = torch.device("cpu")
    model  = SimpleCNN(in_channels=1, num_classes=10).to(device)
    model.eval()

    images = torch.rand(4, 1, 28, 28).to(device)
    labels = torch.randint(0, 10, (4,)).to(device)

    attacker = IFGSMAttack(model, epsilon=0.3, num_steps=10)
    adv      = attacker(images, labels)

    print(attacker)
    print(f"Original  shape : {images.shape}")
    print(f"Adversarial shape: {adv.shape}")
    print(f"Max perturbation : {(adv - images).abs().max():.4f}")
    print(f"Stats: {attacker.last_stats}")
