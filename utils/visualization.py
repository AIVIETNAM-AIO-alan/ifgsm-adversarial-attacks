"""
utils/visualization.py
─────────────────────────────────────────────────────────────
Các hàm vẽ đồ thị cho project I-FGSM.
"""

import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from typing import List, Dict, Optional

SAVE_DIR = "./results/figures"
os.makedirs(SAVE_DIR, exist_ok=True)


# ─────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Tensor [C,H,W] hoặc [H,W] → numpy uint8 [H,W,C] hoặc [H,W]."""
    img = t.detach().cpu().clamp(0, 1).numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)   # CHW → HWC
    return img

def _label_name(idx: int, class_names: Optional[List[str]] = None) -> str:
    if class_names:
        return class_names[idx]
    return str(idx)


# ─────────────────────────────────────────────────────────────
# 1. So sánh ảnh: Original | Perturbation | Adversarial
# ─────────────────────────────────────────────────────────────

def plot_adversarial_examples(
    original     : torch.Tensor,
    adversarial  : torch.Tensor,
    orig_labels  : List[int],
    adv_labels   : List[int],
    epsilon      : float,
    num_steps    : int,
    class_names  : Optional[List[str]] = None,
    dataset_name : str                 = "",
    n_cols       : int                 = 5,
    save_path    : Optional[str]       = None,
) -> None:
    """
    Vẽ lưới ảnh: mỗi cột 1 mẫu, mỗi hàng 1 loại (gốc / nhiễu / đối kháng).

    Args:
        original    : ảnh gốc [N, C, H, W]
        adversarial : ảnh đối kháng [N, C, H, W]
        orig_labels : nhãn dự đoán trên ảnh gốc
        adv_labels  : nhãn dự đoán trên ảnh đối kháng
        epsilon     : ε đã dùng
        num_steps   : số bước I-FGSM
        class_names : tên lớp (tùy chọn)
        n_cols      : số mẫu hiển thị
        save_path   : đường dẫn lưu file (None → không lưu)
    """
    n     = min(n_cols, original.size(0))
    perturb = adversarial - original     # nhiễu thực tế
    is_gray = (original.shape[1] == 1)

    fig, axes = plt.subplots(3, n, figsize=(2.5 * n, 7))
    ds_title = f" — {dataset_name}" if dataset_name else ""
    fig.suptitle(
        f"I-FGSM Attack{ds_title}  |  ε={epsilon}  |  steps={num_steps}"
        f"\n(Chỉ tấn công mẫu dự đoán đúng)",
        fontsize=12, fontweight="bold", y=1.02,
    )

    row_titles = ["Ảnh gốc", "Nhiễu (×10)", "Ảnh đối kháng"]
    for row_ax, title in zip(axes[:, 0], row_titles):
        row_ax.set_ylabel(title, fontsize=11, fontweight="bold")

    for col in range(n):
        orig_img  = _to_numpy(original[col])
        adv_img   = _to_numpy(adversarial[col])
        noise_img = _to_numpy((perturb[col] * 10 + 0.5).clamp(0, 1))

        cmap = "gray" if is_gray else None

        # Hàng 1: ảnh gốc
        axes[0, col].imshow(orig_img.squeeze(), cmap=cmap)
        axes[0, col].set_title(
            f"GT: {_label_name(orig_labels[col], class_names)}",
            fontsize=9, color="green"
        )

        # Hàng 2: nhiễu
        axes[1, col].imshow(noise_img.squeeze(), cmap=cmap)
        axes[1, col].set_title(f"L∞={perturb[col].abs().max():.3f}", fontsize=9)

        # Hàng 3: ảnh đối kháng
        axes[2, col].imshow(adv_img.squeeze(), cmap=cmap)
        color = "red" if adv_labels[col] != orig_labels[col] else "green"
        axes[2, col].set_title(
            f"Pred: {_label_name(adv_labels[col], class_names)}",
            fontsize=9, color=color
        )

    for ax in axes.flat:
        ax.axis("off")

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, f"examples_eps{epsilon}.png"))


# ─────────────────────────────────────────────────────────────
# 2. Accuracy vs Epsilon
# ─────────────────────────────────────────────────────────────

def plot_accuracy_vs_epsilon(
    results      : List[Dict],
    dataset_name : str           = "",
    save_path    : Optional[str] = None,
) -> None:
    """
    Vẽ đường Clean / FGSM / I-FGSM accuracy theo epsilon.
    Clean accuracy là đường ngang cố định (baseline).
    FGSM và I-FGSM là accuracy sau tấn công trên mẫu đúng.

    Args:
        results      : output của AdversarialEvaluator.evaluate_epsilon_range()
        dataset_name : tên dataset để hiển thị trên tiêu đề
    """
    epsilons   = [r["epsilon"]   for r in results]
    clean_accs = [r["clean_acc"] for r in results]
    fgsm_accs  = [r["fgsm_acc"]  for r in results]
    ifgsm_accs = [r["ifgsm_acc"] for r in results]

    # clean_acc không đổi theo epsilon — lấy giá trị đầu tiên
    clean_baseline = clean_accs[0] if clean_accs else 0.0
    n_correct  = results[0].get("n_correct", "?") if results else "?"
    total_test = results[0].get("total_test", "?") if results else "?"

    fig, ax = plt.subplots(figsize=(8, 5))

    # Clean: đường ngang (baseline)
    ax.axhline(
        clean_baseline, color="steelblue", linestyle="--", linewidth=2,
        label=f"Clean ({clean_baseline:.1f}%)",
    )
    ax.plot(epsilons, fgsm_accs,  "s-", color="orange",
            label="FGSM (1 bước)", linewidth=2, markersize=7)
    ax.plot(epsilons, ifgsm_accs, "^-", color="red",
            label="I-FGSM", linewidth=2.5, markersize=8)

    # Annotate drop tại epsilon lớn nhất
    if results:
        last = results[-1]
        ax.annotate(
            f"↓{last['ifgsm_drop']:.1f}%",
            xy=(last["epsilon"], last["ifgsm_acc"]),
            xytext=(0, -18), textcoords="offset points",
            ha="center", fontsize=9, color="red",
        )

    ds_title = f" — {dataset_name}" if dataset_name else ""
    ax.set_xlabel("Epsilon (ε)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"Accuracy vs Epsilon{ds_title}",
        fontsize=14, fontweight="bold",
    )
    ax.set_subtitle = None  # matplotlib không có subtitle native
    fig.text(
        0.5, 0.91,
        f"Tấn công trên {n_correct}/{total_test} mẫu dự đoán đúng",
        ha="center", fontsize=9, color="gray",
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 105)
    ax.set_xticks(epsilons)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, "acc_vs_epsilon.png"))


# ─────────────────────────────────────────────────────────────
# 3. Accuracy vs Số bước lặp
# ─────────────────────────────────────────────────────────────

def plot_accuracy_vs_steps(
    results      : List[Dict],
    epsilon      : float,
    dataset_name : str           = "",
    save_path    : Optional[str] = None,
) -> None:
    """
    Vẽ I-FGSM accuracy khi thay đổi num_steps.
    Thêm đường ngang clean_acc làm baseline tham chiếu.
    """
    steps = [r["num_steps"] for r in results]
    accs  = [r["adv_acc"]   for r in results]

    clean_baseline = results[0].get("clean_acc") if results else None
    n_correct  = results[0].get("n_correct",  "?") if results else "?"
    total_test = results[0].get("total_test", "?") if results else "?"

    fig, ax = plt.subplots(figsize=(7, 4))

    # Đường ngang clean accuracy
    if clean_baseline is not None:
        ax.axhline(
            clean_baseline, color="steelblue", linestyle="--", linewidth=1.8,
            label=f"Clean ({clean_baseline:.1f}%)",
        )

    ax.plot(steps, accs, "^-", color="red", linewidth=2.5, markersize=8,
            label="I-FGSM (adv acc)")

    for x, y in zip(steps, accs):
        ax.annotate(f"{y:.1f}%", (x, y), textcoords="offset points",
                    xytext=(0, 8), ha="center", fontsize=9)

    ds_title = f" — {dataset_name}" if dataset_name else ""
    ax.set_xlabel("Số bước lặp (T)", fontsize=12)
    ax.set_ylabel("Accuracy (%)", fontsize=12)
    ax.set_title(
        f"I-FGSM: Accuracy vs Số bước lặp  (ε={epsilon}){ds_title}",
        fontsize=12, fontweight="bold",
    )
    fig.text(
        0.5, 0.91,
        f"Tấn công trên {n_correct}/{total_test} mẫu dự đoán đúng",
        ha="center", fontsize=9, color="gray",
    )
    ax.set_xticks(steps)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, f"acc_vs_steps_eps{epsilon}.png"))


# ─────────────────────────────────────────────────────────────
# 4. Training history
# ─────────────────────────────────────────────────────────────

def plot_training_history(
    history   : Dict[str, List[float]],
    save_path : Optional[str] = None,
) -> None:
    """
    Vẽ train/val loss và accuracy qua các epoch.
    """
    epochs = range(1, len(history["train_loss"]) + 1)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    ax1.plot(epochs, history["train_loss"], "b-o", markersize=4, label="Train")
    ax1.plot(epochs, history["val_loss"],   "r-s", markersize=4, label="Val")
    ax1.set_xlabel("Epoch"); ax1.set_ylabel("Loss")
    ax1.set_title("Loss theo Epoch"); ax1.legend(); ax1.grid(alpha=0.3)

    # Accuracy
    ax2.plot(epochs, history["train_acc"], "b-o", markersize=4, label="Train")
    ax2.plot(epochs, history["val_acc"],   "r-s", markersize=4, label="Val")
    ax2.set_xlabel("Epoch"); ax2.set_ylabel("Accuracy (%)")
    ax2.set_title("Accuracy theo Epoch"); ax2.legend(); ax2.grid(alpha=0.3)

    plt.suptitle("Training History", fontsize=13, fontweight="bold")
    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, "training_history.png"))


# ─────────────────────────────────────────────────────────────
# 5. Loss evolution qua các bước I-FGSM
# ─────────────────────────────────────────────────────────────

def plot_loss_evolution(
    loss_history : List[float],
    epsilon      : float,
    save_path    : Optional[str] = None,
) -> None:
    """Vẽ loss tăng dần qua các bước lặp I-FGSM."""
    steps = list(range(1, len(loss_history) + 1))

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(steps, loss_history, "r-", linewidth=2)
    ax.fill_between(steps, loss_history, alpha=0.15, color="red")
    ax.set_xlabel("Bước lặp (t)"); ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title(f"Loss tăng dần qua các bước I-FGSM  (ε={epsilon})",
                 fontsize=12, fontweight="bold")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, "loss_evolution.png"))


# ─────────────────────────────────────────────────────────────
# Helper lưu file
# ─────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
