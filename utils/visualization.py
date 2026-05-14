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

# Normalization params — dùng để denorm ảnh về [0,1] khi hiển thị
_NORM_PARAMS = {
    "CIFAR10":    {"mean": (0.4914, 0.4822, 0.4465), "std": (0.2470, 0.2435, 0.2616)},
    "IMAGENETTE": {"mean": (0.485,  0.456,  0.406),  "std": (0.229,  0.224,  0.225)},
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _denormalize(tensor: torch.Tensor, dataset_name: str) -> torch.Tensor:
    """Đảo normalization về [0,1] để hiển thị. MNIST không cần denorm."""
    key = dataset_name.upper()
    if key not in _NORM_PARAMS:
        return tensor.clamp(0, 1)
    p    = _NORM_PARAMS[key]
    mean = torch.tensor(p["mean"], dtype=tensor.dtype).view(-1, 1, 1)
    std  = torch.tensor(p["std"],  dtype=tensor.dtype).view(-1, 1, 1)
    return (tensor * std + mean).clamp(0, 1)

def _to_numpy(t: torch.Tensor) -> np.ndarray:
    """Tensor [C,H,W] hoặc [H,W] → numpy [H,W,C] hoặc [H,W]."""
    img = t.detach().cpu().clamp(0, 1).numpy()
    if img.ndim == 3:
        img = img.transpose(1, 2, 0)
    return img

def _label_name(idx: int, class_names: Optional[List[str]] = None) -> str:
    if class_names:
        return class_names[idx]
    return str(idx)

def _show_img(ax, tensor: torch.Tensor, dataset_name: str) -> None:
    """Hiển thị 1 ảnh (có denorm), tự nhận grayscale hay RGB."""
    img = _to_numpy(_denormalize(tensor.cpu(), dataset_name))
    cmap = "gray" if img.ndim == 2 or img.shape[2] == 1 else None
    ax.imshow(img.squeeze(), cmap=cmap)
    ax.axis("off")


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
    ax.set_ylim(0, 105)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, f"acc_vs_steps_eps{epsilon}.png"))


# ─────────────────────────────────────────────────────────────
# 4. Prediction probability bar chart (trước và sau tấn công)
# ─────────────────────────────────────────────────────────────

def plot_prediction_probs(
    model        : "torch.nn.Module",
    original     : "torch.Tensor",
    adversarial  : "torch.Tensor",
    true_labels  : List[int],
    class_names  : Optional[List[str]] = None,
    dataset_name : str                 = "",
    n_cols       : int                 = 5,
    save_path    : Optional[str]       = None,
) -> None:
    """
    Với mỗi mẫu, vẽ bar chart xác suất softmax trước (xanh) và sau (đỏ) tấn công.
    Cột đúng (true label) được đánh dấu bằng viền đậm.

    Args:
        model       : model đã eval
        original    : ảnh gốc  [N, C, H, W]
        adversarial : ảnh đối kháng [N, C, H, W]
        true_labels : nhãn thực [N]
        class_names : tên lớp (tuỳ chọn)
        dataset_name: tên dataset (cho tiêu đề)
        n_cols      : số mẫu hiển thị
        save_path   : đường dẫn lưu file
    """
    import torch.nn.functional as F

    model.eval()
    n     = min(n_cols, original.size(0))
    device = next(model.parameters()).device

    with torch.no_grad():
        orig_logits = model(original[:n].to(device))
        adv_logits  = model(adversarial[:n].to(device))

    orig_probs = F.softmax(orig_logits, dim=1).cpu().numpy()
    adv_probs  = F.softmax(adv_logits,  dim=1).cpu().numpy()

    num_classes = orig_probs.shape[1]
    x = np.arange(num_classes)
    labels = class_names if class_names else [str(i) for i in range(num_classes)]

    fig, axes = plt.subplots(2, n, figsize=(2.8 * n, 6), sharey=False)
    if n == 1:
        axes = axes.reshape(2, 1)

    ds_title = f" — {dataset_name}" if dataset_name else ""
    fig.suptitle(
        f"Xác suất dự đoán trước & sau I-FGSM{ds_title}",
        fontsize=13, fontweight="bold",
    )

    for col in range(n):
        true_cls = true_labels[col]
        orig_pred = int(np.argmax(orig_probs[col]))
        adv_pred  = int(np.argmax(adv_probs[col]))

        for row, (probs, pred, row_label, color) in enumerate([
            (orig_probs[col], orig_pred, "Gốc",        "steelblue"),
            (adv_probs[col],  adv_pred,  "Đối kháng",  "tomato"),
        ]):
            ax = axes[row, col]
            bars = ax.bar(x, probs, color=color, alpha=0.75, width=0.7)

            # Đánh dấu true label bằng viền đen
            bars[true_cls].set_edgecolor("black")
            bars[true_cls].set_linewidth(2.5)

            # Highlight predicted bar
            bars[pred].set_alpha(1.0)

            ax.set_xticks(x)
            ax.set_xticklabels(labels, fontsize=7, rotation=45, ha="right")
            ax.set_ylim(0, 1.05)
            ax.set_yticks([0, 0.5, 1.0])
            ax.tick_params(axis="y", labelsize=7)

            pred_color = "green" if pred == true_cls else "red"
            ax.set_title(
                f"{row_label}: {labels[pred]} ({probs[pred]*100:.1f}%)",
                fontsize=8, color=pred_color, fontweight="bold",
            )
            if col == 0:
                ax.set_ylabel("P(class)", fontsize=8)

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, "prediction_probs.png"))


# ─────────────────────────────────────────────────────────────
# 5. Training history
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
# 6. So sánh adversarial ở nhiều mức Epsilon (presentation grid)
# ─────────────────────────────────────────────────────────────

def plot_epsilon_grid(
    model        : "torch.nn.Module",
    images       : "torch.Tensor",
    labels       : List[int],
    epsilon_list : List[float],
    num_steps    : int,
    dataset_name : str                 = "",
    class_names  : Optional[List[str]] = None,
    n_samples    : int                 = 5,
    save_path    : Optional[str]       = None,
) -> None:
    """
    Grid presentation: hàng = mẫu ảnh, cột = Original + mỗi epsilon.

    Mỗi ô adversarial hiển thị:
      - Ảnh bị tấn công
      - Nhãn dự đoán (xanh=đúng, đỏ=sai) + confidence %
    """
    import torch.nn.functional as F
    from attacks.ifgsm import IFGSMAttack
    from utils.data_loader import get_clip_values

    model.eval()
    device    = next(model.parameters()).device
    n         = min(n_samples, images.size(0))
    imgs      = images[:n].to(device)
    lbls      = torch.tensor(labels[:n], device=device)
    clip_min, clip_max = get_clip_values(dataset_name)

    n_cols = len(epsilon_list) + 1          # Original + K epsilons
    fig, axes = plt.subplots(n, n_cols, figsize=(2.6 * n_cols, 2.8 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    # Header cột
    axes[0, 0].set_title("Original", fontsize=10, fontweight="bold", pad=6)
    for j, eps in enumerate(epsilon_list):
        axes[0, j + 1].set_title(f"ε = {eps}", fontsize=10, fontweight="bold", pad=6)

    # Lấy dự đoán clean 1 lần
    with torch.no_grad():
        clean_preds = model(imgs).argmax(1).cpu().tolist()

    for i in range(n):
        # Cột 0: ảnh gốc
        _show_img(axes[i, 0], imgs[i], dataset_name)
        axes[i, 0].set_ylabel(
            f"GT: {_label_name(labels[i], class_names)}", fontsize=8,
            rotation=0, labelpad=60, va="center",
        )

        # Cột 1..K: adversarial tại từng epsilon
        for j, eps in enumerate(epsilon_list):
            attacker = IFGSMAttack(model, epsilon=eps, num_steps=num_steps,
                                   clip_min=clip_min, clip_max=clip_max)
            adv = attacker(imgs[i:i+1], lbls[i:i+1])

            with torch.no_grad():
                logits = model(adv)
                probs  = F.softmax(logits, dim=1)
                pred   = int(logits.argmax(1).item())
                conf   = float(probs[0, pred].item()) * 100

            _show_img(axes[i, j + 1], adv[0], dataset_name)

            correct = (pred == labels[i])
            color   = "green" if correct else "red"
            mark    = "✓" if correct else "✗"
            axes[i, j + 1].set_title(
                f"{mark} {_label_name(pred, class_names)}\n{conf:.1f}%",
                fontsize=8, color=color,
            )

    ds_title = f" — {dataset_name}" if dataset_name else ""
    fig.suptitle(
        f"Ảnh đối kháng I-FGSM theo ε{ds_title}  (T={num_steps} bước)\n"
        f"Xanh = dự đoán đúng  |  Đỏ = dự đoán sai",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, f"grid_epsilon_{dataset_name.lower()}.png"))


# ─────────────────────────────────────────────────────────────
# 7. So sánh adversarial ở nhiều mức Steps T (presentation grid)
# ─────────────────────────────────────────────────────────────

def plot_steps_grid(
    model        : "torch.nn.Module",
    images       : "torch.Tensor",
    labels       : List[int],
    steps_list   : List[int],
    epsilon      : float,
    dataset_name : str                 = "",
    class_names  : Optional[List[str]] = None,
    n_samples    : int                 = 5,
    save_path    : Optional[str]       = None,
) -> None:
    """
    Grid presentation: hàng = mẫu ảnh, cột = Original + mỗi giá trị T.

    T=1 tương đương FGSM (1 bước), các T lớn hơn là I-FGSM.
    """
    import torch.nn.functional as F
    from attacks.ifgsm import IFGSMAttack
    from utils.data_loader import get_clip_values

    model.eval()
    device    = next(model.parameters()).device
    n         = min(n_samples, images.size(0))
    imgs      = images[:n].to(device)
    lbls      = torch.tensor(labels[:n], device=device)
    clip_min, clip_max = get_clip_values(dataset_name)

    n_cols = len(steps_list) + 1
    fig, axes = plt.subplots(n, n_cols, figsize=(2.6 * n_cols, 2.8 * n))

    if n == 1:
        axes = axes[np.newaxis, :]

    # Header cột
    axes[0, 0].set_title("Original", fontsize=10, fontweight="bold", pad=6)
    for j, T in enumerate(steps_list):
        label = f"T={T}\n(FGSM)" if T == 1 else f"T={T}"
        axes[0, j + 1].set_title(label, fontsize=10, fontweight="bold", pad=6)

    with torch.no_grad():
        clean_preds = model(imgs).argmax(1).cpu().tolist()

    for i in range(n):
        _show_img(axes[i, 0], imgs[i], dataset_name)
        axes[i, 0].set_ylabel(
            f"GT: {_label_name(labels[i], class_names)}", fontsize=8,
            rotation=0, labelpad=60, va="center",
        )

        for j, T in enumerate(steps_list):
            attacker = IFGSMAttack(model, epsilon=epsilon, num_steps=T,
                                   clip_min=clip_min, clip_max=clip_max)
            adv = attacker(imgs[i:i+1], lbls[i:i+1])

            with torch.no_grad():
                logits = model(adv)
                probs  = F.softmax(logits, dim=1)
                pred   = int(logits.argmax(1).item())
                conf   = float(probs[0, pred].item()) * 100

            _show_img(axes[i, j + 1], adv[0], dataset_name)

            correct = (pred == labels[i])
            color   = "green" if correct else "red"
            mark    = "✓" if correct else "✗"
            axes[i, j + 1].set_title(
                f"{mark} {_label_name(pred, class_names)}\n{conf:.1f}%",
                fontsize=8, color=color,
            )

    ds_title = f" — {dataset_name}" if dataset_name else ""
    fig.suptitle(
        f"Ảnh đối kháng I-FGSM theo số bước T{ds_title}  (ε={epsilon})\n"
        f"Xanh = dự đoán đúng  |  Đỏ = dự đoán sai",
        fontsize=12, fontweight="bold", y=1.02,
    )

    plt.tight_layout()
    _save_or_show(fig, save_path or os.path.join(SAVE_DIR, f"grid_steps_{dataset_name.lower()}.png"))


# ─────────────────────────────────────────────────────────────
# Helper lưu file
# ─────────────────────────────────────────────────────────────

def _save_or_show(fig: plt.Figure, save_path: str) -> None:
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  ✓ Saved: {save_path}")
