"""
experiments/exp3_visualize.py
─────────────────────────────────────────────────────────────
Thí nghiệm 3: Trực quan hóa adversarial examples

Flow:
  1. Lấy 1 batch từ test set
  2. Lọc các mẫu được model dự đoán ĐÚNG
  3. Tấn công I-FGSM chỉ trên những mẫu đó
  4. Hiển thị: ảnh gốc | nhiễu×10 | ảnh đối kháng

Output:
  - Lưới ảnh với tiêu đề nhãn (đỏ=sai, xanh=đúng)
  - Biểu đồ loss tăng dần qua từng bước I-FGSM
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml

from models              import SimpleCNN, get_resnet18_imagenette, get_mobilenetv2_imagenette
from attacks.ifgsm       import IFGSMAttack
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size, get_clip_values
from utils.visualization import (
    plot_adversarial_examples,
    plot_loss_evolution,
    plot_prediction_probs,
)

MNIST_CLASSES      = [str(i) for i in range(10)]
CIFAR10_CLASSES    = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]


def run(config_path: str = "../configs/config.yaml", dataset: str = None, model: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset:
        cfg["dataset"]["name"] = dataset
    if model:
        cfg["model"]["name"] = model

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    ds_name = cfg["dataset"]["name"]
    if ds_name.upper() == "MNIST":
        class_names = MNIST_CLASSES
    elif ds_name.upper() == "CIFAR10":
        class_names = CIFAR10_CLASSES
    else:
        class_names = IMAGENETTE_CLASSES
    print(f"[Exp3] Dataset: {ds_name} | Device: {device}")

    # ── Load model ────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    model_arch = cfg["model"]["name"]

    if ds_name.upper() == "IMAGENETTE":
        if model_arch == "MobileNetV2":
            model    = get_mobilenetv2_imagenette(num_classes=10)
            ckpt_tag = "imagenette_mobilenetv2"
        else:
            model    = get_resnet18_imagenette(num_classes=10)
            ckpt_tag = "imagenette_resnet18"
    else:
        model    = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)
        ckpt_tag = ds_name.lower()

    ckpt_path = os.path.join(cfg["train"]["save_dir"], f"{ckpt_tag}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Không tìm thấy checkpoint. Chạy train.py --dataset {ds_name} trước!")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    model.eval()
    print(f"  Loaded checkpoint: {ckpt_path}")

    # ── Load 1 batch lớn để lọc đủ mẫu ──────────────────────
    num_show  = cfg["vis"]["num_examples"]
    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = num_show * 4,   # lấy nhiều hơn cần để sau khi lọc vẫn đủ
    )

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    # ── Phase 1: lọc mẫu dự đoán đúng ────────────────────────
    with torch.no_grad():
        preds = model(images).argmax(1)
        mask  = preds == labels

    correct_images = images[mask]
    correct_labels = labels[mask]
    correct_preds  = preds[mask].tolist()

    n_total   = len(labels)
    n_correct = len(correct_labels)
    print(f"  Batch: {n_total} mẫu | Đúng: {n_correct} ({100*n_correct/n_total:.1f}%)")

    if n_correct == 0:
        print("  [WARNING] Không có mẫu nào đúng trong batch này.")
        return

    # ── Phase 2: tấn công I-FGSM trên mẫu đúng ───────────────
    epsilon   = cfg["attack"]["epsilon"]
    num_steps = cfg["attack"]["num_steps"]

    clip_min, clip_max = get_clip_values(ds_name)
    attacker   = IFGSMAttack(model, epsilon=epsilon, num_steps=num_steps,
                             clip_min=clip_min, clip_max=clip_max)
    adv_images = attacker(correct_images, correct_labels)

    with torch.no_grad():
        adv_preds = model(adv_images).argmax(1).tolist()

    n_show     = min(num_show, n_correct)
    n_adv_corr = sum(p == l for p, l in zip(adv_preds, correct_labels.tolist()))
    print(f"  Sau tấn công : {n_adv_corr}/{n_correct} vẫn đúng "
          f"({100*n_adv_corr/n_correct:.1f}% trên mẫu đúng)")

    # ── Vẽ lưới ảnh ───────────────────────────────────────────
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    plot_adversarial_examples(
        original    = correct_images[:n_show].cpu(),
        adversarial = adv_images[:n_show].cpu(),
        orig_labels = correct_preds[:n_show],
        adv_labels  = adv_preds[:n_show],
        epsilon     = epsilon,
        num_steps   = num_steps,
        class_names = class_names,
        dataset_name= ds_name,
        save_path   = os.path.join(fig_dir, f"exp3_examples_{ds_name.lower()}.png"),
    )

    # ── Vẽ loss evolution ──────────────────────────────────────
    plot_loss_evolution(
        loss_history = attacker.last_stats["loss_history"],
        epsilon      = epsilon,
        save_path    = os.path.join(fig_dir, f"exp3_loss_evolution_{ds_name.lower()}.png"),
    )

    # ── Vẽ xác suất dự đoán trước/sau tấn công ────────────────
    plot_prediction_probs(
        model        = model,
        original     = correct_images[:n_show],
        adversarial  = adv_images[:n_show],
        true_labels  = correct_labels[:n_show].tolist(),
        class_names  = class_names,
        dataset_name = ds_name,
        n_cols       = n_show,
        save_path    = os.path.join(fig_dir, f"exp3_pred_probs_{ds_name.lower()}.png"),
    )

    print(f"\n[Exp3 — {ds_name}] Hoàn tất!")


if __name__ == "__main__":
    run()
