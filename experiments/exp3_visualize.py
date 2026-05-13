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

from models              import SimpleCNN
from attacks.ifgsm       import IFGSMAttack
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size
from utils.visualization import plot_adversarial_examples, plot_loss_evolution

MNIST_CLASSES   = [str(i) for i in range(10)]
CIFAR10_CLASSES = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]


def run(config_path: str = "../configs/config.yaml", dataset: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset:
        cfg["dataset"]["name"] = dataset

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    ds_name     = cfg["dataset"]["name"]
    class_names = MNIST_CLASSES if ds_name.upper() == "MNIST" else CIFAR10_CLASSES
    print(f"[Exp3] Dataset: {ds_name} | Device: {device}")

    # ── Load model ────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    model      = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)

    ckpt_path = os.path.join(cfg["train"]["save_dir"], f"{ds_name.lower()}_best.pth")
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

    attacker   = IFGSMAttack(model, epsilon=epsilon, num_steps=num_steps)
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

    print(f"\n[Exp3 — {ds_name}] Hoàn tất!")


if __name__ == "__main__":
    run()
