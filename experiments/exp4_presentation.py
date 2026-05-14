"""
experiments/exp4_presentation.py
─────────────────────────────────────────────────────────────
Thí nghiệm 4: Xuất ảnh so sánh adversarial cho presentation

Xuất 2 grid ảnh:
  Grid 1 — Epsilon comparison:
    Hàng = mẫu ảnh  |  Cột = Original | ε=0.05 | ε=0.10 | ε=0.20 | ε=0.30
    → Thấy rõ nhiễu tăng dần và model bắt đầu sai từ epsilon nào

  Grid 2 — Steps comparison:
    Hàng = mẫu ảnh  |  Cột = Original | T=1(FGSM) | T=5 | T=20 | T=40
    → Thấy rõ tấn công mạnh dần theo số bước, dù epsilon cố định

Mỗi ô adversarial hiển thị: nhãn dự đoán (xanh=đúng / đỏ=sai) + confidence %
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml

from models              import SimpleCNN, get_resnet18_imagenette, get_mobilenetv2_imagenette
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size
from utils.visualization import plot_epsilon_grid, plot_steps_grid

MNIST_CLASSES      = [str(i) for i in range(10)]
CIFAR10_CLASSES    = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
]
IMAGENETTE_CLASSES = [
    "tench", "English springer", "cassette player", "chain saw", "church",
    "French horn", "garbage truck", "gas pump", "golf ball", "parachute",
]


def run(
    config_path : str = "../configs/config.yaml",
    dataset     : str = None,
    model       : str = None,
    n_samples   : int = 5,
):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset:
        cfg["dataset"]["name"] = dataset
    if model:
        cfg["model"]["name"] = model

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    ds_name    = cfg["dataset"]["name"]
    model_arch = cfg["model"]["name"]

    if ds_name.upper() == "MNIST":
        class_names = MNIST_CLASSES
    elif ds_name.upper() == "CIFAR10":
        class_names = CIFAR10_CLASSES
    else:
        class_names = IMAGENETTE_CLASSES

    print(f"[Exp4] Dataset: {ds_name} | Model: {model_arch} | Device: {device}")

    # ── Load model ────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)

    if ds_name.upper() == "IMAGENETTE":
        if model_arch == "MobileNetV2":
            net      = get_mobilenetv2_imagenette(num_classes=10)
            ckpt_tag = "imagenette_mobilenetv2"
        else:
            net      = get_resnet18_imagenette(num_classes=10)
            ckpt_tag = "imagenette_resnet18"
    else:
        net      = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)
        ckpt_tag = ds_name.lower()

    ckpt_path = os.path.join(cfg["train"]["save_dir"], f"{ckpt_tag}_best.pth")
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Không tìm thấy checkpoint: {ckpt_path}")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    net.load_state_dict(ckpt["model_state"])
    net = net.to(device)
    net.eval()
    print(f"  Loaded: {ckpt_path}")

    # ── Load test data ────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = n_samples * 6,   # lấy nhiều để lọc đủ mẫu đúng
    )

    images, labels = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    with torch.no_grad():
        preds = net(images).argmax(1)
        mask  = preds == labels

    correct_images = images[mask]
    correct_labels = labels[mask].tolist()

    n_avail = len(correct_labels)
    n_show  = min(n_samples, n_avail)
    print(f"  Lọc được {n_avail} mẫu đúng, hiển thị {n_show}")

    if n_show == 0:
        print("  [WARNING] Không có mẫu đúng.")
        return

    # ── Tham số tấn công ─────────────────────────────────────
    epsilon_list = cfg["experiment"]["epsilon_list"]          # [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
    steps_list   = [1] + cfg["experiment"]["steps_list"]      # [1, 5, 10, 20, 40]
    fixed_eps    = cfg["experiment"].get("steps_epsilon", 0.1)
    fixed_T      = cfg["attack"]["num_steps"]

    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # ── Grid 1: so sánh theo Epsilon ─────────────────────────
    print(f"\n  Đang tạo Grid 1 — Epsilon comparison (T={fixed_T}) ...")
    plot_epsilon_grid(
        model        = net,
        images       = correct_images,
        labels       = correct_labels,
        epsilon_list = epsilon_list,
        num_steps    = fixed_T,
        dataset_name = ds_name,
        class_names  = class_names,
        n_samples    = n_show,
        save_path    = os.path.join(fig_dir, f"exp4_grid_epsilon_{ckpt_tag}.png"),
    )

    # ── Grid 2: so sánh theo số bước T ───────────────────────
    print(f"  Đang tạo Grid 2 — Steps comparison (ε={fixed_eps}) ...")
    plot_steps_grid(
        model        = net,
        images       = correct_images,
        labels       = correct_labels,
        steps_list   = steps_list,
        epsilon      = fixed_eps,
        dataset_name = ds_name,
        class_names  = class_names,
        n_samples    = n_show,
        save_path    = os.path.join(fig_dir, f"exp4_grid_steps_{ckpt_tag}.png"),
    )

    print(f"\n[Exp4 — {ds_name}] Hoàn tất!")
    print(f"  → results/figures/exp4_grid_epsilon_{ckpt_tag}.png")
    print(f"  → results/figures/exp4_grid_steps_{ckpt_tag}.png")


if __name__ == "__main__":
    run()
