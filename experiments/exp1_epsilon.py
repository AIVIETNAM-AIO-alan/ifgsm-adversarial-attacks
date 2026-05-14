"""
experiments/exp1_epsilon.py
─────────────────────────────────────────────────────────────
Thí nghiệm 1: Khảo sát ảnh hưởng của Epsilon lên Accuracy

Flow:
  1. Load model checkpoint
  2. Dự đoán test set → lọc mẫu đúng (clean correct)
  3. Tấn công FGSM + I-FGSM trên mẫu đúng với từng epsilon
  4. So sánh accuracy giảm bao nhiêu

Kết quả kỳ vọng:
  - Accuracy giảm khi ε tăng
  - I-FGSM luôn mạnh hơn FGSM
  - Với ε đủ lớn, accuracy về ~0% (trên mẫu đã đúng)
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml
import json

from models            import SimpleCNN
from utils.data_loader import get_dataloaders, get_in_channels, get_input_size, get_clip_values
from utils.evaluator   import AdversarialEvaluator
from utils.visualization import plot_accuracy_vs_epsilon


def run(config_path: str = "../configs/config.yaml", dataset: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset:
        cfg["dataset"]["name"] = dataset

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    ds_name = cfg["dataset"]["name"]
    print(f"[Exp1] Dataset: {ds_name} | Device: {device}")

    # ── Load model ────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    model      = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)

    ckpt_path = os.path.join(
        cfg["train"]["save_dir"],
        f"{ds_name.lower()}_best.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Không tìm thấy checkpoint: {ckpt_path}")
        print(f"  Chạy train.py --dataset {ds_name} trước!")
        return

    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model_state"])
    model = model.to(device)
    print(f"  Loaded checkpoint: {ckpt_path}")

    # ── Load test data ────────────────────────────────────────
    _, _, test_loader = get_dataloaders(
        ds_name,
        root       = cfg["dataset"]["root"],
        batch_size = cfg["dataset"]["batch_size"],
    )

    # ── Đánh giá (flow 2 pha) ─────────────────────────────────
    clip_min, clip_max = get_clip_values(ds_name)
    evaluator = AdversarialEvaluator(model, device=device, clip_min=clip_min, clip_max=clip_max)

    eps_list = cfg["experiment"]["epsilon_list"]
    results  = evaluator.evaluate_epsilon_range(
        loader       = test_loader,
        epsilon_list = eps_list,
        num_steps    = cfg["attack"]["num_steps"],
        max_batches  = 20,
    )

    # ── Lưu kết quả ──────────────────────────────────────────
    log_dir = os.path.join(ROOT, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"exp1_epsilon_{ds_name.lower()}.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved log: {log_path}")

    # ── Vẽ biểu đồ ───────────────────────────────────────────
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_accuracy_vs_epsilon(
        results,
        dataset_name = ds_name,
        save_path    = os.path.join(fig_dir, f"exp1_acc_vs_epsilon_{ds_name.lower()}.png"),
    )

    print(f"\n[Exp1 — {ds_name}] Hoàn tất!")


if __name__ == "__main__":
    run()
