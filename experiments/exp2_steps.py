"""
experiments/exp2_steps.py
─────────────────────────────────────────────────────────────
Thí nghiệm 2: Khảo sát ảnh hưởng của số bước lặp T

Flow:
  1. Load model checkpoint
  2. Dự đoán test set → lọc mẫu đúng (clean correct)
  3. Tấn công I-FGSM trên mẫu đúng với từng num_steps
  4. So sánh accuracy giảm theo số bước

Câu hỏi nghiên cứu:
  - Bao nhiêu bước thì I-FGSM "hội tụ"?
  - Thêm bước có luôn làm tăng attack strength?

Kết quả kỳ vọng:
  - Accuracy giảm nhanh từ T=1→10, chậm dần sau T=20
  - Sau một ngưỡng nào đó, thêm bước không còn hiệu quả nhiều
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import torch
import yaml
import json

from models              import SimpleCNN
from utils.data_loader   import get_dataloaders, get_in_channels, get_input_size
from utils.evaluator     import AdversarialEvaluator
from utils.visualization import plot_accuracy_vs_steps


def run(config_path: str = "../configs/config.yaml", dataset: str = None):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    if dataset:
        cfg["dataset"]["name"] = dataset

    device = torch.device(
        cfg["experiment"]["device"] if torch.cuda.is_available() else "cpu"
    )
    ds_name = cfg["dataset"]["name"]
    print(f"[Exp2] Dataset: {ds_name} | Device: {device}")

    # ── Load model ────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    model      = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)

    ckpt_path = os.path.join(
        cfg["train"]["save_dir"],
        f"{ds_name.lower()}_best.pth"
    )
    if not os.path.exists(ckpt_path):
        print(f"  [WARNING] Không tìm thấy checkpoint. Chạy train.py --dataset {ds_name} trước!")
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
    evaluator  = AdversarialEvaluator(model, device=device)
    epsilon    = cfg["attack"]["epsilon"]
    steps_list = cfg["experiment"]["steps_list"]

    results = evaluator.evaluate_steps(
        loader      = test_loader,
        epsilon     = epsilon,
        steps_list  = steps_list,
        max_batches = 20,
    )

    # ── Lưu kết quả ──────────────────────────────────────────
    log_dir = os.path.join(ROOT, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"exp2_steps_{ds_name.lower()}.json")
    with open(log_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"  Saved log: {log_path}")

    # ── Vẽ biểu đồ ───────────────────────────────────────────
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)
    plot_accuracy_vs_steps(
        results,
        epsilon      = epsilon,
        dataset_name = ds_name,
        save_path    = os.path.join(fig_dir, f"exp2_acc_vs_steps_{ds_name.lower()}.png"),
    )

    print(f"\n[Exp2 — {ds_name}] Hoàn tất!")


if __name__ == "__main__":
    run()
