"""
experiments/exp5_transfer.py
─────────────────────────────────────────────────────────────
Thí nghiệm 5: Cross-Architecture Transfer Attack — CIFAR-10

Câu hỏi nghiên cứu:
  Ảnh đối kháng sinh ra từ mô hình A (source) có thể đánh lừa
  mô hình B hoàn toàn khác kiến trúc (target) không?

Flow:
  1. Load tất cả model có checkpoint sẵn trên CIFAR-10
     (SimpleCNN | ResNet18 | MobileNetV2 — bỏ qua nếu chưa train)
  2. Với mỗi source model:
     a. Phase 1: lọc mẫu test source phân loại đúng
     b. Phase 2: sinh ảnh đối kháng FGSM + I-FGSM với tham số cố định
     c. Phase 3: đánh giá trên TẤT CẢ model đã load
        - source == target → White-box (WB) baseline
        - source != target → Transfer ASR (black-box)
  3. Xây dựng ma trận N×N cho từng epsilon
  4. Visualize: heatmap + line chart ASR vs Epsilon

Output:
  results/logs/exp5_transfer_cifar10.json
  results/figures/exp5_transfer_heatmap_eps{ε}.png
  results/figures/exp5_transfer_asr_vs_epsilon.png
"""

import sys, os
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

import json
import numpy as np
import torch
import yaml
from tqdm import tqdm

from attacks.fgsm        import fgsm_attack
from attacks.ifgsm       import IFGSMAttack
from models              import SimpleCNN, get_resnet18, get_mobilenetv2_cifar10
from utils.data_loader   import get_dataloaders, get_clip_values
from utils.visualization import plot_transfer_heatmap, plot_transfer_lines


# ── Registry: 3 model CIFAR-10 ───────────────────────────────
MODEL_REGISTRY = {
    "SimpleCNN": {
        "build":     lambda: SimpleCNN(in_channels=3, num_classes=10, input_size=32),
        "ckpt_name": "cifar10_best.pth",
    },
    "ResNet18": {
        "build":     lambda: get_resnet18(in_channels=3, num_classes=10),
        "ckpt_name": "cifar10_resnet18_best.pth",
    },
    "MobileNetV2": {
        "build":     lambda: get_mobilenetv2_cifar10(num_classes=10),
        "ckpt_name": "cifar10_mobilenetv2_best.pth",
    },
}


# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _load_model(name: str, ckpt_dir: str, device: torch.device):
    """Load model từ checkpoint. Trả về None nếu chưa có checkpoint."""
    cfg  = MODEL_REGISTRY[name]
    path = os.path.join(ckpt_dir, cfg["ckpt_name"])
    if not os.path.exists(path):
        print(f"  [SKIP] {name}: chưa có checkpoint ({cfg['ckpt_name']})")
        return None
    model = cfg["build"]().to(device)
    ckpt  = torch.load(path, map_location=device, weights_only=True)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"  [OK]   {name}: {path}")
    return model


def _collect_correct(model, loader, max_batches, device):
    """Phase 1: thu thập mẫu source model phân loại đúng."""
    imgs_acc, lbls_acc = [], []
    total = 0
    with torch.no_grad():
        for i, (imgs, lbls) in enumerate(
            tqdm(loader, desc="    Phase 1: lọc mẫu đúng", leave=False)
        ):
            if max_batches and i >= max_batches:
                break
            imgs, lbls = imgs.to(device), lbls.to(device)
            mask = model(imgs).argmax(1) == lbls
            if mask.any():
                imgs_acc.append(imgs[mask])
                lbls_acc.append(lbls[mask])
            total += lbls.size(0)

    images = torch.cat(imgs_acc) if imgs_acc else torch.empty(0, device=device)
    labels = torch.cat(lbls_acc) if lbls_acc else torch.empty(0, dtype=torch.long, device=device)
    return images, labels, total


def _count_correct_on(model, adv_images, labels, batch_sz: int = 128) -> int:
    """Đếm số mẫu đối kháng target model vẫn phân loại đúng."""
    model.eval()
    correct = 0
    with torch.no_grad():
        for i in range(0, len(labels), batch_sz):
            imgs = adv_images[i : i + batch_sz]
            lbls = labels[i : i + batch_sz]
            correct += (model(imgs).argmax(1) == lbls).sum().item()
    return correct


# ─────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────

def run(config_path: str = "configs/config.yaml"):
    with open(config_path) as f:
        cfg = yaml.safe_load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{'='*62}")
    print(f"  [Exp5] Cross-Architecture Transfer Attack — CIFAR-10")
    print(f"  Device: {device}")
    print(f"{'='*62}")

    ckpt_dir    = cfg["train"]["save_dir"]
    clip_min, clip_max = get_clip_values("CIFAR10")
    eps_list    = cfg["experiment"]["epsilon_list"]
    num_steps   = cfg["attack"]["num_steps"]
    max_batches = 20   # giữ nhất quán với exp1–exp3

    # ── Bước 1: Load models ───────────────────────────────────
    print("\n[Bước 1] Load models CIFAR-10...")
    models_loaded = {}
    for name in MODEL_REGISTRY:
        m = _load_model(name, ckpt_dir, device)
        if m is not None:
            models_loaded[name] = m

    model_names = list(models_loaded.keys())

    if len(model_names) < 2:
        print(
            f"\n  [WARNING] Chỉ load được {len(model_names)} model — "
            "cần ít nhất 2 để đo transfer.\n"
            "  Train thêm:\n"
            "    python train.py --dataset CIFAR10 --model ResNet18\n"
            "    python train.py --dataset CIFAR10 --model MobileNetV2"
        )
        if len(model_names) == 0:
            return

    print(f"\n  Ma trận {len(model_names)}×{len(model_names)}: {model_names}")

    # ── Bước 2: Load test data ────────────────────────────────
    _, _, test_loader = get_dataloaders(
        "CIFAR10",
        root       = cfg["dataset"]["root"],
        batch_size = cfg["dataset"]["batch_size"],
    )

    # ── Bước 3: Experiment ────────────────────────────────────
    # results[src][tgt]["fgsm" | "ifgsm"] = [asr_tại_eps0, asr_tại_eps1, ...]
    results = {
        src: {tgt: {"fgsm": [], "ifgsm": []} for tgt in model_names}
        for src in model_names
    }
    clean_accs = {}

    for src_name in model_names:
        src_model = models_loaded[src_name]
        print(f"\n{'─'*62}")
        print(f"  Source: {src_name}")

        # Phase 1: lọc mẫu đúng từ source
        images, labels, total = _collect_correct(
            src_model, test_loader, max_batches, device
        )
        n_correct = len(labels)
        clean_acc = 100.0 * n_correct / total
        clean_accs[src_name] = clean_acc
        print(f"  Clean correct: {n_correct}/{total} ({clean_acc:.2f}%)")
        print(f"  Epsilon list : {eps_list}")
        print(f"{'─'*62}")

        for eps in eps_list:
            attacker = IFGSMAttack(
                src_model,
                epsilon   = eps,
                num_steps = num_steps,
                clip_min  = clip_min,
                clip_max  = clip_max,
            )

            # Phase 2: sinh ảnh đối kháng trên source (1 lần / epsilon)
            fgsm_adv_list, ifgsm_adv_list = [], []
            batch_sz = 64
            for i in tqdm(
                range(0, n_correct, batch_sz),
                desc=f"  ε={eps:.3f} | Sinh ảnh đối kháng",
                leave=False,
            ):
                imgs = images[i : i + batch_sz]
                lbls = labels[i : i + batch_sz]
                fgsm_adv_list.append(
                    fgsm_attack(src_model, imgs, lbls, eps, clip_min, clip_max)
                )
                ifgsm_adv_list.append(attacker(imgs, lbls))

            fgsm_adv  = torch.cat(fgsm_adv_list)
            ifgsm_adv = torch.cat(ifgsm_adv_list)

            # Phase 3: đánh giá trên từng target
            for tgt_name in model_names:
                tgt_model = models_loaded[tgt_name]
                tag = "WB      " if src_name == tgt_name else "Transfer"

                fgsm_corr  = _count_correct_on(tgt_model, fgsm_adv,  labels)
                ifgsm_corr = _count_correct_on(tgt_model, ifgsm_adv, labels)

                fgsm_asr  = 100.0 * (n_correct - fgsm_corr)  / n_correct
                ifgsm_asr = 100.0 * (n_correct - ifgsm_corr) / n_correct

                results[src_name][tgt_name]["fgsm"].append(fgsm_asr)
                results[src_name][tgt_name]["ifgsm"].append(ifgsm_asr)

                print(
                    f"  ε={eps:.3f} | {src_name}→{tgt_name} [{tag}] | "
                    f"FGSM={fgsm_asr:.1f}%  I-FGSM={ifgsm_asr:.1f}%"
                )

    # ── Bước 4: Lưu log ───────────────────────────────────────
    log_dir  = os.path.join(ROOT, "results", "logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "exp5_transfer_cifar10.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(
            {"epsilon_list": eps_list, "clean_accs": clean_accs, "results": results},
            f, indent=2, ensure_ascii=False,
        )
    print(f"\n  Saved log: {log_path}")

    # ── Bước 5: Visualize ─────────────────────────────────────
    fig_dir = os.path.join(ROOT, "results", "figures")
    os.makedirs(fig_dir, exist_ok=True)

    # Heatmap tại epsilon ở giữa danh sách
    mid_idx = len(eps_list) // 2
    eps_mid = eps_list[mid_idx]
    matrix_fgsm = np.array([
        [results[src][tgt]["fgsm"][mid_idx] for tgt in model_names]
        for src in model_names
    ])
    matrix_ifgsm = np.array([
        [results[src][tgt]["ifgsm"][mid_idx] for tgt in model_names]
        for src in model_names
    ])
    plot_transfer_heatmap(
        matrix_fgsm, matrix_ifgsm, model_names, eps_mid,
        save_path=os.path.join(fig_dir, f"exp5_transfer_heatmap_eps{eps_mid}.png"),
    )

    # Line chart ASR vs Epsilon (tất cả pairs)
    plot_transfer_lines(
        results, model_names, eps_list,
        save_path=os.path.join(fig_dir, "exp5_transfer_asr_vs_epsilon.png"),
    )

    print(f"\n[Exp5 — Transfer] Hoàn tất! Models: {model_names}")


if __name__ == "__main__":
    run("configs/config.yaml")
