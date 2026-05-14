"""
main.py
─────────────────────────────────────────────────────────────
Pipeline chính — chạy cho MNIST, CIFAR-10 hoặc ImageNette:

  Với mỗi dataset:
    1. Train model (SimpleCNN / ResNet-18 cho ImageNette)
    2. Exp1: Accuracy vs Epsilon  (FGSM + I-FGSM)
    3. Exp2: Accuracy vs Num Steps (I-FGSM)
    4. Exp3: Visualize adversarial examples

  Flow đánh giá (Exp1–3):
    Phase 1 → dự đoán test set → lọc mẫu đúng
    Phase 2 → tấn công FGSM / I-FGSM chỉ trên mẫu đúng

Cách dùng:
    python main.py                          # chạy MNIST (mặc định)
    python main.py --dataset CIFAR10        # chỉ CIFAR-10
    python main.py --dataset ImageNette     # chỉ ImageNette (ResNet-18)
    python main.py --dataset both           # MNIST + CIFAR-10
    python main.py --skip-train             # bỏ qua train
    python main.py --exp 1 2               # chỉ chạy exp 1 và 2
"""

import argparse
import os
import sys
import time
import yaml


def parse_args():
    parser = argparse.ArgumentParser(
        description="I-FGSM Project — Full Pipeline"
    )
    parser.add_argument("--config",     type=str, default="configs/config.yaml")
    parser.add_argument(
        "--dataset", type=str, default="MNIST",
        choices=["MNIST", "CIFAR10", "ImageNette", "both"],
        help="Dataset để chạy: MNIST | CIFAR10 | ImageNette | both (mặc định: MNIST)",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        choices=["SimpleCNN", "ResNet18", "MobileNetV2"],
        help="Override model: SimpleCNN | ResNet18 | MobileNetV2 (MobileNetV2 chỉ dùng với ImageNette)",
    )
    parser.add_argument("--skip-train", action="store_true",
                        help="Bỏ qua bước train (cần checkpoint có sẵn)")
    parser.add_argument("--exp",        nargs="*", type=int,
                        help="Chỉ chạy experiment cụ thể: --exp 1 3")
    return parser.parse_args()


def print_header(title: str) -> None:
    print(f"\n{'═'*62}")
    print(f"  {title}")
    print(f"{'═'*62}")


def main():
    args = parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    exps_to_run = args.exp if args.exp else [1, 2, 3]

    from experiments.exp1_epsilon   import run as run_exp1
    from experiments.exp2_steps     import run as run_exp2
    from experiments.exp3_visualize import run as run_exp3

    exp_map = {
        1: ("Exp1 — Accuracy vs Epsilon",           run_exp1),
        2: ("Exp2 — Accuracy vs Num Steps",          run_exp2),
        3: ("Exp3 — Visualize Adversarial Examples", run_exp3),
    }

    # ── Tạo thư mục output ────────────────────────────────────
    for d in ["results/figures", "results/logs", "results/checkpoints"]:
        os.makedirs(d, exist_ok=True)

    datasets_to_run = ["MNIST", "CIFAR10"] if args.dataset == "both" else [args.dataset]
    print_header("I-FGSM Adversarial Attack Project")
    print(f"  Dataset  : {args.dataset}")
    print(f"  Epsilon  : {cfg['attack']['epsilon']}")
    print(f"  Steps    : {cfg['attack']['num_steps']}")
    print(f"  Flow     : train → lọc mẫu đúng → FGSM + I-FGSM")

    # ── Vòng lặp qua từng dataset ─────────────────────────────
    for ds_name in datasets_to_run:
        print_header(f"Dataset: {ds_name}")

        # ── 1. Train ──────────────────────────────────────────
        if not args.skip_train:
            print_header(f"[{ds_name}] Bước 1: Huấn luyện Model")
            from train import main as train_main
            sys.argv = ["train.py", "--config", args.config, "--dataset", ds_name]
            if args.model:
                sys.argv += ["--model", args.model]
            t0 = time.time()
            train_main()
            print(f"  Hoàn tất train ({time.time()-t0:.1f}s)")
        else:
            print(f"\n  [Bỏ qua train — dùng checkpoint có sẵn]")

        # ── 2. Experiments ────────────────────────────────────
        for exp_id in exps_to_run:
            if exp_id not in exp_map:
                print(f"  [WARNING] Exp {exp_id} không tồn tại, bỏ qua.")
                continue
            title, run_fn = exp_map[exp_id]
            print_header(f"[{ds_name}] {title}")
            t0 = time.time()
            run_fn(config_path=args.config, dataset=ds_name, model=args.model)
            print(f"  Hoàn tất ({time.time()-t0:.1f}s)")

    # ── Tổng kết ──────────────────────────────────────────────
    print_header("Tổng kết")
    print("  Kết quả đã lưu tại:")
    print("  ├── results/figures/      ← Biểu đồ & ảnh minh họa")
    print("  ├── results/logs/         ← Số liệu JSON")
    print("  └── results/checkpoints/  ← Model checkpoint")
    print()
    figures = sorted(os.listdir("results/figures"))
    if figures:
        print("  Figures:")
        for f in figures:
            print(f"    results/figures/{f}")
    print()


if __name__ == "__main__":
    main()
