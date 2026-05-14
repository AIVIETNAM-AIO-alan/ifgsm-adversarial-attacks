"""
train.py
─────────────────────────────────────────────────────────────
Script huấn luyện model phân lớp ảnh (MNIST / CIFAR-10).

Cách dùng:
    python train.py                          # dùng config mặc định
    python train.py --dataset CIFAR10        # override dataset
    python train.py --epochs 30 --lr 0.0005  # override hyperparams
"""

import argparse
import os
import sys
import torch
import torch.optim as optim
import yaml

from models            import SimpleCNN, get_resnet18, get_resnet18_imagenette, get_mobilenetv2_imagenette, get_mobilenetv2_cifar10
from utils.data_loader import get_dataloaders, get_in_channels, get_input_size
from utils.trainer     import Trainer
from utils.visualization import plot_training_history


def parse_args():
    parser = argparse.ArgumentParser(description="Train image classifier")
    parser.add_argument("--config",  type=str, default="configs/config.yaml")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Override dataset: MNIST | CIFAR10")
    parser.add_argument("--model",   type=str, default=None,
                        help="Override model: SimpleCNN | ResNet18")
    parser.add_argument("--epochs",  type=int, default=None)
    parser.add_argument("--lr",      type=float, default=None)
    parser.add_argument("--batch",   type=int, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    # ── Load config ───────────────────────────────────────────
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # Override từ CLI
    if args.dataset : cfg["dataset"]["name"]    = args.dataset
    if args.model   : cfg["model"]["name"]       = args.model
    if args.epochs  : cfg["train"]["epochs"]     = args.epochs
    if args.lr      : cfg["train"]["lr"]         = args.lr
    if args.batch   : cfg["dataset"]["batch_size"] = args.batch

    # ── Device ────────────────────────────────────────────────
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Device: {device}")

    # ── Seed ──────────────────────────────────────────────────
    torch.manual_seed(cfg["experiment"]["seed"])

    # ── Dataset ───────────────────────────────────────────────
    ds_name = cfg["dataset"]["name"]
    train_loader, val_loader, test_loader = get_dataloaders(
        ds_name,
        root        = cfg["dataset"]["root"],
        batch_size  = cfg["dataset"]["batch_size"],
        val_split   = cfg["dataset"]["val_split"],
        num_workers = cfg["dataset"]["num_workers"],
        seed        = cfg["experiment"]["seed"],
    )

    # ── Model ─────────────────────────────────────────────────
    in_ch      = get_in_channels(ds_name)
    input_size = get_input_size(ds_name)
    model_name = cfg["model"]["name"]

    if ds_name.upper() == "IMAGENETTE":
        if model_name == "MobileNetV2":
            model      = get_mobilenetv2_imagenette(num_classes=10)
            save_tag   = "imagenette_mobilenetv2"
            model_name = "MobileNetV2"
        else:
            model      = get_resnet18_imagenette(num_classes=10)
            save_tag   = "imagenette_resnet18"
            model_name = "ResNet18"
    elif model_name == "SimpleCNN":
        model    = SimpleCNN(in_channels=in_ch, num_classes=10, input_size=input_size)
        save_tag = ds_name.lower()
    elif model_name == "ResNet18":
        model    = get_resnet18(in_channels=in_ch, num_classes=10)
        save_tag = f"{ds_name.lower()}_resnet18"
    elif model_name == "MobileNetV2":
        model    = get_mobilenetv2_cifar10(num_classes=10)
        save_tag = f"{ds_name.lower()}_mobilenetv2"
    else:
        raise ValueError(f"Unknown model: {model_name}")

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Model: {model_name} | Params: {n_params:,} | Checkpoint tag: {save_tag}")

    # ── Optimizer & Scheduler ─────────────────────────────────
    lr = cfg["train"]["lr"]
    if cfg["train"]["optimizer"] == "Adam":
        optimizer = optim.Adam(model.parameters(), lr=lr,
                               weight_decay=cfg["train"]["weight_decay"])
    else:
        optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9,
                              weight_decay=cfg["train"]["weight_decay"])

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size = cfg["train"]["step_size"],
        gamma     = cfg["train"]["gamma"],
    )

    # ── Train ─────────────────────────────────────────────────
    save_dir = cfg["train"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)

    trainer = Trainer(
        model     = model,
        optimizer = optimizer,
        scheduler = scheduler,
        device    = device,
        save_dir  = save_dir,
    )

    history = trainer.fit(
        train_loader = train_loader,
        val_loader   = val_loader,
        epochs       = cfg["train"]["epochs"],
        model_name   = save_tag,
    )

    # ── Evaluate trên test set ────────────────────────────────
    print("\n── Đánh giá trên Test set ──")
    trainer.load_checkpoint(f"{save_tag}_best.pth")
    trainer.evaluate(test_loader)

    # ── Vẽ training history ───────────────────────────────────
    os.makedirs("results/figures", exist_ok=True)
    plot_training_history(
        history,
        save_path=f"results/figures/training_history_{save_tag}.png"
    )

    print(f"\n✓ Checkpoint lưu tại: {save_dir}/{save_tag}_best.pth")


if __name__ == "__main__":
    main()
