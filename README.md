# I-FGSM Adversarial Attack — Image Classifier

Implementation of **I-FGSM (Iterative Fast Gradient Sign Method)** adversarial attacks on MNIST and CIFAR-10 image classifiers, with full training, evaluation, and visualization pipelines.

> Paper: *Adversarial Examples in the Physical World* — Kurakin, Goodfellow & Bengio (2016)
> https://arxiv.org/abs/1607.02533

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [What is I-FGSM?](#what-is-i-fgsm)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Models](#models)
- [Attack API](#attack-api)
- [Training](#training)
- [Experiment Results](#experiment-results)
- [Configuration](#configuration)
- [Testing](#testing)
- [References](#references)

---

## What is I-FGSM?

**FGSM** (Goodfellow et al., 2015) crafts adversarial examples in a single gradient step:

```
x_adv = x + ε · sign(∇ₓ J(θ, x, y))
```

**I-FGSM** (also known as **BIM — Basic Iterative Method**) extends this by taking many small steps, clipping the perturbation after each one to stay within the ε-ball:

```
x₀    = x
xₜ₊₁ = Clip_{x,ε} [ xₜ + α · sign(∇ₓ J(θ, xₜ, y)) ]
```

| Symbol | Meaning |
|---|---|
| `ε` | Maximum L∞ perturbation magnitude |
| `α` | Step size per iteration (default: `ε / T`) |
| `T` | Number of iterations |
| `Clip_{x,ε}` | Keeps perturbation in `[-ε, +ε]` and pixels in `[0, 1]` |

**Why is I-FGSM stronger than FGSM?**

FGSM takes a single large step and often overshoots. I-FGSM refines the adversarial direction iteratively — each step re-computes the gradient from the current (already adversarial) image, finding a much more targeted direction. At the same ε budget, I-FGSM is dramatically more effective.

**PGD variant:** setting `random_start=True` initializes the perturbation with uniform noise in `[-ε, +ε]` before iterating, turning I-FGSM into the **PGD attack** (Madry et al., 2018).

---

## Project Structure

```
ifgsm_project/
│
├── attacks/
│   ├── fgsm.py               # FGSM baseline (1-step, functional API)
│   └── ifgsm.py              # I-FGSM: class-based + functional API
│
├── models/
│   ├── cnn.py                # SimpleCNN (auto-adapts to MNIST & CIFAR-10)
│   └── resnet.py             # ResNet-18 wrapper (torchvision)
│
├── utils/
│   ├── data_loader.py        # MNIST / CIFAR-10 dataloaders
│   ├── trainer.py            # Training loop with checkpoint saving
│   ├── evaluator.py          # Batch evaluation under adversarial attack
│   └── visualization.py      # Plot generation utilities
│
├── experiments/
│   ├── exp1_epsilon.py       # Sweep ε → measure accuracy drop
│   ├── exp2_steps.py         # Sweep T (num steps) → measure accuracy drop
│   └── exp3_visualize.py     # Side-by-side clean vs. adversarial images
│
├── configs/
│   └── config.yaml           # All hyperparameters in one place
│
├── results/
│   ├── checkpoints/          # Saved model weights (.pth)
│   ├── figures/              # Generated plots (.png)
│   └── logs/                 # Experiment metrics (.json)
│
├── tests/
│   └── test_ifgsm.py         # pytest unit tests (15 tests)
│
├── train.py                  # Standalone training script
├── main.py                   # Full pipeline (train + all experiments)
└── requirements.txt
```

---

## Setup

```bash
# Clone the repo
git clone https://github.com/wotttoo/ifgsm-adversarial-attacks.git
cd ifgsm-adversarial-attacks

# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate       # Linux/Mac
# venv\Scripts\activate        # Windows

# Install dependencies
pip install -r requirements.txt
```

**Requirements:** Python 3.10+, PyTorch 2.0+, torchvision, matplotlib, tqdm, pyyaml. See `requirements.txt` for pinned versions.

---

## Usage

### Run the full pipeline

```bash
# Train model + run all 3 experiments
python main.py

# Skip training (reuse saved checkpoint)
python main.py --skip-train

# Run specific experiments only (e.g., exp 1 and 3)
python main.py --skip-train --exp 1 3
```

### Train only

```bash
# Train on MNIST (default)
python train.py

# Train on CIFAR-10
python train.py --dataset CIFAR10

# Train with ResNet-18 instead of SimpleCNN
python train.py --model ResNet18
```

### Run experiments individually

```bash
python experiments/exp1_epsilon.py   # accuracy vs epsilon sweep
python experiments/exp2_steps.py     # accuracy vs iteration count
python experiments/exp3_visualize.py # visual comparison
```

### Run unit tests

```bash
python -m pytest tests/ -v
```

---

## Models

### SimpleCNN

A compact CNN that auto-scales its architecture based on the input dataset.

**MNIST architecture** (`in_channels=1`, `28×28`):

```
Block 1:  Conv2d(1→32, 3×3) → BN → ReLU → Conv2d(32→32, 3×3) → BN → ReLU
          → MaxPool2d(2×2) → Dropout2d(0.25)
          Output: [B, 32, 14, 14]

Classifier: Flatten → Linear(6272→512) → ReLU → Dropout(0.5) → Linear(512→10)
```

**CIFAR-10 architecture** (`in_channels=3`, `32×32`):

```
Block 1:  Conv2d(3→32, 3×3) → BN → ReLU → Conv2d(32→32, 3×3) → BN → ReLU
          → MaxPool2d(2×2) → Dropout2d(0.25)
Block 2:  Conv2d(32→64, 3×3) → BN → ReLU → Conv2d(64→64, 3×3) → BN → ReLU
          → MaxPool2d(2×2) → Dropout2d(0.25)
          Output: [B, 64, 8, 8]

Classifier: Flatten → Linear(4096→512) → ReLU → Dropout(0.5) → Linear(512→10)
```

### ResNet-18

A modified torchvision ResNet-18 adapted for small images:

- **Grayscale input** (`in_channels=1`): the first `Conv2d` is replaced with a `3×3, stride=1` layer and `maxpool` is replaced with `Identity()` to avoid over-downsampling `28×28` images.
- **Output layer**: the final `fc` is replaced with `Linear(512 → num_classes)`.

---

## Attack API

### Class-based (recommended)

```python
from attacks.ifgsm import IFGSMAttack

attacker = IFGSMAttack(
    model       = model,
    epsilon     = 0.3,       # L∞ budget
    num_steps   = 40,        # iterations T
    alpha       = None,      # None → auto = epsilon / num_steps
    targeted    = False,     # True for targeted attack
    random_start= False,     # True → PGD-style random initialization
    clip_min    = 0.0,
    clip_max    = 1.0,
)

adv_images = attacker(images, labels)       # returns adversarial images
adv_images, perturbation = attacker.get_perturbation(images, labels)

# Attack statistics after each call
print(attacker.last_stats)
# {
#   'epsilon': 0.3, 'alpha': 0.0075, 'num_steps': 40,
#   'loss_history': [...],        # cross-entropy loss at each step
#   'final_loss': 4.23,
#   'perturbation_l2': 1.84,      # mean L2 norm of perturbation
#   'perturbation_linf': 0.3,     # max L∞ norm (should equal epsilon)
# }
```

### Functional API (quick use)

```python
from attacks.ifgsm import ifgsm_attack
from attacks.fgsm  import fgsm_attack

adv = ifgsm_attack(model, images, labels, epsilon=0.3, num_steps=40)
adv = fgsm_attack(model, images, labels, epsilon=0.3)
```

### Targeted attack

For a targeted attack, supply the **target class labels** (not ground truth) and set `targeted=True`. The attack minimizes the loss towards the target class instead of maximizing it away from the true class.

```python
target_labels = torch.full_like(labels, fill_value=3)  # fool model into predicting class 3
attacker = IFGSMAttack(model, epsilon=0.2, num_steps=40, targeted=True)
adv = attacker(images, target_labels)
```

---

## Training

The `Trainer` class handles the full training loop:

| Feature | Details |
|---|---|
| Loss function | Cross-entropy |
| Checkpointing | Saves best model by val accuracy to `results/checkpoints/` |
| LR scheduling | `StepLR` (step=10, γ=0.1) |
| History tracking | `train_loss`, `train_acc`, `val_loss`, `val_acc` per epoch |
| Progress bar | `tqdm` per batch |

### MNIST training settings

| Setting | Value |
|---|---|
| Dataset split | 54,000 train / 6,000 val / 10,000 test |
| Model | SimpleCNN |
| Optimizer | Adam (lr=0.001, weight_decay=1e-4) |
| Scheduler | StepLR (step=10, γ=0.1) |
| Epochs | 20 |
| Batch size | 64 |
| **Best val accuracy** | **98.98%** |
| **Test accuracy** | **99.45%** |

Training history (loss and accuracy curves):

![Training History](results/figures/training_history_mnist.png)

---

## Experiment Results

### Exp 1 — Accuracy vs Epsilon (ε)

Fixed: `T=40` steps. Evaluated on MNIST test set (10,000 images).

| ε | Clean Acc | FGSM Acc | I-FGSM Acc | I-FGSM Drop |
|---|---|---|---|---|
| 0.05 | 99.45% | 94.53% | 85.23% | −14.22% |
| 0.10 | 99.45% | 70.63% | 16.56% | −82.89% |
| 0.15 | 99.45% | 42.73% | 0.70% | −98.75% |
| 0.20 | 99.45% | 25.16% | 0.00% | −99.45% |
| 0.25 | 99.45% | 15.94% | 0.00% | −99.45% |
| 0.30 | 99.45% | 11.17% | 0.00% | −99.45% |

**Key takeaways:**
- At `ε=0.10`, I-FGSM drops accuracy from 99.45% → 16.56%, while FGSM only drops it to 70.63% — a 4× larger attack impact at the same perturbation budget.
- At `ε=0.20`, I-FGSM achieves **100% attack success rate** (0% accuracy remaining).
- FGSM retains ~11% accuracy even at `ε=0.30`, showing its saturation limit vs. iterative methods.

![Exp1](results/figures/exp1_acc_vs_epsilon_mnist.png)

---

### Exp 2 — Accuracy vs Number of Steps (T)

Fixed: `ε=0.3`. Evaluated on MNIST test set.

| T (steps) | I-FGSM Acc |
|---|---|
| 5 | 0.00% |
| 10 | 0.00% |
| 20 | 0.00% |
| 40 | 0.00% |

At `ε=0.3`, the attack fully saturates in as few as 5 steps — accuracy drops to 0% regardless of T. To observe the step-count effect, use a smaller budget (`ε=0.05–0.1`) where each additional step provides incremental gain.

![Exp2](results/figures/exp2_acc_vs_steps_mnist.png)

---

### Exp 3 — Adversarial Example Visualization

Side-by-side comparison of:
1. **Original image** — correctly classified
2. **Perturbation** (×10 amplified for visibility) — the `sign(∇)` pattern
3. **Adversarial image** — visually identical but misclassified

![Exp3 Examples](results/figures/exp3_examples_mnist.png)

Loss evolution across iterations, showing the attack converging as the cross-entropy loss rises:

![Exp3 Loss Evolution](results/figures/exp3_loss_evolution_mnist.png)

---

## Configuration

All hyperparameters are centralized in `configs/config.yaml`:

```yaml
dataset:
  name: "MNIST"           # MNIST | CIFAR10
  batch_size: 64

model:
  name: "SimpleCNN"       # SimpleCNN | ResNet18

train:
  epochs: 20
  lr: 0.001
  weight_decay: 0.0001
  optimizer: "Adam"
  scheduler: "StepLR"
  scheduler_step: 10
  scheduler_gamma: 0.1

attack:
  epsilon: 0.3            # max L∞ perturbation
  num_steps: 40           # iterations T
  alpha: null             # null → auto = epsilon / num_steps
  targeted: false
  random_start: false     # true → PGD-style

experiment:
  epsilon_list: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  steps_list: [5, 10, 20, 40]
  num_samples: 1000       # images per experiment run
  device: "cuda"          # cuda | cpu | mps
```

---

## Testing

The test suite covers the full attack pipeline using `pytest`. Tests run on CPU with a randomly-initialized `SimpleCNN` so no checkpoint is required.

```bash
python -m pytest tests/ -v
```

| Test class | What is tested |
|---|---|
| `TestIFGSMAttack` | Output shape, L∞ constraint, pixel range `[0,1]`, no gradient leakage, `last_stats` keys, auto-alpha computation, custom clip range, `__repr__` |
| `TestIfgsmFunction` | Functional API produces same result as class API (same seed), output shape |
| `TestFGSMBaseline` | FGSM output shape, L∞ constraint |
| `TestSimpleCNN` | MNIST and CIFAR-10 output shapes `[B, 10]` |

---

## References

| Paper | Link |
|---|---|
| FGSM — Goodfellow et al. (2015) | https://arxiv.org/abs/1412.6572 |
| **I-FGSM / BIM** — Kurakin et al. (2016) | https://arxiv.org/abs/1607.02533 |
| PGD — Madry et al. (2018) | https://arxiv.org/abs/1706.06083 |
| MI-FGSM — Dong et al. (2018) | https://arxiv.org/abs/1710.06081 |
| DI-FGSM — Xie et al. (2019) | https://arxiv.org/abs/1803.06978 |
