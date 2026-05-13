# I-FGSM Adversarial Attack — Image Classifier

Implementation of **FGSM** and **I-FGSM (Iterative Fast Gradient Sign Method)** adversarial attacks on MNIST and CIFAR-10 image classifiers, with full training, evaluation, and visualization pipelines.

> Paper: *Adversarial Examples in the Physical World* — Kurakin, Goodfellow & Bengio (2016)
> https://arxiv.org/abs/1607.02533

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Table of Contents

- [What is I-FGSM?](#what-is-i-fgsm)
- [Evaluation Methodology](#evaluation-methodology)
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

## Evaluation Methodology

All experiments follow a strict **2-phase evaluation** to measure attack effectiveness fairly:

```
Phase 1 — Predict & Filter
  Feed entire test set through the model (no attack)
  → Record clean accuracy and n_correct
  → Keep only the samples the model predicted CORRECTLY

Phase 2 — Attack
  Run FGSM and I-FGSM only on the correctly classified samples
  → Measure robust accuracy, ASR, perturbation size, and attack time
  → Report absolute accuracy over the full test set
```

**Why attack only correctly classified samples?**

Attacking misclassified samples is meaningless — they are already wrong. This methodology isolates the true attack strength: *of the examples the model gets right, how many can be fooled?*

**Metrics reported for each experiment:**

| Metric | Description |
|---|---|
| `clean_acc` | Baseline accuracy before any attack (%) |
| `fgsm_acc` / `ifgsm_acc` | Accuracy remaining after attack, over full test set (%) |
| `fgsm_asr` / `ifgsm_asr` | **Attack Success Rate** — % of correctly classified samples that were fooled |
| `fgsm_drop` / `ifgsm_drop` | Absolute accuracy drop in percentage points |
| `perturbation_linf` | Max L∞ perturbation magnitude (should equal ε) |
| `perturbation_l2` | Mean L2 perturbation norm across the batch |
| `fgsm_time_s` / `ifgsm_time_s` | Wall-clock time of the attack in seconds |

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
│   ├── evaluator.py          # 2-phase evaluation: filter correct → attack
│   └── visualization.py      # 5 plot types including probability bar charts
│
├── experiments/
│   ├── exp1_epsilon.py       # Sweep ε → accuracy, ASR, timing
│   ├── exp2_steps.py         # Sweep T → accuracy, ASR, timing  (uses steps_epsilon)
│   └── exp3_visualize.py     # Images + perturbation + prediction probabilities
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
│   └── test_ifgsm.py         # pytest unit tests
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
# Train + run all 3 experiments on MNIST (default)
python main.py

# Choose dataset: MNIST | CIFAR10 | both
python main.py --dataset MNIST
python main.py --dataset CIFAR10
python main.py --dataset both

# Skip training (reuse saved checkpoints)
python main.py --dataset CIFAR10 --skip-train

# Run specific experiments only (e.g., exp 1 and 3)
python main.py --dataset MNIST --skip-train --exp 1 3
```

### Train only

```bash
# Train on MNIST (default)
python train.py

# Train on CIFAR-10
python train.py --dataset CIFAR10

# Train with ResNet-18 instead of SimpleCNN
python train.py --model ResNet18

# Override hyperparameters from CLI
python train.py --dataset CIFAR10 --epochs 30 --lr 0.0005
```

### Run experiments individually

```bash
python experiments/exp1_epsilon.py   # accuracy + ASR + timing vs epsilon
python experiments/exp2_steps.py     # accuracy + ASR + timing vs num steps
python experiments/exp3_visualize.py # images + perturbation + prediction probs
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
    model        = model,
    epsilon      = 0.3,       # L∞ budget
    num_steps    = 40,        # iterations T
    alpha        = None,      # None → auto = epsilon / num_steps
    targeted     = False,     # True for targeted attack
    random_start = False,     # True → PGD-style random initialization
    clip_min     = 0.0,
    clip_max     = 1.0,
)

adv_images = attacker(images, labels)
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
| Optimizer | Adam (lr=0.001, weight\_decay=1e-4) |
| Scheduler | StepLR (step=10, γ=0.1) |
| Epochs | 20 |
| Batch size | 64 |
| **Best val accuracy** | **98.98%** |
| **Test accuracy** | **99.45%** |

### CIFAR-10 training settings

| Setting | Value |
|---|---|
| Dataset split | 45,000 train / 5,000 val / 10,000 test |
| Model | SimpleCNN |
| Optimizer | Adam (lr=0.001, weight\_decay=1e-4) |
| Scheduler | StepLR (step=10, γ=0.1) |
| Epochs | 20 |
| Batch size | 64 |
| Augmentation | RandomCrop(32, padding=4) + RandomHorizontalFlip |

Training history (loss and accuracy curves):

![Training History MNIST](results/figures/training_history_mnist.png)

---

## Experiment Results

> All results use the **2-phase evaluation**: attacks are applied only to the correctly classified subset.
> Accuracy values use the full test set as denominator — clean and adversarial numbers are directly comparable.

---

### Exp 1 — Accuracy & ASR vs Epsilon (ε)

Fixed: `T=40` steps. Evaluated on MNIST test set (10,000 total, 9,945 correctly classified).

| ε | Clean Acc | FGSM Acc | FGSM ASR | I-FGSM Acc | I-FGSM ASR |
|---|---|---|---|---|---|
| 0.05 | 99.45% | 94.53% | 4.9% | 85.23% | 14.3% |
| 0.10 | 99.45% | 70.63% | 28.9% | 16.56% | 83.4% |
| 0.15 | 99.45% | 42.73% | 57.0% | 0.70% | 99.3% |
| 0.20 | 99.45% | 25.16% | 74.7% | 0.00% | 100.0% |
| 0.25 | 99.45% | 15.94% | 84.0% | 0.00% | 100.0% |
| 0.30 | 99.45% | 11.17% | 88.8% | 0.00% | 100.0% |

> **ASR** = % of correctly classified samples successfully fooled by the attack.

**Key takeaways:**
- At `ε=0.10`, I-FGSM achieves **83.4% ASR** vs. FGSM's 28.9% — nearly 3× more effective at the same perturbation budget.
- At `ε=0.20`, I-FGSM reaches **100% ASR**: every correctly classified sample is fooled.
- FGSM tops out at ~88.8% ASR even at `ε=0.30`, showing the fundamental ceiling of single-step attacks.

![Exp1 MNIST](results/figures/exp1_acc_vs_epsilon_mnist.png)

---

### Exp 2 — Accuracy & ASR vs Number of Steps (T)

Fixed: **`ε=0.1`** (`steps_epsilon` in config — chosen so that step count produces a visible difference).
Evaluated on MNIST test set (1,280 samples, 1,273 correctly classified).

| T (steps) | Clean Acc | I-FGSM Acc | ASR | Drop |
|---|---|---|---|---|
| 5  | 99.45% | 26.64% | 73.2% | −72.81% |
| 10 | 99.45% | 21.25% | 78.6% | −78.20% |
| 20 | 99.45% | 18.20% | 81.7% | −81.25% |
| 40 | 99.45% | 16.56% | 83.4% | −82.89% |

**Key takeaways:**
- Attack strength increases with step count, but with **diminishing returns**: the jump from T=5→10 (5.4 pp drop) is larger than T=20→40 (1.6 pp drop).
- At T=40 the attack is approaching convergence — further iterations yield minimal gain.
- The horizontal clean baseline line in the chart makes the accuracy gap visually clear.

![Exp2 MNIST](results/figures/exp2_acc_vs_steps_mnist.png)

---

### Exp 3 — Adversarial Example Visualization

All visualizations are generated from the **correctly classified** subset only.

**1. Side-by-side image comparison** — original | perturbation (×10) | adversarial:

![Exp3 Examples MNIST](results/figures/exp3_examples_mnist.png)

**2. Prediction probability bar chart** — softmax probabilities before (blue) and after (red) I-FGSM attack. The true-label bar is outlined in black; the predicted bar is fully opaque:

![Exp3 Prediction Probs MNIST](results/figures/exp3_pred_probs_mnist.png)

Before attack the model assigns ~100% confidence to the correct class. After attack, that confidence shifts entirely to a **wrong class** — also with ~100% certainty — demonstrating how adversarial examples exploit the model's decision boundary.

**3. Loss evolution** — cross-entropy loss rising across I-FGSM iterations:

![Exp3 Loss Evolution MNIST](results/figures/exp3_loss_evolution_mnist.png)

---

## Configuration

All hyperparameters are centralized in `configs/config.yaml`:

```yaml
dataset:
  name: "MNIST"           # MNIST | CIFAR10
  batch_size: 64
  val_split: 0.1

model:
  name: "SimpleCNN"       # SimpleCNN | ResNet18

train:
  epochs: 20
  lr: 0.001
  weight_decay: 0.0001
  optimizer: "Adam"
  scheduler: "StepLR"
  step_size: 10
  gamma: 0.1

attack:
  method: "ifgsm"
  epsilon: 0.3            # max L∞ perturbation (used by Exp1 and Exp3)
  alpha: 0.01             # step size (null → auto = epsilon / num_steps)
  num_steps: 40           # iterations T
  targeted: false
  clip_min: 0.0
  clip_max: 1.0

experiment:
  epsilon_list: [0.05, 0.1, 0.15, 0.2, 0.25, 0.3]
  steps_list: [5, 10, 20, 40]
  steps_epsilon: 0.1      # epsilon used by Exp2 — smaller so step count matters
  seed: 42
  device: "cuda"          # cuda | cpu | mps
  num_samples: 1000

vis:
  num_examples: 10
  save_dir: "./results/figures"
```

The `--dataset` CLI argument overrides `dataset.name` without modifying this file:

```bash
python main.py --dataset MNIST    # MNIST only (default)
python main.py --dataset CIFAR10  # CIFAR-10 only
python main.py --dataset both     # run both sequentially
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
