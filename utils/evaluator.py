"""
utils/evaluator.py
─────────────────────────────────────────────────────────────
Evaluator — đánh giá model trên clean và adversarial examples.

Flow 2 pha:
  Phase 1 — Dự đoán toàn bộ test set, lọc lấy mẫu đúng (clean correct)
  Phase 2 — Tấn công FGSM + I-FGSM chỉ trên những mẫu đó
             → đo accuracy giảm còn bao nhiêu
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from attacks.ifgsm import IFGSMAttack
from attacks.fgsm  import fgsm_attack


class AdversarialEvaluator:
    """
    Đánh giá model dưới tấn công FGSM và I-FGSM.

    Args:
        model    : nn.Module đã được train
        device   : torch.device
        clip_min : giá trị pixel nhỏ nhất (mặc định 0.0)
        clip_max : giá trị pixel lớn nhất (mặc định 1.0)
    """

    def __init__(
        self,
        model    : nn.Module,
        device   : torch.device = torch.device("cpu"),
        clip_min : float        = 0.0,
        clip_max : float        = 1.0,
    ):
        self.model    = model.to(device)
        self.device   = device
        self.clip_min = clip_min
        self.clip_max = clip_max

    # ─────────────────────────────────────────────────────────
    # Phase 1: Thu thập mẫu dự đoán đúng
    # ─────────────────────────────────────────────────────────

    def _collect_correct(
        self,
        loader      : DataLoader,
        max_batches : Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """
        Duyệt loader, chỉ giữ lại các mẫu model dự đoán ĐÚNG.

        Returns:
            correct_images : Tensor [N_correct, C, H, W]
            correct_labels : Tensor [N_correct]
            total_seen     : tổng số mẫu đã duyệt qua
        """
        self.model.eval()
        all_imgs: List[torch.Tensor] = []
        all_lbls: List[torch.Tensor] = []
        total = 0

        with torch.no_grad():
            for i, (images, labels) in enumerate(
                tqdm(loader, desc="  [Phase 1] Dự đoán & lọc mẫu đúng", leave=False)
            ):
                if max_batches and i >= max_batches:
                    break
                images = images.to(self.device)
                labels = labels.to(self.device)

                preds = self.model(images).argmax(1)
                mask  = preds == labels

                if mask.any():
                    all_imgs.append(images[mask])
                    all_lbls.append(labels[mask])
                total += labels.size(0)

        if all_imgs:
            return torch.cat(all_imgs, dim=0), torch.cat(all_lbls, dim=0), total

        return (
            torch.empty(0, device=self.device),
            torch.empty(0, dtype=torch.long, device=self.device),
            total,
        )

    # ─────────────────────────────────────────────────────────
    # Exp 1 — Accuracy vs Epsilon (FGSM + I-FGSM)
    # ─────────────────────────────────────────────────────────

    def evaluate_epsilon_range(
        self,
        loader       : DataLoader,
        epsilon_list : List[float],
        num_steps    : int,
        alpha        : Optional[float] = None,
        max_batches  : Optional[int]   = None,
    ) -> List[Dict]:
        """
        Thu thập mẫu đúng 1 lần, sau đó quét qua từng epsilon.
        Mỗi epsilon đánh giá cả FGSM (1 bước) lẫn I-FGSM (nhiều bước).

        Returns:
            list of dict, mỗi dict là kết quả tại 1 epsilon:
            {
                epsilon, total_test, n_correct, clean_acc,
                fgsm_acc, ifgsm_acc, fgsm_drop, ifgsm_drop
            }
        """
        # ── Phase 1 ───────────────────────────────────────────
        images, labels, total = self._collect_correct(loader, max_batches)
        n_correct = len(labels)

        if total == 0:
            print("  [WARNING] Không có mẫu nào để đánh giá.")
            return []

        clean_acc = 100.0 * n_correct / total

        print(f"\n{'─'*62}")
        print(f"  [Phase 1] Tổng mẫu : {total:,}")
        print(f"            Đúng (clean): {n_correct:,}  ({clean_acc:.2f}%)")
        print(f"  [Phase 2] Tấn công {n_correct:,} mẫu | steps={num_steps}")
        print(f"            epsilon list: {epsilon_list}")
        print(f"{'─'*62}")

        results: List[Dict] = []
        batch_sz = 64

        # ── Phase 2 ───────────────────────────────────────────
        for eps in epsilon_list:
            attacker = IFGSMAttack(
                self.model,
                epsilon  = eps,
                alpha    = alpha,
                num_steps= num_steps,
                clip_min = self.clip_min,
                clip_max = self.clip_max,
            )
            fgsm_surv = ifgsm_surv = 0

            for i in tqdm(
                range(0, n_correct, batch_sz),
                desc=f"  ε={eps:.3f}", leave=False,
            ):
                imgs = images[i : i + batch_sz]
                lbls = labels[i : i + batch_sz]

                # FGSM (1 bước)
                adv_f = fgsm_attack(
                    self.model, imgs, lbls, eps,
                    self.clip_min, self.clip_max,
                )
                with torch.no_grad():
                    fgsm_surv += (self.model(adv_f).argmax(1) == lbls).sum().item()

                # I-FGSM (nhiều bước)
                adv_i = attacker(imgs, lbls)
                with torch.no_grad():
                    ifgsm_surv += (self.model(adv_i).argmax(1) == lbls).sum().item()

            fgsm_acc  = 100.0 * fgsm_surv  / total
            ifgsm_acc = 100.0 * ifgsm_surv / total

            r = {
                "epsilon"    : eps,
                "total_test" : total,
                "n_correct"  : n_correct,
                "clean_acc"  : clean_acc,
                "fgsm_acc"   : fgsm_acc,
                "ifgsm_acc"  : ifgsm_acc,
                "fgsm_drop"  : clean_acc - fgsm_acc,
                "ifgsm_drop" : clean_acc - ifgsm_acc,
            }
            results.append(r)
            print(
                f"  ε={eps:.3f} | Clean={clean_acc:.2f}% → "
                f"FGSM={fgsm_acc:.2f}% (↓{r['fgsm_drop']:.2f}%) | "
                f"I-FGSM={ifgsm_acc:.2f}% (↓{r['ifgsm_drop']:.2f}%)"
            )

        return results

    # ─────────────────────────────────────────────────────────
    # Exp 2 — Accuracy vs Num Steps (I-FGSM)
    # ─────────────────────────────────────────────────────────

    def evaluate_steps(
        self,
        loader      : DataLoader,
        epsilon     : float,
        steps_list  : List[int],
        max_batches : Optional[int] = None,
    ) -> List[Dict]:
        """
        Thu thập mẫu đúng 1 lần, sau đó quét qua từng num_steps.
        Chỉ dùng I-FGSM (khảo sát ảnh hưởng của số bước lặp).

        Returns:
            list of dict:
            {num_steps, total_test, n_correct, clean_acc, adv_acc, acc_drop}
        """
        # ── Phase 1 ───────────────────────────────────────────
        images, labels, total = self._collect_correct(loader, max_batches)
        n_correct = len(labels)

        if total == 0:
            print("  [WARNING] Không có mẫu nào để đánh giá.")
            return []

        clean_acc = 100.0 * n_correct / total

        print(f"\n{'─'*62}")
        print(f"  [Phase 1] Tổng mẫu : {total:,}")
        print(f"            Đúng (clean): {n_correct:,}  ({clean_acc:.2f}%)")
        print(f"  [Phase 2] Tấn công I-FGSM | ε={epsilon}")
        print(f"            steps list: {steps_list}")
        print(f"{'─'*62}")

        results: List[Dict] = []
        batch_sz = 64

        # ── Phase 2 ───────────────────────────────────────────
        for steps in steps_list:
            attacker = IFGSMAttack(
                self.model,
                epsilon  = epsilon,
                num_steps= steps,
                clip_min = self.clip_min,
                clip_max = self.clip_max,
            )
            survived = 0

            for i in tqdm(
                range(0, n_correct, batch_sz),
                desc=f"  steps={steps}", leave=False,
            ):
                imgs = images[i : i + batch_sz]
                lbls = labels[i : i + batch_sz]
                adv  = attacker(imgs, lbls)
                with torch.no_grad():
                    survived += (self.model(adv).argmax(1) == lbls).sum().item()

            adv_acc = 100.0 * survived / total
            r = {
                "num_steps"  : steps,
                "total_test" : total,
                "n_correct"  : n_correct,
                "clean_acc"  : clean_acc,
                "adv_acc"    : adv_acc,
                "acc_drop"   : clean_acc - adv_acc,
            }
            results.append(r)
            print(
                f"  steps={steps:3d} | "
                f"Clean={clean_acc:.2f}% → I-FGSM={adv_acc:.2f}% "
                f"(↓{r['acc_drop']:.2f}%)"
            )

        return results
