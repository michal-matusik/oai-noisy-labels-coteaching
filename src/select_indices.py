from typing import List

import torch

# Sampling rates per class [class 0, class 1], found empirically (see SOLUTION.md,
# "Chosen Approach") so that a selected batch is close to class-balanced (~50/50)
# despite the raw training set being skewed ~63%/37% toward class 0.
SAMPLING_RATES = [0.5, 0.93]


def default_select_indices(targets: torch.Tensor, losses: List[torch.Tensor]) -> List[torch.Tensor]:
    """Baseline: use every sample for both models (no denoising, no rebalancing)."""
    device = targets.device
    return [torch.arange(targets.shape[0], device=device) for _ in range(2)]


def your_select_indices(targets: torch.Tensor, losses: List[torch.Tensor]) -> List[torch.Tensor]:
    """Co-teaching-style cross sample selection with class-specific sampling rates.

    For each class, each model trains on the `SAMPLING_RATES[class]` fraction of that
    class's samples with the LOWEST loss **according to the other model** (cross-teaching).
    Selecting low-loss samples filters out likely-mislabeled examples (noisy labels tend
    to produce high loss early in training); using per-class rates simultaneously corrects
    the training set's class imbalance, since it lets us take almost all of the minority
    class while filtering more aggressively out of the majority class.

    Reference: Han et al., "Co-teaching: Robust Training of Deep Neural Networks with
    Extremely Noisy Labels" (arXiv:1804.06872).
    """
    device = targets.device
    selected_indices: List[List[int]] = [[], []]

    for class_idx in torch.unique(targets):
        class_mask = targets == class_idx
        class_indices = class_mask.nonzero(as_tuple=True)[0]
        rate = SAMPLING_RATES[int(class_idx)]
        take_n = int(len(class_indices) * rate)
        if take_n == 0:
            continue

        for i in range(2):
            # Cross-teaching: use the *other* model's losses to pick this model's data.
            other_losses = losses[1 - i].clone().to(device)
            other_losses[targets != class_idx] = float("inf")
            available = int((other_losses != float("inf")).sum().item())
            k = min(take_n, available)
            _, best_idx = torch.topk(-other_losses, k=k)
            selected_indices[i].extend(best_idx.cpu().tolist())

    return [torch.tensor(sel, dtype=torch.long, device=device) for sel in selected_indices]
