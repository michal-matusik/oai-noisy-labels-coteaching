import os
from typing import Callable, List

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import balanced_accuracy_score
from torch.optim import AdamW

from .dataset import load_data
from .evaluate import performance, predict_and_evaluate
from .model import SmallMobileNet
from .select_indices import your_select_indices


def seed_everything(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_loop(
    model1, model2, optimizer1, optimizer2, criterion,
    train_loader, val_loader, num_epochs: int, device,
    select_indices_fn: Callable, verbose: bool = True,
):
    for epoch in range(1, num_epochs + 1):
        model1.train(), model2.train()
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.squeeze().long().to(device)
            outputs = [m(inputs) for m in (model1, model2)]
            losses = [criterion(out, targets) for out in outputs]
            selected_indices = select_indices_fn(targets, losses)
            for i, (model, optim) in enumerate([(model1, optimizer1), (model2, optimizer2)]):
                sel_idx = selected_indices[i]
                if len(sel_idx) == 0:
                    continue
                optim.zero_grad()
                loss = criterion(model(inputs[sel_idx]), targets[sel_idx]).mean()
                loss.backward()
                optim.step()

        if verbose:
            bac_1 = predict_and_evaluate(model1, val_loader, device)
            bac_2 = predict_and_evaluate(model2, val_loader, device)
            print(f"Epoch {epoch}/{num_epochs} - val BAC model1={bac_1:.4f}, model2={bac_2:.4f}")


def run(
    train_path: str, val_path: str, seed: int = 123, lr: float = 1e-2,
    num_epochs: int = 6, batch_size: int = 128, weight_decay: float = 1e-3,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    device = torch.device(device)
    train_loader, val_loader = load_data(train_path, val_path, batch_size)

    seed_everything(seed)
    model1 = SmallMobileNet().to(device)
    model2 = SmallMobileNet().to(device)
    optimizer1 = AdamW(model1.parameters(), lr=lr, weight_decay=weight_decay)
    optimizer2 = AdamW(model2.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss(reduction="none")

    seed_everything(seed)
    train_loop(model1, model2, optimizer1, optimizer2, criterion, train_loader, val_loader,
               num_epochs, device, your_select_indices)

    bac_1 = predict_and_evaluate(model1, val_loader, device, verbose=True)
    bac_2 = predict_and_evaluate(model2, val_loader, device, verbose=True)
    performance(bac_1, bac_2)
    return model1, model2


if __name__ == "__main__":
    run(train_path="train", val_path="val")
