import torch
from sklearn.metrics import balanced_accuracy_score


def predict_and_evaluate(model, val_loader, device, verbose: bool = False) -> float:
    model.eval()
    all_preds, all_targets = [], []
    with torch.no_grad():
        for inputs, targets in val_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())
    bac = balanced_accuracy_score(all_targets, all_preds)
    if verbose:
        print(f"Balanced Accuracy: {bac}")
    return bac


def performance(bac_1: float, bac_2: float) -> int:
    """Competition scoring: 0 pts at BAC_mean<=0.5, 100 pts at BAC_mean>=0.8, linear between."""
    bac_mean = (bac_1 + bac_2) / 2
    if bac_mean <= 0.5:
        points = 0
    elif 0.5 < bac_mean < 0.8:
        points = int(round((bac_mean - 0.5) / (0.8 - 0.5) * 100))
    else:
        points = 100
    print(f"Mean balanced accuracy = {round(bac_mean, 5)} -> {points}/100 points.")
    return points
