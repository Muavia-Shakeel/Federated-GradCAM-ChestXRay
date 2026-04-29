import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix
)


def compute_classification_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    threshold: float = 0.5,
) -> dict:
    probs = torch.sigmoid(logits).numpy()
    targets = labels.numpy()
    preds = (probs >= threshold).astype(int)

    # Macro metrics across 14 classes
    metrics = {
        "accuracy": accuracy_score(targets.flatten(), preds.flatten()),
        "f1_macro": f1_score(targets, preds, average="macro", zero_division=0),
        "f1_micro": f1_score(targets, preds, average="micro", zero_division=0),
        "precision_macro": precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro": recall_score(targets, preds, average="macro", zero_division=0),
    }

    # AUC-ROC per class, then macro average
    try:
        auc_scores = []
        for c in range(targets.shape[1]):
            if targets[:, c].sum() > 0:
                auc_scores.append(roc_auc_score(targets[:, c], probs[:, c]))
        metrics["auc_roc_macro"] = float(np.mean(auc_scores)) if auc_scores else 0.0
    except Exception:
        metrics["auc_roc_macro"] = 0.0

    return metrics


def faithfulness_score(
    model,
    images: torch.Tensor,
    saliency_map: np.ndarray,
    labels: torch.Tensor,
    device: torch.device,
    top_k_pct: float = 0.1,
) -> float:
    """
    Drop in prediction confidence when top_k_pct most salient pixels are masked.
    Higher drop = more faithful explanation.
    """
    model.eval()
    images = images.to(device)
    labels = labels.to(device)

    with torch.no_grad():
        orig_logits = torch.sigmoid(model(images))

    # Build mask: zero out top-k% pixels
    flat_map = saliency_map.flatten()
    threshold = np.percentile(flat_map, (1 - top_k_pct) * 100)
    mask = torch.tensor(saliency_map >= threshold, dtype=torch.float32).to(device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    masked_images = images * (1 - mask)

    with torch.no_grad():
        masked_logits = torch.sigmoid(model(masked_images))

    # Mean confidence drop across batch and classes
    drop = (orig_logits - masked_logits).mean().item()
    return drop
