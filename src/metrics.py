import numpy as np
import torch
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    precision_score, recall_score, confusion_matrix, hamming_loss
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
    # hamming_score = fraction of labels correctly predicted (standard multi-label metric)
    # exact_match   = all 14 labels must match (strict)
    # accuracy_elementwise is misleading for multi-label (inflated by true negatives)
    metrics = {
        "hamming_score":        1.0 - float(hamming_loss(targets, preds)),
        "exact_match_accuracy": float(accuracy_score(targets, preds)),
        "accuracy":             accuracy_score(targets.flatten(), preds.flatten()),
        "f1_macro":             f1_score(targets, preds, average="macro",  zero_division=0),
        "f1_micro":             f1_score(targets, preds, average="micro",  zero_division=0),
        "precision_macro":      precision_score(targets, preds, average="macro", zero_division=0),
        "recall_macro":         recall_score(targets, preds, average="macro", zero_division=0),
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


# ImageNet normalization mean — used as mean-fill baseline for masked pixels.
# Filling with dataset mean instead of zeros avoids out-of-distribution inputs
# for ImageNet-pretrained backbones (Fong & Vedaldi, 2017).
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)


def faithfulness_score(
    model,
    images: torch.Tensor,
    saliency_map: np.ndarray,
    labels: torch.Tensor,
    device: torch.device,
    top_k_pct: float = 0.2,
    class_idx: int = 0,
) -> float:
    """
    Drop in prediction confidence (for class_idx) when top_k_pct most salient
    pixels are replaced with dataset mean (mean-fill baseline, Fong & Vedaldi 2017).
    Restricted to positive examples (labels[:, class_idx]==1) to measure disease-
    region attribution, not suppressive background activations.
    Higher (more positive) drop = more faithful explanation.
    """
    model.eval()
    images = images.to(device)
    labels_cpu = labels if labels.device.type == "cpu" else labels.cpu()

    # Restrict to positive examples — avoids GradCAM suppression artifact
    pos_mask = (labels_cpu[:, class_idx] == 1)
    if pos_mask.sum() == 0:
        return 0.0
    images = images[pos_mask]

    with torch.no_grad():
        orig_probs = torch.sigmoid(model(images))[:, class_idx]  # (B_pos,)

    # Build pixel mask for top-k% most salient regions
    flat_map = saliency_map.flatten()
    threshold = np.percentile(flat_map, (1 - top_k_pct) * 100)
    mask = torch.tensor(saliency_map >= threshold, dtype=torch.float32).to(device)
    mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, H, W)

    # Mean-fill baseline: replace masked pixels with ImageNet channel mean
    mean_fill = _IMAGENET_MEAN.to(device).expand_as(images)
    masked_images = images * (1 - mask) + mean_fill * mask

    with torch.no_grad():
        masked_probs = torch.sigmoid(model(masked_images))[:, class_idx]  # (B_pos,)

    drop = (orig_probs - masked_probs).mean().item()
    return drop


def faithfulness_score_multiclass(
    model,
    images: torch.Tensor,
    saliency_maps: dict,
    labels: torch.Tensor,
    device: torch.device,
    class_indices: list = None,
    top_k_pct: float = 0.2,
) -> dict:
    """
    Compute faithfulness across multiple high-support classes and return per-class
    and averaged scores. Reduces noise from single-class evaluation on small batches.

    Args:
        saliency_maps: dict of {class_idx: np.ndarray (H, W)} — one map per class
        class_indices:  list of class indices to evaluate (default: [0, 2, 3])
    """
    if class_indices is None:
        class_indices = [0, 2, 3]  # Atelectasis, Effusion, Infiltration

    scores = {}
    for c in class_indices:
        if c not in saliency_maps:
            continue
        s = faithfulness_score(model, images, saliency_maps[c], labels, device,
                               top_k_pct=top_k_pct, class_idx=c)
        scores[c] = s

    valid = [v for v in scores.values() if v != 0.0]
    scores["mean"] = float(np.mean(valid)) if valid else 0.0
    return scores


# ---------------------------------------------------------------------------
# Per-class metrics (Sir feedback: per-pathology breakdown)
# ---------------------------------------------------------------------------

def compute_per_class_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    pathology_labels: list,
    threshold: float = 0.5,
) -> dict:
    """
    Per-pathology AUC-ROC, F1, Precision, Recall, and support.
    Returns dict keyed by pathology name.
    """
    probs = torch.sigmoid(logits).numpy()
    targets = labels.numpy()
    preds = (probs >= threshold).astype(int)

    per_class = {}
    for i, label in enumerate(pathology_labels):
        col_targets = targets[:, i]
        col_preds = preds[:, i]
        col_probs = probs[:, i]

        try:
            auc = float(roc_auc_score(col_targets, col_probs)) if col_targets.sum() > 0 else 0.0
        except Exception:
            auc = 0.0

        per_class[label] = {
            "auc":       auc,
            "f1":        float(f1_score(col_targets, col_preds, zero_division=0)),
            "precision": float(precision_score(col_targets, col_preds, zero_division=0)),
            "recall":    float(recall_score(col_targets, col_preds, zero_division=0)),
            "support":   int(col_targets.sum()),
        }
    return per_class


# ---------------------------------------------------------------------------
# Quantitative XAI evaluation (Sir feedback: insertion/deletion, AOPC)
# ---------------------------------------------------------------------------

def insertion_deletion_auc(
    model,
    images: torch.Tensor,
    saliency_map: np.ndarray,
    device: torch.device,
    steps: int = 10,
) -> dict:
    """
    Insertion and Deletion AUC (Petsiuk et al., BMVC 2018).

    Deletion: Progressively zero out pixels in descending saliency order.
              Lower AUC = more faithful (confidence drops faster).
    Insertion: Starting from blurred baseline, reveal pixels by saliency rank.
               Higher AUC = more faithful (confidence rises faster).

    Returns dict with 'insertion_auc', 'deletion_auc', and raw score lists.
    """
    model.eval()
    images = images.to(device)                          # (B, C, H, W)
    _, _, H, W = images.shape
    n_pixels = H * W

    # Saliency rank (highest first)
    flat_sal = saliency_map.flatten()
    sorted_idx = np.argsort(-flat_sal)                  # descending

    # Blurred baseline for insertion metric
    blurred = F.avg_pool2d(images, kernel_size=11, stride=1, padding=5)

    step_pixels = max(1, n_pixels // steps)

    del_scores, ins_scores = [], []

    with torch.no_grad():
        for step in range(1, steps + 1):
            n = min(step * step_pixels, n_pixels)
            top_idx = sorted_idx[:n]
            rows = top_idx // W
            cols = top_idx % W

            # Deletion: mask top saliency pixels with zero
            del_img = images.clone()
            del_img[:, :, rows, cols] = 0.0
            del_prob = torch.sigmoid(model(del_img)).mean().item()

            # Insertion: reveal top saliency pixels from blurred image
            ins_img = blurred.clone()
            ins_img[:, :, rows, cols] = images[:, :, rows, cols]
            ins_prob = torch.sigmoid(model(ins_img)).mean().item()

            del_scores.append(del_prob)
            ins_scores.append(ins_prob)

    x = np.linspace(0, 1, steps)
    del_auc = float(np.trapz(del_scores, x))
    ins_auc = float(np.trapz(ins_scores, x))

    return {
        "insertion_auc":    ins_auc,
        "deletion_auc":     del_auc,
        "insertion_scores": ins_scores,
        "deletion_scores":  del_scores,
    }


def aopc_score(
    model,
    images: torch.Tensor,
    saliency_map: np.ndarray,
    device: torch.device,
    k_steps: int = 10,
    class_idx: int = 0,
) -> float:
    """
    Area Over Perturbation Curve (Samek et al., 2017 JSTSP).
    Measures average drop in model output for class_idx when top-k% pixels
    are removed in MoRF order. Evaluated class-conditionally.
    Higher AOPC = more faithful explanation.

    AOPC = (1 / (k+1)) * sum_{i=0}^{k} [ f(x) - f(x_i^MoRF) ]
    """
    model.eval()
    images = images.to(device)
    _, _, H, W = images.shape
    n_pixels = H * W

    flat_sal = saliency_map.flatten()
    sorted_idx = np.argsort(-flat_sal)          # MoRF order

    step_size = max(1, n_pixels // k_steps)

    with torch.no_grad():
        orig_prob = torch.sigmoid(model(images))[:, class_idx].mean().item()

    drops = []   # step 0: no perturbation (Samek et al. definition starts at step 1)
    with torch.no_grad():
        for step in range(1, k_steps + 1):
            n = min(step * step_size, n_pixels)
            top_idx = sorted_idx[:n]
            rows = top_idx // W
            cols = top_idx % W
            perturbed = images.clone()
            perturbed[:, :, rows, cols] = 0.0
            pert_prob = torch.sigmoid(model(perturbed))[:, class_idx].mean().item()
            drops.append(orig_prob - pert_prob)

    aopc = float(np.mean(drops))
    return aopc


def map_similarity_score(
    map1: np.ndarray,
    map2: np.ndarray,
) -> dict:
    """
    Compute structural and statistical similarity between two saliency maps.
    Used to quantitatively compare aggregation strategy maps against an oracle.

    Returns:
        pearson_r  : Pearson correlation coefficient
        spearman_r : Spearman rank correlation (rank-order agreement)
        mse        : Mean squared error between maps
        ssim       : Structural similarity (simplified luminance+structure)
    """
    a = map1.flatten().astype(np.float64)
    b = map2.flatten().astype(np.float64)

    # Pearson / Spearman undefined when either input is constant
    if a.std() < 1e-8 or b.std() < 1e-8:
        pr, sr = 0.0, 0.0
    else:
        pr, _ = pearsonr(a, b)
        sr, _ = spearmanr(a, b)
        if np.isnan(pr): pr = 0.0
        if np.isnan(sr): sr = 0.0
    mse = float(np.mean((a - b) ** 2))

    # Simplified SSIM (luminance + structure, single channel)
    mu_a, mu_b = a.mean(), b.mean()
    sigma_a = a.std()
    sigma_b = b.std()
    sigma_ab = np.mean((a - mu_a) * (b - mu_b))
    C1, C2 = 0.01 ** 2, 0.03 ** 2   # standard SSIM constants
    ssim = ((2 * mu_a * mu_b + C1) * (2 * sigma_ab + C2)) / (
        (mu_a ** 2 + mu_b ** 2 + C1) * (sigma_a ** 2 + sigma_b ** 2 + C2)
    )

    return {
        "pearson_r":  round(float(pr),   4),
        "spearman_r": round(float(sr),   4),
        "mse":        round(float(mse),  6),
        "ssim":       round(float(ssim), 4),
    }
