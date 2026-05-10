"""
Per-client GradCAM generation + dataset-size-weighted global aggregation.
No patient data leaves the client — only heatmaps (float tensors) are transmitted.

Aggregation strategies (Sir feedback: compare multiple approaches):
  - "weighted"     : dataset-size-weighted average (proposed method)
  - "uniform"      : equal weight for all clients
  - "performance"  : weight by per-client validation AUC
  - "max_pool"     : element-wise maximum across clients
"""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget

from model import get_gradcam_target_layer


def generate_client_gradcam(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    class_idx: int = 0,
    max_batches: int = 10,
) -> np.ndarray:
    """
    Generate averaged GradCAM map for one client across up to max_batches batches.
    Returns array of shape (H, W) averaged over all samples.
    """
    target_layer = get_gradcam_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]

    accumulated = None
    count = 0

    model.eval()
    for batch_idx, (images, _) in enumerate(loader):
        if batch_idx >= max_batches:
            break
        images = images.to(device)
        grayscale_cams = cam(input_tensor=images, targets=targets)  # (B, H, W)
        batch_mean = grayscale_cams.mean(axis=0)  # (H, W)
        if accumulated is None:
            accumulated = batch_mean
        else:
            accumulated += batch_mean
        count += 1

    if accumulated is None or count == 0:
        raise RuntimeError("No batches processed in GradCAM generation.")

    return accumulated / count  # (H, W) — per-client mean saliency map


def aggregate_gradcam_maps(
    client_maps: list[np.ndarray],
    client_sizes: list[int],
    strategy: str = "weighted",
    client_aucs: list[float] | None = None,
) -> np.ndarray:
    """
    Aggregate per-client saliency maps into a single global explanation.

    Args:
        client_maps:  list of (H, W) GradCAM arrays, one per client
        client_sizes: number of training samples per client
        strategy:     aggregation method —
                        "weighted"    dataset-size-weighted average (proposed)
                        "uniform"     equal weights
                        "performance" AUC-weighted average (requires client_aucs)
                        "max_pool"    element-wise maximum
        client_aucs:  per-client AUC scores (required for strategy="performance")

    Returns: normalized global explanation map (H, W) in [0, 1].
    """
    if strategy == "weighted":
        total = sum(client_sizes)
        weights = [s / total for s in client_sizes]
        global_map = sum(w * m for w, m in zip(weights, client_maps))

    elif strategy == "uniform":
        global_map = np.mean(np.stack(client_maps, axis=0), axis=0)

    elif strategy == "performance":
        if client_aucs is None:
            raise ValueError("client_aucs must be provided for strategy='performance'")
        total_auc = sum(client_aucs)
        if total_auc < 1e-8:
            weights = [1.0 / len(client_maps)] * len(client_maps)
        else:
            weights = [a / total_auc for a in client_aucs]
        global_map = sum(w * m for w, m in zip(weights, client_maps))

    elif strategy == "max_pool":
        global_map = np.max(np.stack(client_maps, axis=0), axis=0)

    else:
        raise ValueError(f"Unknown aggregation strategy: '{strategy}'. "
                         f"Choose from: weighted, uniform, performance, max_pool")

    # Normalize to [0, 1]
    min_val, max_val = global_map.min(), global_map.max()
    if max_val - min_val > 1e-8:
        global_map = (global_map - min_val) / (max_val - min_val)
    return global_map


def compare_aggregation_strategies(
    client_maps: list[np.ndarray],
    client_sizes: list[int],
    client_aucs: list[float] | None = None,
) -> dict[str, np.ndarray]:
    """
    Run all four strategies and return a dict of strategy_name → global_map.
    Useful for ablation / comparison figures.
    """
    strategies = ["weighted", "uniform", "max_pool"]
    if client_aucs is not None:
        strategies.append("performance")

    results = {}
    for strat in strategies:
        results[strat] = aggregate_gradcam_maps(
            client_maps, client_sizes,
            strategy=strat,
            client_aucs=client_aucs,
        )
    return results


def generate_oracle_gradcam(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    class_idx: int = 0,
) -> np.ndarray:
    """
    Generate a GradCAM oracle map directly from the global model on a specific
    test batch. Used as the reference ground-truth explanation for evaluating
    how closely each aggregation strategy reproduces what the final global
    model actually attends to.

    Args:
        model:     the global federated model (already trained)
        images:    test batch tensor (B, C, H, W) — CPU or GPU
        device:    target device
        class_idx: class to target (default 0 = Atelectasis)

    Returns: mean saliency map of shape (H, W), normalized to [0, 1].
    """
    target_layer = get_gradcam_target_layer(model)
    cam = GradCAM(model=model, target_layers=[target_layer])
    targets = [ClassifierOutputTarget(class_idx)]

    model.eval()
    images = images.to(device)
    grayscale_cams = cam(input_tensor=images, targets=targets)  # (B, H, W)
    oracle_map = grayscale_cams.mean(axis=0)                    # (H, W)

    # Normalize to [0, 1]
    lo, hi = oracle_map.min(), oracle_map.max()
    if hi - lo > 1e-8:
        oracle_map = (oracle_map - lo) / (hi - lo)

    return oracle_map
