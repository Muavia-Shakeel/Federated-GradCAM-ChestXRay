"""
Per-client GradCAM generation + dataset-size-weighted global aggregation.
No patient data leaves the client — only heatmaps (float tensors) are transmitted.
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
) -> np.ndarray:
    """
    Weighted average of per-client saliency maps by dataset size.
    client_maps: list of (H, W) arrays
    client_sizes: number of training samples per client
    Returns global explanation map (H, W), values in [0, 1].
    """
    total = sum(client_sizes)
    weights = [s / total for s in client_sizes]
    global_map = sum(w * m for w, m in zip(weights, client_maps))
    # Normalize to [0, 1]
    min_val, max_val = global_map.min(), global_map.max()
    if max_val - min_val > 1e-8:
        global_map = (global_map - min_val) / (max_val - min_val)
    return global_map
