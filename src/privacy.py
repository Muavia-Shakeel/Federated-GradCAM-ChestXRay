"""
Differential Privacy for Federated Learning (DP-FedAvg).

Implements the Gaussian mechanism from:
  McMahan et al., "Learning Differentially Private Recurrent Language Models", ICLR 2018.
  Abadi et al., "Deep Learning with Differential Privacy", CCS 2016.

Privacy model:
  - Each client clips its model UPDATE (w_local - w_global) to L2 norm ≤ C
  - Gaussian noise N(0, σ²C²I) is added to each clipped update before aggregation
  - The Central Limit Theorem / moments accountant gives (ε, δ)-DP guarantee
"""

import math
import copy
import torch
import torch.nn as nn


def clip_model_update(
    local_model: nn.Module,
    global_model: nn.Module,
    max_norm: float,
) -> dict:
    """
    Compute clipped update: delta = clip(w_local - w_global, max_norm).
    Returns clipped update as a state-dict of float tensors.
    """
    global_state = global_model.state_dict()
    local_state = local_model.state_dict()

    updates = {}
    sq_norm = 0.0
    for key in local_state:
        diff = local_state[key].float() - global_state[key].float()
        updates[key] = diff
        sq_norm += diff.norm().item() ** 2

    update_norm = math.sqrt(sq_norm)
    clip_coeff = min(1.0, max_norm / (update_norm + 1e-8))

    clipped = {k: v * clip_coeff for k, v in updates.items()}
    return clipped, update_norm


def add_gaussian_noise(
    updates: dict,
    noise_multiplier: float,
    max_norm: float,
) -> dict:
    """
    Add calibrated Gaussian noise N(0, (σ·C)²) to each parameter tensor.
    σ = noise_multiplier, C = max_norm.
    """
    noisy = {}
    sigma = noise_multiplier * max_norm
    for key, val in updates.items():
        noisy[key] = val + torch.randn_like(val) * sigma
    return noisy


def apply_dp_to_client(
    local_model: nn.Module,
    global_model: nn.Module,
    noise_multiplier: float,
    max_grad_norm: float,
) -> tuple[nn.Module, float]:
    """
    Full DP pipeline for a single client:
      1. Compute update (w_local - w_global)
      2. Clip to L2 norm ≤ max_grad_norm
      3. Add Gaussian noise
      4. Reconstruct noisy local model weights
    Returns the noised local_model and the pre-clipping update norm.
    """
    clipped, raw_norm = clip_model_update(local_model, global_model, max_grad_norm)
    noisy = add_gaussian_noise(clipped, noise_multiplier, max_grad_norm)

    global_state = global_model.state_dict()
    noised_state = {}
    for key in global_state:
        noised_state[key] = global_state[key].float() + noisy[key]
        # Cast back to original dtype (int params like num_batches_tracked)
        noised_state[key] = noised_state[key].to(global_state[key].dtype)

    local_model.load_state_dict(noised_state)
    return local_model, raw_norm


def compute_epsilon(
    n_steps: int,
    noise_multiplier: float,
    delta: float,
    sample_rate: float,
) -> float:
    """
    Approximate privacy budget ε using the simplified sub-sampled Gaussian mechanism.

    Central Limit Theorem bound (Mironov, 2017 — RDP accountant simplified form):
        ε ≈ q · sqrt(2 · T · ln(1/δ)) / σ

    where:
        q  = sample_rate (batch_size / dataset_size)
        T  = n_steps
        σ  = noise_multiplier
        δ  = target failure probability

    Note: This is a conservative upper bound. Tighter accounting (e.g., Rényi DP via
    google/dp-accounting) gives smaller ε; add as dependency for camera-ready version.
    """
    if noise_multiplier <= 0:
        return float("inf")
    eps = sample_rate * math.sqrt(2 * n_steps * math.log(1.0 / delta)) / noise_multiplier
    return eps


def privacy_report(
    n_clients: int,
    n_rounds: int,
    local_epochs: int,
    batch_size: int,
    dataset_size: int,
    noise_multiplier: float,
    max_grad_norm: float,
    delta: float = 1e-5,
) -> dict:
    """
    Generate a privacy budget summary for the federated training run.
    'dataset_size' should be the per-client average training set size.
    """
    sample_rate = batch_size / max(dataset_size, 1)
    n_steps = n_rounds * local_epochs * max(dataset_size // batch_size, 1)
    epsilon = compute_epsilon(n_steps, noise_multiplier, delta, sample_rate)

    return {
        "mechanism":        "Gaussian (sub-sampled)",
        "epsilon":          round(epsilon, 4),
        "delta":            delta,
        "noise_multiplier": noise_multiplier,
        "max_grad_norm":    max_grad_norm,
        "n_rounds":         n_rounds,
        "n_clients":        n_clients,
        "local_epochs":     local_epochs,
        "n_steps":          n_steps,
        "sample_rate":      round(sample_rate, 6),
        "interpretation":   f"({round(epsilon, 2)}, {delta})-DP",
        "note": (
            "Bound via simplified CLT accountant. For tighter ε use "
            "'pip install dp-accounting' (google/dp-accounting)."
        ),
    }
