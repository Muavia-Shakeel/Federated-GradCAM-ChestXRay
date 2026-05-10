"""
Non-IID Robustness and Stress Testing for Federated Learning.

Tests the framework under:
  (A) Alpha sweep: Dirichlet α ∈ {0.1, 0.3, 0.5, 1.0}
      Lower α = more heterogeneous data (extreme non-IID).
  (B) Client dropout: A random fraction of clients skip each round,
      simulating real-world hospital availability issues.
  (C) Reports AUC vs α table and AUC vs dropout rate table.

Run: python src/stress_test.py
     Saves: outputs/metrics/stress_test_results.json
             outputs/plots/stress_alpha_sweep.png
             outputs/plots/stress_dropout_sweep.png
"""

import os
import sys
import json
import copy
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_DIR, SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR, PATHOLOGY_LABELS,
    STRESS_ALPHAS, STRESS_DROPOUT_RATE,
)
from utils import set_seed, ensure_dirs, get_device
from partition import partition_data
from dataset import build_client_loaders, ChestXrayDataset
from model import build_model, get_optimizer, copy_model
from train_client import train_one_round, evaluate
from fedavg import fedavg, broadcast_weights
from metrics import compute_classification_metrics


# ── Slim training loop for stress experiments ────────────────────────────────

def _run_fl(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    img_dir: str,
    device: torch.device,
    n_clients: int,
    n_rounds: int,
    alpha: float,
    dropout_rate: float = 0.0,
    seed: int = SEED,
) -> dict:
    """
    Lightweight FL run for stress testing.
    Returns final test metrics dict.
    """
    set_seed(seed)
    partitions    = partition_data(train_df, n_clients, mode="non_iid", alpha=alpha)
    client_loaders = build_client_loaders(partitions, img_dir, BATCH_SIZE)
    client_sizes  = [cl["n_train"] for cl in client_loaders]

    test_ds     = ChestXrayDataset(test_df, img_dir, split="val")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=2, pin_memory=True)

    global_model    = build_model().to(device)
    client_models   = [copy_model(global_model).to(device) for _ in range(n_clients)]
    client_optims   = [get_optimizer(m, LEARNING_RATE, WEIGHT_DECAY) for m in client_models]

    best_auc = 0.0
    N_LOCAL  = 2    # fewer epochs for speed in stress tests

    for round_idx in range(n_rounds):
        # Simulate client dropout
        active = [
            i for i in range(n_clients)
            if random.random() > dropout_rate
        ]
        if not active:
            active = [0]   # at least one client must participate

        for c_idx in active:
            train_one_round(
                client_models[c_idx], client_loaders[c_idx]["train"],
                client_optims[c_idx], device, N_LOCAL,
            )

        active_models = [client_models[i] for i in active]
        active_sizes  = [client_sizes[i]  for i in active]
        global_model  = fedavg(global_model, active_models, active_sizes)
        broadcast_weights(global_model, client_models)

    result = evaluate(global_model, test_loader, device)
    m      = compute_classification_metrics(result["logits"], result["labels"])
    return m


# ── Main stress test driver ──────────────────────────────────────────────────

def load_data(seed: int = SEED):
    csv_path = os.path.join(DATA_DIR, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    for label in PATHOLOGY_LABELS:
        df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)

    img_dir   = DATA_DIR
    available = set(f for f in os.listdir(img_dir) if f.endswith(".png"))
    df = df[df["Image Index"].isin(available)].reset_index(drop=True)

    test_df  = df.sample(frac=0.10, random_state=seed)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)
    return train_df, test_df, img_dir


def run_alpha_sweep(train_df, test_df, img_dir, device,
                    alphas=STRESS_ALPHAS, n_clients=5, n_rounds=10):
    """Test AUC under increasing data heterogeneity (lower α = more non-IID)."""
    print("\n── Alpha Sweep (Non-IID Stress Test) ──────────────────────────")
    results = {}
    for alpha in alphas:
        print(f"  Running α={alpha} ...", end=" ", flush=True)
        m = _run_fl(train_df, test_df, img_dir, device,
                    n_clients=n_clients, n_rounds=n_rounds, alpha=alpha)
        results[str(alpha)] = {
            "alpha":        alpha,
            "auc_roc_macro": round(m["auc_roc_macro"], 4),
            "f1_macro":      round(m["f1_macro"], 4),
            "accuracy":      round(m["accuracy"], 4),
        }
        print(f"AUC={m['auc_roc_macro']:.4f}")
    return results


def run_dropout_sweep(train_df, test_df, img_dir, device,
                      dropout_rates=None, alpha=0.5, n_clients=5, n_rounds=10):
    """Test AUC under increasing client dropout rates."""
    if dropout_rates is None:
        dropout_rates = [0.0, 0.1, 0.2, 0.3, 0.5]

    print("\n── Client Dropout Sweep (Robustness Test) ──────────────────────")
    results = {}
    for rate in dropout_rates:
        print(f"  Dropout rate={rate:.1f} ...", end=" ", flush=True)
        m = _run_fl(train_df, test_df, img_dir, device,
                    n_clients=n_clients, n_rounds=n_rounds,
                    alpha=alpha, dropout_rate=rate)
        results[str(rate)] = {
            "dropout_rate":  rate,
            "auc_roc_macro": round(m["auc_roc_macro"], 4),
            "f1_macro":      round(m["f1_macro"], 4),
            "accuracy":      round(m["accuracy"], 4),
        }
        print(f"AUC={m['auc_roc_macro']:.4f}")
    return results


def plot_sweep_results(results_dict: dict, x_key: str, x_label: str,
                       title: str, save_path: str):
    """Plot AUC vs sweep variable."""
    try:
        import matplotlib.pyplot as plt

        x_vals   = [v[x_key]        for v in results_dict.values()]
        auc_vals = [v["auc_roc_macro"] for v in results_dict.values()]
        f1_vals  = [v["f1_macro"]    for v in results_dict.values()]

        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(x_vals, auc_vals, "o-", color="steelblue", label="AUC-ROC (macro)")
        ax1.plot(x_vals, f1_vals,  "s--", color="coral",    label="F1 (macro)")
        ax1.set_xlabel(x_label)
        ax1.set_ylabel("Score")
        ax1.set_title(title)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Plot saved → {save_path}")
    except Exception as e:
        print(f"  Plot skipped ({e})")


def main():
    set_seed(SEED)
    device = get_device(DEVICE)
    ensure_dirs(CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR)

    print("=== Federated Learning Stress Test ===")
    train_df, test_df, img_dir = load_data()

    # (A) Alpha sweep — non-IID heterogeneity
    alpha_results = run_alpha_sweep(train_df, test_df, img_dir, device)

    # (B) Client dropout sweep — robustness
    dropout_results = run_dropout_sweep(train_df, test_df, img_dir, device)

    # Combine and save
    stress_results = {
        "alpha_sweep":   alpha_results,
        "dropout_sweep": dropout_results,
    }

    out_path = os.path.join(METRICS_DIR, "stress_test_results.json")
    with open(out_path, "w") as f:
        json.dump(stress_results, f, indent=2)
    print(f"\nStress test results saved → {out_path}")

    # Tables
    print("\n── Alpha Sweep Summary ─────────────────────────────────────────")
    print(f"  {'Alpha':<8} {'AUC':>8} {'F1':>8} {'Acc':>8}")
    for v in alpha_results.values():
        print(f"  {v['alpha']:<8} {v['auc_roc_macro']:>8.4f} {v['f1_macro']:>8.4f} {v['accuracy']:>8.4f}")

    print("\n── Client Dropout Summary ──────────────────────────────────────")
    print(f"  {'Dropout':>8} {'AUC':>8} {'F1':>8} {'Acc':>8}")
    for v in dropout_results.values():
        print(f"  {v['dropout_rate']:>8.2f} {v['auc_roc_macro']:>8.4f} {v['f1_macro']:>8.4f} {v['accuracy']:>8.4f}")

    # Plots
    plot_sweep_results(
        alpha_results, "alpha", "Dirichlet α (heterogeneity)",
        "Non-IID Stress: AUC vs α",
        os.path.join(PLOTS_DIR, "stress_alpha_sweep.png"),
    )
    plot_sweep_results(
        dropout_results, "dropout_rate", "Client Dropout Rate",
        "Robustness: AUC vs Dropout Rate",
        os.path.join(PLOTS_DIR, "stress_dropout_sweep.png"),
    )


if __name__ == "__main__":
    main()
