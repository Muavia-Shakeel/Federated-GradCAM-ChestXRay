"""
Statistical Validation: Multi-seed federated training runs.

Addresses Sir's feedback: "Provide statistical validation (multiple runs)".

Runs main.py with three different random seeds, collects AUC / F1 / Accuracy,
and reports mean ± std across runs — standard practice for ML papers.

Run:  python src/multi_run.py [--seeds 42 123 456] [--mode non_iid] [--algorithm fedavg]
      Saves: outputs/metrics/multi_run_stats.json
             outputs/plots/multi_run_auc_distribution.png
"""

import os
import sys
import json
import argparse
import subprocess

import numpy as np

sys.path.insert(0, os.path.dirname(__file__))

from config import SEED, METRICS_DIR, PLOTS_DIR
from utils import ensure_dirs


METRIC_KEYS = ["auc_roc_macro", "f1_macro", "f1_micro", "accuracy",
               "precision_macro", "recall_macro"]


def parse_args():
    parser = argparse.ArgumentParser(description="Multi-seed statistical validation")
    parser.add_argument("--seeds",     type=int, nargs="+", default=[42, 123, 456])
    parser.add_argument("--mode",      default="non_iid", choices=["non_iid", "iid"])
    parser.add_argument("--algorithm", default="fedavg",  choices=["fedavg", "fedprox"])
    parser.add_argument("--rounds",    type=int, default=20)
    parser.add_argument("--clients",   type=int, default=5)
    return parser.parse_args()


def run_single_seed(seed: int, mode: str, algorithm: str,
                    rounds: int, clients: int) -> dict:
    """
    Launch main.py as a subprocess with the given seed.
    Reads the resulting test_metrics JSON and returns it.
    """
    run_id   = f"seed{seed}"
    suffix   = f"{mode}_{algorithm}_{run_id}"   # matches main.py naming: {mode}_{algorithm}_{run_id}
    out_file = os.path.join(METRICS_DIR, f"test_metrics_{suffix}.json")

    cmd = [
        sys.executable,
        os.path.join(os.path.dirname(__file__), "main.py"),
        "--mode",      mode,
        "--algorithm", algorithm,
        "--rounds",    str(rounds),
        "--clients",   str(clients),
        "--seed",      str(seed),
        "--run_id",    run_id,
        "--ablation",  "none",
    ]

    print(f"\n── Seed {seed} ──────────────────────────────────────────────────")
    print(f"  cmd: {' '.join(cmd)}")

    result = subprocess.run(cmd, capture_output=False, text=True)
    if result.returncode != 0:
        print(f"  [WARNING] Seed {seed} run returned non-zero exit code.")

    if os.path.isfile(out_file):
        with open(out_file) as f:
            metrics = json.load(f)
        return metrics
    else:
        print(f"  [WARNING] Metrics file not found: {out_file}")
        return {}


def aggregate_runs(all_metrics: list[dict]) -> dict:
    """Compute mean ± std for each scalar metric across runs."""
    stats = {}
    for key in METRIC_KEYS:
        values = [m[key] for m in all_metrics if key in m]
        if values:
            stats[key] = {
                "mean":   round(float(np.mean(values)), 4),
                "std":    round(float(np.std(values)),  4),
                "min":    round(float(np.min(values)),  4),
                "max":    round(float(np.max(values)),  4),
                "values": [round(v, 4) for v in values],
                "n_runs": len(values),
            }
    return stats


def plot_distribution(stats: dict, save_path: str):
    """Bar plot of mean ± std for key metrics."""
    try:
        import matplotlib.pyplot as plt

        keys   = [k for k in METRIC_KEYS if k in stats]
        means  = [stats[k]["mean"] for k in keys]
        stds   = [stats[k]["std"]  for k in keys]
        labels = [k.replace("_macro", "\n(macro)").replace("_micro", "\n(micro)")
                  for k in keys]

        fig, ax = plt.subplots(figsize=(10, 5))
        x = np.arange(len(keys))
        bars = ax.bar(x, means, yerr=stds, capsize=5,
                      color="steelblue", alpha=0.8, ecolor="black")

        # Annotate bars
        for bar, m, s in zip(bars, means, stds):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + s + 0.005,
                    f"{m:.3f}±{s:.3f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=9)
        ax.set_ylabel("Score")
        ax.set_ylim(0, 1.15)
        ax.set_title("Multi-Run Statistical Validation (mean ± std)")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_path, dpi=150)
        plt.close()
        print(f"  Distribution plot saved → {save_path}")
    except Exception as e:
        print(f"  Plot skipped ({e})")


def main():
    args = parse_args()
    ensure_dirs(METRICS_DIR, PLOTS_DIR)

    print(f"=== Multi-Seed Statistical Validation ===")
    print(f"  Seeds: {args.seeds} | Mode: {args.mode} | Algorithm: {args.algorithm}")
    print(f"  Rounds: {args.rounds} | Clients: {args.clients}")

    all_metrics = []
    for seed in args.seeds:
        m = run_single_seed(seed, args.mode, args.algorithm,
                            args.rounds, args.clients)
        if m:
            m["seed"] = seed
            all_metrics.append(m)

    if not all_metrics:
        print("[ERROR] No successful runs. Exiting.")
        return

    stats     = aggregate_runs(all_metrics)
    per_run   = [{k: m.get(k) for k in METRIC_KEYS + ["seed"]} for m in all_metrics]

    output = {
        "config": {
            "seeds":     args.seeds,
            "mode":      args.mode,
            "algorithm": args.algorithm,
            "rounds":    args.rounds,
            "clients":   args.clients,
        },
        "per_run":    per_run,
        "statistics": stats,
    }

    out_path = os.path.join(METRICS_DIR, "multi_run_stats.json")
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n── Multi-Run Summary ───────────────────────────────────────────")
    print(f"  {'Metric':<22} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print(f"  {'-'*54}")
    for key, s in stats.items():
        print(f"  {key:<22} {s['mean']:>8.4f} {s['std']:>8.4f} "
              f"{s['min']:>8.4f} {s['max']:>8.4f}")

    plot_distribution(stats, os.path.join(PLOTS_DIR, "multi_run_auc_distribution.png"))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
