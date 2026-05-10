"""
Centralized baseline: train EfficientNet-B0 on ALL data (no federation).

Provides the upper-bound AUC for FL comparison and per-class XAI analysis
without FL communication overhead.

Run: python src/centralized.py [--epochs 20]
     Saves: outputs/metrics/test_metrics_centralized.json
"""

import os
import sys
import argparse
import json

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from config import (
    DATA_DIR, SEED, BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY,
    DEVICE, CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR,
    PATHOLOGY_LABELS,
)
from utils import set_seed, ensure_dirs, get_device
from dataset import ChestXrayDataset
from model import build_model, get_optimizer
from train_client import train_one_round, evaluate
from metrics import (
    compute_classification_metrics,
    compute_per_class_metrics,
    faithfulness_score,
)
from gradcam_aggregation import generate_client_gradcam
from visualize import plot_training_curves, plot_roc_curves


def parse_args():
    parser = argparse.ArgumentParser(description="Centralized baseline training")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--seed",   type=int, default=SEED)
    return parser.parse_args()


def load_metadata(data_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    for label in PATHOLOGY_LABELS:
        df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)
    return df


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(DEVICE)
    ensure_dirs(CHECKPOINT_DIR, METRICS_DIR, PLOTS_DIR)

    print(f"=== Centralized Baseline | device={device} | epochs={args.epochs} ===")

    # Load data
    df = load_metadata(DATA_DIR)
    img_dir = DATA_DIR
    available = set(f for f in os.listdir(img_dir) if f.endswith(".png"))
    df = df[df["Image Index"].isin(available)].reset_index(drop=True)
    print(f"Images matched: {len(df)}")

    test_df  = df.sample(frac=0.10, random_state=args.seed)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    train_ds = ChestXrayDataset(train_df, img_dir, split="train")
    test_ds  = ChestXrayDataset(test_df,  img_dir, split="val")

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                              num_workers=4, pin_memory=True)
    test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False,
                              num_workers=4, pin_memory=True)

    # pos_weight to handle NIH ChestX-ray14 class imbalance
    import torch as _torch
    _lbl = train_df[PATHOLOGY_LABELS].values
    _pos = _lbl.sum(axis=0).clip(min=1)
    pos_weight = _torch.tensor((_lbl.shape[0] - _pos) / _pos, dtype=_torch.float32)
    print(f"pos_weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    model     = build_model().to(device)
    optimizer = get_optimizer(model, LEARNING_RATE, WEIGHT_DECAY)

    history  = {"train_loss": [], "val_loss": [], "val_auc": [], "val_f1": []}
    best_auc = 0.0
    ckpt_path = os.path.join(CHECKPOINT_DIR, "best_centralized_model.pth")

    epoch_bar = tqdm(range(args.epochs), desc="Centralized Epochs")
    for epoch in epoch_bar:
        h      = train_one_round(model, train_loader, optimizer, device, local_epochs=1,
                                pos_weight=pos_weight)
        result = evaluate(model, test_loader, device)
        m      = compute_classification_metrics(result["logits"], result["labels"])

        history["train_loss"].append(h["train_loss"][0])
        history["val_loss"].append(result["val_loss"])
        history["val_auc"].append(m["auc_roc_macro"])
        history["val_f1"].append(m["f1_macro"])

        epoch_bar.set_postfix({"auc": f"{m['auc_roc_macro']:.4f}",
                               "f1":  f"{m['f1_macro']:.4f}"})

        if m["auc_roc_macro"] > best_auc:
            best_auc = m["auc_roc_macro"]
            torch.save(model.state_dict(), ckpt_path)

    # ── Final evaluation ─────────────────────────────────────────────────────
    print("\n=== Final Test Evaluation (Centralized) ===")
    model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
    result      = evaluate(model, test_loader, device)
    test_metrics = compute_classification_metrics(result["logits"], result["labels"])
    per_class    = compute_per_class_metrics(result["logits"], result["labels"], PATHOLOGY_LABELS)
    test_metrics["per_class"] = per_class

    print(f"  AUC-ROC (macro): {test_metrics['auc_roc_macro']:.4f}")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):      {test_metrics['f1_macro']:.4f}")
    print("\n  Per-class AUC:")
    for label, vals in per_class.items():
        print(f"    {label:<22} AUC={vals['auc']:.4f}  F1={vals['f1']:.4f}  "
              f"support={vals['support']}")

    # ── XAI — GradCAM faithfulness on centralized model ─────────────────────
    print("\n  Computing GradCAM + faithfulness score ...")
    try:
        sample_images, sample_labels = next(iter(test_loader))
        cam_map = generate_client_gradcam(model, test_loader, device, class_idx=0, max_batches=5)
        faith   = faithfulness_score(model, sample_images, cam_map, sample_labels, device)
        test_metrics["faithfulness_score"] = round(float(faith), 6)
        print(f"  Faithfulness score (pixel masking): {faith:.4f}")
    except Exception as e:
        print(f"  GradCAM skipped ({e})")

    # ── Save ─────────────────────────────────────────────────────────────────
    out_path = os.path.join(METRICS_DIR, "test_metrics_centralized.json")
    with open(out_path, "w") as f:
        json.dump(test_metrics, f, indent=2)

    hist_path = os.path.join(METRICS_DIR, "history_centralized.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, title="Centralized_Training",
                         save_path=os.path.join(PLOTS_DIR, "curves_centralized.png"))
    plot_roc_curves(result["logits"], result["labels"],
                    title="Centralized_ROC",
                    save_path=os.path.join(PLOTS_DIR, "roc_centralized.png"))

    print(f"\nMetrics saved → {out_path}")
    return test_metrics


if __name__ == "__main__":
    main()
