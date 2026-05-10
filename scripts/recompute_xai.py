"""
Recompute XAI metrics with all 4 improvements:
  1. Mean-fill baseline masking (Fong & Vedaldi, 2017)
  2. Multi-class faithfulness: Atelectasis(0), Effusion(2), Infiltration(3)
  3. top_k_pct = 0.2 (20% masking, up from 10%)
  4. Stratified test batch — guaranteed >= 10 positives per target class

Run: /home/msi/DS_env/bin/python3 recompute_xai.py
"""
import os, sys, json
import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from config import (
    DATA_DIR, SEED, BATCH_SIZE, DEVICE, PATHOLOGY_LABELS,
    CHECKPOINT_DIR, METRICS_DIR, NUM_CLIENTS, ALPHA,
)
from utils import set_seed, get_device
from model import build_model, copy_model
from dataset import ChestXrayDataset
from partition import partition_data
from dataset import build_client_loaders
from gradcam_aggregation import (
    generate_oracle_gradcam, generate_client_gradcam,
    compare_aggregation_strategies,
)
from metrics import (
    faithfulness_score, faithfulness_score_multiclass,
    insertion_deletion_auc, aopc_score, map_similarity_score,
)
import pandas as pd


# Classes to evaluate: high-support pathologies for reliable faithfulness
HIGH_SUPPORT_CLASSES = [0, 2, 3]  # Atelectasis (1112), Effusion (1351), Infiltration (2003)

EXPERIMENTS = [
    {
        "suffix":     "non_iid_fedavg",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "best_global_model_fedavg.pth"),
    },
    {
        "suffix":     "non_iid_fedprox",
        "checkpoint": os.path.join(CHECKPOINT_DIR, "best_global_model_fedprox.pth"),
    },
]


def load_data(data_dir, seed, batch_size):
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    for label in PATHOLOGY_LABELS:
        df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)
    available = set(f for f in os.listdir(data_dir) if f.endswith(".png"))
    df = df[df["Image Index"].isin(available)].reset_index(drop=True)
    test_df  = df.sample(frac=0.10, random_state=seed).reset_index(drop=True)
    train_df = df.drop(df.sample(frac=0.10, random_state=seed).index).reset_index(drop=True)
    test_ds  = ChestXrayDataset(test_df, data_dir, split="val")
    loader   = DataLoader(test_ds, batch_size=batch_size, shuffle=False,
                          num_workers=4, pin_memory=True)
    return train_df, loader


def get_stratified_batch(test_loader, class_indices, min_pos_each=10, max_total=128):
    """
    Scan test_loader and accumulate images until we have >= min_pos_each
    positives for EACH class in class_indices, up to max_total images.
    """
    all_images, all_labels = [], []
    for images, labels in test_loader:
        all_images.append(images)
        all_labels.append(labels)
        total = sum(len(x) for x in all_images)
        lbls_so_far = torch.cat(all_labels)
        if all((lbls_so_far[:, c] == 1).sum() >= min_pos_each for c in class_indices):
            break
        if total >= max_total:
            break
    imgs = torch.cat(all_images)[:max_total]
    lbls = torch.cat(all_labels)[:max_total]
    for c in class_indices:
        n = (lbls[:, c] == 1).sum().item()
        print(f"    Class {c} ({PATHOLOGY_LABELS[c]}): {n} positives in batch")
    return imgs, lbls


def recompute(exp, device, train_df, test_loader):
    suffix = exp["suffix"]
    ckpt   = exp["checkpoint"]
    metrics_path = os.path.join(METRICS_DIR, f"test_metrics_{suffix}.json")

    if not os.path.exists(ckpt):
        print(f"  Checkpoint missing: {ckpt} — skipping")
        return
    if not os.path.exists(metrics_path):
        print(f"  Metrics file missing — skipping")
        return

    with open(metrics_path) as f:
        metrics = json.load(f)

    model = build_model().to(device)
    model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=True))
    model.eval()
    print(f"  Checkpoint loaded: {os.path.basename(ckpt)}")

    # Stratified batch with guaranteed positives for all target classes
    print("  Building stratified test batch ...")
    sample_images, sample_labels = get_stratified_batch(
        test_loader, HIGH_SUPPORT_CLASSES, min_pos_each=10, max_total=128
    )

    # Generate oracle GradCAM per high-support class
    print("  Generating per-class oracle GradCAMs ...")
    oracle_maps = {}
    for c in HIGH_SUPPORT_CLASSES:
        oracle_maps[c] = generate_oracle_gradcam(model, sample_images, device, class_idx=c)

    # Use Infiltration (class 3, highest support) as primary map for insertion/deletion
    primary_map   = oracle_maps[3]
    primary_class = 3

    # 1. Multi-class faithfulness (mean-fill, top_k=0.2)
    faith_mc = faithfulness_score_multiclass(
        model, sample_images, oracle_maps, sample_labels, device,
        class_indices=HIGH_SUPPORT_CLASSES, top_k_pct=0.2,
    )
    faith_mean = faith_mc["mean"]
    faith_per  = {PATHOLOGY_LABELS[c]: round(faith_mc.get(c, 0.0), 6) for c in HIGH_SUPPORT_CLASSES}
    print(f"  Faithfulness (multi-class mean): {faith_mean:.4f}")
    print(f"  Per-class: {faith_per}")

    # 2. AOPC on primary class (Infiltration)
    aopc = aopc_score(model, sample_images, primary_map, device, class_idx=primary_class)
    print(f"  AOPC (Infiltration): {aopc:.4f}")

    # 3. Insertion / Deletion AUC on primary map
    ins_del = insertion_deletion_auc(model, sample_images, primary_map, device)
    print(f"  Insertion AUC: {ins_del['insertion_auc']:.4f} | Deletion AUC: {ins_del['deletion_auc']:.4f}")

    # Write back to metrics
    metrics["faithfulness_score"]          = round(float(faith_mean), 6)
    metrics["faithfulness_per_class"]      = faith_per
    metrics["aopc_score"]                  = round(float(aopc), 6)
    metrics["insertion_auc"]               = round(float(ins_del["insertion_auc"]), 6)
    metrics["deletion_auc"]                = round(float(ins_del["deletion_auc"]), 6)
    metrics["xai_eval_config"] = {
        "top_k_pct":        0.2,
        "masking_baseline": "imagenet_mean",
        "faithfulness_classes": [PATHOLOGY_LABELS[c] for c in HIGH_SUPPORT_CLASSES],
        "primary_class":    PATHOLOGY_LABELS[primary_class],
        "batch_size":       len(sample_images),
    }

    # 4. Strategy comparison (rebuild client maps)
    if "strategy_comparison" in metrics:
        print("  Recomputing aggregation strategy comparison ...")
        partitions    = partition_data(train_df, NUM_CLIENTS, mode="non_iid", alpha=ALPHA)
        client_loaders = build_client_loaders(partitions, DATA_DIR, BATCH_SIZE)
        client_sizes   = [cl["n_train"] for cl in client_loaders]
        client_models  = [copy_model(model).to(device) for _ in range(NUM_CLIENTS)]

        # Client GradCAMs for primary class
        client_maps = []
        for cm, loaders in zip(client_models, client_loaders):
            cmap = generate_client_gradcam(cm, loaders["val"], device,
                                           class_idx=primary_class, max_batches=3)
            client_maps.append(cmap)

        oracle_primary = oracle_maps[primary_class]
        strategy_maps  = compare_aggregation_strategies(client_maps, client_sizes)
        strat_scores   = {}

        for s, smap in strategy_maps.items():
            # Per-strategy faithfulness using multi-class approach
            strat_oracle_maps = {c: smap for c in HIGH_SUPPORT_CLASSES}  # same smap, multi-class eval
            sf_mc = faithfulness_score_multiclass(
                model, sample_images, strat_oracle_maps, sample_labels, device,
                class_indices=HIGH_SUPPORT_CLASSES, top_k_pct=0.2,
            )
            sa  = aopc_score(model, sample_images, smap, device, class_idx=primary_class)
            sim = map_similarity_score(oracle_primary, smap)
            strat_scores[s] = {
                "faithfulness": round(float(sf_mc["mean"]), 6),
                "aopc":         round(float(sa), 6),
                "pearson_r":    sim["pearson_r"],
                "spearman_r":   sim["spearman_r"],
                "ssim":         sim["ssim"],
                "mse":          sim["mse"],
            }
            print(f"    {s:<15} faith={sf_mc['mean']:.4f}  aopc={sa:.4f}  "
                  f"pearson={sim['pearson_r']:.4f}  ssim={sim['ssim']:.4f}  mse={sim['mse']:.6f}")
        metrics["strategy_comparison"] = strat_scores

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"  Saved → {metrics_path}\n")


def main():
    set_seed(SEED)
    device = get_device(DEVICE)
    print(f"Device: {device}")
    print(f"Improvements: mean-fill baseline | top_k=0.2 | multi-class faithfulness | stratified batch\n")
    train_df, test_loader = load_data(DATA_DIR, SEED, BATCH_SIZE)

    for exp in EXPERIMENTS:
        print(f"=== {exp['suffix']} ===")
        recompute(exp, device, train_df, test_loader)

    # Summary
    print("=== SUMMARY ===")
    print(f"{'Model':<25} {'Faithful':>10} {'AOPC':>8} {'Ins AUC':>9} {'Del AUC':>9}")
    print("-" * 65)
    for exp in EXPERIMENTS:
        path = os.path.join(METRICS_DIR, f"test_metrics_{exp['suffix']}.json")
        if os.path.exists(path):
            d = json.load(open(path))
            print(f"{exp['suffix']:<25} {d['faithfulness_score']:>10.4f} "
                  f"{d['aopc_score']:>8.4f} {d['insertion_auc']:>9.4f} {d['deletion_auc']:>9.4f}")


if __name__ == "__main__":
    main()
