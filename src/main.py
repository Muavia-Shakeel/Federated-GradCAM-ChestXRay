"""
Main federated training loop.
Run: python main.py [--mode non_iid|iid] [--ablation no_gradcam]
                    [--algorithm fedavg|fedprox]
                    [--clients N] [--rounds N] [--alpha FLOAT]
                    [--seed INT] [--run_id STR]
                    [--dp_noise FLOAT]   # 0 = disabled
"""
import os
import sys
import argparse
import json
import copy

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from config import (
    DATA_DIR, SEED, NUM_CLIENTS, NUM_ROUNDS, LOCAL_EPOCHS,
    BATCH_SIZE, LEARNING_RATE, WEIGHT_DECAY, DEVICE,
    PARTITION_MODE, ALPHA, CHECKPOINT_DIR, GRADCAM_DIR,
    METRICS_DIR, PLOTS_DIR, PATHOLOGY_LABELS,
    FEDPROX_MU, DP_MAX_GRAD_NORM, DP_DELTA,
)
from utils import set_seed, ensure_dirs, get_device
from partition import partition_data
from dataset import build_client_loaders, ChestXrayDataset, get_transforms
from model import build_model, get_optimizer, copy_model
from train_client import train_one_round, train_one_round_fedprox, evaluate
from fedavg import fedavg, fedprox_aggregate, broadcast_weights
from gradcam_aggregation import (
    generate_client_gradcam, aggregate_gradcam_maps, compare_aggregation_strategies,
    generate_oracle_gradcam,
)
from metrics import (
    compute_classification_metrics, compute_per_class_metrics,
    faithfulness_score, insertion_deletion_auc, aopc_score, map_similarity_score,
)
from privacy import apply_dp_to_client, privacy_report
from visualize import (
    plot_training_curves, plot_roc_curves, plot_comparison_bar,
    plot_global_gradcam, plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode",      choices=["non_iid", "iid"],     default=PARTITION_MODE)
    parser.add_argument("--ablation",  choices=["none", "no_gradcam"], default="none")
    parser.add_argument("--algorithm", choices=["fedavg", "fedprox"],  default="fedavg")
    parser.add_argument("--rounds",    type=int,   default=NUM_ROUNDS)
    parser.add_argument("--clients",   type=int,   default=NUM_CLIENTS)
    parser.add_argument("--alpha",     type=float, default=ALPHA)
    parser.add_argument("--seed",      type=int,   default=SEED)
    parser.add_argument("--run_id",    type=str,   default="")
    parser.add_argument("--dp_noise",  type=float, default=0.0)
    return parser.parse_args()


def load_metadata(data_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    from config import PATHOLOGY_LABELS
    for label in PATHOLOGY_LABELS:
        df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)
    return df


def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device(DEVICE)
    ensure_dirs(CHECKPOINT_DIR, GRADCAM_DIR, METRICS_DIR, PLOTS_DIR)

    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Algorithm: {args.algorithm} | Ablation: {args.ablation}")
    print(f"Rounds: {args.rounds} | Clients: {args.clients} | Alpha: {args.alpha}")
    print(f"Seed: {args.seed} | DP noise: {args.dp_noise if args.dp_noise > 0 else 'disabled'}")

    # Load metadata
    df = load_metadata(DATA_DIR)
    img_dir = DATA_DIR
    available = set(f for f in os.listdir(img_dir) if f.endswith(".png"))
    before = len(df)
    df = df[df["Image Index"].isin(available)].reset_index(drop=True)
    print(f"Images on disk: {len(available)} | Metadata rows matched: {len(df)} (dropped {before - len(df)} missing)")

    # Hold out global test set (10%) before partitioning
    test_df  = df.sample(frac=0.10, random_state=SEED)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df  = test_df.reset_index(drop=True)

    # Partition training data across clients
    partitions    = partition_data(train_df, args.clients, mode=args.mode, alpha=args.alpha)
    client_loaders = build_client_loaders(partitions, img_dir, BATCH_SIZE)
    client_sizes   = [cl["n_train"] for cl in client_loaders]
    print(f"Client sizes: {client_sizes}")

    # Compute per-class pos_weight from combined training data to handle imbalance.
    # pos_weight[c] = (num_negatives_c / num_positives_c) — downweights majority negatives.
    all_labels_np = pd.concat(partitions)[PATHOLOGY_LABELS].values
    pos_counts    = all_labels_np.sum(axis=0).clip(min=1)
    neg_counts    = len(all_labels_np) - pos_counts
    pos_weight    = torch.tensor(neg_counts / pos_counts, dtype=torch.float32)
    print(f"pos_weight range: [{pos_weight.min():.1f}, {pos_weight.max():.1f}]")

    # Global test loader
    from torch.utils.data import DataLoader
    test_ds     = ChestXrayDataset(test_df, img_dir, split="val")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Initialize models
    global_model      = build_model().to(device)
    client_models     = [copy_model(global_model).to(device) for _ in range(args.clients)]
    client_optimizers = [get_optimizer(m, LEARNING_RATE, WEIGHT_DECAY) for m in client_models]

    history = {"train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []}
    best_val_auc       = 0.0
    patience_counter   = 0
    global_maps_per_round = []

    # Federated training loop
    round_bar = tqdm(range(args.rounds), desc="Federated Rounds", unit="round")
    for round_idx in round_bar:
        round_bar.set_postfix({"best_auc": f"{best_val_auc:.4f}"})
        round_train_losses = []

        client_bar = tqdm(
            zip(client_models, client_optimizers, client_loaders),
            total=args.clients, desc=f"  Round {round_idx+1} clients", leave=False
        )
        for c_idx, (model, optimizer, loaders) in enumerate(client_bar):
            client_bar.set_description(f"  Client {c_idx}")
            if args.algorithm == "fedprox":
                h = train_one_round_fedprox(
                    model, loaders["train"], optimizer, device,
                    LOCAL_EPOCHS, global_model, FEDPROX_MU, pos_weight=pos_weight
                )
            else:
                h = train_one_round(model, loaders["train"], optimizer, device, LOCAL_EPOCHS,
                                    pos_weight=pos_weight)
            if args.dp_noise > 0:
                apply_dp_to_client(model, global_model, args.dp_noise, DP_MAX_GRAD_NORM)
            round_train_losses.extend(h["train_loss"])
            client_bar.set_postfix({"train_loss": f"{h['train_loss'][-1]:.4f}"})

        # Aggregate
        global_model = fedavg(global_model, client_models, client_sizes)
        broadcast_weights(global_model, client_models)

        # Validation on global model using client val sets.
        # Track per-client AUC here (used by performance-weighted GradCAM aggregation).
        val_logits_all, val_labels_all, val_losses = [], [], []
        client_val_aucs = []
        for loaders in client_loaders:
            result = evaluate(global_model, loaders["val"], device)
            val_losses.append(result["val_loss"])
            val_logits_all.append(result["logits"])
            val_labels_all.append(result["labels"])
            c_m = compute_classification_metrics(result["logits"], result["labels"])
            client_val_aucs.append(c_m["auc_roc_macro"])

        val_logits  = torch.cat(val_logits_all)
        val_labels  = torch.cat(val_labels_all)
        val_metrics = compute_classification_metrics(val_logits, val_labels)

        history["train_loss"].append(np.mean(round_train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_auc"].append(val_metrics["auc_roc_macro"])

        print(f"  Val — loss={np.mean(val_losses):.4f} | F1={val_metrics['f1_macro']:.4f} | AUC={val_metrics['auc_roc_macro']:.4f}")

        # GradCAM aggregation (skip if ablation=no_gradcam)
        if args.ablation != "no_gradcam":
            client_maps = []
            for model, loaders in tqdm(zip(client_models, client_loaders), total=args.clients,
                                       desc=f"  Round {round_idx+1} GradCAM", leave=False):
                cam_map = generate_client_gradcam(model, loaders["val"], device, class_idx=0)
                client_maps.append(cam_map)

            global_map = aggregate_gradcam_maps(client_maps, client_sizes)
            global_maps_per_round.append(global_map)
            np.save(os.path.join(GRADCAM_DIR, f"global_map_round{round_idx+1}.npy"), global_map)
            plot_global_gradcam(global_map, title=f"Global_GradCAM_Round_{round_idx+1}",
                                save_path=os.path.join(PLOTS_DIR, f"gradcam_round{round_idx+1}.png"))

        # Early stopping on AUC
        if val_metrics["auc_roc_macro"] > best_val_auc:
            best_val_auc     = val_metrics["auc_roc_macro"]
            patience_counter = 0
            torch.save(global_model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_global_model.pth"))
            print(f"  New best AUC: {best_val_auc:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

    # ── Final Test Evaluation ──────────────────────────────────────────────────
    print("\n=== Final Test Evaluation ===")
    global_model.load_state_dict(torch.load(
        os.path.join(CHECKPOINT_DIR, "best_global_model.pth"),
        map_location=device, weights_only=True))
    # Sync client models with best checkpoint before XAI evaluation
    broadcast_weights(global_model, client_models)
    test_result  = evaluate(global_model, test_loader, device)
    test_metrics = compute_classification_metrics(test_result["logits"], test_result["labels"])

    per_class = compute_per_class_metrics(test_result["logits"], test_result["labels"], PATHOLOGY_LABELS)
    test_metrics["per_class"] = per_class

    print(f"  AUC-ROC (macro): {test_metrics['auc_roc_macro']:.4f}")
    print(f"  Accuracy:        {test_metrics['accuracy']:.4f}")
    print(f"  F1 (macro):      {test_metrics['f1_macro']:.4f}")
    print("\n  Per-class AUC:")
    for label, vals in per_class.items():
        print(f"    {label:<22} AUC={vals['auc']:.4f}  F1={vals['f1']:.4f}  support={vals['support']}")

    # XAI quantitative evaluation
    print("\n  Computing XAI metrics ...")
    try:
        sample_images, sample_labels = next(iter(test_loader))
        if global_maps_per_round:
            # Use oracle map from best checkpoint (not last-round map which may be from DP-corrupted training)
            cam_map = generate_oracle_gradcam(global_model, sample_images, device, class_idx=0)
            faith   = faithfulness_score(global_model, sample_images, cam_map, sample_labels, device, class_idx=0)
            aopc    = aopc_score(global_model, sample_images, cam_map, device, class_idx=0)
            ins_del = insertion_deletion_auc(global_model, sample_images, cam_map, device)
            ins_auc = ins_del["insertion_auc"]
            del_auc = ins_del["deletion_auc"]
            test_metrics["faithfulness_score"] = round(float(faith), 6)
            test_metrics["aopc_score"]         = round(float(aopc),  6)
            test_metrics["insertion_auc"]      = round(float(ins_auc), 6)
            test_metrics["deletion_auc"]       = round(float(del_auc), 6)
            print(f"  Faithfulness={faith:.4f} | AOPC={aopc:.4f} | Ins={ins_auc:.4f} | Del={del_auc:.4f}")

            if args.ablation != "no_gradcam":
                # Generate oracle map: what the global model itself attends to on the test batch
                oracle_map = generate_oracle_gradcam(global_model, sample_images, device, class_idx=0)

                client_maps_final = []
                for model, loaders in zip(client_models, client_loaders):
                    cm = generate_client_gradcam(model, loaders["val"], device, class_idx=0, max_batches=3)
                    client_maps_final.append(cm)
                strategy_maps = compare_aggregation_strategies(client_maps_final, client_sizes, client_val_aucs)
                strat_scores  = {}
                for s, smap in strategy_maps.items():
                    sf   = faithfulness_score(global_model, sample_images, smap, sample_labels, device, class_idx=0)
                    sa   = aopc_score(global_model, sample_images, smap, device, class_idx=0)
                    sim  = map_similarity_score(oracle_map, smap)
                    strat_scores[s] = {
                        "faithfulness": round(float(sf), 6),
                        "aopc":         round(float(sa), 6),
                        "pearson_r":    sim["pearson_r"],
                        "spearman_r":   sim["spearman_r"],
                        "ssim":         sim["ssim"],
                        "mse":          sim["mse"],
                    }
                test_metrics["strategy_comparison"] = strat_scores
                print("\n  GradCAM strategy comparison (vs oracle):")
                print(f"  {'Strategy':<15} {'Faith':>7} {'AOPC':>8} {'Pearson-r':>10} {'Spearman-r':>11} {'SSIM':>7} {'MSE':>8}")
                for s, v in strat_scores.items():
                    print(f"  {s:<15} {v['faithfulness']:>7.4f} {v['aopc']:>8.4f} "
                          f"{v['pearson_r']:>10.4f} {v['spearman_r']:>11.4f} "
                          f"{v['ssim']:>7.4f} {v['mse']:>8.6f}")
    except Exception as e:
        print(f"  XAI evaluation skipped: {e}")

    # DP privacy report
    if args.dp_noise > 0:
        report = privacy_report(
            args.clients, args.rounds, LOCAL_EPOCHS,
            BATCH_SIZE, len(train_df) // args.clients,
            args.dp_noise, DP_MAX_GRAD_NORM, DP_DELTA
        )
        test_metrics["privacy_report"] = report
        print(f"\n  DP: epsilon={report.get('epsilon', 'N/A'):.2f}, delta={DP_DELTA}")

    # Save
    suffix = f"{args.mode}_{args.algorithm}"
    if args.ablation != "none":
        suffix += f"_{args.ablation}"
    if args.run_id:
        suffix += f"_{args.run_id}"

    with open(os.path.join(METRICS_DIR, f"test_metrics_{suffix}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(METRICS_DIR, f"history_{suffix}.json"), "w") as f:
        json.dump(history, f, indent=2)

    plot_training_curves(history, title=f"Training_Curves_{suffix}",
                         save_path=os.path.join(PLOTS_DIR, f"curves_{suffix}.png"))
    plot_roc_curves(test_result["logits"], test_result["labels"],
                    title=f"ROC_Curves_{suffix}",
                    save_path=os.path.join(PLOTS_DIR, f"roc_{suffix}.png"))

    return test_metrics


if __name__ == "__main__":
    main()
