"""
Main federated training loop.
Run: python main.py [--mode non_iid|iid] [--ablation no_gradcam]
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
    METRICS_DIR, PLOTS_DIR,
)
from utils import set_seed, ensure_dirs, get_device
from partition import partition_data
from dataset import build_client_loaders, ChestXrayDataset, get_transforms
from model import build_model, get_optimizer, copy_model
from train_client import train_one_round, evaluate
from fedavg import fedavg, broadcast_weights
from gradcam_aggregation import generate_client_gradcam, aggregate_gradcam_maps
from metrics import compute_classification_metrics
from visualize import (
    plot_training_curves, plot_roc_curves, plot_comparison_bar,
    plot_global_gradcam, plot_confusion_matrix,
)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["non_iid", "iid"], default=PARTITION_MODE)
    parser.add_argument("--ablation", choices=["none", "no_gradcam"], default="none")
    parser.add_argument("--rounds", type=int, default=NUM_ROUNDS)
    parser.add_argument("--clients", type=int, default=NUM_CLIENTS)
    return parser.parse_args()


def load_metadata(data_dir: str) -> pd.DataFrame:
    csv_path = os.path.join(data_dir, "Data_Entry_2017.csv")
    df = pd.read_csv(csv_path)
    # Parse multi-label column into binary columns
    from config import PATHOLOGY_LABELS
    for label in PATHOLOGY_LABELS:
        df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)
    return df


def main():
    args = parse_args()
    set_seed(SEED)
    device = get_device(DEVICE)
    ensure_dirs(CHECKPOINT_DIR, GRADCAM_DIR, METRICS_DIR, PLOTS_DIR)

    print(f"Device: {device}")
    print(f"Mode: {args.mode} | Ablation: {args.ablation} | Rounds: {args.rounds}")

    # Load metadata — filter to only images present on disk (subset download)
    df = load_metadata(DATA_DIR)
    img_dir = DATA_DIR
    available = set(f for f in os.listdir(img_dir) if f.endswith(".png"))
    before = len(df)
    df = df[df["Image Index"].isin(available)].reset_index(drop=True)
    print(f"Images on disk: {len(available)} | Metadata rows matched: {len(df)} (dropped {before - len(df)} missing)")

    # Hold out global test set (10%) before partitioning
    test_df = df.sample(frac=0.10, random_state=SEED)
    train_df = df.drop(test_df.index).reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)

    # Partition training data across clients
    partitions = partition_data(train_df, args.clients, mode=args.mode, alpha=ALPHA)
    client_loaders = build_client_loaders(partitions, img_dir, BATCH_SIZE)
    client_sizes = [cl["n_train"] for cl in client_loaders]
    print(f"Client sizes: {client_sizes}")

    # Global test loader
    from torch.utils.data import DataLoader
    test_ds = ChestXrayDataset(test_df, img_dir, split="val")
    test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False,
                             num_workers=4, pin_memory=True)

    # Initialize models
    global_model = build_model().to(device)
    client_models = [copy_model(global_model).to(device) for _ in range(args.clients)]
    client_optimizers = [get_optimizer(m, LEARNING_RATE, WEIGHT_DECAY) for m in client_models]

    history = {
        "train_loss": [], "val_loss": [], "val_f1": [], "val_auc": []
    }
    best_val_auc = 0.0
    patience_counter = 0
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
            h = train_one_round(model, loaders["train"], optimizer, device, LOCAL_EPOCHS)
            round_train_losses.extend(h["train_loss"])
            client_bar.set_postfix({"train_loss": f"{h['train_loss'][-1]:.4f}"})

        # FedAvg
        global_model = fedavg(global_model, client_models, client_sizes)
        broadcast_weights(global_model, client_models)

        # Validation on global model using client val sets
        val_logits_all, val_labels_all = [], []
        val_losses = []
        for c_idx, loaders in enumerate(client_loaders):
            result = evaluate(global_model, loaders["val"], device)
            val_losses.append(result["val_loss"])
            val_logits_all.append(result["logits"])
            val_labels_all.append(result["labels"])

        val_logits = torch.cat(val_logits_all)
        val_labels = torch.cat(val_labels_all)
        val_metrics = compute_classification_metrics(val_logits, val_labels)

        history["train_loss"].append(np.mean(round_train_losses))
        history["val_loss"].append(np.mean(val_losses))
        history["val_f1"].append(val_metrics["f1_macro"])
        history["val_auc"].append(val_metrics["auc_roc_macro"])

        print(f"  Val — loss={np.mean(val_losses):.4f} | F1={val_metrics['f1_macro']:.4f} | AUC={val_metrics['auc_roc_macro']:.4f}")

        # GradCAM aggregation (skip if ablation=no_gradcam)
        if args.ablation != "no_gradcam":
            client_maps = []
            for c_idx, (model, loaders) in enumerate(
                tqdm(zip(client_models, client_loaders), total=args.clients,
                     desc=f"  Round {round_idx+1} GradCAM", leave=False)
            ):
                cam_map = generate_client_gradcam(model, loaders["val"], device, class_idx=0)
                client_maps.append(cam_map)

            global_map = aggregate_gradcam_maps(client_maps, client_sizes)
            global_maps_per_round.append(global_map)
            np.save(os.path.join(GRADCAM_DIR, f"global_map_round{round_idx+1}.npy"), global_map)
            plot_global_gradcam(global_map, title=f"Global_GradCAM_Round_{round_idx+1}",
                                save_path=os.path.join(PLOTS_DIR, f"gradcam_round{round_idx+1}.png"))

        # Early stopping on AUC
        if val_metrics["auc_roc_macro"] > best_val_auc:
            best_val_auc = val_metrics["auc_roc_macro"]
            patience_counter = 0
            torch.save(global_model.state_dict(), os.path.join(CHECKPOINT_DIR, "best_global_model.pth"))
            print(f"  New best AUC: {best_val_auc:.4f} — checkpoint saved.")
        else:
            patience_counter += 1
            if patience_counter >= 5:
                print("Early stopping triggered.")
                break

    # Final evaluation on test set
    print("\n=== Final Test Evaluation ===")
    global_model.load_state_dict(torch.load(os.path.join(CHECKPOINT_DIR, "best_global_model.pth")))
    test_result = evaluate(global_model, test_loader, device)
    test_metrics = compute_classification_metrics(test_result["logits"], test_result["labels"])
    print(f"Test metrics: {test_metrics}")

    # Save metrics
    suffix = f"{args.mode}_{args.ablation}"
    with open(os.path.join(METRICS_DIR, f"test_metrics_{suffix}.json"), "w") as f:
        json.dump(test_metrics, f, indent=2)
    with open(os.path.join(METRICS_DIR, f"history_{suffix}.json"), "w") as f:
        json.dump(history, f, indent=2)

    # Plots
    plot_training_curves(history, title=f"Training_Curves_{suffix}",
                         save_path=os.path.join(PLOTS_DIR, f"curves_{suffix}.png"))
    plot_roc_curves(test_result["logits"], test_result["labels"],
                    title=f"ROC_Curves_{suffix}",
                    save_path=os.path.join(PLOTS_DIR, f"roc_{suffix}.png"))

    return test_metrics


if __name__ == "__main__":
    main()
