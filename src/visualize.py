import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from sklearn.metrics import roc_curve, auc, confusion_matrix
import seaborn as sns
import torch

from config import PATHOLOGY_LABELS, PLOTS_DIR


def plot_training_curves(history: dict, title: str = "Training Curves", save_path: str = None):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    if "train_loss" in history:
        axes[0].plot(history["train_loss"], label="Train")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="Val")
    axes[0].set_title("Loss")
    axes[0].set_xlabel("Round")
    axes[0].legend()

    if "val_f1" in history:
        axes[1].plot(history["val_f1"], label="Val F1")
    if "val_auc" in history:
        axes[1].plot(history["val_auc"], label="Val AUC-ROC")
    axes[1].set_title("Performance")
    axes[1].set_xlabel("Round")
    axes[1].legend()

    fig.suptitle(title)
    plt.tight_layout()
    path = save_path or os.path.join(PLOTS_DIR, f"{title.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_roc_curves(logits: torch.Tensor, labels: torch.Tensor, title: str = "ROC Curves", save_path: str = None):
    probs = torch.sigmoid(logits).numpy()
    targets = labels.numpy()
    fig, ax = plt.subplots(figsize=(10, 8))

    for i, label in enumerate(PATHOLOGY_LABELS):
        if targets[:, i].sum() == 0:
            continue
        fpr, tpr, _ = roc_curve(targets[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        ax.plot(fpr, tpr, label=f"{label} (AUC={roc_auc:.2f})")

    ax.plot([0, 1], [0, 1], "k--")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title(title)
    ax.legend(loc="lower right", fontsize=7)
    plt.tight_layout()
    path = save_path or os.path.join(PLOTS_DIR, f"{title.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_confusion_matrix(logits: torch.Tensor, labels: torch.Tensor, class_idx: int = 0, save_path: str = None):
    probs = torch.sigmoid(logits).numpy()
    preds = (probs[:, class_idx] >= 0.5).astype(int)
    targets = labels.numpy()[:, class_idx].astype(int)
    cm = confusion_matrix(targets, preds)
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["Neg", "Pos"], yticklabels=["Neg", "Pos"])
    ax.set_title(f"Confusion Matrix — {PATHOLOGY_LABELS[class_idx]}")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    plt.tight_layout()
    label_name = PATHOLOGY_LABELS[class_idx].replace(" ", "_")
    path = save_path or os.path.join(PLOTS_DIR, f"cm_{label_name}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_global_gradcam(global_map: np.ndarray, title: str = "Global GradCAM", save_path: str = None):
    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(global_map, cmap="jet", interpolation="bilinear")
    plt.colorbar(im, ax=ax)
    ax.set_title(title)
    ax.axis("off")
    plt.tight_layout()
    path = save_path or os.path.join(PLOTS_DIR, f"{title.replace(' ', '_')}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")


def plot_comparison_bar(results: dict, metric: str = "f1_macro", save_path: str = None):
    """
    results: {"Model Name": metric_value, ...}
    """
    fig, ax = plt.subplots(figsize=(8, 5))
    names = list(results.keys())
    values = [results[n] for n in names]
    bars = ax.bar(names, values, color=["steelblue", "tomato", "seagreen", "orange"])
    ax.bar_label(bars, fmt="%.3f", padding=3)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel(metric)
    ax.set_title(f"Model Comparison — {metric}")
    plt.tight_layout()
    path = save_path or os.path.join(PLOTS_DIR, f"comparison_{metric}.png")
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {path}")
