# Federated Learning with Weighted GradCAM Aggregation for Explainable Chest X-Ray Diagnosis

## Overview

A novel Federated Learning (FL) framework for multi-label chest pathology classification. Addresses the intersection of **patient data privacy** and **clinical explainability**. Multiple simulated hospital clients collaboratively train an `EfficientNet-B0` model on the **NIH ChestX-ray14 dataset** (14 pathology labels, 112,120 images) without sharing raw data, while producing aggregated global GradCAM visual explanations.

## Problem Statement

Current federated learning models for multi-site medical image classification produce predictions without clinically usable explanations. FL preserves patient privacy, but the resulting global model operates as a black box — clinicians get predictions with no indication of which anatomical regions drove the decision. This leaves federated medical AI systems technically privacy-compliant but clinically undeployable.

## Research Gap

1. **No Global Explanation:** Existing federated XAI approaches generate and evaluate maps locally only. No principled method exists to produce a coherent global explanation reflecting cross-institution variability.
2. **Unfair Weighting:** Current explanation aggregations ignore dataset size imbalance across institutions.
3. **Non-IID Impact Unknown:** Non-IID data distributions (realistic clinical scenarios) are rarely the focus of ablation studies in federated XAI work.

## Novelty & Proposed Solution

**Dataset-Size-Weighted GradCAM Aggregation** — instead of only aggregating model weights, the Central Aggregation Server receives local GradCAM saliency maps from each hospital and computes a weighted average proportional to each client's dataset size.

**Key Contributions:**
- **Federated XAI:** Global Explanation Map reflecting collective attention without transferring patient data
- **Size-Proportional Fairness:** Both model weights (`FedAvg`) and GradCAM maps weighted by local dataset size
- **Non-IID Robustness:** Tested across heavily imbalanced (Dirichlet α=0.5) non-IID distributions across clients
- **XAI Faithfulness Evaluation:** Insertion/Deletion AUC, AOPC, and per-class faithfulness scoring

## Methodology

- **Dataset:** NIH ChestX-ray14 (112,120 frontal-view X-ray images, 14 pathology labels)
- **Data Partitioning:** Non-IID Dirichlet (α=0.5) partitioning across 3–5 simulated hospital clients
- **Local Training:** `EfficientNet-B0` CNN trained locally per round
- **Aggregation:**
  - Model Weights: Standard `FedAvg` / `FedProx`
  - XAI Maps: Dataset-size-weighted average of `conv_head` layer activations

## Repository Structure

```text
.
├── assignments/            # Course assignment documents
├── docs/                   # Project documentation and reports
├── logs/                   # Training run logs
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── outputs/
│   ├── checkpoints/        # Saved model weights (.pth) — git-ignored
│   ├── gradcam_maps/       # Generated heatmaps (.npy) — git-ignored
│   ├── metrics/            # Evaluation metrics (.json)
│   └── plots/              # Loss curves, ROC curves, GradCAM visualizations
├── scripts/                # Utility scripts (figure generation, reporting)
├── src/
│   ├── config.py           # Hyperparameters and global settings
│   ├── dataset.py          # PyTorch Dataset and DataLoader definitions
│   ├── fedavg.py           # Federated weight aggregation logic
│   ├── gradcam_aggregation.py  # Novel weighted GradCAM aggregation
│   ├── main.py             # Main training orchestration
│   ├── metrics.py          # AUC, F1, Accuracy, XAI faithfulness
│   ├── model.py            # EfficientNet-B0 initialization
│   ├── partition.py        # Dirichlet non-IID and IID splitting
│   ├── train_client.py     # Local client training and evaluation
│   └── visualize.py        # Plotting utilities
├── requirements.txt
└── run_all.sh              # Full experiment pipeline runner
```

## Results

### Summary Comparison

| Method | AUC-ROC (Macro) | Accuracy | F1-Macro | Precision | Recall |
|--------|:-:|:-:|:-:|:-:|:-:|
| **Centralized Baseline** | **0.8291** | **81.55%** | **0.2440** | 0.1569 | 0.6799 |
| FedAvg + GradCAM (ours) | 0.8235 | 79.53% | 0.2266 | 0.1429 | 0.7143 |
| FedAvg (no GradCAM) | 0.8256 | 80.36% | 0.2268 | — | — |
| FedProx + GradCAM | 0.8210 | 82.63% | 0.2480 | 0.1620 | 0.6501 |
| FedProx multi-run (mean±std) | 0.8139±0.0000 | 95.04%±0.01% | 0.0999±0.003 | 0.3284±0.004 | 0.0635±0.002 |

> Note: FedAvg/FedProx results from non-IID Dirichlet (α=0.5) partitioning, 3 clients, 10–20 rounds. FedProx multi-run = 20 rounds, 5 clients, seeds {123, 456}.

### XAI Faithfulness (FedAvg + GradCAM)

| Metric | Score |
|--------|-------|
| Faithfulness Score | −0.000766 |
| AOPC Score | −0.0996 |
| Insertion AUC | 0.3433 |
| Deletion AUC | 0.4176 |

**GradCAM Aggregation Strategy Comparison (FedAvg):**

| Strategy | Faithfulness | AOPC | SSIM |
|----------|:-:|:-:|:-:|
| **Weighted (ours)** | **0.002146** | **−0.0694** | **−0.2417** |
| Uniform | −0.000247 | −0.0664 | −0.1078 |
| Max-Pool | −0.039139 | −0.0752 | −0.0797 |

### Per-Class AUC (FedAvg + GradCAM)

| Pathology | AUC | F1 | Support |
|-----------|:-:|:-:|:-:|
| Emphysema | 0.9188 | 0.260 | 240 |
| Edema | 0.8897 | 0.139 | 223 |
| Pneumothorax | 0.8864 | 0.305 | 518 |
| Cardiomegaly | 0.8962 | 0.208 | 278 |
| Mass | 0.8430 | 0.290 | 578 |
| Effusion | 0.8683 | 0.483 | 1351 |
| Fibrosis | 0.8243 | 0.094 | 170 |
| Atelectasis | 0.7931 | 0.327 | 1112 |
| Pleural_Thickening | 0.7963 | 0.142 | 320 |
| Consolidation | 0.7887 | 0.154 | 414 |
| Nodule | 0.7542 | 0.215 | 616 |
| Pneumonia | 0.7352 | 0.067 | 143 |
| Infiltration | 0.7156 | 0.409 | 2003 |
| Hernia | 0.8188 | 0.079 | 21 |

### Artifacts

- GradCAM heatmaps aggregated per round (rounds 1–20+) in `outputs/plots/`
- Loss and ROC curves for all experiment configurations in `outputs/plots/`
- Full per-class metrics in `outputs/metrics/`

## How to Run

> **Dataset not included.** Download NIH ChestX-ray14 from [NIH Clinical Center](https://nihcc.app.box.com/v/ChestXray-NIHCC) and extract into `data/raw/` with `Data_Entry_2017.csv` in `data/`.

```bash
pip install -r requirements.txt
```

```bash
# Non-IID FedAvg with weighted GradCAM aggregation
python src/main.py --mode non_iid --ablation none

# Ablation: FedAvg without GradCAM aggregation
python src/main.py --mode non_iid --ablation no_gradcam

# FedProx
python src/main.py --mode non_iid --algorithm fedprox

# Centralized baseline
python src/main.py --mode centralized

# Full experiment suite
bash run_all.sh
```

## Requirements

- Python 3.8+
- PyTorch, torchvision
- pandas, numpy, scikit-learn
- albumentations, opencv-python
- matplotlib, tqdm
- efficientnet-pytorch (or torchvision ≥0.13)

See `requirements.txt` for pinned versions.
