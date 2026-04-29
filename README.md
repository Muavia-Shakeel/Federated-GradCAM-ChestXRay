# Federated Learning with Weighted GradCAM Aggregation for Explainable Chest X-Ray Diagnosis

## 📌 Overview
This repository contains the implementation of a novel Federated Learning (FL) framework designed for multi-label chest pathology classification. The project addresses the critical intersection of **patient data privacy** and **clinical explainability**. We implement a system where multiple simulated hospital clients collaboratively train an `EfficientNet-B0` model on the **NIH ChestX-ray14 dataset** without sharing raw images, while simultaneously producing aggregated global visual explanations (GradCAM).

## 🛑 Problem Statement
Current federated learning models for multi-site medical image classification produce predictions without clinically usable explanations. Although FL preserves patient privacy by keeping raw imaging data at each hospital, the resulting global model operates as a black box. Clinicians receive a diagnostic prediction with no indication of which anatomical regions drove the decision. This leaves federated medical AI systems technically privacy-compliant but clinically undeployable due to lack of trust.

## 🔍 Research Gap
1. **No Global Explanation:** Existing federated XAI approaches mostly generate and evaluate maps locally. There is no principled method to produce a coherent global explanation reflecting cross-institution variability.
2. **Unfair Weighting:** Current explanation aggregations ignore the massive imbalance in dataset sizes across clinical institutions. A rural clinic with 200 scans should not carry the same explanatory weight as a tertiary hospital with 12,000.
3. **Non-IID Impact Unknown:** Non-IID data distributions (realistic clinical scenarios) are rarely the focus of ablation studies in federated XAI work, meaning their effect on explanation quality is largely unknown.

## 🚀 Novelty & Proposed Solution
Our framework introduces **Dataset-Size-Weighted GradCAM Aggregation**. 
Instead of only aggregating model weights, our Central Aggregation Server securely receives local GradCAM saliency maps from each hospital. It then computes a weighted average of these maps—proportional to each client's dataset size. 

**Key Contributions:**
* **Federated XAI:** Generates a Global Explanation Map reflecting collective attention without transferring patient data.
* **Size-Proportional Fairness:** Both model weights (`FedAvg`) and GradCAM maps are weighted by the participating hospital's local dataset size.
* **Non-IID Robustness:** Tested across heavily imbalanced (Dirichlet $\alpha=0.5$) non-IID data distributions across clients.

## 🏗️ Methodology
* **Dataset:** NIH ChestX-ray14 (112,120 frontal-view X-ray images, 14 pathology labels).
* **Data Partitioning:** Non-IID Dirichlet partitioning to simulate 3 specialized hospital clients.
* **Local Training:** `EfficientNet-B0` CNN trained locally for fixed epochs.
* **Aggregation Strategy:** 
  * Weights: Standard `FedAvg`.
  * XAI: Dataset-size-weighted average of `conv_head` layer activations.

## 📂 Repository Structure
```text
.
├── data/
│   ├── raw/                # Extracted PNG images and metadata CSVs (Ignored in Git)
│   └── partitions/         # Partition logic/outputs
├── notebooks/              # Jupyter notebooks for EDA and prototyping
├── outputs/
│   ├── checkpoints/        # Saved model weights (.pth)
│   ├── gradcam_maps/       # Generated heatmaps (.npy arrays)
│   ├── metrics/            # Evaluation metrics (.json)
│   └── plots/              # Loss curves, ROC curves, and GradCAM visualizations
└── src/
    ├── config.py           # Hyperparameters and global settings
    ├── dataset.py          # PyTorch Dataset and DataLoader definitions
    ├── fedavg.py           # Federated weight aggregation logic
    ├── gradcam_aggregation.py # Novel weighted GradCAM aggregation logic
    ├── main.py             # Main training orchestration script
    ├── metrics.py          # AUC, F1, Accuracy computations
    ├── model.py            # EfficientNet-B0 initialization
    ├── partition.py        # Dirichlet non-IID and IID splitting logic
    ├── train_client.py     # Local client training and evaluation loops
    └── visualize.py        # Plotting utilities
```

## 📊 Current Results (Round 10/10)
After 10 Federated Rounds across 3 simulated Non-IID hospital clients, the model achieved the following performance on a held-out global test set (10% of total data):

* **AUC-ROC (Macro):** `0.8146` (Strong discriminatory capability across 14 pathologies)
* **Accuracy:** `94.93%` (High overall pixel/class accuracy)
* **F1-Score (Macro):** `0.1415` (Reflects expected performance drops on highly imbalanced rare classes)
* **Precision (Macro):** `0.3759`
* **Recall (Macro):** `0.0945`

### Generated Artifacts:
- Visual explanation heatmaps (GradCAM) successfully generated and aggregated proportionally across clients per round.
- Loss and ROC curves plotted and saved in `outputs/plots/`.

*Note: Ablation studies (comparing performance w/o GradCAM aggregation) and Centralized Baseline comparisons are currently underway.*

## ⚙️ How to Run
1. Ensure the dataset is extracted into `data/raw/` along with metadata files.
2. Install dependencies: `pip install torch torchvision pandas numpy albumentations opencv-python scikit-learn matplotlib tqdm`
3. Run the main pipeline:
```bash
# Standard Non-IID federated training with GradCAM aggregation
python src/main.py --mode non_iid --ablation none

# Ablation study: training without GradCAM aggregation
python src/main.py --mode non_iid --ablation no_gradcam
```
