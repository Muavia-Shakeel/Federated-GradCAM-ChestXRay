#!/bin/bash
# ============================================================
# Full Pipeline Runner
# Runs: FedAvg → FedProx → Centralized Baseline → Ablation
# Usage: bash run_all.sh
# Log:   outputs/run_all.log
# ============================================================

set -e

PYTHON="/home/msi/DS_env/bin/python3"
PROJ="/home/msi/Desktop/Projects/A5_Implementation"
LOG="$PROJ/outputs/run_all.log"
CKPT="$PROJ/outputs/checkpoints"
METRICS="$PROJ/outputs/metrics"

cd "$PROJ/src"

mkdir -p "$PROJ/outputs/checkpoints" "$PROJ/outputs/metrics" \
         "$PROJ/outputs/plots"       "$PROJ/outputs/gradcam_maps"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1" | tee -a "$LOG"; }

log "=========================================="
log "Pipeline START"
log "GPU: $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader 2>/dev/null || echo 'N/A')"
log "=========================================="

# ----------------------------------------------------------
# STEP 1: Centralized Baseline (upper bound)
# ----------------------------------------------------------
log "STEP 1/4: Centralized baseline..."
$PYTHON centralized.py --epochs 20 --seed 42 2>&1 | tee -a "$LOG"
cp "$CKPT/best_centralized_model.pth" "$CKPT/best_centralized_model_backup.pth"
log "STEP 1 DONE → test_metrics_centralized.json"

# ----------------------------------------------------------
# STEP 2: FedAvg non-IID (main result)
# ----------------------------------------------------------
log "STEP 2/4: FedAvg non-IID (seed=42)..."
$PYTHON main.py --mode non_iid --algorithm fedavg --rounds 20 --clients 5 \
                --alpha 0.5 --seed 42 2>&1 | tee -a "$LOG"
cp "$CKPT/best_global_model.pth" "$CKPT/best_global_model_fedavg.pth"
log "STEP 2 DONE → test_metrics_non_iid_fedavg.json"

# ----------------------------------------------------------
# STEP 3: FedProx non-IID (comparison)
# ----------------------------------------------------------
log "STEP 3/4: FedProx non-IID (seed=42)..."
$PYTHON main.py --mode non_iid --algorithm fedprox --rounds 20 --clients 5 \
                --alpha 0.5 --seed 42 2>&1 | tee -a "$LOG"
cp "$CKPT/best_global_model.pth" "$CKPT/best_global_model_fedprox.pth"
log "STEP 3 DONE → test_metrics_non_iid_fedprox.json"

# ----------------------------------------------------------
# STEP 4: Ablation — FedAvg without GradCAM aggregation
# ----------------------------------------------------------
log "STEP 4/4: Ablation (no_gradcam, seed=42)..."
$PYTHON main.py --mode non_iid --algorithm fedavg --rounds 20 --clients 5 \
                --alpha 0.5 --seed 42 --ablation no_gradcam 2>&1 | tee -a "$LOG"
log "STEP 4 DONE → test_metrics_non_iid_fedavg_no_gradcam.json"

# ----------------------------------------------------------
# Summary
# ----------------------------------------------------------
log "=========================================="
log "ALL STEPS COMPLETE"
log "Results:"

$PYTHON - 2>&1 | tee -a "$LOG" << 'EOF'
import json, os, glob

metrics_dir = "../outputs/metrics"
files = {
    "Centralized":      "test_metrics_centralized.json",
    "FedAvg":           "test_metrics_non_iid_fedavg.json",
    "FedProx":          "test_metrics_non_iid_fedprox.json",
    "FedAvg (no GradCAM)": "test_metrics_non_iid_fedavg_no_gradcam.json",
}

print(f"\n{'Model':<22} {'AUC':>7} {'F1-macro':>9} {'Hamming':>9} {'Exact':>7}")
print("-" * 60)
for name, fn in files.items():
    path = os.path.join(metrics_dir, fn)
    if not os.path.exists(path):
        print(f"{name:<22}  (missing)")
        continue
    d = json.load(open(path))
    auc  = d.get("auc_roc_macro", 0)
    f1   = d.get("f1_macro", 0)
    ham  = d.get("hamming_score", d.get("accuracy", 0))
    exm  = d.get("exact_match_accuracy", 0)
    print(f"{name:<22} {auc:>7.4f} {f1:>9.4f} {ham:>9.4f} {exm:>7.4f}")
EOF

log "Full log: $LOG"
log "=========================================="
