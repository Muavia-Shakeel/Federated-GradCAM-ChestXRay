"""
Generate publication-quality figures for the research paper.
Saves PNGs to outputs/plots/paper_figures/
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

os.makedirs("outputs/plots/paper_figures", exist_ok=True)

# ── Colorblind-safe palette (Wong 2011) ──────────────────────────────
BLUE   = "#0072B2"
ORANGE = "#E69F00"
GREEN  = "#009E73"
RED    = "#D55E00"
PURPLE = "#CC79A7"
LBLUE  = "#56B4E9"
GRAY   = "#999999"

plt.rcParams.update({
    "font.family":      "DejaVu Serif",
    "font.size":        11,
    "axes.titlesize":   12,
    "axes.labelsize":   11,
    "xtick.labelsize":  10,
    "ytick.labelsize":  10,
    "legend.fontsize":  10,
    "figure.dpi":       150,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "axes.grid":        True,
    "grid.alpha":       0.3,
    "grid.linestyle":   "--",
})

# ════════════════════════════════════════════════════════════════════
# Figure 1 — Overall Classification Performance Comparison
# ════════════════════════════════════════════════════════════════════
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

methods = ["Centralized\n(Oracle)", "FedAvg", "FedProx\n(μ=0.01)"]
auc     = [0.8261, 0.8125, 0.8115]
auc_err = [0.0,    0.004,  0.0]
f1      = [0.1316, 0.1679, 0.1106]
colors  = [GREEN, BLUE, ORANGE]

# — AUC-ROC bar chart
ax = axes[0]
bars = ax.bar(methods, auc, yerr=auc_err, capsize=5,
              color=colors, edgecolor="white", linewidth=0.8,
              error_kw={"elinewidth": 1.5, "ecolor": "#333333"})
ax.set_ylim(0.78, 0.845)
ax.set_ylabel("Macro-Averaged AUC-ROC")
ax.set_title("(a) AUC-ROC Performance", fontweight="bold")
for bar, val in zip(bars, auc):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
            f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
ax.axhline(auc[0], color=GREEN, linestyle=":", linewidth=1.2, alpha=0.6,
           label="Centralized oracle")
ax.legend(frameon=False, fontsize=9)

# — F1 bar chart
ax2 = axes[1]
bars2 = ax2.bar(methods, f1, color=colors, edgecolor="white", linewidth=0.8)
ax2.set_ylim(0.0, 0.22)
ax2.set_ylabel("Macro-Averaged F1 Score")
ax2.set_title("(b) Macro F1 Score", fontweight="bold")
for bar, val in zip(bars2, f1):
    ax2.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.002,
             f"{val:.4f}", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

fig.suptitle("Figure 1. Classification Performance: Centralized vs. Federated Methods",
             fontsize=12, fontweight="bold", y=1.01)
fig.tight_layout()
fig.savefig("outputs/plots/paper_figures/fig1_classification_performance.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("Figure 1 saved.")

# ════════════════════════════════════════════════════════════════════
# Figure 2 — GradCAM Consistency: FedAvg vs FedProx
# ════════════════════════════════════════════════════════════════════
strategies = ["Weighted", "Uniform", "Max-Pool", "Performance"]

pearson_fedavg   = [0.7619, 0.7453, 0.8210, 0.7453]
pearson_fedprox  = [0.9706, 0.9687, 0.9552, 0.9687]
ssim_fedavg      = [0.7633, 0.7458, 0.8203, 0.7458]
ssim_fedprox     = [0.9694, 0.9678, 0.9394, 0.9678]

x    = np.arange(len(strategies))
w    = 0.35

fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))

# — Pearson-r
ax = axes[0]
b1 = ax.bar(x - w/2, pearson_fedavg,  w, label="FedAvg",  color=BLUE,   edgecolor="white")
b2 = ax.bar(x + w/2, pearson_fedprox, w, label="FedProx", color=ORANGE, edgecolor="white")
ax.set_xticks(x)
ax.set_xticklabels(strategies)
ax.set_ylim(0.60, 1.02)
ax.set_ylabel("Pearson Correlation (r)")
ax.set_title("(a) Pearson-r with Centralized Oracle", fontweight="bold")
ax.legend(frameon=False)
# annotate top of bars
for bars in [b1, b2]:
    for bar in bars:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

# — SSIM
ax2 = axes[1]
b3 = ax2.bar(x - w/2, ssim_fedavg,  w, label="FedAvg",  color=BLUE,   edgecolor="white")
b4 = ax2.bar(x + w/2, ssim_fedprox, w, label="FedProx", color=ORANGE, edgecolor="white")
ax2.set_xticks(x)
ax2.set_xticklabels(strategies)
ax2.set_ylim(0.60, 1.02)
ax2.set_ylabel("Structural Similarity Index (SSIM)")
ax2.set_title("(b) SSIM with Centralized Oracle", fontweight="bold")
ax2.legend(frameon=False)
for bars in [b3, b4]:
    for bar in bars:
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                 f"{bar.get_height():.3f}", ha="center", va="bottom", fontsize=8.5)

fig.suptitle("Figure 2. GradCAM Explanation Consistency: FedAvg vs. FedProx\n"
             "Across Four Aggregation Strategies (Higher = More Consistent with Centralized Oracle)",
             fontsize=11.5, fontweight="bold", y=1.03)
fig.tight_layout()
fig.savefig("outputs/plots/paper_figures/fig2_gradcam_consistency.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("Figure 2 saved.")

# ════════════════════════════════════════════════════════════════════
# Figure 3 — Dirichlet α Robustness Sweep
# ════════════════════════════════════════════════════════════════════
alpha_vals = [0.1, 0.3, 0.5]
auc_alpha  = [0.8044, 0.8109, 0.8115]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
ax.plot(alpha_vals, auc_alpha, "o-", color=ORANGE, linewidth=2.2,
        markersize=9, markerfacecolor="white", markeredgewidth=2.5,
        label="FedProx AUC-ROC")

for xa, ya in zip(alpha_vals, auc_alpha):
    ax.annotate(f"{ya:.4f}", (xa, ya), textcoords="offset points",
                xytext=(0, 12), ha="center", fontsize=10, fontweight="bold", color=ORANGE)

ax.fill_between(alpha_vals, [0.795]*3, auc_alpha, alpha=0.12, color=ORANGE)
ax.set_xlabel("Dirichlet Concentration Parameter (α)\n← More Heterogeneous       More Homogeneous →",
              fontsize=10.5)
ax.set_ylabel("Macro-Averaged AUC-ROC")
ax.set_title("Figure 3. Robustness to Data Heterogeneity\n"
             "FedProx Performance Across Dirichlet α Levels", fontweight="bold")
ax.set_xlim(0.0, 0.65)
ax.set_ylim(0.795, 0.820)
ax.set_xticks(alpha_vals)
ax.set_xticklabels([f"α={a}" for a in alpha_vals])

# annotation arrows for heterogeneity labels
ax.annotate("Very High\nHeterogeneity", xy=(0.1, 0.8044), xytext=(0.17, 0.8055),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
            fontsize=9, color=GRAY, ha="left")
ax.annotate("Moderate\nHeterogeneity\n(Main Exp.)", xy=(0.5, 0.8115), xytext=(0.38, 0.8107),
            arrowprops=dict(arrowstyle="->", color=GRAY, lw=1.2),
            fontsize=9, color=GRAY, ha="right")

ax.legend(frameon=False)
fig.tight_layout()
fig.savefig("outputs/plots/paper_figures/fig3_alpha_sweep.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("Figure 3 saved.")

# ════════════════════════════════════════════════════════════════════
# Figure 4 — Per-Class AUC (FedAvg vs FedProx)
# ════════════════════════════════════════════════════════════════════
diseases = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural\nThickening", "Hernia"
]
auc_fedavg_cls = [
    0.8024, 0.8851, 0.8767, 0.7124, 0.8479, 0.7416, 0.7410,
    0.8760, 0.7816, 0.8886, 0.9168, 0.7915, 0.7970, 0.7165
]
auc_fedprox_cls = [
    0.7943, 0.8853, 0.8752, 0.7133, 0.8390, 0.7496, 0.7313,
    0.8835, 0.7720, 0.8905, 0.9178, 0.7799, 0.7972, 0.7102
]

y = np.arange(len(diseases))
fig, ax = plt.subplots(figsize=(10, 7))

ax.barh(y + 0.2, auc_fedavg_cls,  0.38, label="FedAvg",  color=BLUE,   alpha=0.85, edgecolor="white")
ax.barh(y - 0.2, auc_fedprox_cls, 0.38, label="FedProx", color=ORANGE, alpha=0.85, edgecolor="white")
ax.axvline(0.5,  color="#bbbbbb", linestyle="--", linewidth=1, label="AUC=0.5 (random)")
ax.axvline(0.80, color=GRAY,      linestyle=":",  linewidth=1, alpha=0.6, label="AUC=0.80")

ax.set_yticks(y)
ax.set_yticklabels(diseases, fontsize=10)
ax.set_xlabel("AUC-ROC")
ax.set_xlim(0.55, 0.97)
ax.set_title("Figure 4. Per-Class AUC-ROC: FedAvg vs. FedProx\n"
             "NIH ChestX-ray14 Test Set (14 Pathology Classes)", fontweight="bold")
ax.legend(frameon=False, loc="lower right")
ax.invert_yaxis()

fig.tight_layout()
fig.savefig("outputs/plots/paper_figures/fig4_perclass_auc.png",
            bbox_inches="tight", dpi=150)
plt.close()
print("Figure 4 saved.")

# ════════════════════════════════════════════════════════════════════
# Figure 5 — Training Convergence Curves
# ════════════════════════════════════════════════════════════════════
import json

def load_auc_history(path, key="val_auc_roc_macro"):
    with open(path) as f:
        h = json.load(f)
    rounds = list(range(1, len(h) + 1))
    aucs   = [r.get(key, r.get("auc_roc_macro", 0)) for r in h]
    return rounds, aucs

try:
    r_fa, a_fa = load_auc_history("outputs/metrics/history_non_iid_fedavg.json")
    r_fp, a_fp = load_auc_history("outputs/metrics/history_non_iid_fedprox.json")
    r_ct, a_ct = load_auc_history("outputs/metrics/history_centralized.json")

    fig, ax = plt.subplots(figsize=(8, 4.8))
    ax.plot(r_ct, a_ct, "-", color=GREEN,  linewidth=2.0, label="Centralized (Oracle)")
    ax.plot(r_fa, a_fa, "-", color=BLUE,   linewidth=2.0, label="FedAvg")
    ax.plot(r_fp, a_fp, "--",color=ORANGE, linewidth=2.0, label="FedProx (μ=0.01)")

    # early stop marker for FedAvg
    if len(r_fa) < 20:
        ax.axvline(r_fa[-1], color=BLUE, linestyle=":", linewidth=1.2, alpha=0.7)
        ax.text(r_fa[-1] + 0.2, min(a_fa) + 0.001, f"Early stop\n(Round {r_fa[-1]})",
                color=BLUE, fontsize=8.5)

    ax.set_xlabel("Federated Round / Epoch")
    ax.set_ylabel("Validation AUC-ROC (Macro)")
    ax.set_title("Figure 5. Training Convergence Curves\n"
                 "Validation AUC-ROC Over Training Rounds", fontweight="bold")
    ax.legend(frameon=False)
    fig.tight_layout()
    fig.savefig("outputs/plots/paper_figures/fig5_convergence.png",
                bbox_inches="tight", dpi=150)
    plt.close()
    print("Figure 5 saved.")
except Exception as e:
    print(f"Figure 5 skipped: {e}")

print("\nAll figures saved to outputs/plots/paper_figures/")
