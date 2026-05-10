import os

# Reproducibility
SEED = 42

# Dataset
DATA_DIR = os.path.join(os.path.dirname(__file__), "../data/raw")
PARTITION_DIR = os.path.join(os.path.dirname(__file__), "../data/partitions")
IMG_SIZE = 224
NUM_CLASSES = 14
PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

# Federated setup — increased scale per Sir's feedback
NUM_CLIENTS = 5          # simulated hospital clients (was 3)
PARTITION_MODE = "non_iid"  # "non_iid" or "iid"
ALPHA = 0.5              # Dirichlet concentration: lower = more non-IID

# Training — increased rounds for convergence
NUM_ROUNDS = 20          # (was 10)
LOCAL_EPOCHS = 3
BATCH_SIZE = 32
LEARNING_RATE = 1e-4
OPTIMIZER = "adam"
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
EARLY_STOPPING_PATIENCE = 5

# Model
MODEL_NAME = "efficientnet_b0"
PRETRAINED = True

# FedProx proximal regularization
FEDPROX_MU = 0.01        # proximal term weight (Li et al., 2020)

# Differential Privacy (DP-FedAvg)
DP_NOISE_MULTIPLIER = 1.0   # Gaussian σ multiplier
DP_MAX_GRAD_NORM = 1.0      # gradient/update clipping norm
DP_DELTA = 1e-5             # target δ for (ε, δ)-DP guarantee

# Stress testing — heterogeneity sweep
STRESS_ALPHAS = [0.1, 0.3, 0.5, 1.0]   # Dirichlet α values
STRESS_DROPOUT_RATE = 0.2               # fraction of clients that may drop per round

# Paths
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "../outputs/checkpoints")
GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "../outputs/gradcam_maps")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "../outputs/metrics")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "../outputs/plots")

# Device
DEVICE = "cuda"
