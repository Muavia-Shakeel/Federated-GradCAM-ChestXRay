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

# Federated setup
NUM_CLIENTS = 3          # simulated hospital clients
PARTITION_MODE = "non_iid"  # "non_iid" or "iid"
ALPHA = 0.5              # Dirichlet concentration: lower = more non-IID

# Training
NUM_ROUNDS = 10
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

# Paths
CHECKPOINT_DIR = os.path.join(os.path.dirname(__file__), "../outputs/checkpoints")
GRADCAM_DIR = os.path.join(os.path.dirname(__file__), "../outputs/gradcam_maps")
METRICS_DIR = os.path.join(os.path.dirname(__file__), "../outputs/metrics")
PLOTS_DIR = os.path.join(os.path.dirname(__file__), "../outputs/plots")

# Device
DEVICE = "cuda"
