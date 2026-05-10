"""
Non-IID partitioning via Dirichlet distribution (LDA).
Each client gets a different class distribution, simulating hospital specialization.
"""
import numpy as np
import pandas as pd

from config import PATHOLOGY_LABELS, SEED


def dirichlet_partition(
    df: pd.DataFrame,
    num_clients: int,
    alpha: float = 0.5,
) -> list[pd.DataFrame]:
    """
    Splits df into num_clients partitions using Dirichlet(alpha) over label distributions.
    Lower alpha = more heterogeneous (non-IID). alpha >= 100 ≈ IID.
    Uses primary label (first positive) for stratification.
    """
    rng = np.random.default_rng(SEED)

    # Assign primary label = first positive class.
    # All-zero rows (No Finding) → bin index 14 (one beyond PATHOLOGY_LABELS).
    # np.argmax was wrong: returns 0 for all-zero rows, misclassifying healthy
    # images as Atelectasis and skewing every client's distribution.
    label_matrix = df[PATHOLOGY_LABELS].values  # (N, 14)
    no_finding_mask = label_matrix.sum(axis=1) == 0
    primary = label_matrix.argmax(axis=1)
    primary[no_finding_mask] = len(PATHOLOGY_LABELS)  # 14 = "No Finding" bin

    client_indices = [[] for _ in range(num_clients)]

    for cls in range(len(PATHOLOGY_LABELS) + 1):  # +1 to include No Finding bin
        cls_idx = np.where(primary == cls)[0]
        if len(cls_idx) == 0:
            continue
        rng.shuffle(cls_idx)
        proportions = rng.dirichlet(np.repeat(alpha, num_clients))
        proportions = (proportions * len(cls_idx)).astype(int)
        # Fix rounding: assign remainder to largest client
        diff = len(cls_idx) - proportions.sum()
        proportions[proportions.argmax()] += diff

        start = 0
        for c, count in enumerate(proportions):
            client_indices[c].extend(cls_idx[start:start + count].tolist())
            start += count

    return [df.iloc[sorted(idxs)].reset_index(drop=True) for idxs in client_indices]


def iid_partition(df: pd.DataFrame, num_clients: int) -> list[pd.DataFrame]:
    df_shuffled = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
    chunks = np.array_split(df_shuffled, num_clients)
    return [c.reset_index(drop=True) for c in chunks]


def partition_data(
    df: pd.DataFrame,
    num_clients: int,
    mode: str = "non_iid",
    alpha: float = 0.5,
) -> list[pd.DataFrame]:
    if mode == "non_iid":
        return dirichlet_partition(df, num_clients, alpha)
    return iid_partition(df, num_clients)
