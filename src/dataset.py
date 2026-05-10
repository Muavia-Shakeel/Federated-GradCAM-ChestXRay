import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset, DataLoader
import albumentations as A
from albumentations.pytorch import ToTensorV2

from config import PATHOLOGY_LABELS, IMG_SIZE, SEED


def get_transforms(split: str = "train") -> A.Compose:
    if split == "train":
        return A.Compose([
            A.Resize(IMG_SIZE, IMG_SIZE),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.4),
            A.Affine(translate_percent=0.05, scale=(0.9, 1.1), rotate=(-10, 10), p=0.3),
            A.GaussNoise(p=0.3),
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8), p=0.3),
            A.CoarseDropout(num_holes_range=(1, 8), hole_height_range=(8, 16),
                            hole_width_range=(8, 16), p=0.2),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    return A.Compose([
        A.Resize(IMG_SIZE, IMG_SIZE),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])


class ChestXrayDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_dir: str, split: str = "train"):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir
        self.transform = get_transforms(split)
        self.labels = PATHOLOGY_LABELS

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.img_dir, row["Image Index"])
        image = np.array(Image.open(img_path).convert("RGB"))
        label = torch.tensor(row[self.labels].values.astype(np.float32))
        augmented = self.transform(image=image)
        return augmented["image"], label


def build_client_loaders(
    partition_dfs: list[pd.DataFrame],
    img_dir: str,
    batch_size: int = 32,
    val_ratio: float = 0.15,
) -> list[dict]:
    """
    Returns list of dicts with 'train' and 'val' DataLoaders per client.
    """
    client_loaders = []

    for df in partition_dfs:
        df = df.sample(frac=1, random_state=SEED).reset_index(drop=True)
        split_idx = int(len(df) * (1 - val_ratio))
        train_df = df.iloc[:split_idx]
        val_df = df.iloc[split_idx:]

        train_ds = ChestXrayDataset(train_df, img_dir, split="train")
        val_ds = ChestXrayDataset(val_df, img_dir, split="val")

        client_loaders.append({
            "train": DataLoader(train_ds, batch_size=batch_size, shuffle=True,
                                num_workers=4, pin_memory=True),
            "val": DataLoader(val_ds, batch_size=batch_size, shuffle=False,
                              num_workers=4, pin_memory=True),
            "n_train": len(train_ds),
            "n_val": len(val_ds),
        })

    return client_loaders
