"""
Generates a synthetic NIH ChestX-ray14 compatible dataset for pipeline testing.
Creates N grayscale 1024x1024 PNG images + Data_Entry_2017.csv + BBox_List_2017.csv
with correct format — drop-in replacement for real data.

Usage:
    source ~/DS_env/bin/activate
    python generate_synthetic_data.py --n 3000
"""
import os
import argparse
import random
import numpy as np
import pandas as pd
from PIL import Image
from tqdm import tqdm

PATHOLOGY_LABELS = [
    "Atelectasis", "Cardiomegaly", "Effusion", "Infiltration", "Mass",
    "Nodule", "Pneumonia", "Pneumothorax", "Consolidation", "Edema",
    "Emphysema", "Fibrosis", "Pleural_Thickening", "Hernia"
]

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RAW_DIR = os.path.join(SCRIPT_DIR, "data", "raw")
IMG_DIR = os.path.join(RAW_DIR, "images")


def generate_chest_xray(seed: int, size: int = 224) -> np.ndarray:
    """Synthetic chest X-ray: dark background, bright oval lung fields, noise."""
    rng = np.random.default_rng(seed)
    img = np.zeros((size, size), dtype=np.float32)

    cx, cy = size // 2, size // 2

    # Two lung ovals
    for ox in [cx - size // 6, cx + size // 6]:
        for y in range(size):
            for x in range(size):
                if ((x - ox) / (size // 8)) ** 2 + ((y - cy) / (size // 3)) ** 2 < 1:
                    img[y, x] = rng.uniform(0.4, 0.85)

    # Spine (bright vertical bar center)
    spine_w = size // 20
    x1, x2 = cx - spine_w // 2, cx - spine_w // 2 + spine_w
    img[:, x1:x2] = rng.uniform(0.6, 0.9, (size, x2 - x1))

    # Gaussian noise
    img += rng.normal(0, 0.05, (size, size))
    img = np.clip(img, 0, 1)
    return (img * 255).astype(np.uint8)


def random_finding_labels(rng: np.random.Generator) -> str:
    """Randomly assign 0–3 pathology labels, or 'No Finding'."""
    n = rng.integers(0, 4)
    if n == 0:
        return "No Finding"
    chosen = rng.choice(PATHOLOGY_LABELS, size=n, replace=False).tolist()
    return "|".join(chosen)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n", type=int, default=3000, help="Number of synthetic images")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    rng = np.random.default_rng(args.seed)
    os.makedirs(IMG_DIR, exist_ok=True)

    print(f"Generating {args.n} synthetic chest X-rays → {IMG_DIR}")
    rows = []

    for i in tqdm(range(1, args.n + 1), unit="img"):
        patient_id = i
        img_name = f"{patient_id:08d}_000.png"
        img_path = os.path.join(IMG_DIR, img_name)

        if not os.path.exists(img_path):
            arr = generate_chest_xray(seed=args.seed + i)
            Image.fromarray(arr, mode="L").convert("RGB").save(img_path)

        finding_labels = random_finding_labels(rng)
        rows.append({
            "Image Index": img_name,
            "Finding Labels": finding_labels,
            "Follow-up #": 0,
            "Patient ID": patient_id,
            "Patient Age": int(rng.integers(20, 85)),
            "Patient Gender": rng.choice(["M", "F"]),
            "View Position": "PA",
            "OriginalImage[Width": 1024,
            "Height]": 1024,
            "OriginalImagePixelSpacing[x": 0.143,
            "y]": 0.143,
        })

    # Write Data_Entry_2017.csv
    df = pd.DataFrame(rows)
    for label in PATHOLOGY_LABELS:
        df[label] = df["Finding Labels"].apply(lambda x: 1 if label in x else 0)

    csv_path = os.path.join(RAW_DIR, "Data_Entry_2017.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved: {csv_path} ({len(df)} rows)")

    # Write minimal BBox_List_2017.csv (subset with fake bboxes for 20% of images)
    bbox_rows = []
    sample_idx = rng.choice(len(df), size=len(df) // 5, replace=False)
    for idx in sample_idx:
        row = df.iloc[idx]
        labels_present = [l for l in PATHOLOGY_LABELS if row[l] == 1]
        if not labels_present:
            continue
        label = rng.choice(labels_present)
        # Random bounding box (x, y, w, h) — in 1024x1024 space
        x = int(rng.integers(100, 700))
        y = int(rng.integers(100, 700))
        w = int(rng.integers(100, 300))
        h = int(rng.integers(100, 300))
        bbox_rows.append({
            "Image Index": row["Image Index"],
            "Finding Label": label,
            "Bbox [x": x,
            "y": y,
            "w": w,
            "h]": h,
        })

    bbox_df = pd.DataFrame(bbox_rows)
    bbox_path = os.path.join(RAW_DIR, "BBox_List_2017.csv")
    bbox_df.to_csv(bbox_path, index=False)
    print(f"Saved: {bbox_path} ({len(bbox_df)} rows)")

    print(f"\nDone. {args.n} synthetic images ready.")
    print(f"Run training: cd src && python main.py --mode non_iid")


if __name__ == "__main__":
    main()
