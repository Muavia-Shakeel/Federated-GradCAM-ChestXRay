#!/usr/bin/env bash
# Downloads NIH ChestX-ray14 SUBSET via Kaggle API
# Downloads: metadata CSVs + images_001 through images_003 (~8 GB, ~28k images)
# Enough for 3-client FL simulation. Full dataset = 45 GB / 112k images.
#
# To get more images, change BATCH_END below (max 12).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RAW_DIR="$SCRIPT_DIR/data/raw"
IMG_DIR="$RAW_DIR/images"
BATCH_START=1
BATCH_END=3   # change to 12 for full dataset

echo "=== NIH ChestX-ray14 Subset Download ==="
echo "Target   : $RAW_DIR"
echo "Batches  : images_00${BATCH_START} to images_00${BATCH_END} (~$((BATCH_END * 9))k images, ~$((BATCH_END * 3)) GB)"
echo "Disk free: $(df -h "$SCRIPT_DIR" | awk 'NR==2{print $4}')"
echo ""

mkdir -p "$IMG_DIR"

DATASET="nih-chest-xrays/data"

# ── Step 1: Metadata files ─────────────────────────────────────────────────
echo "[1/3] Downloading metadata CSVs..."

for FILE in "Data_Entry_2017.csv" "BBox_List_2017.csv" "train_val_list.txt" "test_list.txt"; do
    if [ -f "$RAW_DIR/$FILE" ]; then
        echo "  SKIP (exists): $FILE"
        continue
    fi
    echo "  Fetching: $FILE"
    kaggle datasets download \
        --dataset "$DATASET" \
        --file "$FILE" \
        --path "$RAW_DIR" 2>&1 | tail -2
    # Kaggle downloads single files as FILE.zip — unzip if needed
    if [ -f "$RAW_DIR/${FILE}.zip" ]; then
        unzip -q -o "$RAW_DIR/${FILE}.zip" -d "$RAW_DIR"
        rm -f "$RAW_DIR/${FILE}.zip"
    fi
done

echo ""

# ── Step 2: Image batches ──────────────────────────────────────────────────
echo "[2/3] Downloading image batches ${BATCH_START}–${BATCH_END}..."

for i in $(seq -f "%03g" $BATCH_START $BATCH_END); do
    ZIP_NAME="images_${i}.zip"
    ZIP_PATH="$RAW_DIR/$ZIP_NAME"

    # Count already-extracted images to skip completed batches
    EXISTING=$(find "$IMG_DIR" -name "*.png" 2>/dev/null | wc -l)

    if [ -d "$IMG_DIR" ] && [ "$EXISTING" -ge $(( (i - BATCH_START + 1) * 9000 )) ]; then
        echo "  SKIP (already extracted batch $i, ~$EXISTING images present)"
        continue
    fi

    echo ""
    echo "  ── Batch $i ──"
    kaggle datasets download \
        --dataset "$DATASET" \
        --file "$ZIP_NAME" \
        --path "$RAW_DIR"

    echo "  Extracting $ZIP_NAME..."
    unzip -q -o "$ZIP_PATH" -d "$RAW_DIR" &
    UNZIP_PID=$!
    while kill -0 $UNZIP_PID 2>/dev/null; do
        COUNT=$(find "$IMG_DIR" -name "*.png" 2>/dev/null | wc -l)
        printf "\r    Images extracted: %-7s" "$COUNT"
        sleep 1
    done
    wait $UNZIP_PID
    echo ""

    # Move images if unzipped into subdirectory
    for SUBDIR in "$RAW_DIR/images_${i}" "$RAW_DIR/images"; do
        if [ -d "$SUBDIR" ] && [ "$SUBDIR" != "$IMG_DIR" ]; then
            mv "$SUBDIR"/*.png "$IMG_DIR"/ 2>/dev/null || true
            rmdir "$SUBDIR" 2>/dev/null || true
        fi
    done

    rm -f "$ZIP_PATH"
    TOTAL=$(find "$IMG_DIR" -name "*.png" | wc -l)
    echo "    Batch $i done. Total images: $TOTAL"
done

echo ""

# ── Step 3: Verify ─────────────────────────────────────────────────────────
echo "[3/3] Verifying..."

if [ ! -f "$RAW_DIR/Data_Entry_2017.csv" ]; then
    echo "ERROR: Data_Entry_2017.csv missing."
    exit 1
fi

TOTAL_IMGS=$(find "$IMG_DIR" -name "*.png" | wc -l)
echo "  Data_Entry_2017.csv — OK"
echo "  BBox_List_2017.csv  — $([ -f "$RAW_DIR/BBox_List_2017.csv" ] && echo OK || echo MISSING)"
echo "  Images              — $TOTAL_IMGS PNG files"

if [ "$TOTAL_IMGS" -lt 5000 ]; then
    echo "WARNING: Very few images. Check download."
fi

echo ""
echo "=== Done ==="
echo "Run: cd src && python main.py --mode non_iid"
echo ""
echo "Note: Using ~$TOTAL_IMGS images (subset). For full 112k, set BATCH_END=12 and rerun."
