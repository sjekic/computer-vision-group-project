"""
train_test_split.py
====================
Creates a reproducible, stratified train/test split of the processed image
dataset and writes two manifest CSVs:

    data/splits/train.csv   — database (index) images
    data/splits/test.csv    — query images for evaluation

The split is *per location*: each location contributes a configurable
fraction of its images to the test set (default 20 %).  Using a per-location
stratified split ensures every location is represented in both the database
and the query set, which is essential for fair place-recognition evaluation.

Augmented variants are NEVER placed in the test set — they stay in the
database only when --include-aug is passed.

Usage::

    # Default 80/20 split, originals only
    python src/train_test_split.py

    # Custom ratio
    python src/train_test_split.py --test-ratio 0.25

    # Include augmented images in the database (train) split only
    python src/train_test_split.py --include-aug

    # Reproduce a specific run
    python src/train_test_split.py --seed 123

    # Use a different processed directory
    python src/train_test_split.py --dir data/processed --out-dir data/splits
"""

import argparse
import logging
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
SPLITS_DIR    = Path("data/splits")
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}


# ─── Helpers ────────────────────────────────────────────────────────────────

def _is_augmented(name: str) -> bool:
    """Detect augmented images by filename prefix convention (aug_*)."""
    return name.startswith("aug_")


def collect_originals(data_dir: Path) -> dict[str, list[dict]]:
    """
    Walk data_dir/<location>/*.{jpg,png,...} and collect *original* images
    grouped by location.

    Returns a dict  {location_name: [{'path': ..., 'image_id': ...}, ...]}
    """
    manifest_path = data_dir / "manifest.csv"
    groups: dict[str, list[dict]] = defaultdict(list)

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            aug = str(row.get("augmented", "false")).strip().lower()
            if aug in {"1", "true", "yes", "y"}:
                continue  # skip augmented — handled separately
            p = Path(row["path"])
            if p.exists() and p.suffix.lower() in IMG_EXTS:
                loc = row.get("location", p.parent.name)
                groups[loc].append({"path": str(p), "image_id": row["image_id"], "location": loc})
    else:
        # Fallback: directory walk
        for loc_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
            for img_path in sorted(f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                if _is_augmented(img_path.name):
                    continue
                image_id = f"{loc_dir.name}/{img_path.stem}"
                groups[loc_dir.name].append({
                    "path": str(img_path),
                    "image_id": image_id,
                    "location": loc_dir.name,
                })

    return dict(groups)


def collect_augmented(data_dir: Path) -> list[dict]:
    """Collect augmented images (for optional inclusion in the train/database set)."""
    manifest_path = data_dir / "manifest.csv"
    aug_rows = []

    if manifest_path.exists():
        df = pd.read_csv(manifest_path)
        for _, row in df.iterrows():
            aug = str(row.get("augmented", "false")).strip().lower()
            if aug in {"1", "true", "yes", "y"}:
                p = Path(row["path"])
                if p.exists() and p.suffix.lower() in IMG_EXTS:
                    loc = row.get("location", p.parent.name)
                    aug_rows.append({"path": str(p), "image_id": row["image_id"], "location": loc,
                                     "augmented": True})
    else:
        for loc_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
            for img_path in sorted(f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                if _is_augmented(img_path.name):
                    image_id = f"{loc_dir.name}/{img_path.stem}"
                    aug_rows.append({"path": str(img_path), "image_id": image_id,
                                     "location": loc_dir.name, "augmented": True})
    return aug_rows


# ─── Core split logic ─────────────────────────────────────────────────────────

def stratified_split(
    groups: dict[str, list[dict]],
    test_ratio: float,
    rng: random.Random,
) -> tuple[list[dict], list[dict]]:
    """
    For each location, reserve *ceil(n * test_ratio)* images as test queries
    (minimum 1 if the location has >= 2 images) and the rest go to train/database.

    Returns (train_records, test_records).
    """
    import math

    train_records: list[dict] = []
    test_records:  list[dict] = []

    for location, records in sorted(groups.items()):
        shuffled = records.copy()
        rng.shuffle(shuffled)

        n = len(shuffled)
        n_test = max(1, math.ceil(n * test_ratio)) if n >= 2 else 0

        if n_test == 0:
            logger.warning(f"Location '{location}' has only 1 image — placing it in train only.")
            train_records.extend(shuffled)
        elif n_test >= n:
            # Safety guard: always keep at least 1 image in train
            n_test = n - 1

        test_records.extend(shuffled[:n_test])
        train_records.extend(shuffled[n_test:])

    return train_records, test_records


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Stratified train/test split for image retrieval evaluation"
    )
    parser.add_argument("--dir",         type=Path, default=PROCESSED_DIR,
                        help="Root of processed images (default: data/processed)")
    parser.add_argument("--out-dir",     type=Path, default=SPLITS_DIR,
                        help="Output directory for split CSVs (default: data/splits)")
    parser.add_argument("--test-ratio",  type=float, default=0.2,
                        help="Fraction of each location's images reserved for test (default: 0.2)")
    parser.add_argument("--include-aug", action="store_true", default=False,
                        help="Add augmented images to the train (database) split")
    parser.add_argument("--seed",        type=int, default=42,
                        help="Random seed for reproducibility (default: 42)")
    args = parser.parse_args()

    if not (0.0 < args.test_ratio < 1.0):
        parser.error("--test-ratio must be strictly between 0 and 1")

    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    args.out_dir.mkdir(parents=True, exist_ok=True)

    # ---- Collect originals grouped by location ----------------------------
    logger.info(f"Scanning {args.dir} for original images...")
    groups = collect_originals(args.dir)

    if not groups:
        logger.error(f"No images found under {args.dir}. Run preprocess.py first.")
        return

    n_locations = len(groups)
    n_total = sum(len(v) for v in groups.values())
    logger.info(f"Found {n_total} original images across {n_locations} locations.")

    # ---- Stratified split ------------------------------------------------
    train_records, test_records = stratified_split(groups, args.test_ratio, rng)

    # ---- Optionally add augmented images to train ------------------------
    if args.include_aug:
        aug_records = collect_augmented(args.dir)
        logger.info(f"Adding {len(aug_records)} augmented images to train split.")
        train_records.extend(aug_records)

    # ---- Save CSVs -------------------------------------------------------
    train_df = pd.DataFrame(train_records)
    test_df  = pd.DataFrame(test_records)

    # Ensure consistent column order
    for df in (train_df, test_df):
        for col in ("augmented",):
            if col not in df.columns:
                df[col] = False

    train_path = args.out_dir / "train.csv"
    test_path  = args.out_dir / "test.csv"

    train_df.to_csv(train_path, index=False)
    test_df.to_csv(test_path,   index=False)

    # ---- Summary ---------------------------------------------------------
    logger.info("")
    logger.info("Split summary")
    logger.info(f"  Seed            : {args.seed}")
    logger.info(f"  Test ratio      : {args.test_ratio:.0%}")
    logger.info(f"  Train (database): {len(train_df)} images  →  {train_path}")
    logger.info(f"  Test  (queries) : {len(test_df)} images   →  {test_path}")
    logger.info("")
    logger.info("Per-location breakdown:")
    logger.info(f"  {'Location':<30}  {'Train':>5}  {'Test':>5}  {'Total':>5}")
    logger.info(f"  {'-'*30}  {'-----':>5}  {'----':>5}  {'-----':>5}")

    train_counts = train_df[train_df.get("augmented", False) != True]["location"].value_counts()
    test_counts  = test_df["location"].value_counts()
    all_locations = sorted(set(train_counts.index) | set(test_counts.index))

    for loc in all_locations:
        tr = train_counts.get(loc, 0)
        te = test_counts.get(loc, 0)
        logger.info(f"  {loc:<30}  {tr:>5}  {te:>5}  {tr+te:>5}")

    logger.info("")
    logger.info("Next steps:")
    logger.info("  1. Build index from train split:")
    logger.info("       python src/extract.py --split-csv data/splits/train.csv")
    logger.info("  2. Evaluate on test split:")
    logger.info("       python src/retrieve.py --query-dir data/splits/test.csv")


if __name__ == "__main__":
    main()
