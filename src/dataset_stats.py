"""
Dataset Statistics & Validation
================================
Run after preprocessing to get a quick health check on your dataset:
  - Per-location image counts
  - Resolution distribution
  - Channel/brightness statistics
  - Detects near-duplicate images using perceptual hashing
  - Saves a summary report to results/dataset_report.txt

Usage:
    python src/dataset_stats.py                        # checks processed/
    python src/dataset_stats.py --dir data/raw         # checks raw/
    python src/dataset_stats.py --no-duplicates        # skip duplicate detection
"""

import argparse
import logging
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    from pillow_heif import register_heif_opener
    from PIL import Image as _PIL_Image
    register_heif_opener()
    _HEIC_SUPPORT = True
except ImportError:
    _HEIC_SUPPORT = False

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic", ".heif"}
RESULTS_DIR = Path("results")


# ─── Perceptual hash (pHash) ──────────────────────────────────────────────────

def phash(img_path: Path, hash_size: int = 16) -> np.ndarray:
    """Compute perceptual hash of an image (DCT-based)."""
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (hash_size * 4, hash_size * 4))
    img_f = np.float32(img)
    dct = cv2.dct(dct := cv2.dct(img_f))           # 2D DCT
    dct_low = dct[:hash_size, :hash_size]
    med = np.median(dct_low)
    return (dct_low > med).flatten()


def hamming(h1: np.ndarray, h2: np.ndarray) -> int:
    return int(np.sum(h1 != h2))


# ─── Main stats function ──────────────────────────────────────────────────────

def compute_stats(data_dir: Path, find_duplicates: bool) -> str:
    location_dirs = sorted([d for d in data_dir.iterdir() if d.is_dir()])
    if not location_dirs:
        location_dirs = [data_dir]

    all_records = []
    hash_map = {}            # path → phash

    lines = []
    lines.append("=" * 60)
    lines.append(f"Dataset Report: {data_dir}")
    lines.append("=" * 60)

    total_images = 0

    for loc_dir in location_dirs:
        location = loc_dir.name
        images = sorted([f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        total_images += len(images)

        heights, widths, means_r, means_g, means_b = [], [], [], [], []

        for img_path in tqdm(images, desc=f"Analysing {location}", leave=False):
            if img_path.suffix.lower() in {".heic", ".heif"} and _HEIC_SUPPORT:
                try:
                    pil_img = _PIL_Image.open(img_path).convert("RGB")
                    img_rgb = np.array(pil_img)
                    h, w = img_rgb.shape[:2]
                except Exception:
                    continue
            else:
                img = cv2.imread(str(img_path))
                if img is None:
                    continue
                h, w = img.shape[:2]
                img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            heights.append(h)
            widths.append(w)
            means_r.append(img_rgb[:, :, 0].mean())
            means_g.append(img_rgb[:, :, 1].mean())
            means_b.append(img_rgb[:, :, 2].mean())

            if find_duplicates:
                h_val = phash(img_path)
                if h_val is not None:
                    hash_map[img_path] = h_val

            all_records.append({
                "location": location,
                "filename": img_path.name,
                "height":   h,
                "width":    w,
                "mean_r":   img_rgb[:, :, 0].mean(),
                "mean_g":   img_rgb[:, :, 1].mean(),
                "mean_b":   img_rgb[:, :, 2].mean(),
                "brightness": img_rgb.mean(),
            })

        if heights:
            lines.append(f"\nLocation: {location}  ({len(images)} images)")
            lines.append(f"  Resolutions  : {min(widths)}x{min(heights)} to {max(widths)}x{max(heights)}")
            lines.append(f"  Avg size     : {np.mean(widths):.0f} x {np.mean(heights):.0f}")
            lines.append(f"  Brightness   : mean={np.mean(means_r + means_g + means_b):.1f}  "
                         f"std={np.std(means_r + means_g + means_b):.1f}")
            lines.append(f"  Channel means: R={np.mean(means_r):.1f}  G={np.mean(means_g):.1f}  B={np.mean(means_b):.1f}")

    # ─ Overall summary ─
    df = pd.DataFrame(all_records)
    lines.append("\n" + "─" * 60)
    lines.append("OVERALL SUMMARY")
    lines.append(f"  Total images      : {total_images}")
    lines.append(f"  Locations         : {len(location_dirs)}")
    if not df.empty:
        counts = df.groupby("location").size()
        lines.append(f"  Min per location  : {counts.min()} ({counts.idxmin()})")
        lines.append(f"  Max per location  : {counts.max()} ({counts.idxmax()})")
        lines.append(f"  Mean brightness   : {df['brightness'].mean():.1f}  (ideal: 100-160)")
        dark = df[df["brightness"] < 60]
        bright = df[df["brightness"] > 220]
        if len(dark):
            lines.append(f"  ⚠ Very dark images : {len(dark)} — consider recapturing")
        if len(bright):
            lines.append(f"  ⚠ Overexposed      : {len(bright)} — consider recapturing")

    # ─ Duplicate detection ─
    if find_duplicates and hash_map:
        lines.append("\n" + "─" * 60)
        lines.append("DUPLICATE DETECTION  (pHash, threshold ≤ 8 bits)")
        paths = list(hash_map.keys())
        hashes = list(hash_map.values())
        dup_pairs = []
        for i in range(len(paths)):
            for j in range(i + 1, len(paths)):
                dist = hamming(hashes[i], hashes[j])
                if dist <= 8:
                    dup_pairs.append((paths[i], paths[j], dist))
        if dup_pairs:
            lines.append(f"  Found {len(dup_pairs)} near-duplicate pair(s):")
            for p1, p2, d in dup_pairs[:20]:    # cap display at 20
                lines.append(f"    dist={d:2d}  {p1.name}  <->  {p2.name}")
            if len(dup_pairs) > 20:
                lines.append(f"    ... and {len(dup_pairs) - 20} more")
        else:
            lines.append("  No near-duplicates found.")

    report = "\n".join(lines)

    # Save CSV and report
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_DIR / "dataset_stats.csv", index=False)
    with open(RESULTS_DIR / "dataset_report.txt", "w", encoding="utf-8") as f:
        f.write(report)

    logger.info(f"Report saved to results/dataset_report.txt")
    return report


# ─── CLI ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Dataset statistics and validation")
    p.add_argument("--dir",           type=Path, default=Path("data/processed"))
    p.add_argument("--no-duplicates", action="store_true", default=False)
    args = p.parse_args()

    report = compute_stats(args.dir, find_duplicates=not args.no_duplicates)
    print(report)
