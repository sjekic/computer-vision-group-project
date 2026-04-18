"""
Image Preprocessing Pipeline for IE Tower Visual Place Recognition
=================================================================
This script:
  1. Reads raw images organized by location (data/raw/<location>/<image>.jpg)
  2. Validates, resizes, normalizes and saves processed images
  3. Applies optional augmentations to increase intra-class variability
  4. Generates a dataset manifest (CSV) for downstream retrieval use

Expected raw structure:
    data/raw/
        <location_name>/
            img_001.jpg
            img_002.jpg
            ...

Output structure:
    data/processed/
        <location_name>/
            img_001.jpg     <- cleaned, resized original
            img_001_aug0.jpg  <- augmented variants (if --augment)
            ...
    data/processed/manifest.csv
"""

import os
import argparse
import logging
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from PIL import Image, ImageEnhance, ImageFilter
from tqdm import tqdm

# Register HEIC/HEIF support via pillow-heif (iPhone photos)
# Falls back gracefully if not installed
try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass  # pillow-heif not installed; HEIC files will be skipped

# ─── Configuration ────────────────────────────────────────────────────────────

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
QUERY_DIR = Path("data/query")

# Target size for all images fed into the retrieval system
TARGET_SIZE = (640, 480)          # (width, height)  — good for both SIFT & ViT

# ImageNet-style normalization stats (used at feature extraction, not saved to disk)
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Supported image extensions (HEIC/HEIF added for iPhone photos)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".heic", ".heif"}

# ─── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


# ─── Core helpers ─────────────────────────────────────────────────────────────

def is_valid_image(path: Path) -> bool:
    """Check that the file is a readable, non-corrupt image."""
    try:
        img = Image.open(path)
        img.verify()          # catches truncated / corrupt files
        return True
    except Exception as e:
        logger.warning(f"Skipping corrupt image {path}: {e}")
        return False


def load_image(path: Path) -> np.ndarray:
    """Load an image as RGB numpy array. Uses Pillow for HEIC, OpenCV for everything else."""
    suffix = path.suffix.lower()
    if suffix in {".heic", ".heif"}:
        # pillow-heif registered as a Pillow plugin, so Image.open works directly
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img)
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"cv2 could not read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img


def resize_and_pad(img: np.ndarray, target: tuple[int, int]) -> np.ndarray:
    """
    Resize while preserving aspect ratio, then pad to exact target size.
    Padding colour: mean grey (128, 128, 128) — neutral for most descriptors.
    """
    tw, th = target
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)

    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)

    # Centre-pad to exact target
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    y_off = (th - nh) // 2
    x_off = (tw - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


def normalize_for_model(img_uint8: np.ndarray) -> np.ndarray:
    """
    Return a float32 array normalised with ImageNet stats.
    Shape: (H, W, 3)  |  dtype: float32  |  range: approx [-2, 2]
    NOTE: this is returned for use in model inference; the saved files
    remain standard uint8 PNGs for storage efficiency.
    """
    img_f = img_uint8.astype(np.float32) / 255.0
    img_f = (img_f - IMAGENET_MEAN) / IMAGENET_STD
    return img_f


def equalize_histogram(img: np.ndarray) -> np.ndarray:
    """CLAHE on L channel (LAB) — improves contrast robustness across lighting."""
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)


def save_image(img: np.ndarray, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    cv2.imwrite(str(path), bgr, [cv2.IMWRITE_JPEG_QUALITY, 95])


# ─── Augmentation suite ───────────────────────────────────────────────────────

def augment(img: np.ndarray) -> list[tuple[str, np.ndarray]]:
    """
    Returns a list of (suffix, augmented_image) tuples.
    Augmentations simulate real-world query variability:
      - Brightness shifts   (morning vs evening light)
      - Horizontal flip     (mirrored viewpoint)
      - Slight rotation     (handheld camera tilt)
      - Gaussian blur       (motion / focus blur)
      - Saturation shift    (different phone camera profiles)
    """
    pil = Image.fromarray(img)
    results = []

    # 1. Brightness +30%
    bright = ImageEnhance.Brightness(pil).enhance(1.3)
    results.append(("aug_bright", np.array(bright)))

    # 2. Brightness -30%
    dark = ImageEnhance.Brightness(pil).enhance(0.7)
    results.append(("aug_dark", np.array(dark)))

    # 3. Horizontal flip
    flipped = np.fliplr(img)
    results.append(("aug_flip", flipped))

    # 4. Slight CW rotation (5°)
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 5, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    results.append(("aug_rot5", rotated))

    # 5. Gaussian blur (simulates motion)
    blurred = np.array(pil.filter(ImageFilter.GaussianBlur(radius=2)))
    results.append(("aug_blur", blurred))

    # 6. Low saturation (greyscale-ish — cloudy day)
    desat = ImageEnhance.Color(pil).enhance(0.4)
    results.append(("aug_desat", np.array(desat)))

    return results


# ─── Main pipeline ────────────────────────────────────────────────────────────

def process_dataset(
    raw_dir: Path,
    out_dir: Path,
    target_size: tuple[int, int],
    apply_clahe: bool,
    apply_augment: bool,
    dry_run: bool,
) -> pd.DataFrame:
    """
    Walk raw_dir/<location>/*.img, preprocess, optionally augment,
    and write to out_dir/<location>/*.jpg.
    Returns a manifest DataFrame.
    """
    records = []
    skipped = 0

    location_dirs = sorted([d for d in raw_dir.iterdir() if d.is_dir()])
    if not location_dirs:
        # Flat dataset — treat all images as one "unknown" class
        location_dirs = [raw_dir]
        logger.warning(
            "No sub-directories found in raw_dir. "
            "Place images in data/raw/<location_name>/ for best results."
        )

    logger.info(f"Found {len(location_dirs)} location(s): {[d.name for d in location_dirs]}")

    for loc_dir in tqdm(location_dirs, desc="Locations"):
        location = loc_dir.name
        images = sorted([f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS])
        logger.info(f"  {location}: {len(images)} images")

        for img_path in tqdm(images, desc=f"  {location}", leave=False):
            if not is_valid_image(img_path):
                skipped += 1
                continue

            try:
                img = load_image(img_path)
            except Exception as e:
                logger.warning(f"Load error {img_path}: {e}")
                skipped += 1
                continue

            # Resize + optional CLAHE
            img = resize_and_pad(img, target_size)
            if apply_clahe:
                img = equalize_histogram(img)

            stem = img_path.stem
            out_path = out_dir / location / f"{stem}.jpg"

            if not dry_run:
                save_image(img, out_path)

            records.append({
                "image_id":    f"{location}/{stem}",
                "location":    location,
                "filename":    out_path.name,
                "path":        str(out_path),
                "original":    str(img_path),
                "width":       target_size[0],
                "height":      target_size[1],
                "augmented":   False,
                "aug_type":    "original",
            })

            # Augmentations
            if apply_augment:
                for aug_suffix, aug_img in augment(img):
                    aug_path = out_dir / location / f"{stem}_{aug_suffix}.jpg"
                    if not dry_run:
                        save_image(aug_img, aug_path)
                    records.append({
                        "image_id":  f"{location}/{stem}_{aug_suffix}",
                        "location":  location,
                        "filename":  aug_path.name,
                        "path":      str(aug_path),
                        "original":  str(img_path),
                        "width":     target_size[0],
                        "height":    target_size[1],
                        "augmented": True,
                        "aug_type":  aug_suffix,
                    })

    df = pd.DataFrame(records)
    if not dry_run and not df.empty:
        manifest_path = out_dir / "manifest.csv"
        df.to_csv(manifest_path, index=False)
        logger.info(f"Manifest saved → {manifest_path}  ({len(df)} entries, {skipped} skipped)")

    return df


# ─── CLI ──────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Preprocess IE Tower image dataset")
    p.add_argument("--raw-dir",    type=Path, default=RAW_DIR,       help="Root folder with raw images")
    p.add_argument("--out-dir",    type=Path, default=PROCESSED_DIR, help="Output folder for processed images")
    p.add_argument("--size",       type=int,  nargs=2, default=list(TARGET_SIZE), metavar=("W", "H"))
    p.add_argument("--clahe",      action="store_true", default=True,  help="Apply CLAHE histogram equalisation")
    p.add_argument("--no-clahe",   dest="clahe", action="store_false")
    p.add_argument("--augment",    action="store_true", default=False, help="Generate augmented variants")
    p.add_argument("--dry-run",    action="store_true", default=False, help="Run without writing files")
    p.add_argument("--query-mode", action="store_true", default=False,
                   help="Process images from --raw-dir as query images (no augmentation, saved to data/query/)")
    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()

    out = QUERY_DIR if args.query_mode else args.out_dir
    df = process_dataset(
        raw_dir=args.raw_dir,
        out_dir=out,
        target_size=tuple(args.size),
        apply_clahe=args.clahe,
        apply_augment=(args.augment and not args.query_mode),
        dry_run=args.dry_run,
    )

    print(f"\n{'DRY RUN — ' if args.dry_run else ''}Summary")
    print(f"  Total images   : {len(df)}")
    print(f"  Locations      : {df['location'].nunique() if not df.empty else 0}")
    print(f"  Augmented      : {df['augmented'].sum() if not df.empty else 0}")
