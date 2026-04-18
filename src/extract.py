"""
Feature Extraction Pipeline
=============================
Extracts image descriptors using two methods for comparison:

  Method A — Classical: SIFT keypoints + VLAD encoding
    - Scale/rotation invariant local descriptors
    - VLAD aggregates local SIFT descriptors into a fixed-size global vector
    - Fast, interpretable, no GPU needed

  Method B — Deep: DINOv2 ViT-S/14 global embedding
    - Self-supervised Vision Transformer (Meta AI)
    - State-of-the-art visual place recognition accuracy
    - Requires torch (CPU works, GPU is ~10x faster)

Outputs (saved to models/):
    sift_vlad_descriptors.npy   — (N, vlad_dim) float32 array
    sift_vlad_labels.npy        — (N,) string array of image_ids
    dinov2_descriptors.npy      — (N, 384) float32 array
    dinov2_labels.npy           — (N,) string array of image_ids
    sift_codebook.npy           — VLAD visual vocabulary (K centroids)

Usage:
    python src/extract.py                        # both methods, processed dir
    python src/extract.py --method sift          # SIFT only
    python src/extract.py --method dinov2        # DINOv2 only
    python src/extract.py --dir data/processed   # custom dir
    python src/extract.py --no-aug               # skip augmented images (originals only)
"""

import argparse
import logging
import pickle
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

PROCESSED_DIR = Path("data/processed")
MODELS_DIR    = Path("models")
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

# VLAD vocabulary size (number of visual words)
VLAD_K = 64

# DINOv2 model — ViT-S/14 is the smallest, good balance of speed vs accuracy
# Options: vits14, vitb14, vitl14, vitg14  (larger = more accurate, slower)
DINOV2_MODEL = "facebookresearch/dinov2"
DINOV2_VARIANT = "dinov2_vits14"
DINOV2_DIM = 384   # ViT-S embedding dimension


# ─── Image loading ────────────────────────────────────────────────────────────

try:
    from pillow_heif import register_heif_opener
    from PIL import Image as _PIL
    register_heif_opener()
    IMG_EXTS.add(".heic")
    IMG_EXTS.add(".heif")
except ImportError:
    pass


def load_image_rgb(path: Path) -> np.ndarray:
    """Load image as uint8 RGB numpy array."""
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def load_image_gray(path: Path) -> np.ndarray:
    """Load image as uint8 grayscale numpy array (for SIFT)."""
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        return np.array(Image.open(path).convert("L"))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return img


def collect_image_paths(data_dir: Path, skip_aug: bool) -> list[tuple[Path, str]]:
    """
    Walk data_dir/<location>/*.jpg and return (path, image_id) pairs.
    If skip_aug=True, only include images without 'aug_' in their filename.
    """
    entries = []
    manifest = data_dir / "manifest.csv"

    if manifest.exists():
        df = pd.read_csv(manifest)
        if skip_aug:
            df = df[~df["augmented"].astype(bool)]
        for _, row in df.iterrows():
            p = Path(row["path"])
            if p.exists() and p.suffix.lower() in IMG_EXTS:
                entries.append((p, row["image_id"]))
    else:
        # Fallback: walk directory
        for loc_dir in sorted(d for d in data_dir.iterdir() if d.is_dir()):
            for img_path in sorted(f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                if skip_aug and "aug_" in img_path.name:
                    continue
                image_id = f"{loc_dir.name}/{img_path.stem}"
                entries.append((img_path, image_id))

    logger.info(f"Found {len(entries)} images in {data_dir}")
    return entries


# ─────────────────────────────────────────────────────────────────────────────
# METHOD A: SIFT + VLAD
# ─────────────────────────────────────────────────────────────────────────────

def extract_sift_descriptors_raw(path: Path, sift) -> np.ndarray | None:
    """Extract raw SIFT local descriptors from one image. Returns (N, 128) or None."""
    img = load_image_gray(path)
    _, desc = sift.detectAndCompute(img, None)
    return desc  # None if no keypoints found


def build_vlad_vocabulary(entries: list[tuple[Path, str]], k: int) -> np.ndarray:
    """
    Build a visual vocabulary (k-means codebook) from SIFT descriptors
    sampled across the dataset. Returns codebook of shape (k, 128).
    """
    logger.info(f"Building VLAD vocabulary (k={k}) from SIFT descriptors...")
    sift = cv2.SIFT_create(nfeatures=500)
    all_descs = []

    for path, _ in tqdm(entries, desc="Sampling SIFT descriptors"):
        desc = extract_sift_descriptors_raw(path, sift)
        if desc is not None:
            # Sample up to 50 descriptors per image to keep memory manageable
            idx = np.random.choice(len(desc), min(50, len(desc)), replace=False)
            all_descs.append(desc[idx])

    all_descs = np.vstack(all_descs).astype(np.float32)
    logger.info(f"Running k-means on {len(all_descs)} descriptors...")

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, _, codebook = cv2.kmeans(
        all_descs, k, None, criteria,
        attempts=5, flags=cv2.KMEANS_PP_CENTERS
    )
    logger.info("Vocabulary built.")
    return codebook


def compute_vlad(descriptors: np.ndarray, codebook: np.ndarray) -> np.ndarray:
    """
    Compute VLAD encoding for a set of local descriptors.
    Returns a (k * 128,) float32 vector, L2-normalised.
    """
    k, d = codebook.shape
    vlad = np.zeros((k, d), dtype=np.float32)

    if descriptors is None or len(descriptors) == 0:
        return vlad.flatten()

    # Assign each descriptor to nearest visual word
    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
    matches = flann.knnMatch(descriptors.astype(np.float32), codebook, k=1)

    for i, match in enumerate(matches):
        if match:
            word_idx = match[0].trainIdx
            vlad[word_idx] += descriptors[i] - codebook[word_idx]

    # Intra-normalisation + L2 normalisation (improves retrieval quality)
    for i in range(k):
        norm = np.linalg.norm(vlad[i])
        if norm > 0:
            vlad[i] /= norm

    vlad = vlad.flatten()
    norm = np.linalg.norm(vlad)
    if norm > 0:
        vlad /= norm

    return vlad


def extract_sift_vlad(entries: list[tuple[Path, str]], save_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
    Full SIFT+VLAD pipeline over all images.
    Returns (descriptors array, labels list) and saves to disk.
    """
    codebook_path = save_dir / "sift_codebook.npy"

    # Build or load vocabulary
    if codebook_path.exists():
        logger.info(f"Loading existing VLAD codebook from {codebook_path}")
        codebook = np.load(codebook_path)
    else:
        codebook = build_vlad_vocabulary(entries, VLAD_K)
        np.save(codebook_path, codebook)
        logger.info(f"Codebook saved to {codebook_path}")

    sift = cv2.SIFT_create(nfeatures=500)
    descriptors, labels = [], []

    for path, image_id in tqdm(entries, desc="SIFT+VLAD extraction"):
        raw = extract_sift_descriptors_raw(path, sift)
        vlad_vec = compute_vlad(raw, codebook)
        descriptors.append(vlad_vec)
        labels.append(image_id)

    descriptors = np.array(descriptors, dtype=np.float32)
    labels = np.array(labels)

    np.save(save_dir / "sift_vlad_descriptors.npy", descriptors)
    np.save(save_dir / "sift_vlad_labels.npy",      labels)
    logger.info(f"SIFT+VLAD: saved {len(descriptors)} vectors of dim {descriptors.shape[1]}")

    return descriptors, list(labels)


# ─────────────────────────────────────────────────────────────────────────────
# METHOD B: DINOv2
# ─────────────────────────────────────────────────────────────────────────────

def load_dinov2_model():
    """Load DINOv2 ViT-S/14 from torch.hub. Downloads once, cached automatically."""
    try:
        import torch
    except ImportError:
        raise ImportError("torch is required for DINOv2. Run: pip install torch torchvision")

    logger.info(f"Loading {DINOV2_VARIANT} from torch.hub (downloads on first run)...")
    model = torch.hub.load(DINOV2_MODEL, DINOV2_VARIANT, pretrained=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device).eval()
    logger.info(f"DINOv2 loaded on {device}")
    return model, device


def get_dinov2_transform():
    """Standard ImageNet preprocessing transform for DINOv2."""
    from torchvision import transforms
    return transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])


def extract_dinov2(entries: list[tuple[Path, str]], save_dir: Path) -> tuple[np.ndarray, list[str]]:
    """
    Extract DINOv2 global embeddings for all images.
    Returns (descriptors array, labels list) and saves to disk.
    """
    import torch
    from PIL import Image

    model, device = load_dinov2_model()
    transform = get_dinov2_transform()
    descriptors, labels = [], []

    BATCH_SIZE = 32

    # Process in batches for efficiency
    batch_imgs, batch_ids = [], []

    def flush_batch():
        if not batch_imgs:
            return
        tensor = torch.stack(batch_imgs).to(device)
        with torch.no_grad():
            feats = model(tensor)           # (B, 384)
        descriptors.extend(feats.cpu().numpy())
        labels.extend(batch_ids)
        batch_imgs.clear()
        batch_ids.clear()

    for path, image_id in tqdm(entries, desc="DINOv2 extraction"):
        try:
            img = Image.open(path).convert("RGB")
            t = transform(img)
            batch_imgs.append(t)
            batch_ids.append(image_id)
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

        if len(batch_imgs) >= BATCH_SIZE:
            flush_batch()

    flush_batch()  # remaining

    descriptors = np.array(descriptors, dtype=np.float32)

    # L2 normalise (cosine similarity = dot product after normalisation)
    norms = np.linalg.norm(descriptors, axis=1, keepdims=True)
    descriptors = descriptors / np.maximum(norms, 1e-8)

    labels_arr = np.array(labels)
    np.save(save_dir / "dinov2_descriptors.npy", descriptors)
    np.save(save_dir / "dinov2_labels.npy",      labels_arr)
    logger.info(f"DINOv2: saved {len(descriptors)} vectors of dim {descriptors.shape[1]}")

    return descriptors, list(labels)


# ─── CLI ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Extract image features for retrieval")
    parser.add_argument("--method",  choices=["sift", "dinov2", "both"], default="both")
    parser.add_argument("--dir",     type=Path, default=PROCESSED_DIR)
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--no-aug",  action="store_true", default=False,
                        help="Only extract from original images (skip augmented variants)")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    entries = collect_image_paths(args.dir, skip_aug=args.no_aug)

    if not entries:
        logger.error(f"No images found in {args.dir}. Run preprocess.py first.")
        return

    if args.method in ("sift", "both"):
        logger.info("=== METHOD A: SIFT + VLAD ===")
        extract_sift_vlad(entries, args.out_dir)

    if args.method in ("dinov2", "both"):
        logger.info("=== METHOD B: DINOv2 ===")
        extract_dinov2(entries, args.out_dir)

    logger.info("Feature extraction complete. Next: python src/index.py")


if __name__ == "__main__":
    main()
