"""
FAISS Index Builder
====================
Builds fast approximate nearest-neighbour indexes from extracted descriptors.
One index per method (SIFT+VLAD and DINOv2).

Index types:
  - SIFT+VLAD  → IndexFlatL2   (exact L2 search — vectors are ~8K dim, fast enough)
  - DINOv2     → IndexFlatIP   (exact inner-product / cosine — vectors are L2-normalised)

For very large datasets (>100K images) swap to IndexIVFFlat for ~10x faster search.

Usage:
    python src/index.py              # build both indexes
    python src/index.py --method sift
    python src/index.py --method dinov2
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import faiss

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def build_index(descriptors: np.ndarray, metric: str) -> faiss.Index:
    """
    Build a FAISS flat index.
    metric: 'l2' for SIFT+VLAD, 'ip' (inner product) for DINOv2.
    """
    dim = descriptors.shape[1]
    if metric == "l2":
        index = faiss.IndexFlatL2(dim)
    else:
        index = faiss.IndexFlatIP(dim)

    index.add(descriptors.astype(np.float32))
    logger.info(f"Index built: {index.ntotal} vectors, dim={dim}, metric={metric}")
    return index


def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval indexes")
    parser.add_argument("--method",  choices=["sift", "dinov2", "both"], default="both")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    if args.method in ("sift", "both"):
        desc_path = args.out_dir / "sift_vlad_descriptors.npy"
        if not desc_path.exists():
            logger.error(f"{desc_path} not found. Run extract.py --method sift first.")
        else:
            descs = np.load(desc_path)
            index = build_index(descs, metric="l2")
            faiss.write_index(index, str(args.out_dir / "sift_vlad.index"))
            logger.info(f"Saved: models/sift_vlad.index")

    if args.method in ("dinov2", "both"):
        desc_path = args.out_dir / "dinov2_descriptors.npy"
        if not desc_path.exists():
            logger.error(f"{desc_path} not found. Run extract.py --method dinov2 first.")
        else:
            descs = np.load(desc_path)
            index = build_index(descs, metric="ip")
            faiss.write_index(index, str(args.out_dir / "dinov2.index"))
            logger.info(f"Saved: models/dinov2.index")

    logger.info("Indexing complete. Next: python src/retrieve.py --query <image_path>")


if __name__ == "__main__":
    main()
