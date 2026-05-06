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
import json
import logging
from pathlib import Path

import numpy as np
import faiss

from retrieval_config import (
    descriptor_filename,
    index_filename,
    label_filename,
    method_display_name,
    method_metric,
    resolve_methods,
)

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")


def build_index(
    descriptors: np.ndarray,
    metric: str,
    index_type: str = "flat",
    nlist: int = 64,
    hnsw_m: int = 32,
    nprobe: int = 10,
) -> faiss.Index:
    """
    Build a FAISS index.
    metric: 'l2' for SIFT+VLAD, 'ip' (inner product) for DINOv2.
    """
    dim = descriptors.shape[1]
    descs = descriptors.astype(np.float32)
    faiss_metric = faiss.METRIC_L2 if metric == "l2" else faiss.METRIC_INNER_PRODUCT

    if index_type == "flat" and metric == "l2":
        index = faiss.IndexFlatL2(dim)
    elif index_type == "flat":
        index = faiss.IndexFlatIP(dim)
    elif index_type == "hnsw":
        index = faiss.IndexHNSWFlat(dim, hnsw_m, faiss_metric)
        index.hnsw.efConstruction = 80
        index.hnsw.efSearch = 64
    elif index_type == "ivf":
        adjusted_nlist = max(1, min(nlist, len(descs)))
        if adjusted_nlist != nlist:
            logger.warning(f"Adjusted nlist from {nlist} to {adjusted_nlist} for {len(descs)} vectors")
        quantizer = faiss.IndexFlatL2(dim) if metric == "l2" else faiss.IndexFlatIP(dim)
        index = faiss.IndexIVFFlat(quantizer, dim, adjusted_nlist, faiss_metric)
        index.train(descs)
        index.nprobe = max(1, min(nprobe, adjusted_nlist))
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index.add(descs)
    logger.info(f"Index built: {index.ntotal} vectors, dim={dim}, metric={metric}, type={index_type}")
    return index


def save_index_metadata(
    out_dir: Path,
    method: str,
    index_type: str,
    index: faiss.Index,
    descriptor_path: Path,
    label_path: Path,
    args,
):
    try:
        serialized_mb = faiss.serialize_index(index).size / (1024 ** 2)
    except Exception:
        serialized_mb = None

    metadata = {
        "method": method,
        "method_display": method_display_name(method),
        "index_type": index_type,
        "metric": method_metric(method),
        "vectors": int(index.ntotal),
        "dimension": int(index.d),
        "descriptor_file": descriptor_path.name,
        "label_file": label_path.name,
        "index_file": index_filename(method, index_type),
        "serialized_index_mb": serialized_mb,
        "nlist": args.nlist if index_type == "ivf" else None,
        "nprobe": args.nprobe if index_type == "ivf" else None,
        "hnsw_m": args.hnsw_m if index_type == "hnsw" else None,
    }
    meta_path = out_dir / f"{method}_{index_type}_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata: {meta_path}")


def main():
    parser = argparse.ArgumentParser(description="Build FAISS retrieval indexes")
    parser.add_argument("--method",  choices=["sift", "orb", "dinov2", "anyloc", "both", "all"],
                        default="both", help="'both'=SIFT+DINOv2, 'all'=includes ORB+AnyLoc")
    parser.add_argument("--out-dir", type=Path, default=MODELS_DIR)
    parser.add_argument("--index-type", choices=["flat", "ivf", "hnsw"], default="flat",
                        help="FAISS index type. flat is exact; ivf/hnsw are scalable approximate indexes.")
    parser.add_argument("--nlist", type=int, default=64, help="IVF cluster count")
    parser.add_argument("--nprobe", type=int, default=10, help="IVF clusters probed at query time")
    parser.add_argument("--hnsw-m", type=int, default=32, help="HNSW graph connectivity")
    args = parser.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)

    for method in resolve_methods(args.method):
        desc_path = args.out_dir / descriptor_filename(method)
        label_path = args.out_dir / label_filename(method)
        if not desc_path.exists():
            logger.error(f"{desc_path} not found. Run extract.py --method {method} first.")
            continue

        descs = np.load(desc_path)
        index = build_index(
            descs,
            metric=method_metric(method),
            index_type=args.index_type,
            nlist=args.nlist,
            hnsw_m=args.hnsw_m,
            nprobe=args.nprobe,
        )
        out_path = args.out_dir / index_filename(method, args.index_type)
        faiss.write_index(index, str(out_path))
        logger.info(f"Saved: {out_path}")
        save_index_metadata(args.out_dir, method, args.index_type, index, desc_path, label_path, args)

    logger.info("Indexing complete. Next: python src/retrieve.py --query <image_path>")


if __name__ == "__main__":
    main()
