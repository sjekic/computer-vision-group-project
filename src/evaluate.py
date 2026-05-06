"""
Evaluation Script
==================
Computes quantitative performance metrics for both retrieval methods:

  - Top-K Accuracy  (K = 1, 3, 5, 10)
  - Mean Average Precision (mAP)
  - Mean query latency (ms)
  - Index memory footprint (MB)

Protocol:
  - Uses augmented images that were NOT used to build the index as queries,
    OR any images in data/query/<location>/ as a held-out test set.
  - Ground truth: the location folder name must match the query's location.

Usage:
    # Evaluate using augmented variants as queries (leave-one-out style)
    python src/evaluate.py --method both

    # Evaluate using a dedicated query folder (recommended)
    python src/evaluate.py --method both --query-dir data/query/

    # Quick test on first 50 queries only
    python src/evaluate.py --method both --max-queries 50

    # Save results table to CSV
    python src/evaluate.py --method both --save
"""

import argparse
import logging
import time
import os
import sys
from pathlib import Path
from collections import defaultdict

import cv2
import numpy as np
import faiss

from anyloc_features import encode_anyloc_image
from metrics import (
    average_precision_at_k,
    location_from_label,
    relevant_counts_by_location,
    topk_hit,
)
from retrieval_config import index_filename, resolve_methods
from split_protocol import exclude_indexed_queries, select_synthetic_query_records

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")
IMG_EXTS      = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".heic", ".heif"}

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ─── Image loading ────────────────────────────────────────────────────────────

def load_gray(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        import numpy as np
        return np.array(Image.open(path).convert("L"))
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return img


def load_rgb(path: Path) -> np.ndarray:
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        return np.array(Image.open(path).convert("RGB"))
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Cannot read {path}")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def preprocess(img_rgb: np.ndarray, target=(640, 480)) -> np.ndarray:
    tw, th = target
    h, w = img_rgb.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img_rgb, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    canvas[(th-nh)//2:(th-nh)//2+nh, (tw-nw)//2:(tw-nw)//2+nw] = resized
    return canvas


# ─── Query vector builders ────────────────────────────────────────────────────

def encode_sift_vlad(img_rgb: np.ndarray, sift, codebook: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, raw = sift.detectAndCompute(gray, None)
    k, d = codebook.shape
    vlad = np.zeros((k, d), dtype=np.float32)
    if raw is not None and len(raw) > 0:
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
        matches = flann.knnMatch(raw.astype(np.float32), codebook, k=1)
        for i, m in enumerate(matches):
            if m:
                vlad[m[0].trainIdx] += raw[i] - codebook[m[0].trainIdx]
        for i in range(k):
            n = np.linalg.norm(vlad[i])
            if n > 0: vlad[i] /= n
    vec = vlad.flatten()
    n = np.linalg.norm(vec)
    return vec / n if n > 0 else vec


def encode_orb(img_rgb: np.ndarray, orb, codebook: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    _, raw = orb.detectAndCompute(gray, None)
    k = codebook.shape[0]
    bow = np.zeros(k, dtype=np.float32)
    if raw is not None and len(raw) > 0:
        raw_f = raw.astype(np.float32)
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
        matches = flann.knnMatch(raw_f, codebook.astype(np.float32), k=1)
        for m in matches:
            if m: bow[m[0].trainIdx] += 1
    n = np.linalg.norm(bow)
    return bow / n if n > 0 else bow


def encode_dinov2(img_rgb: np.ndarray, model, device, transform) -> np.ndarray:
    import torch
    from PIL import Image
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.maximum(norm, 1e-8)


def encode_anyloc(img_rgb: np.ndarray, model, device, transform, codebook: np.ndarray) -> np.ndarray:
    return encode_anyloc_image(img_rgb, model, device, transform, codebook)


# ─── Metrics ──────────────────────────────────────────────────────────────────

def average_precision(
    retrieved_labels: list[str],
    gt_location: str | None,
    total_relevant: int,
    k: int | None = None,
) -> float:
    """Compute AP for a single query."""
    return average_precision_at_k(retrieved_labels, gt_location, total_relevant, k)


def estimate_index_memory_mb(index: faiss.Index) -> float:
    """Estimate FAISS index footprint in MB."""
    try:
        return faiss.serialize_index(index).size / (1024 ** 2)
    except Exception:
        return (index.ntotal * index.d * 4) / (1024 ** 2)


def evaluate_method(
    method_name: str,
    queries: list[tuple[Path, str | None, str]],   # (path, gt_location, query_image_id)
    index: faiss.Index,
    labels: list[str],
    encode_fn,                          # callable(img_rgb) -> (1, D) or (D,) np array
    k_values: list[int],
    max_k: int,
) -> dict:
    """
    Run all queries through the index and compute metrics.
    Returns a results dict.
    """
    top_k_correct = defaultdict(int)
    ap_scores = []
    full_latencies = []
    encode_latencies = []
    search_latencies = []
    index_mem_mb = estimate_index_memory_mb(index)
    relevant_counts = relevant_counts_by_location(labels)

    for path, gt_loc, _query_id in queries:
        if not gt_loc:
            logger.warning(f"Skipping {path}: no ground-truth location available")
            continue

        total_relevant = relevant_counts.get(gt_loc, 0)
        if total_relevant == 0:
            logger.warning(f"Skipping {path}: ground-truth location '{gt_loc}' not found in index labels")
            continue

        t_full = time.perf_counter()
        try:
            img = preprocess(load_rgb(path))
        except Exception as e:
            logger.warning(f"Skipping {path}: {e}")
            continue

        t_encode = time.perf_counter()
        vec = encode_fn(img)
        encode_latencies.append((time.perf_counter() - t_encode) * 1000)
        vec = np.atleast_2d(vec).astype(np.float32)

        t_search = time.perf_counter()
        _, indices = index.search(vec, max_k)
        search_latencies.append((time.perf_counter() - t_search) * 1000)
        full_latencies.append((time.perf_counter() - t_full) * 1000)

        retrieved = [location_from_label(labels[i]) for i in indices[0] if i >= 0]

        for k in k_values:
            if topk_hit(retrieved, gt_loc, k):
                top_k_correct[k] += 1

        ap_scores.append(average_precision(retrieved, gt_loc, total_relevant, max_k))

    n = len(ap_scores)
    results = {
        "method":     method_name,
        "n_queries":  n,
        "mAP":        np.mean(ap_scores) if ap_scores else 0.0,
        "latency_ms": np.mean(full_latencies) if full_latencies else 0.0,
        "encode_ms":  np.mean(encode_latencies) if encode_latencies else 0.0,
        "search_ms":  np.mean(search_latencies) if search_latencies else 0.0,
        "index_mb":   index_mem_mb,
    }
    for k in k_values:
        results[f"top{k}_acc"] = top_k_correct[k] / n if n > 0 else 0.0

    return results


# ─── Print & save ─────────────────────────────────────────────────────────────

def print_results_table(all_results: list[dict], k_values: list[int]):
    print("\n" + "=" * 75)
    print("  EVALUATION RESULTS")
    print("=" * 75)

    # Header
    k_cols = "".join(f"  Top-{k:<3}" for k in k_values)
    print(f"  {'Method':<18}  {'mAP':>6}  {k_cols}  {'Full(ms)':>8}  {'Search':>8}  {'Mem(MB)':>8}")
    print("  " + "-" * 82)

    for r in all_results:
        k_vals = "".join(f"  {r[f'top{k}_acc']:>6.1%}" for k in k_values)
        print(
            f"  {r['method']:<18}  {r['mAP']:>6.3f}  {k_vals}"
            f"  {r['latency_ms']:>8.2f}  {r['search_ms']:>8.2f}  {r['index_mb']:>8.1f}"
        )

    print("=" * 75)
    print(f"  Queries evaluated: {all_results[0]['n_queries']}")


def save_results(all_results: list[dict], k_values: list[int]):
    import pandas as pd
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(all_results)
    path = RESULTS_DIR / "evaluation_results.csv"
    df.to_csv(path, index=False)
    logger.info(f"Results saved to {path}")


# ─── Query collection ─────────────────────────────────────────────────────────

def _collect_queries_legacy(query_dir: Path, processed_dir: Path,
                            aug_types: list[str], max_queries: int) -> list[tuple[Path, str]]:
    """
    If query_dir exists and has images, use those.
    Otherwise, use a subset of augmented images from processed_dir as queries.
    """
    queries = []

    if query_dir and query_dir.exists():
        for loc_dir in sorted(d for d in query_dir.iterdir() if d.is_dir()):
            for img_path in sorted(f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                queries.append((img_path, loc_dir.name))
        if not queries:
            # Flat query dir without location sub-folders
            for img_path in sorted(f for f in query_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                queries.append((img_path, None))
        logger.info(f"Using {len(queries)} query images from {query_dir}")

    else:
        # Use one specific augmentation type per location as a synthetic test set
        logger.info("No query dir provided — using aug_blur variants from processed/ as queries")
        for loc_dir in sorted(d for d in processed_dir.iterdir() if d.is_dir()):
            for aug_type in aug_types:
                imgs = sorted(f for f in loc_dir.iterdir()
                              if f.suffix.lower() in IMG_EXTS and aug_type in f.name)
                for img_path in imgs:
                    queries.append((img_path, loc_dir.name))

    if max_queries and len(queries) > max_queries:
        np.random.shuffle(queries)
        queries = queries[:max_queries]

    return queries


def collect_queries(query_dir: Path, processed_dir: Path,
                    aug_types: list[str], max_queries: int,
                    indexed_image_ids: set[str] | None = None,
                    seed: int = 42) -> list[tuple[Path, str | None, str]]:
    """Collect held-out query images without reusing indexed synthetic images."""
    queries = []

    if query_dir and query_dir.exists():
        for loc_dir in sorted(d for d in query_dir.iterdir() if d.is_dir()):
            for img_path in sorted(f for f in loc_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                queries.append((img_path, loc_dir.name, f"{loc_dir.name}/{img_path.stem}"))
        if not queries:
            for img_path in sorted(f for f in query_dir.iterdir() if f.suffix.lower() in IMG_EXTS):
                queries.append((img_path, None, img_path.stem))
        logger.info(f"Using {len(queries)} query images from {query_dir}")
    else:
        logger.info("No query dir provided - using selected augmented variants from processed/ as queries")
        manifest = processed_dir / "manifest.csv"
        if manifest.exists():
            import pandas as pd
            records = select_synthetic_query_records(pd.read_csv(manifest).to_dict("records"), aug_types)
            for row in records:
                img_path = Path(row["path"])
                if img_path.exists() and img_path.suffix.lower() in IMG_EXTS:
                    queries.append((img_path, row.get("location"), row.get("image_id")))
        else:
            for loc_dir in sorted(d for d in processed_dir.iterdir() if d.is_dir()):
                for aug_type in aug_types:
                    imgs = sorted(f for f in loc_dir.iterdir()
                                  if f.suffix.lower() in IMG_EXTS and aug_type in f.name)
                    for img_path in imgs:
                        queries.append((img_path, loc_dir.name, f"{loc_dir.name}/{img_path.stem}"))

        if indexed_image_ids:
            before = len(queries)
            queries = exclude_indexed_queries(queries, indexed_image_ids)
            removed = before - len(queries)
            if removed:
                logger.info(f"Removed {removed} synthetic queries already present in the index")

    if max_queries and len(queries) > max_queries:
        rng = np.random.default_rng(seed)
        rng.shuffle(queries)
        queries = queries[:max_queries]

    return queries


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate image retrieval methods")
    parser.add_argument("--method",      choices=["sift", "orb", "dinov2", "anyloc", "both", "all"],
                        default="both", help="'both'=SIFT+DINOv2, 'all'=includes ORB+AnyLoc")
    parser.add_argument("--query-dir",   type=Path, default=None,
                        help="Directory of query images, organised as <location>/<img>")
    parser.add_argument("--models-dir",  type=Path, default=MODELS_DIR)
    parser.add_argument("--data-dir",    type=Path, default=PROCESSED_DIR)
    parser.add_argument("--max-queries", type=int,  default=None)
    parser.add_argument("--k",           type=int,  nargs="+", default=[1, 3, 5, 10])
    parser.add_argument("--index-type",  choices=["flat", "ivf", "hnsw"], default="flat")
    parser.add_argument("--nprobe",      type=int, default=10,
                        help="Number of IVF lists to probe when evaluating IVF indexes")
    parser.add_argument("--seed",        type=int, default=42)
    parser.add_argument("--save",        action="store_true")
    args = parser.parse_args()

    max_k = max(args.k)
    all_results = []

    # Determine which methods to run
    methods = resolve_methods(args.method)
    indexed_image_ids = set()

    # Load resources per method
    sift_res = orb_res = dino_res = anyloc_res = None

    if "sift" in methods:
        idx_p = args.models_dir / index_filename("sift", args.index_type)
        cb_p  = args.models_dir / "sift_codebook.npy"
        lb_p  = args.models_dir / "sift_vlad_labels.npy"
        if all(p.exists() for p in [idx_p, cb_p, lb_p]):
            sift_res = {
                "index":    faiss.read_index(str(idx_p)),
                "codebook": np.load(cb_p),
                "labels":   list(np.load(lb_p)),
                "sift":     cv2.SIFT_create(nfeatures=500),
            }
            if hasattr(sift_res["index"], "nprobe"):
                sift_res["index"].nprobe = args.nprobe
            indexed_image_ids.update(sift_res["labels"])
        else:
            logger.warning("SIFT resources not found — skipping SIFT evaluation")
            methods.remove("sift")

    if "orb" in methods:
        idx_p = args.models_dir / index_filename("orb", args.index_type)
        cb_p  = args.models_dir / "orb_codebook.npy"
        lb_p  = args.models_dir / "orb_labels.npy"
        if all(p.exists() for p in [idx_p, cb_p, lb_p]):
            orb_res = {
                "index":    faiss.read_index(str(idx_p)),
                "codebook": np.load(cb_p),
                "labels":   list(np.load(lb_p)),
                "orb":      cv2.ORB_create(nfeatures=500),
            }
            if hasattr(orb_res["index"], "nprobe"):
                orb_res["index"].nprobe = args.nprobe
            indexed_image_ids.update(orb_res["labels"])
        else:
            logger.warning("ORB resources not found — skipping ORB evaluation")
            methods.remove("orb")

    if "dinov2" in methods:
        idx_p = args.models_dir / index_filename("dinov2", args.index_type)
        lb_p  = args.models_dir / "dinov2_labels.npy"
        if all(p.exists() for p in [idx_p, lb_p]):
            import torch
            from torchvision import transforms
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            dino_res = {
                "index":     faiss.read_index(str(idx_p)),
                "labels":    list(np.load(lb_p)),
                "model":     model, "device": device, "transform": transform,
            }
            if hasattr(dino_res["index"], "nprobe"):
                dino_res["index"].nprobe = args.nprobe
            indexed_image_ids.update(dino_res["labels"])
        else:
            logger.warning("DINOv2 resources not found — skipping DINOv2 evaluation")
            methods.remove("dinov2")

    if "anyloc" in methods:
        idx_p = args.models_dir / index_filename("anyloc", args.index_type)
        cb_p  = args.models_dir / "anyloc_dinov2_codebook.npy"
        lb_p  = args.models_dir / "anyloc_dinov2_vlad_labels.npy"
        if all(p.exists() for p in [idx_p, cb_p, lb_p]):
            import torch
            from torchvision import transforms
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            transform = transforms.Compose([
                transforms.Resize(256), transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225]),
            ])
            anyloc_res = {
                "index":     faiss.read_index(str(idx_p)),
                "codebook":  np.load(cb_p),
                "labels":    list(np.load(lb_p)),
                "model":     model, "device": device, "transform": transform,
            }
            if hasattr(anyloc_res["index"], "nprobe"):
                anyloc_res["index"].nprobe = args.nprobe
            indexed_image_ids.update(anyloc_res["labels"])
        else:
            logger.warning("AnyLoc resources not found - skipping AnyLoc evaluation")
            methods.remove("anyloc")

    if not methods:
        logger.error("No methods available. Run extract.py and index.py first.")
        sys.exit(1)

    # Collect queries
    aug_types = ["aug_blur", "aug_dark", "aug_bright", "aug_skew_h"]
    queries = collect_queries(
        args.query_dir,
        args.data_dir,
        aug_types,
        args.max_queries,
        indexed_image_ids=indexed_image_ids,
        seed=args.seed,
    )

    if not queries:
        logger.error("No query images found.")
        sys.exit(1)

    logger.info(f"Evaluating {len(queries)} queries, methods: {methods}, K={args.k}")

    # Run evaluation per method
    if "sift" in methods and sift_res:
        r = sift_res
        def sift_encode(img): return encode_sift_vlad(img, r["sift"], r["codebook"])
        results = evaluate_method("SIFT+VLAD", queries, r["index"], r["labels"],
                                  sift_encode, args.k, max_k)
        all_results.append(results)

    if "orb" in methods and orb_res:
        r = orb_res
        def orb_encode(img): return encode_orb(img, r["orb"], r["codebook"])
        results = evaluate_method("ORB+BoW", queries, r["index"], r["labels"],
                                  orb_encode, args.k, max_k)
        all_results.append(results)

    if "dinov2" in methods and dino_res:
        r = dino_res
        def dino_encode(img): return encode_dinov2(img, r["model"], r["device"], r["transform"])
        results = evaluate_method("DINOv2 ViT-S/14", queries, r["index"], r["labels"],
                                  dino_encode, args.k, max_k)
        all_results.append(results)

    if "anyloc" in methods and anyloc_res:
        r = anyloc_res
        def anyloc_encode(img): return encode_anyloc(img, r["model"], r["device"], r["transform"], r["codebook"])
        results = evaluate_method("AnyLoc-DINOv2-VLAD", queries, r["index"], r["labels"],
                                  anyloc_encode, args.k, max_k)
        all_results.append(results)

    if not all_results:
        logger.error("No evaluation results were produced.")
        sys.exit(1)

    print_results_table(all_results, args.k)

    if args.save:
        save_results(all_results, args.k)


if __name__ == "__main__":
    main()
