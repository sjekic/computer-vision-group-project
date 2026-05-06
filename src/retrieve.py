"""
Image Retrieval — Query Interface
===================================
Given a query image, retrieves the top-K most similar images from the database
using either SIFT+VLAD or DINOv2.

Usage:
    # Single query, both methods
    python src/retrieve.py --query data/query/my_photo.jpg

    # Top-5, DINOv2 only
    python src/retrieve.py --query data/query/my_photo.jpg --method dinov2 --k 5

    # Show results visually
    python src/retrieve.py --query data/query/my_photo.jpg --show

    # Batch evaluation over query folder
    python src/retrieve.py --query-dir data/query/ --method both --k 5
"""

import argparse
import logging
import time
from pathlib import Path
from collections import Counter

import cv2
import numpy as np
import faiss

from anyloc_features import encode_anyloc_image
from retrieval_config import index_filename, resolve_methods

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")


# ─── Preprocessing helpers (mirrors preprocess.py) ───────────────────────────

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


def load_and_preprocess(path: Path, target_size=(640, 480)) -> np.ndarray:
    """Load and resize query image to match database images."""
    if path.suffix.lower() in {".heic", ".heif"}:
        from PIL import Image
        img = np.array(Image.open(path).convert("RGB"))
    else:
        img = cv2.imread(str(path))
        if img is None:
            raise ValueError(f"Cannot read {path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    tw, th = target_size
    h, w = img.shape[:2]
    scale = min(tw / w, th / h)
    nw, nh = int(w * scale), int(h * scale)
    resized = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LANCZOS4)
    canvas = np.full((th, tw, 3), 128, dtype=np.uint8)
    y_off, x_off = (th - nh) // 2, (tw - nw) // 2
    canvas[y_off:y_off + nh, x_off:x_off + nw] = resized
    return canvas


# ─── SIFT + VLAD query ────────────────────────────────────────────────────────

def query_sift_vlad(img_rgb: np.ndarray, index: faiss.Index,
                    codebook: np.ndarray, labels: list[str], k: int):
    img_gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    sift = cv2.SIFT_create(nfeatures=500)
    _, raw = sift.detectAndCompute(img_gray, None)

    # VLAD encode
    vlad_k, d = codebook.shape
    vlad = np.zeros((vlad_k, d), dtype=np.float32)
    if raw is not None and len(raw) > 0:
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
        matches = flann.knnMatch(raw.astype(np.float32), codebook, k=1)
        for i, match in enumerate(matches):
            if match:
                wi = match[0].trainIdx
                vlad[wi] += raw[i] - codebook[wi]
        for i in range(vlad_k):
            n = np.linalg.norm(vlad[i])
            if n > 0:
                vlad[i] /= n
    vec = vlad.flatten()
    n = np.linalg.norm(vec)
    if n > 0:
        vec /= n

    vec = vec.reshape(1, -1).astype(np.float32)
    t0 = time.perf_counter()
    distances, indices = index.search(vec, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [(labels[i], float(distances[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
    return results, latency_ms


# ─── DINOv2 query ─────────────────────────────────────────────────────────────

def query_orb_bow(img_rgb: np.ndarray, index: faiss.Index,
                  codebook: np.ndarray, labels: list[str], k: int):
    gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
    orb = cv2.ORB_create(nfeatures=500)
    _, raw = orb.detectAndCompute(gray, None)

    bow = np.zeros(codebook.shape[0], dtype=np.float32)
    if raw is not None and len(raw) > 0:
        flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
        matches = flann.knnMatch(raw.astype(np.float32), codebook.astype(np.float32), k=1)
        for m in matches:
            if m:
                bow[m[0].trainIdx] += 1

    norm = np.linalg.norm(bow)
    if norm > 0:
        bow /= norm

    vec = bow.reshape(1, -1).astype(np.float32)
    t0 = time.perf_counter()
    distances, indices = index.search(vec, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [(labels[i], float(distances[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
    return results, latency_ms


def query_dinov2(img_rgb: np.ndarray, index: faiss.Index,
                 labels: list[str], k: int, model, device, transform):
    from PIL import Image
    import torch

    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)

    with torch.no_grad():
        feat = model(tensor).cpu().numpy().astype(np.float32)

    # L2 normalise
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    feat = feat / np.maximum(norm, 1e-8)

    t0 = time.perf_counter()
    scores, indices = index.search(feat, k)
    latency_ms = (time.perf_counter() - t0) * 1000

    results = [(labels[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
    return results, latency_ms


def query_anyloc(img_rgb: np.ndarray, index: faiss.Index,
                 codebook: np.ndarray, labels: list[str], k: int,
                 model, device, transform):
    vec = encode_anyloc_image(img_rgb, model, device, transform, codebook)
    vec = vec.reshape(1, -1).astype(np.float32)
    t0 = time.perf_counter()
    scores, indices = index.search(vec, k)
    latency_ms = (time.perf_counter() - t0) * 1000
    results = [(labels[i], float(scores[0][j])) for j, i in enumerate(indices[0]) if i >= 0]
    return results, latency_ms


# ─── Result display ───────────────────────────────────────────────────────────

def extract_location(image_id: str) -> str:
    """Extract location label from image_id like 'cafeteria/IMG_001'."""
    return image_id.split("/")[0]


def print_results(method: str, results: list, latency_ms: float, query_location: str = None):
    print(f"\n{'─'*55}")
    print(f"  {method.upper()} — Top-{len(results)} results  ({latency_ms:.1f} ms)")
    print(f"{'─'*55}")
    for rank, (image_id, score) in enumerate(results, 1):
        loc = extract_location(image_id)
        correct = ""
        if query_location and loc == query_location:
            correct = "  ✓"
        print(f"  #{rank:2d}  {loc:<30}  score={score:.4f}{correct}")
    print(f"{'─'*55}")

    if query_location:
        top1_loc = extract_location(results[0][0]) if results else ""
        top5_locs = [extract_location(r[0]) for r in results[:5]]
        print(f"  Top-1 correct: {'YES' if top1_loc == query_location else 'NO'}")
        print(f"  Top-5 correct: {'YES' if query_location in top5_locs else 'NO'}")

    if results:
        votes = Counter(extract_location(image_id) for image_id, _ in results)
        predicted, count = votes.most_common(1)[0]
        confidence = count / len(results)
        print(f"  Predicted location: {predicted}  (top-{len(results)} vote={confidence:.0%})")


def show_results_grid(query_path: Path, results: list[tuple[str, float]], method: str):
    """Display query + top results in a grid using OpenCV."""
    import cv2

    THUMB = (200, 150)
    imgs = []

    # Query image
    q = cv2.imread(str(query_path))
    if q is not None:
        q = cv2.resize(q, THUMB)
        cv2.putText(q, "QUERY", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        imgs.append(q)

    # Retrieved images
    for rank, (image_id, score) in enumerate(results, 1):
        parts = image_id.split("/")
        if len(parts) == 2:
            img_path = Path("data/processed") / parts[0] / f"{parts[1]}.jpg"
            r = cv2.imread(str(img_path))
            if r is not None:
                r = cv2.resize(r, THUMB)
                loc = parts[0][:15]
                cv2.putText(r, f"#{rank} {loc}", (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
                imgs.append(r)

    if imgs:
        grid = np.hstack(imgs)
        cv2.imshow(f"Retrieval results — {method}", grid)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


# ─── Main ─────────────────────────────────────────────────────────────────────

def load_resources(method: str, models_dir: Path, index_type: str = "flat", nprobe: int = 10):
    """Load index, labels and (for dinov2) model."""
    resources = {}
    methods = resolve_methods(method)

    if "sift" in methods:
        idx_path = models_dir / index_filename("sift", index_type)
        cb_path  = models_dir / "sift_codebook.npy"
        lb_path  = models_dir / "sift_vlad_labels.npy"
        if not all(p.exists() for p in [idx_path, cb_path, lb_path]):
            logger.error("SIFT resources missing. Run extract.py and index.py first.")
        else:
            resources["sift"] = {
                "index":    faiss.read_index(str(idx_path)),
                "codebook": np.load(cb_path),
                "labels":   list(np.load(lb_path)),
            }
            if hasattr(resources["sift"]["index"], "nprobe"):
                resources["sift"]["index"].nprobe = nprobe

    if "orb" in methods:
        idx_path = models_dir / index_filename("orb", index_type)
        cb_path  = models_dir / "orb_codebook.npy"
        lb_path  = models_dir / "orb_labels.npy"
        if not all(p.exists() for p in [idx_path, cb_path, lb_path]):
            logger.error("ORB resources missing. Run extract.py --method orb and index.py --method orb first.")
        else:
            resources["orb"] = {
                "index":    faiss.read_index(str(idx_path)),
                "codebook": np.load(cb_path),
                "labels":   list(np.load(lb_path)),
            }
            if hasattr(resources["orb"]["index"], "nprobe"):
                resources["orb"]["index"].nprobe = nprobe

    if "dinov2" in methods:
        idx_path = models_dir / index_filename("dinov2", index_type)
        lb_path  = models_dir / "dinov2_labels.npy"
        if not all(p.exists() for p in [idx_path, lb_path]):
            logger.error("DINOv2 resources missing. Run extract.py and index.py first.")
        else:
            import torch
            from torchvision import transforms

            logger.info("Loading DINOv2 model for inference...")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()

            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            resources["dinov2"] = {
                "index":     faiss.read_index(str(idx_path)),
                "labels":    list(np.load(lb_path)),
                "model":     model,
                "device":    device,
                "transform": transform,
            }
            if hasattr(resources["dinov2"]["index"], "nprobe"):
                resources["dinov2"]["index"].nprobe = nprobe

    if "anyloc" in methods:
        idx_path = models_dir / index_filename("anyloc", index_type)
        cb_path  = models_dir / "anyloc_dinov2_codebook.npy"
        lb_path  = models_dir / "anyloc_dinov2_vlad_labels.npy"
        if not all(p.exists() for p in [idx_path, cb_path, lb_path]):
            logger.error("AnyLoc resources missing. Run extract.py --method anyloc and index.py --method anyloc first.")
        else:
            import torch
            from torchvision import transforms

            logger.info("Loading DINOv2 model for AnyLoc inference...")
            model = torch.hub.load("facebookresearch/dinov2", "dinov2_vits14", pretrained=True)
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = model.to(device).eval()
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225]),
            ])

            resources["anyloc"] = {
                "index":     faiss.read_index(str(idx_path)),
                "codebook":  np.load(cb_path),
                "labels":    list(np.load(lb_path)),
                "model":     model,
                "device":    device,
                "transform": transform,
            }
            if hasattr(resources["anyloc"]["index"], "nprobe"):
                resources["anyloc"]["index"].nprobe = nprobe

    return resources


def run_query(query_path: Path, resources: dict, k: int, show: bool,
              query_location: str = None):
    img = load_and_preprocess(query_path)

    if "sift" in resources:
        r = resources["sift"]
        results, latency = query_sift_vlad(
            img, r["index"], r["codebook"], r["labels"], k
        )
        print_results("SIFT+VLAD", results, latency, query_location)
        if show:
            show_results_grid(query_path, results, "SIFT+VLAD")

    if "orb" in resources:
        r = resources["orb"]
        results, latency = query_orb_bow(
            img, r["index"], r["codebook"], r["labels"], k
        )
        print_results("ORB+BoW", results, latency, query_location)
        if show:
            show_results_grid(query_path, results, "ORB+BoW")

    if "dinov2" in resources:
        r = resources["dinov2"]
        results, latency = query_dinov2(
            img, r["index"], r["labels"], k,
            r["model"], r["device"], r["transform"]
        )
        print_results("DINOv2", results, latency, query_location)
        if show:
            show_results_grid(query_path, results, "DINOv2")

    if "anyloc" in resources:
        r = resources["anyloc"]
        results, latency = query_anyloc(
            img, r["index"], r["codebook"], r["labels"], k,
            r["model"], r["device"], r["transform"]
        )
        print_results("AnyLoc-DINOv2-VLAD", results, latency, query_location)
        if show:
            show_results_grid(query_path, results, "AnyLoc-DINOv2-VLAD")


def main():
    parser = argparse.ArgumentParser(description="Query the image retrieval system")
    parser.add_argument("--query",        type=Path, default=None, help="Single query image path")
    parser.add_argument("--query-dir",    type=Path, default=None, help="Folder of query images")
    parser.add_argument("--method",       choices=["sift", "orb", "dinov2", "anyloc", "both", "all"], default="both")
    parser.add_argument("--k",            type=int, default=5,    help="Number of results to return")
    parser.add_argument("--show",         action="store_true",    help="Display results visually")
    parser.add_argument("--models-dir",   type=Path, default=MODELS_DIR)
    parser.add_argument("--index-type",   choices=["flat", "ivf", "hnsw"], default="flat")
    parser.add_argument("--nprobe",       type=int, default=10, help="Number of IVF lists to probe")
    args = parser.parse_args()

    if args.query is None and args.query_dir is None:
        parser.error("Provide --query <image> or --query-dir <folder>")

    resources = load_resources(args.method, args.models_dir, args.index_type, args.nprobe)
    if not resources:
        logger.error("No resources loaded. Aborting.")
        return

    IMG_EXTS = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".bmp", ".webp"}

    if args.query:
        run_query(args.query, resources, args.k, args.show)

    if args.query_dir:
        queries = [f for f in args.query_dir.rglob("*") if f.suffix.lower() in IMG_EXTS]
        logger.info(f"Running {len(queries)} queries from {args.query_dir}")
        for qp in sorted(queries):
            print(f"\nQuery: {qp.name}")
            # If query images are in location sub-folders, use folder name as ground truth
            gt = qp.parent.name if qp.parent != args.query_dir else None
            run_query(qp, resources, args.k, args.show, query_location=gt)


if __name__ == "__main__":
    main()
