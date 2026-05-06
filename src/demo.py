"""
Demo — Visual Place Recognition
=================================
Interactive demo that takes any query image and displays a side-by-side
visual grid of top-K retrieved results for both methods.

Designed for the recorded project demonstration.

Usage:
    python src/demo.py --query data/query/my_photo.jpg
    python src/demo.py --query data/query/my_photo.jpg --k 5 --method both
    python src/demo.py --query data/query/my_photo.jpg --save   # saves grid to results/
"""

import argparse
import logging
import time
from pathlib import Path

import cv2
import numpy as np
import faiss

logging.basicConfig(format="%(asctime)s [%(levelname)s] %(message)s",
                    level=logging.INFO, datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

MODELS_DIR    = Path("models")
PROCESSED_DIR = Path("data/processed")
RESULTS_DIR   = Path("results")

try:
    from pillow_heif import register_heif_opener
    register_heif_opener()
except ImportError:
    pass


# ─── Image utilities ─────────────────────────────────────────────────────────

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


def make_thumbnail(img_bgr: np.ndarray, size=(240, 180)) -> np.ndarray:
    return cv2.resize(img_bgr, size, interpolation=cv2.INTER_LANCZOS4)


def add_label(img: np.ndarray, text: str, subtext: str = "",
              color=(255, 255, 255), bg_color=(0, 0, 0)) -> np.ndarray:
    """Add a label bar at the bottom of a thumbnail."""
    h, w = img.shape[:2]
    bar_h = 44
    out = np.zeros((h + bar_h, w, 3), dtype=np.uint8)
    out[:h] = img
    out[h:] = bg_color

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(out, text[:22],     (5, h + 16), font, 0.45, color,       1, cv2.LINE_AA)
    cv2.putText(out, subtext[:28],  (5, h + 34), font, 0.38, (180,180,180), 1, cv2.LINE_AA)
    return out


def find_image_in_processed(image_id: str) -> Path | None:
    """Locate a processed image by its image_id (location/stem)."""
    parts = image_id.split("/")
    if len(parts) != 2:
        return None
    loc, stem = parts
    for ext in [".jpg", ".jpeg", ".png"]:
        p = PROCESSED_DIR / loc / f"{stem}{ext}"
        if p.exists():
            return p
    return None


# ─── Encoding ─────────────────────────────────────────────────────────────────

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


def encode_dinov2(img_rgb: np.ndarray, model, device, transform) -> np.ndarray:
    import torch
    from PIL import Image
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    with torch.no_grad():
        feat = model(tensor).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(feat, axis=1, keepdims=True)
    return feat / np.maximum(norm, 1e-8)


# ─── Grid builder ─────────────────────────────────────────────────────────────

THUMB_SIZE  = (240, 180)
BORDER      = 6
HEADER_H    = 50
LABEL_H     = 44

# Colour scheme
COL_BG      = (30,  30,  30)
COL_QUERY   = (0,  200,  80)    # green border for query
COL_CORRECT = (0,  200,  80)    # green border for correct match
COL_WRONG   = (60,  60,  60)    # grey border for others
COL_HEADER  = (50,  50,  50)


def build_result_grid(
    query_path: Path,
    method_results: list[tuple[str, list[tuple[str, float]], float]],
    # [(method_name, [(image_id, score), ...], latency_ms), ...]
    k: int,
    gt_location: str | None = None,
) -> np.ndarray:
    """
    Build a visual grid:
    Row 0: header
    Row 1: query image + top-K results for method 1
    Row 2: query image + top-K results for method 2
    ...
    """
    tw, th = THUMB_SIZE
    cell_w = tw + BORDER * 2
    cell_h = th + LABEL_H + BORDER * 2
    n_cols  = k + 1   # query + k results

    total_w = cell_w * n_cols
    total_h = HEADER_H + cell_h * len(method_results)

    canvas = np.full((total_h, total_w, 3), COL_BG, dtype=np.uint8)

    # ── Header ──
    cv2.putText(canvas, "Visual Place Recognition — IE Tower",
                (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220, 220, 220), 1, cv2.LINE_AA)

    # ── Rows ──
    query_bgr = cv2.imread(str(query_path))
    if query_bgr is None:
        try:
            from PIL import Image as PILImage
            query_bgr = cv2.cvtColor(np.array(PILImage.open(query_path).convert("RGB")), cv2.COLOR_RGB2BGR)
        except Exception:
            query_bgr = np.zeros((th, tw, 3), dtype=np.uint8)

    query_thumb = make_thumbnail(query_bgr, THUMB_SIZE)

    for row_idx, (method_name, results, latency_ms) in enumerate(method_results):
        row_y = HEADER_H + row_idx * cell_h

        # ── Query cell ──
        q_cell = add_label(
            query_thumb.copy(),
            "QUERY",
            query_path.parent.name if query_path.parent.name != "query" else query_path.stem[:18],
            color=(0, 220, 100),
            bg_color=(20, 60, 20),
        )
        # Green border
        qy1, qx1 = row_y + BORDER, BORDER
        qy2, qx2 = qy1 + th + LABEL_H, qx1 + tw
        canvas[qy1-BORDER:qy2+BORDER, qx1-BORDER:qx2+BORDER] = COL_QUERY
        canvas[qy1:qy2, qx1:qx2] = q_cell

        # Method label on left edge
        cv2.putText(canvas, method_name,
                    (BORDER, row_y + (cell_h // 2)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (200, 200, 200), 1, cv2.LINE_AA)

        # ── Result cells ──
        for col_idx, (image_id, score) in enumerate(results[:k]):
            loc = image_id.split("/")[0]
            is_correct = (gt_location is not None and loc == gt_location)
            border_col = COL_CORRECT if is_correct else COL_WRONG

            img_path = find_image_in_processed(image_id)
            if img_path and img_path.exists():
                cell_bgr = cv2.imread(str(img_path))
                if cell_bgr is None:
                    cell_bgr = np.zeros((th, tw, 3), dtype=np.uint8)
            else:
                cell_bgr = np.zeros((th, tw, 3), dtype=np.uint8)

            thumb = make_thumbnail(cell_bgr, THUMB_SIZE)

            rank_label = f"#{col_idx+1}  {loc}"
            score_label = f"score={score:.3f}{'  CORRECT' if is_correct else ''}"
            cell_img = add_label(thumb, rank_label, score_label,
                                 color=(0, 220, 100) if is_correct else (200, 200, 200),
                                 bg_color=(20, 60, 20) if is_correct else (30, 30, 30))

            cx1 = (col_idx + 1) * cell_w + BORDER
            cy1 = row_y + BORDER
            cx2 = cx1 + tw
            cy2 = cy1 + th + LABEL_H

            canvas[cy1-BORDER:cy2+BORDER, cx1-BORDER:cx2+BORDER] = border_col
            canvas[cy1:cy2, cx1:cx2] = cell_img

        # Latency tag
        lat_txt = f"{latency_ms:.1f} ms"
        cv2.putText(canvas, lat_txt,
                    (total_w - 90, row_y + cell_h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (150, 150, 150), 1, cv2.LINE_AA)

    return canvas


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visual demo of image retrieval system")
    parser.add_argument("--query",      type=Path, required=True, help="Query image path")
    parser.add_argument("--method",     choices=["sift", "dinov2", "both"], default="both")
    parser.add_argument("--k",          type=int,  default=5)
    parser.add_argument("--save",       action="store_true", help="Save grid to results/")
    parser.add_argument("--no-show",    action="store_true", help="Don't display window")
    parser.add_argument("--models-dir", type=Path, default=MODELS_DIR)
    args = parser.parse_args()

    if not args.query.exists():
        logger.error(f"Query image not found: {args.query}")
        return

    # Load resources
    resources = {}

    if args.method in ("sift", "both"):
        idx_p = args.models_dir / "sift_vlad.index"
        cb_p  = args.models_dir / "sift_codebook.npy"
        lb_p  = args.models_dir / "sift_vlad_labels.npy"
        if all(p.exists() for p in [idx_p, cb_p, lb_p]):
            resources["sift"] = {
                "index":    faiss.read_index(str(idx_p)),
                "codebook": np.load(cb_p),
                "labels":   list(np.load(lb_p)),
                "sift":     cv2.SIFT_create(nfeatures=500),
            }

    if args.method in ("dinov2", "both"):
        idx_p = args.models_dir / "dinov2.index"
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
            resources["dinov2"] = {
                "index": faiss.read_index(str(idx_p)),
                "labels": list(np.load(lb_p)),
                "model": model, "device": device, "transform": transform,
            }

    if not resources:
        logger.error("No indexes found. Run extract.py and index.py first.")
        return

    # Run query
    img_rgb = preprocess(load_rgb(args.query))
    gt_location = args.query.parent.name if args.query.parent.name not in ("query", ".") else None

    method_results = []

    if "sift" in resources:
        r = resources["sift"]
        gray = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)
        vec = encode_sift_vlad(img_rgb, r["sift"], r["codebook"])
        vec2d = vec.reshape(1, -1).astype(np.float32)
        t0 = time.perf_counter()
        _, idxs = r["index"].search(vec2d, args.k)
        lat = (time.perf_counter() - t0) * 1000
        results = [(r["labels"][i], 0.0) for i in idxs[0] if i >= 0]
        # Re-run with distances
        dists, idxs2 = r["index"].search(vec2d, args.k)
        results = [(r["labels"][i], float(dists[0][j])) for j, i in enumerate(idxs2[0]) if i >= 0]
        method_results.append(("SIFT + VLAD", results, lat))

        # Print to terminal too
        print(f"\nSIFT+VLAD ({lat:.1f} ms)")
        for rank, (iid, sc) in enumerate(results, 1):
            loc = iid.split("/")[0]
            tick = " ✓" if gt_location and loc == gt_location else ""
            print(f"  #{rank}  {loc:<30}  dist={sc:.4f}{tick}")

    if "dinov2" in resources:
        r = resources["dinov2"]
        vec = encode_dinov2(img_rgb, r["model"], r["device"], r["transform"])
        t0 = time.perf_counter()
        scores, idxs = r["index"].search(vec, args.k)
        lat = (time.perf_counter() - t0) * 1000
        results = [(r["labels"][i], float(scores[0][j])) for j, i in enumerate(idxs[0]) if i >= 0]
        method_results.append(("DINOv2 ViT-S/14", results, lat))

        print(f"\nDINOv2 ({lat:.1f} ms)")
        for rank, (iid, sc) in enumerate(results, 1):
            loc = iid.split("/")[0]
            tick = " ✓" if gt_location and loc == gt_location else ""
            print(f"  #{rank}  {loc:<30}  score={sc:.4f}{tick}")

    # Build and show grid
    grid = build_result_grid(args.query, method_results, args.k, gt_location)

    if args.save:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        out_path = RESULTS_DIR / f"demo_{args.query.stem}.jpg"
        cv2.imwrite(str(out_path), grid)
        logger.info(f"Grid saved to {out_path}")

    if not args.no_show:
        cv2.imshow("IE Tower — Visual Place Recognition Demo", grid)
        logger.info("Press any key to close the demo window.")
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
