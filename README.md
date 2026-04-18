# Fast Image Search — IE Tower Visual Place Recognition

End-to-end image retrieval system for the IE Tower area. Given a query image, the system returns the most visually similar database images and identifies the location.

---

## Project Structure

```
computer-vision-group-project/
├── data/
│   ├── raw/              ← your captured images (one sub-folder per location)
│   │   └── <location>/
│   ├── processed/        ← output of preprocessing pipeline
│   └── query/            ← images used as search queries
├── src/
│   ├── preprocess.py     ← data cleaning, resizing, augmentation
│   ├── dataset_stats.py  ← validation, brightness check, duplicate detection
│   ├── extract.py        ← feature extraction (SIFT / DINOv2)       [TODO]
│   ├── index.py          ← FAISS index build & save                  [TODO]
│   └── retrieve.py       ← query → top-K results                     [TODO]
├── notebooks/            ← exploratory analysis
├── models/               ← saved embeddings / index files
├── results/              ← evaluation outputs, reports
├── requirements.txt
└── README.md
```

---

## Setup

```bash
# 1. Clone
git clone https://github.com/sjekic/computer-vision-group-project.git
cd computer-vision-group-project

# 2. Create virtual environment
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
```

---

## Data Organisation

Place your raw images under `data/raw/` with one sub-folder per location:

```
data/raw/
    lobby/
        img_001.jpg
        img_002.jpg
    rooftop_terrace/
        img_001.jpg
    ...
```

> The location folder name becomes the class label used throughout the pipeline.

---

## Step 1 — Preprocessing

```bash
# Basic: resize to 640×480, apply CLAHE histogram equalisation
python src/preprocess.py

# With augmentation (recommended for building the database)
python src/preprocess.py --augment

# Custom size
python src/preprocess.py --size 800 600 --augment

# Dry run (no files written, just counts)
python src/preprocess.py --dry-run

# Process query images (no augmentation, saved to data/query/)
python src/preprocess.py --raw-dir data/raw/my_queries --query-mode
```

**What the script does:**
| Step | Detail |
|---|---|
| Validation | Skips corrupt / unreadable files |
| Resize + pad | Aspect-ratio-preserving resize → letterbox pad to target size |
| CLAHE | Contrast Limited Adaptive Histogram Equalisation on L channel (LAB) — improves robustness to lighting changes |
| Augmentation | 6 variants per image: brightness ±30%, horizontal flip, 5° rotation, Gaussian blur, desaturation |
| Manifest | `data/processed/manifest.csv` — image_id, location, path, aug_type |

---

## Step 2 — Dataset Validation

```bash
python src/dataset_stats.py                   # checks data/processed/
python src/dataset_stats.py --dir data/raw    # checks raw images
```

Outputs:
- Per-location image counts and resolution stats
- Mean brightness (flags very dark / overexposed images)
- Near-duplicate detection via perceptual hashing (pHash, threshold ≤ 8 bits)
- `results/dataset_report.txt` and `results/dataset_stats.csv`

---

## Methods Compared

| Method | Descriptor | Matching | Notes |
|---|---|---|---|
| Classical | SIFT keypoints + VLAD/BoW | Brute-force / FLANN | Fast, interpretable |
| Deep | DINOv2 ViT-S/14 global embedding | FAISS approximate NN | State-of-the-art accuracy |

---

## Evaluation Metrics

- **Top-K accuracy** (K = 1, 3, 5)
- **Mean Average Precision (mAP)**
- **Query latency** (ms per query)
- **Memory footprint** of the index

---

## Team

IE University — Computer Vision Group Project, 2025/26
