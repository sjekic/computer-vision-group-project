# Fast Image Search — IE Tower Visual Place Recognition

End-to-end image retrieval system for the IE Tower area. Given a query image, the system returns the most visually similar database images and identifies the location. Supports three methods for comparative analysis.

---

## Methods Compared

| Method | Descriptor | Aggregation | Search | Notes |
|---|---|---|---|---|
| **SIFT + VLAD** | SIFT (128-dim) keypoints | VLAD (k=64) | FAISS L2 | Classical, scale/rotation invariant |
| **ORB + BoW** | ORB (32-dim) keypoints | Bag-of-Words (k=64) | FAISS L2 | Fastest, binary descriptors |
| **DINOv2** | ViT-S/14 global embedding | — | FAISS IP (cosine) | State-of-the-art accuracy |

---

## Project Structure

```
computer-vision-group-project/
├── data/
│   ├── raw/              ← original captured images (one sub-folder per location)
│   ├── processed/        ← resized, CLAHE, augmented images + manifest.csv
│   └── query/            ← held-out query images for evaluation
├── src/
│   ├── download_dataset.py   ← Step 0: pull images from Google Drive
│   ├── preprocess.py         ← Step 1: resize, CLAHE, 20× augmentation
│   ├── dataset_stats.py      ← Step 1b: validate dataset, detect duplicates
│   ├── extract.py            ← Step 2: SIFT+VLAD / ORB+BoW / DINOv2 features
│   ├── index.py              ← Step 3: build FAISS indexes
│   ├── retrieve.py           ← Step 4: query → top-K results (terminal)
│   ├── evaluate.py           ← Step 5: Top-K accuracy, mAP, latency, memory
│   └── demo.py               ← Step 6: visual grid demo for recording
├── models/               ← saved descriptors, codebooks, FAISS indexes
├── results/              ← evaluation CSVs, demo grids, reports
├── notebooks/            ← exploratory analysis
├── requirements.txt
└── README.md
```

---

## Setup

```bash
git clone https://github.com/sjekic/computer-vision-group-project.git
cd computer-vision-group-project

python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

---

## Step 0 — Download Dataset from Google Drive

Each teammate needs a `credentials.json` from Google Cloud Console (one-time, ~3 min setup):

1. Go to [console.cloud.google.com/apis/credentials](https://console.cloud.google.com/apis/credentials)
2. Enable **Google Drive API** → **Create Credentials → OAuth 2.0 Client ID → Desktop app**
3. Download JSON → rename to `credentials.json` → place in project root

```bash
python src/download_dataset.py        # browser auth on first run, silent after
```

> `credentials.json` and `token.json` are gitignored — never committed.

---

## Step 1 — Preprocessing

```bash
# Resize to 640×480, apply CLAHE, generate 20 augmentation variants per image
python src/preprocess.py --augment

# Validate dataset health (brightness, duplicates, per-location counts)
python src/dataset_stats.py
```

**Augmentations (20 per image):** brightness ±35%, overexposure, contrast ±, saturation ±, warm/cool tint, blur (×2), noise, JPEG artefacts, flip, rotation ±5°, corner crops, zoom, perspective skew (H+V).

---

## Step 2 — Feature Extraction

```bash
# Extract all three methods (recommended)
python src/extract.py --method all

# Or individually
python src/extract.py --method sift
python src/extract.py --method orb
python src/extract.py --method dinov2

# Quick test on originals only (275 images, ~2 min)
python src/extract.py --method all --no-aug
```

Outputs saved to `models/`: descriptor `.npy` arrays, codebooks, labels.

---

## Step 3 — Build FAISS Indexes

```bash
python src/index.py --method all
```

---

## Step 4 — Query the System

```bash
# Query with a single image
python src/retrieve.py --query data/query/my_photo.jpg --k 5

# Show results visually (OpenCV window)
python src/retrieve.py --query data/query/my_photo.jpg --show

# Use only one method
python src/retrieve.py --query data/query/my_photo.jpg --method dinov2
```

---

## Step 5 — Evaluate

```bash
# Evaluate using augmented variants as synthetic queries
python src/evaluate.py --method all --save

# Evaluate using a dedicated query folder (ground truth = sub-folder name)
python src/evaluate.py --method all --query-dir data/query/ --save

# Quick run on 50 queries
python src/evaluate.py --method all --max-queries 50
```

**Output:**
```
===========================================================================
  EVALUATION RESULTS
===========================================================================
  Method              mAP     Top-1   Top-3   Top-5  Top-10  Lat(ms)  Mem(MB)
  -----------------------------------------------------------------------
  SIFT+VLAD          0.xxx   xx.x%   xx.x%   xx.x%  xx.x%     x.xx     x.x
  ORB+BoW            0.xxx   xx.x%   xx.x%   xx.x%  xx.x%     x.xx     x.x
  DINOv2 ViT-S/14   0.xxx   xx.x%   xx.x%   xx.x%  xx.x%     x.xx     x.x
===========================================================================
```

Results saved to `results/evaluation_results.csv`.

---

## Step 6 — Demo (for recorded demonstration)

```bash
# Show visual grid: query + top-5 matches per method
python src/demo.py --query data/query/my_photo.jpg

# Save the grid image instead of showing it
python src/demo.py --query data/query/my_photo.jpg --save --no-show
```

The demo window shows a colour-coded grid with the query image and retrieved results. Correct location matches are highlighted in green.

---

## Evaluation Metrics

| Metric | Description |
|---|---|
| **Top-K Accuracy** | % of queries where correct location appears in top K results (K=1,3,5,10) |
| **mAP** | Mean Average Precision — accounts for ranking quality |
| **Query Latency** | Mean search time per query in ms (FAISS search only) |
| **Memory** | Index footprint in MB |

---

## Team

IE University — Computer Vision Group Project, 2025/26
