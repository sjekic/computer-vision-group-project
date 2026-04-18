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

## Step 0 — Download the Dataset from Google Drive

The dataset lives in a shared Google Drive folder. Run the download script once and it will mirror all 35 location folders into `data/raw/` automatically.

### One-time credentials setup (each teammate does this once)

You need a `credentials.json` file from Google Cloud Console. This is a free, standard OAuth2 step:

1. Go to [Google Cloud Console](https://console.cloud.google.com/) and create a project (or use an existing one)
2. Navigate to **APIs & Services → Library** and enable the **Google Drive API**
3. Go to **APIs & Services → Credentials → Create Credentials → OAuth 2.0 Client ID**
4. Choose **Desktop app** as the application type, give it any name
5. Click **Download JSON** and rename the file to `credentials.json`
6. Place `credentials.json` in the **project root** (same level as README.md)

> `credentials.json` and `token.json` are in `.gitignore` — they will never be accidentally committed.

### Download

```bash
# First run: opens a browser window for Google sign-in consent (takes ~10 seconds)
python src/download_dataset.py

# Subsequent runs: fully automatic (token is cached in token.json)
python src/download_dataset.py

# Preview what would be downloaded without writing files
python src/download_dataset.py --dry-run

# Force re-download everything (ignore already-existing files)
python src/download_dataset.py --no-resume
```

The script will:
- Authenticate via OAuth2 (browser popup on first run only)
- Walk all 35 location sub-folders recursively
- Download only image files (jpg, png, heic, etc.), skip anything else
- Skip files already present locally (`--resume` is on by default)
- Print a summary of downloaded / skipped / failed counts

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
