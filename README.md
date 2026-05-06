# Fast Image Search - IE Tower Visual Place Recognition

End-to-end image retrieval system for place recognition around the IE Tower area.
Given a query image, the system returns the most visually similar database images
and predicts the most likely location.

## Methods Compared

| Method | Descriptor | Aggregation | Search | Purpose |
|---|---|---|---|---|
| SIFT + VLAD | SIFT local descriptors | VLAD, k=64 | FAISS L2 | Strong classical baseline |
| ORB + BoW | ORB local descriptors | Bag of visual words, k=64 | FAISS L2 | Fast lightweight baseline |
| DINOv2 | ViT-S/14 global embedding | None | FAISS inner product | Deep descriptor baseline |
| AnyLoc-DINOv2-VLAD | DINOv2 patch tokens | VLAD, k=32 | FAISS inner product | VPR-focused foundation-model method |

## Project Structure

```text
computer-vision-group-project/
├── data/
│   ├── raw/          original captured images, one folder per location
│   ├── processed/    resized/CLAHE images, augmentations, manifest.csv
│   └── query/        held-out real query images, one folder per location
├── docs/
│   ├── dataset_card.md
│   ├── experiment_protocol.md
│   └── technical_report_template.md
├── models/           descriptors, codebooks, FAISS indexes, metadata
├── results/          evaluation CSVs, dataset reports, demo grids
├── src/              pipeline scripts
└── tests/            unit tests for metrics and split logic
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

On macOS/Linux, activate with `source .venv/bin/activate`.

## Dataset Layout

Place captured database images here:

```text
data/raw/
├── entrance/
│   ├── img001.jpg
│   └── img002.jpg
├── cafeteria/
└── elevators/
```

Place held-out real query images here:

```text
data/query/
├── entrance/
├── cafeteria/
└── elevators/
```

Fill in `docs/dataset_card.md` with counts, capture devices, lighting, and
location descriptions. This matters for the assignment because the dataset must
be captured by the group and documented.

## 1. Preprocess

```bash
python src/preprocess.py --augment --seed 42
python src/dataset_stats.py
```

Preprocessing resizes to 640x480, pads aspect ratio, optionally applies CLAHE,
and writes `data/processed/manifest.csv`. Augmentations are intended for
robustness testing, not for the default searchable database.

## 2. Extract Features

```bash
python src/extract.py --method all --seed 42 --rebuild-codebooks
```

Important: by default, extraction indexes original images only. This prevents
evaluation leakage when augmented images are used as synthetic queries.

To run a documented ablation where augmentations are included in the database:

```bash
python src/extract.py --method all --include-aug-in-index --seed 42
```

## 3. Build FAISS Indexes

Exact search:

```bash
python src/index.py --method all --index-type flat
```

Scalable approximate search:

```bash
python src/index.py --method all --index-type ivf --nlist 64 --nprobe 10
python src/index.py --method all --index-type hnsw --hnsw-m 32
```

Each index build writes a metadata JSON file in `models/` so the report can
record the index type, metric, vector count, dimension, and memory footprint.

## 4. Query

```bash
python src/retrieve.py --query data/query/entrance/example.jpg --method all --k 5
python src/retrieve.py --query data/query/entrance/example.jpg --method dinov2 --show
python src/retrieve.py --query data/query/entrance/example.jpg --method all --index-type ivf --nprobe 10
```

`retrieve.py` supports `sift`, `orb`, `dinov2`, `anyloc`, `both`, and `all`.

## 5. Evaluate

Recommended real held-out protocol:

```bash
python src/evaluate.py --method all --query-dir data/query --save
```

Synthetic stress-test protocol:

```bash
python src/evaluate.py --method all --save
```

The synthetic protocol uses selected augmented variants from `data/processed/`
and removes any query image ID already present in the index.

Metrics reported:

| Metric | Meaning |
|---|---|
| Top-K accuracy | Whether the correct location appears in top K |
| mAP | AP@maxK normalized by relevant database images |
| Full latency | loading + preprocessing + descriptor extraction + FAISS search |
| Search latency | FAISS search only |
| Memory | serialized FAISS index size when available |

## 6. Demo

```bash
python src/demo.py --query data/query/entrance/example.jpg --method all --save --no-show
```

The saved demo grid can be used in the recorded demonstration.

## Submission Checklist

- Captured dataset or representative subset included/shared.
- `docs/dataset_card.md` completed.
- `results/dataset_report.txt` and `results/dataset_stats.csv` generated.
- `models/` contains descriptors, codebooks, indexes, and metadata.
- `results/evaluation_results.csv` generated for real held-out queries.
- `docs/technical_report_template.md` filled into final report.
- Recorded demo shows at least one successful and one failure/edge case query.

## Verification

```bash
python -m unittest discover -s tests
python -m compileall src
```
