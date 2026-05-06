# Experiment Protocol

Use this protocol for reproducible results in the technical report.

## Hardware

Record the machine used for evaluation:

- CPU:
- RAM:
- GPU:
- OS:
- Python version:

## Commands

```bash
python src/preprocess.py --augment --seed 42
python src/dataset_stats.py
python src/extract.py --method all --seed 42 --rebuild-codebooks
python src/index.py --method all --index-type flat
python src/evaluate.py --method all --query-dir data/query --save
```

Optional scalability experiment:

```bash
python src/index.py --method all --index-type ivf --nlist 64 --nprobe 10
python src/evaluate.py --method all --query-dir data/query --index-type ivf --nprobe 10 --save

python src/index.py --method all --index-type hnsw --hnsw-m 32
python src/evaluate.py --method all --query-dir data/query --index-type hnsw --save
```

## Main Evaluation Table

| Method | Index | Top-1 | Top-3 | Top-5 | mAP | Full latency ms | Search ms | Memory MB |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| SIFT+VLAD | flat | | | | | | | |
| ORB+BoW | flat | | | | | | | |
| DINOv2 | flat | | | | | | | |
| AnyLoc-DINOv2-VLAD | flat | | | | | | | |

## Robustness Table

Run synthetic queries without `--query-dir` and report accuracy by augmentation
family if you extend the evaluator to group results by `aug_type`.

| Query condition | Expected stress | Best method | Worst method | Notes |
|---|---|---|---|---|
| blur | motion/focus blur | | | |
| dark | low light | | | |
| bright | glare | | | |
| skew | viewpoint change | | | |

## Failure Analysis

For at least three failed queries, save the demo grid and explain:

- What the query showed.
- Which wrong location was retrieved.
- Why it likely failed: repeated textures, similar architecture, bad lighting,
  motion blur, occlusion, or too few reference images.
- Which method handled it best and why.
