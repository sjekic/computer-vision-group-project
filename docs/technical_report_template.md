# Technical Report Template

## 1. Introduction

State the problem: fast visual place recognition around the IE Tower area.
Describe the retrieval task and why speed/accuracy trade-offs matter.

## 2. Dataset

Summarize `docs/dataset_card.md`: locations, image counts, capture protocol,
lighting/viewpoint variability, and train/query split.

## 3. Preprocessing

Describe resizing, padding, CLAHE, and augmentation. Explain that augmentations
are used for robustness testing and excluded from the default searchable
database to avoid leakage.

## 4. Methods

### SIFT + VLAD

Explain keypoint detection, local descriptors, k-means vocabulary, VLAD
encoding, L2 normalization, and FAISS L2 search.

### ORB + BoW

Explain ORB descriptors, visual vocabulary, normalized histogram encoding, and
why this is expected to be fast but less discriminative.

### DINOv2

Explain the global ViT-S/14 embedding, L2 normalization, and cosine similarity
implemented as inner product search.

### AnyLoc-DINOv2-VLAD

Explain that this method extracts DINOv2 patch tokens instead of only the global
image embedding. A visual vocabulary is learned from sampled patch tokens, then
VLAD aggregates patch-level residuals into one global place descriptor. This is
inspired by AnyLoc-style visual place recognition: no supervised training, but
more spatial/place detail than the CLS/global embedding.

## 5. Indexing And Scalability

Compare FAISS Flat, IVF, and HNSW when available. Explain exact vs approximate
search and the expected speed/accuracy/memory trade-off.

## 6. Evaluation Protocol

Define Top-K accuracy, AP/mAP, full query latency, FAISS-only search latency,
and memory measurement. State the random seed and hardware.

## 7. Results

Paste the main table from `results/evaluation_results.csv`.

## 8. Discussion

Discuss which method is most accurate, which is fastest, and where each fails.
Include qualitative demo grids for successes and failures.

## 9. Conclusion

Summarize the recommended method for the final system and justify it based on
accuracy, speed, scalability, and robustness.
