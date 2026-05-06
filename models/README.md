# Models Directory

Generated artifacts:

- `*_descriptors.npy`: global retrieval vectors.
- `*_labels.npy`: image IDs aligned with descriptor rows.
- `sift_codebook.npy`, `orb_codebook.npy`, and `anyloc_dinov2_codebook.npy`: visual vocabularies.
- `*.index`: FAISS indexes.
- `*_metadata.json`: index settings and footprint.
- `extraction_metadata.json`: extraction split and seed.

Regenerate with:

```bash
python src/extract.py --method all --seed 42 --rebuild-codebooks
python src/index.py --method all --index-type flat
```
