# Dataset Card - IE Tower Visual Place Recognition

Fill this file before submission. The assignment requires a captured and
documented dataset, so this card should match the actual images you submit.

## Summary

- Dataset name:
- Group members:
- Capture area: IE Tower area
- Capture dates:
- Devices used:
- Total database images:
- Total held-out query images:
- Number of locations/classes:

## Location Classes

| Location | Database images | Query images | Notes |
|---|---:|---:|---|
| entrance | 0 | 0 | Replace with real count |
| cafeteria | 0 | 0 | Replace with real count |

## Capture Protocol

For each location, capture:

- Multiple viewpoints: front, left/right angle, close/far.
- Multiple scales: wide scene and detail views.
- Multiple lighting conditions: daylight, indoor light, shadow/glare when possible.
- Intra-class variability: people present/absent, doors open/closed, small object changes.

Recommended minimum: 15-25 database images and 5-10 held-out query images per
location. More is better if the classes remain balanced.

## Data Split

- Database/reference images: `data/raw/<location>/`
- Held-out real queries: `data/query/<location>/`
- Synthetic queries: generated in `data/processed/` by `src/preprocess.py --augment`

Do not place the same image in both the database and held-out query set.

## Quality Notes

- Very dark images:
- Overexposed images:
- Near duplicates removed:
- Ambiguous locations/classes:

## Ethics And Privacy

- Avoid close-up identifiable faces when possible.
- Blur or remove images with sensitive personal information.
- Confirm that all captured images are from the allowed IE Tower project area.
