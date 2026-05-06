"""Retrieval metrics used by the evaluation pipeline.

The functions in this module are dependency-free on purpose so the scoring
logic can be tested without OpenCV, FAISS, or PyTorch installed.
"""

from collections import Counter
from collections.abc import Iterable


def location_from_label(label: str) -> str:
    """Return the location part of an image id such as ``hall/img001``."""
    return str(label).split("/")[0]


def relevant_counts_by_location(labels: Iterable[str]) -> Counter:
    """Count how many indexed database images belong to each location."""
    return Counter(location_from_label(label) for label in labels)


def topk_hit(retrieved_locations: list[str], gt_location: str | None, k: int) -> bool:
    """Return True when the ground-truth location appears in the first k results."""
    if not gt_location:
        return False
    return gt_location in retrieved_locations[:k]


def average_precision_at_k(
    retrieved_locations: list[str],
    gt_location: str | None,
    total_relevant: int,
    k: int | None = None,
) -> float:
    """Compute AP@K for one query.

    AP is normalized by the number of relevant database items, capped at K.
    Normalizing only by retrieved hits inflates scores when many relevant
    images are missed, so this function follows the standard AP@K protocol.
    """
    if not gt_location or total_relevant <= 0:
        return 0.0

    cutoff = len(retrieved_locations) if k is None else min(k, len(retrieved_locations))
    normalizer = min(total_relevant, cutoff)
    if normalizer <= 0:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for rank, location in enumerate(retrieved_locations[:cutoff], 1):
        if location == gt_location:
            hits += 1
            precision_sum += hits / rank

    return precision_sum / normalizer
