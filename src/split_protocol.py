"""Helpers for keeping database and query splits separate."""

from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any


def _truthy(value: Any) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y"}
    return bool(value)


def select_database_records(
    records: Iterable[Mapping[str, Any]],
    include_augmented: bool = False,
) -> list[Mapping[str, Any]]:
    """Select records allowed in the searchable database.

    The honest default is originals only. Augmented samples should normally be
    reserved for stress testing unless the experiment explicitly says otherwise.
    """
    if include_augmented:
        return list(records)
    return [record for record in records if not _truthy(record.get("augmented", False))]


def select_synthetic_query_records(
    records: Iterable[Mapping[str, Any]],
    aug_types: Iterable[str],
) -> list[Mapping[str, Any]]:
    """Select augmented records that can be used as synthetic query images."""
    allowed = set(aug_types)
    return [
        record
        for record in records
        if _truthy(record.get("augmented", False)) and record.get("aug_type") in allowed
    ]


def exclude_indexed_queries(
    queries: Iterable[tuple[Path, str | None, str]],
    indexed_image_ids: Iterable[str],
) -> list[tuple[Path, str | None, str]]:
    """Remove query images whose image id is already present in the FAISS index."""
    indexed = set(indexed_image_ids)
    return [query for query in queries if query[2] not in indexed]
