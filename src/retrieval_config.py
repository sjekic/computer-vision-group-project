"""Shared method and artifact naming helpers for retrieval scripts."""

METHOD_ARTIFACTS = {
    "sift": {
        "display": "SIFT+VLAD",
        "descriptor": "sift_vlad_descriptors.npy",
        "label": "sift_vlad_labels.npy",
        "index": "sift_vlad.index",
        "codebook": "sift_codebook.npy",
        "metric": "l2",
    },
    "orb": {
        "display": "ORB+BoW",
        "descriptor": "orb_descriptors.npy",
        "label": "orb_labels.npy",
        "index": "orb_bow.index",
        "codebook": "orb_codebook.npy",
        "metric": "l2",
    },
    "dinov2": {
        "display": "DINOv2 ViT-S/14",
        "descriptor": "dinov2_descriptors.npy",
        "label": "dinov2_labels.npy",
        "index": "dinov2.index",
        "codebook": None,
        "metric": "ip",
    },
    "anyloc": {
        "display": "AnyLoc-DINOv2-VLAD",
        "descriptor": "anyloc_dinov2_vlad_descriptors.npy",
        "label": "anyloc_dinov2_vlad_labels.npy",
        "index": "anyloc_dinov2_vlad.index",
        "codebook": "anyloc_dinov2_codebook.npy",
        "metric": "ip",
    },
}


def resolve_methods(choice: str) -> list[str]:
    """Expand a CLI method choice into concrete method keys."""
    if choice == "both":
        return ["sift", "dinov2"]
    if choice == "all":
        return ["sift", "orb", "dinov2", "anyloc"]
    if choice in METHOD_ARTIFACTS:
        return [choice]
    raise ValueError(f"Unknown method: {choice}")


def descriptor_filename(method: str) -> str:
    return METHOD_ARTIFACTS[method]["descriptor"]


def label_filename(method: str) -> str:
    return METHOD_ARTIFACTS[method]["label"]


def codebook_filename(method: str) -> str | None:
    return METHOD_ARTIFACTS[method]["codebook"]


def method_display_name(method: str) -> str:
    return METHOD_ARTIFACTS[method]["display"]


def method_metric(method: str) -> str:
    return METHOD_ARTIFACTS[method]["metric"]


def index_filename(method: str, index_type: str = "flat") -> str:
    """Return the FAISS index filename for a method and index type."""
    base = METHOD_ARTIFACTS[method]["index"]
    if index_type == "flat":
        return base
    stem, suffix = base.rsplit(".", 1)
    return f"{stem}_{index_type}.{suffix}"
