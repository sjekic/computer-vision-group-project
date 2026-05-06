"""AnyLoc-style DINOv2 patch-token aggregation helpers."""

from __future__ import annotations

import cv2
import numpy as np


def extract_dinov2_patch_tokens(tensor, model) -> np.ndarray:
    """Return DINOv2 patch tokens as a (num_patches, dim) float32 array."""
    import torch

    with torch.no_grad():
        if hasattr(model, "forward_features"):
            features = model.forward_features(tensor)
            if isinstance(features, dict) and "x_norm_patchtokens" in features:
                tokens = features["x_norm_patchtokens"]
            elif isinstance(features, dict) and "x_prenorm" in features:
                tokens = features["x_prenorm"][:, 1:, :]
            else:
                raise RuntimeError("DINOv2 forward_features did not expose patch tokens")
        elif hasattr(model, "get_intermediate_layers"):
            tokens = model.get_intermediate_layers(
                tensor,
                n=1,
                reshape=False,
                return_class_token=False,
            )[0]
        else:
            raise RuntimeError("DINOv2 model does not expose patch-token features")

    return tokens.squeeze(0).detach().cpu().numpy().astype(np.float32)


def compute_vlad(tokens: np.ndarray | None, codebook: np.ndarray) -> np.ndarray:
    """VLAD-aggregate local descriptors into one L2-normalized vector."""
    k, d = codebook.shape
    vlad = np.zeros((k, d), dtype=np.float32)

    if tokens is None or len(tokens) == 0:
        return vlad.flatten()

    flann = cv2.FlannBasedMatcher({"algorithm": 1, "trees": 4}, {"checks": 32})
    matches = flann.knnMatch(tokens.astype(np.float32), codebook.astype(np.float32), k=1)

    for i, match in enumerate(matches):
        if match:
            word_idx = match[0].trainIdx
            vlad[word_idx] += tokens[i] - codebook[word_idx]

    for i in range(k):
        norm = np.linalg.norm(vlad[i])
        if norm > 0:
            vlad[i] /= norm

    vec = vlad.flatten()
    norm = np.linalg.norm(vec)
    return vec / norm if norm > 0 else vec


def encode_anyloc_image(img_rgb: np.ndarray, model, device: str, transform, codebook: np.ndarray) -> np.ndarray:
    """Encode an RGB uint8 image as an AnyLoc-DINOv2-VLAD vector."""
    from PIL import Image

    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)
    tokens = extract_dinov2_patch_tokens(tensor, model)
    return compute_vlad(tokens, codebook)
