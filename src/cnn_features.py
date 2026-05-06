"""
cnn_features.py
===============
ResNet-50 feature extractor for image retrieval.

Extracts 2048-d global average-pooled features from the layer before the
classification head (avgpool output), then L2-normalises them so cosine
similarity == inner-product on a FAISS IndexFlatIP index.

Usage (standalone)::

    from cnn_features import build_cnn_model, extract_cnn_features
    model, device, transform = build_cnn_model()
    vec = extract_cnn_features(img_rgb, model, device, transform)  # (2048,)
"""

import numpy as np
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image


def build_cnn_model(device: str | None = None):
    """Load a pretrained ResNet-50, strip the classifier, return (model, device, transform)."""
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    backbone = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
    # Remove the final FC layer — keep up to and including avgpool → output: (B, 2048, 1, 1)
    extractor = nn.Sequential(*list(backbone.children())[:-1])
    extractor = extractor.to(device).eval()

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    return extractor, device, transform


def extract_cnn_features(
    img_rgb: np.ndarray,
    model: nn.Module,
    device: str,
    transform,
) -> np.ndarray:
    """Return an L2-normalised (2048,) float32 descriptor for *img_rgb* (H x W x 3 uint8)."""
    pil = Image.fromarray(img_rgb)
    tensor = transform(pil).unsqueeze(0).to(device)  # (1, 3, 224, 224)

    with torch.no_grad():
        feat = model(tensor)                              # (1, 2048, 1, 1)
        feat = feat.squeeze().cpu().numpy().astype(np.float32)  # (2048,)

    norm = np.linalg.norm(feat)
    if norm > 0:
        feat /= norm
    return feat
