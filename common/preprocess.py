"""Deterministic image preprocessing shared by training and the backend.

This module is the single source of truth for turning a raw spiral or wave
drawing into the fixed-size, three-channel array a model consumes. Training
reads paths from disk; the backend receives base64 PNG bytes from a browser
canvas. Both call preprocess_image, so a live canvas trace is processed
exactly like a training image. The backend copies this file into its Docker
image at build time rather than depending on the training package.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path
from typing import Literal, Union

import cv2
import numpy as np
from PIL import Image

PreprocessMode = Literal["standard", "skeleton_rerender"]
ImageSource = Union[str, "Path", bytes, bytearray, np.ndarray, Image.Image]

WHITE = 255


@dataclass(frozen=True)
class PreprocessConfig:
    """Parameters for preprocess_image. Defaults match the training config
    in configs/data.yaml; keep the two in sync."""

    size: int = 224
    padding_frac: float = 0.08
    binarize_threshold: int = 200
    skeleton_stroke_width: int = 3
    mode: PreprocessMode = "standard"


def load_grayscale(source: ImageSource) -> np.ndarray:
    """Load an image from a path, raw bytes, a PIL Image or an array, as a
    uint8 grayscale array. Accepting all four keeps this function usable
    from both a training dataset (paths) and the backend (bytes)."""
    if isinstance(source, np.ndarray):
        if source.ndim == 3:
            return np.array(Image.fromarray(source.astype(np.uint8)).convert("L"))
        return source.astype(np.uint8)
    if isinstance(source, Image.Image):
        return np.array(source.convert("L"))
    if isinstance(source, bytes | bytearray):
        with Image.open(io.BytesIO(source)) as image:
            return np.array(image.convert("L"))
    with Image.open(source) as image:
        return np.array(image.convert("L"))


def crop_to_content(
    gray: np.ndarray, padding_frac: float = 0.08, threshold: int = 200
) -> np.ndarray:
    """Crop to the drawing's bounding box with padding. A blank image (no
    pixel darker than threshold) is returned unchanged rather than raising,
    since a blank canvas trace is a real, if unhelpful, input."""
    mask = gray < threshold
    if not mask.any():
        return gray
    rows = np.any(mask, axis=1)
    cols = np.any(mask, axis=0)
    top, bottom = np.where(rows)[0][[0, -1]]
    left, right = np.where(cols)[0][[0, -1]]
    height, width = gray.shape
    pad_y = int(round((bottom - top + 1) * padding_frac))
    pad_x = int(round((right - left + 1) * padding_frac))
    top = max(0, top - pad_y)
    bottom = min(height - 1, bottom + pad_y)
    left = max(0, left - pad_x)
    right = min(width - 1, right + pad_x)
    return gray[top : bottom + 1, left : right + 1]


def resize_with_padding(gray: np.ndarray, size: int = 224) -> np.ndarray:
    """Resize keeping aspect ratio, then pad with white to a size x size
    square, so the drawing's proportions (relevant to spiral_deviation and
    similar shape features) are never stretched."""
    height, width = gray.shape
    if height == 0 or width == 0:
        return np.full((size, size), WHITE, dtype=np.uint8)
    scale = size / max(height, width)
    new_height = max(1, round(height * scale))
    new_width = max(1, round(width * scale))
    interpolation = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(gray, (new_width, new_height), interpolation=interpolation)
    canvas = np.full((size, size), WHITE, dtype=np.uint8)
    top = (size - new_height) // 2
    left = (size - new_width) // 2
    canvas[top : top + new_height, left : left + new_width] = resized
    return canvas


def normalize(gray: np.ndarray) -> np.ndarray:
    """Scale to float32 in [0, 1]. Backbone-specific ImageNet mean/std
    normalization happens in the training transform (Phase 3), not here,
    since it depends on which pretrained backbone is used."""
    return gray.astype(np.float32) / 255.0


def to_three_channel(array: np.ndarray) -> np.ndarray:
    """Replicate a single channel into 3 channels, since every backbone in
    CLAUDE.md section 2 expects an ImageNet-shaped input."""
    return np.stack([array, array, array], axis=-1)


def skeleton_rerender(gray: np.ndarray, threshold: int = 200, stroke_width: int = 3) -> np.ndarray:
    """Skeletonize the stroke and redraw it at a fixed width. Strips away
    stroke-thickness and pen-pressure cues (and any scanner artifact riding
    on them), used by the Phase 4 shortcut and robustness experiments to
    test whether the model actually reads the drawing rather than the scan."""
    from skimage.morphology import skeletonize

    binary = gray < threshold
    if not binary.any():
        return gray.copy()
    skeleton = skeletonize(binary)
    kernel = np.ones((stroke_width, stroke_width), dtype=np.uint8)
    dilated = cv2.dilate(skeleton.astype(np.uint8), kernel, iterations=1)
    rerendered = np.full_like(gray, WHITE)
    rerendered[dilated > 0] = 0
    return rerendered


def preprocess_image(source: ImageSource, config: PreprocessConfig | None = None) -> np.ndarray:
    """Run the full deterministic pipeline: optional skeleton re-render,
    crop to content, resize with white padding, normalize, replicate to 3
    channels. Returns a float32 array of shape (size, size, 3) in [0, 1].

    Deterministic: the same input always produces the same output, since
    every step is a fixed function with no randomness. Augmentation is a
    separate, train-only step in training/data.py, applied after this."""
    cfg = config or PreprocessConfig()
    gray = load_grayscale(source)
    if cfg.mode == "skeleton_rerender":
        gray = skeleton_rerender(gray, cfg.binarize_threshold, cfg.skeleton_stroke_width)
    cropped = crop_to_content(gray, cfg.padding_frac, cfg.binarize_threshold)
    resized = resize_with_padding(cropped, cfg.size)
    normalized = normalize(resized)
    return to_three_channel(normalized)
