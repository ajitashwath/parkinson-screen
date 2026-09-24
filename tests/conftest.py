"""Shared pytest fixtures. All test images are synthetic, generated here,
never real Parkinson's drawing data, per CLAUDE.md rule 2."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw


def draw_spiral(
    size: int = 256, turns: float = 4.0, noise: float = 0.0, seed: int = 0
) -> Image.Image:
    """A simple Archimedean spiral, black stroke on white, as a synthetic
    stand-in for a real drawing. noise adds Gaussian jitter to the radius,
    a crude stand-in for hand tremor."""
    rng = np.random.default_rng(seed)
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    center = size / 2
    max_radius = size / 2 - size * 0.1
    steps = 400
    points: list[tuple[float, float]] = []
    for i in range(steps):
        theta = turns * 2 * math.pi * i / steps
        radius = max_radius * i / steps
        jitter = float(rng.normal(0, noise)) if noise else 0.0
        x = center + (radius + jitter) * math.cos(theta)
        y = center + (radius + jitter) * math.sin(theta)
        points.append((x, y))
    draw.line(points, fill=0, width=3)
    return image


def draw_wave(
    size: int = 256, cycles: float = 3.0, noise: float = 0.0, seed: int = 0
) -> Image.Image:
    """A horizontal sine wave, black stroke on white: structurally distinct
    from draw_spiral so the two kinds do not collide under a perceptual
    hash. noise adds Gaussian jitter to the amplitude."""
    rng = np.random.default_rng(seed)
    image = Image.new("L", (size, size), color=255)
    draw = ImageDraw.Draw(image)
    margin = size * 0.1
    width = size - 2 * margin
    mid_y = size / 2
    amplitude = size * 0.2
    steps = 400
    points: list[tuple[float, float]] = []
    for i in range(steps):
        x = margin + width * i / steps
        phase = cycles * 2 * math.pi * i / steps
        jitter = float(rng.normal(0, noise)) if noise else 0.0
        y = mid_y + (amplitude + jitter) * math.sin(phase)
        points.append((x, y))
    draw.line(points, fill=0, width=3)
    return image


@pytest.fixture
def spiral_image_factory():
    return draw_spiral


@pytest.fixture
def blank_image() -> Image.Image:
    return Image.new("L", (256, 256), color=255)


@pytest.fixture
def synthetic_raw_dataset(tmp_path: Path) -> Path:
    """Build a tiny data/raw/<dataset>/<kind>/<label>/*.png fixture tree,
    with one deliberately duplicated file, standing in for a real dataset
    download so tests never touch real (and licensed) Parkinson's drawing
    data. Each image varies turns/cycles and noise by kind, label and index
    so unrelated images do not accidentally collide under a perceptual
    hash; only the deliberate duplicate should share a hash."""
    raw_dir = tmp_path / "raw"
    layout = {
        ("fixture_set", "spiral", "healthy"): 4,
        ("fixture_set", "spiral", "parkinson"): 4,
        ("fixture_set", "wave", "healthy"): 3,
        ("fixture_set", "wave", "parkinson"): 3,
    }
    for (dataset, kind, label), count in layout.items():
        out_dir = raw_dir / dataset / kind / label
        out_dir.mkdir(parents=True, exist_ok=True)
        label_offset = 0.0 if label == "healthy" else 2.0
        noise = 0.0 if label == "healthy" else 0.8
        for i in range(count):
            shape_amount = 3.0 + i * 0.6 + label_offset
            if kind == "spiral":
                image = draw_spiral(turns=shape_amount, noise=noise, seed=i)
            else:
                image = draw_wave(cycles=shape_amount, noise=noise, seed=i)
            image.save(out_dir / f"{label}_{i:02d}.png")
    dup_source = raw_dir / "fixture_set" / "spiral" / "healthy" / "healthy_00.png"
    dup_target = raw_dir / "fixture_set" / "spiral" / "healthy" / "healthy_00_copy.png"
    dup_target.write_bytes(dup_source.read_bytes())
    return raw_dir
