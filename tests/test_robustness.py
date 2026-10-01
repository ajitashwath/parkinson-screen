from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.robustness import (
    apply_blur,
    apply_dilate,
    apply_erode,
    apply_gaussian_noise,
    apply_jpeg_compression,
    apply_rotation,
    build_corrupted_manifest,
    plot_robustness_curves,
    summarize_by_corruption_severity,
)


def test_apply_blur_changes_image(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    blurred = apply_blur(gray, severity=2)
    assert blurred.shape == gray.shape
    assert not np.array_equal(blurred, gray)


def test_apply_blur_stronger_at_higher_severity(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    mild = apply_blur(gray, severity=1)
    strong = apply_blur(gray, severity=3)
    assert np.abs(strong.astype(int) - gray.astype(int)).sum() > np.abs(
        mild.astype(int) - gray.astype(int)
    ).sum()


def test_apply_gaussian_noise_is_deterministic_for_a_given_seed(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    first = apply_gaussian_noise(gray, severity=2, seed=5)
    second = apply_gaussian_noise(gray, severity=2, seed=5)
    assert np.array_equal(first, second)


def test_apply_gaussian_noise_changes_image(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    noisy = apply_gaussian_noise(gray, severity=2, seed=0)
    assert not np.array_equal(noisy, gray)


def test_apply_rotation_preserves_shape(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    rotated = apply_rotation(gray, severity=2)
    assert rotated.shape == gray.shape


def test_apply_dilate_thickens_stroke(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    dilated = apply_dilate(gray, severity=2)
    original_stroke_pixels = int((gray < 200).sum())
    dilated_stroke_pixels = int((dilated < 200).sum())
    assert dilated_stroke_pixels > original_stroke_pixels


def test_apply_erode_thins_stroke(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    eroded = apply_erode(gray, severity=1)
    original_stroke_pixels = int((gray < 200).sum())
    eroded_stroke_pixels = int((eroded < 200).sum())
    assert eroded_stroke_pixels < original_stroke_pixels


def test_apply_jpeg_compression_changes_image(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    compressed = apply_jpeg_compression(gray, severity=3)
    assert compressed.shape == gray.shape
    assert not np.array_equal(compressed, gray)


def test_build_corrupted_manifest_writes_real_files(tmp_path: Path, spiral_image_factory) -> None:
    source_path = tmp_path / "source.png"
    spiral_image_factory().save(source_path)
    test_df = pd.DataFrame(
        {"path": [str(source_path)], "label": ["healthy"], "subject_id": ["S1"], "fold": [0]}
    )
    cache_dir = tmp_path / "cache"
    corrupted = build_corrupted_manifest(test_df, "blur", 2, cache_dir)
    assert len(corrupted) == 1
    assert Path(corrupted.iloc[0]["path"]).exists()
    assert corrupted.iloc[0]["path"] != str(source_path)


def test_summarize_by_corruption_severity_aggregates_mean() -> None:
    results = [
        {"fold": 0, "corruption": "blur", "severity": 1, "roc_auc": 0.6},
        {"fold": 1, "corruption": "blur", "severity": 1, "roc_auc": 0.8},
    ]
    summary = summarize_by_corruption_severity(results)
    row = summary[(summary["corruption"] == "blur") & (summary["severity"] == 1)]
    assert row["roc_auc"].iloc[0] == 0.7


def test_plot_robustness_curves_returns_figure_with_traces() -> None:
    results = [
        {"fold": 0, "corruption": "none", "severity": 0, "roc_auc": 0.7},
        {"fold": 0, "corruption": "blur", "severity": 1, "roc_auc": 0.6},
        {"fold": 0, "corruption": "blur", "severity": 2, "roc_auc": 0.55},
    ]
    summary = summarize_by_corruption_severity(results)
    figure = plot_robustness_curves(summary)
    assert len(figure.data) >= 1
