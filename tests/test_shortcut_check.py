from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from training.shortcut_check import (
    build_background_only_manifest,
    erase_stroke_pixels,
    plot_shortcut_comparison,
    render_finding,
    summarize_condition,
)


def test_erase_stroke_pixels_removes_dark_content(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    original_stroke_pixels = int((gray < 200).sum())
    erased = erase_stroke_pixels(gray)
    erased_stroke_pixels = int((erased < 200).sum())
    assert original_stroke_pixels > 0
    assert erased_stroke_pixels < original_stroke_pixels


def test_erase_stroke_pixels_handles_blank_image(blank_image) -> None:
    gray = np.array(blank_image)
    erased = erase_stroke_pixels(gray)
    assert np.array_equal(erased, gray)


def test_erase_stroke_pixels_preserves_shape(spiral_image_factory) -> None:
    gray = np.array(spiral_image_factory())
    erased = erase_stroke_pixels(gray)
    assert erased.shape == gray.shape


def test_build_background_only_manifest_writes_real_files(
    tmp_path: Path, spiral_image_factory
) -> None:
    source_path = tmp_path / "source.png"
    spiral_image_factory().save(source_path)
    manifest = pd.DataFrame(
        {"path": [str(source_path)], "label": ["healthy"], "subject_id": ["S1"], "fold": [0]}
    )
    cache_dir = tmp_path / "cache"
    background_manifest = build_background_only_manifest(manifest, cache_dir)
    assert len(background_manifest) == 1
    out_path = Path(background_manifest.iloc[0]["path"])
    assert out_path.exists()
    assert out_path != source_path


def test_summarize_condition_averages_across_folds() -> None:
    results = [
        {"condition": "normal", "roc_auc": 0.6, "auc_ci_lower": 0.5},
        {"condition": "normal", "roc_auc": 0.8, "auc_ci_lower": 0.7},
    ]
    summary = summarize_condition(results, "normal")
    assert summary["mean_auc"] == 0.7
    assert summary["mean_auc_ci_lower"] == 0.6
    assert summary["n_folds"] == 2


def test_summarize_condition_missing_condition_returns_nan() -> None:
    summary = summarize_condition([{"condition": "normal", "roc_auc": 0.6}], "background_only")
    assert np.isnan(summary["mean_auc"])
    assert summary["n_folds"] == 0


def test_render_finding_detects_leak_when_background_auc_above_chance() -> None:
    results = [
        {"condition": "normal", "roc_auc": 0.9, "auc_ci_lower": 0.8},
        {"condition": "background_only", "roc_auc": 0.85, "auc_ci_lower": 0.75},
        {"condition": "skeleton_rerender", "roc_auc": 0.8, "auc_ci_lower": 0.7},
    ]
    finding = render_finding(results)
    assert "possible shortcut detected" in finding


def test_render_finding_reports_no_leak_when_background_auc_near_chance() -> None:
    results = [
        {"condition": "normal", "roc_auc": 0.9, "auc_ci_lower": 0.8},
        {"condition": "background_only", "roc_auc": 0.5, "auc_ci_lower": 0.3},
        {"condition": "skeleton_rerender", "roc_auc": 0.8, "auc_ci_lower": 0.7},
    ]
    finding = render_finding(results)
    assert "no shortcut detected" in finding


def test_plot_shortcut_comparison_returns_figure_with_bars() -> None:
    results = [
        {"condition": "normal", "roc_auc": 0.6, "auc_ci_lower": 0.5},
        {"condition": "background_only", "roc_auc": 0.5, "auc_ci_lower": 0.4},
        {"condition": "skeleton_rerender", "roc_auc": 0.55, "auc_ci_lower": 0.45},
    ]
    figure = plot_shortcut_comparison(results)
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 3
