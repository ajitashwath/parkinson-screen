"""Robustness check: apply blur, Gaussian noise, rotation, stroke
thickness change (dilate and erode) and JPEG compression at 3 severities
to the test images only, and evaluate an already-trained model at each
severity. Plots ROC-AUC vs severity per corruption type, per CLAUDE.md
Phase 4 rule 4. Fits one model per fold with training/train.py's
fit_backbone, then reuses it across every corrupted test set via
evaluate_fitted_backbone, rather than retraining for each corruption.

Usage:
    python -m training.robustness --config configs/train_resnet50.yaml --fast
"""

from __future__ import annotations

import argparse
import json
import tempfile
from collections.abc import Callable
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from common.preprocess import load_grayscale
from training.data import assign_folds
from training.train import (
    evaluate_fitted_backbone,
    fit_backbone,
    load_config,
    sample_fast_subset,
    set_all_seeds,
)

SEVERITIES = (1, 2, 3)
STROKE_THRESHOLD = 200


def apply_blur(gray: np.ndarray, severity: int) -> np.ndarray:
    kernel_size = 2 * severity + 1
    return cv2.GaussianBlur(gray, (kernel_size, kernel_size), 0)


def apply_gaussian_noise(gray: np.ndarray, severity: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    std = severity * 10.0
    noisy = gray.astype(np.float32) + rng.normal(0, std, size=gray.shape)
    return np.clip(noisy, 0, 255).astype(np.uint8)


def apply_rotation(gray: np.ndarray, severity: int) -> np.ndarray:
    angle_degrees = severity * 5.0
    height, width = gray.shape
    matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle_degrees, 1.0)
    return cv2.warpAffine(gray, matrix, (width, height), borderValue=(255.0,))


def apply_dilate(gray: np.ndarray, severity: int) -> np.ndarray:
    """Thicker strokes: dilate the stroke mask, then re-render at the new
    thickness rather than leaving stray dilated pixel values."""
    kernel = np.ones((severity + 1, severity + 1), np.uint8)
    binary = (gray < STROKE_THRESHOLD).astype(np.uint8)
    dilated = cv2.dilate(binary, kernel, iterations=1)
    result = np.full_like(gray, 255)
    result[dilated > 0] = 0
    return result


def apply_erode(gray: np.ndarray, severity: int) -> np.ndarray:
    """Thinner strokes: the erosion counterpart to apply_dilate."""
    kernel = np.ones((severity + 1, severity + 1), np.uint8)
    binary = (gray < STROKE_THRESHOLD).astype(np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)
    result = np.full_like(gray, 255)
    result[eroded > 0] = 0
    return result


def apply_jpeg_compression(gray: np.ndarray, severity: int) -> np.ndarray:
    quality = {1: 50, 2: 25, 3: 10}[severity]
    success, encoded = cv2.imencode(".jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if not success:
        return gray
    decoded = cv2.imdecode(encoded, cv2.IMREAD_GRAYSCALE)
    return decoded if decoded is not None else gray


CorruptionFn = Callable[[np.ndarray, int], np.ndarray]
CORRUPTIONS: dict[str, CorruptionFn] = {
    "blur": apply_blur,
    "gaussian_noise": apply_gaussian_noise,
    "rotation": apply_rotation,
    "dilate": apply_dilate,
    "erode": apply_erode,
    "jpeg_compression": apply_jpeg_compression,
}


def build_corrupted_manifest(
    test_df: pd.DataFrame, corruption_name: str, severity: int, cache_dir: Path
) -> pd.DataFrame:
    fn = CORRUPTIONS[corruption_name]
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in test_df.iterrows():
        gray = load_grayscale(row["path"])
        corrupted = fn(gray, severity)
        out_path = cache_dir / f"{Path(row['path']).stem}_{corruption_name}_{severity}.png"
        Image.fromarray(corrupted).save(out_path)
        new_row = row.copy()
        new_row["path"] = str(out_path)
        rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def run_robustness_check(
    config: dict[str, Any], seed: int, fast: bool, cache_dir: Path
) -> list[dict[str, Any]]:
    set_all_seeds(seed)
    manifest = pd.read_csv(config["folds_path"])
    if fast:
        manifest = sample_fast_subset(manifest, config.get("fast_sample_per_class", 6), seed)
        manifest = assign_folds(manifest, n_folds=2, seed=seed)

    backbone = config["backbone"]
    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)

    results: list[dict[str, Any]] = []
    for fold in sorted(int(f) for f in manifest["fold"].unique()):
        train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
        test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        trainer, model, threshold, _ = fit_backbone(
            backbone, train_df, config, seed, max_epochs, pretrained
        )

        clean_eval = evaluate_fitted_backbone(trainer, model, test_df, config, threshold)
        results.append(
            {"fold": fold, "corruption": "none", "severity": 0, "roc_auc": clean_eval["roc_auc"]}
        )

        for corruption_name in CORRUPTIONS:
            for severity in SEVERITIES:
                corrupted_test_df = build_corrupted_manifest(
                    test_df, corruption_name, severity, cache_dir
                )
                corrupted_eval = evaluate_fitted_backbone(
                    trainer, model, corrupted_test_df, config, threshold
                )
                results.append(
                    {
                        "fold": fold,
                        "corruption": corruption_name,
                        "severity": severity,
                        "roc_auc": corrupted_eval["roc_auc"],
                    }
                )
    return results


def summarize_by_corruption_severity(results: list[dict[str, Any]]) -> pd.DataFrame:
    df = pd.DataFrame(results)
    grouped = df.groupby(["corruption", "severity"])["roc_auc"].mean().reset_index()
    return grouped.sort_values(["corruption", "severity"])


def plot_robustness_curves(summary: pd.DataFrame) -> go.Figure:
    figure = go.Figure()
    clean_rows = summary[summary["corruption"] == "none"]
    clean_auc = float(clean_rows["roc_auc"].iloc[0]) if not clean_rows.empty else None
    for corruption_name in sorted(c for c in summary["corruption"].unique() if c != "none"):
        rows = summary[summary["corruption"] == corruption_name].sort_values("severity")
        if clean_auc is not None:
            severities = [0, *rows["severity"].tolist()]
            aucs = [clean_auc, *rows["roc_auc"].tolist()]
        else:
            severities = rows["severity"].tolist()
            aucs = rows["roc_auc"].tolist()
        figure.add_trace(
            go.Scatter(x=severities, y=aucs, mode="markers+lines", name=corruption_name)
        )
    figure.update_layout(
        title="Robustness: ROC-AUC vs corruption severity (0 = clean test set)",
        xaxis_title="Severity",
        yaxis_title="ROC-AUC",
    )
    return figure


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_resnet50.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    reports_dir = Path(args.reports_dir)
    reports_dir.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(prefix="robustness_") as cache_dir:
        results = run_robustness_check(config, seed, args.fast, Path(cache_dir))

    results_path = reports_dir / "robustness_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    summary = summarize_by_corruption_severity(results)
    summary_path = reports_dir / "robustness_summary.csv"
    summary.to_csv(summary_path, index=False)

    figure = plot_robustness_curves(summary)
    figure_path = reports_dir / "robustness.html"
    figure.write_html(figure_path, include_plotlyjs="cdn")

    print(summary.to_string(index=False))
    print(f"\nWrote results to {results_path}")
    print(f"Wrote summary to {summary_path}")
    print(f"Wrote figure to {figure_path}")


if __name__ == "__main__":
    main()
