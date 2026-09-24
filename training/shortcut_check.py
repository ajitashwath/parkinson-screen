"""Shortcut check: erase the stroke pixels (threshold, then fill with the
local background) and train on background only. If performance is
clearly above chance, the dataset leaks paper or scanner cues rather than
the drawing itself. Also compares against the skeleton_rerender
preprocessing mode. Writes the finding plainly, whatever it is, per
CLAUDE.md Phase 4 rule 3: this script never claims "no leak" without
checking, and never hides a leak if one is found.

Usage:
    python -m training.shortcut_check --config configs/train_resnet50.yaml --fast
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path
from typing import Any

import cv2
import pandas as pd
import plotly.graph_objects as go
from PIL import Image

from common.preprocess import load_grayscale
from training.data import assign_folds
from training.train import load_config, sample_fast_subset, set_all_seeds, train_one_backbone

BACKGROUND_ERASE_THRESHOLD = 200
INPAINT_RADIUS = 3
LEAK_AUC_LOWER_BOUND = 0.5


def erase_stroke_pixels(
    gray: Any, threshold: int = BACKGROUND_ERASE_THRESHOLD, inpaint_radius: int = INPAINT_RADIUS
) -> Any:
    """Replace stroke pixels (darker than threshold) with an inpainted
    estimate of the surrounding background, rather than a flat fill:
    a flat fill would also erase any real scanner or paper texture the
    shortcut check is specifically trying to keep intact, so a model
    trained on this can only pick up on background cues, not on any
    trace of the stroke itself."""
    mask = (gray < threshold).astype("uint8") * 255
    if not mask.any():
        return gray.copy()
    return cv2.inpaint(gray, mask, inpaint_radius, cv2.INPAINT_TELEA)


def build_background_only_manifest(manifest: pd.DataFrame, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for _, row in manifest.iterrows():
        gray = load_grayscale(row["path"])
        erased = erase_stroke_pixels(gray)
        out_path = cache_dir / f"{Path(row['path']).stem}_erased.png"
        Image.fromarray(erased).save(out_path)
        new_row = row.copy()
        new_row["path"] = str(out_path)
        rows.append(new_row)
    return pd.DataFrame(rows).reset_index(drop=True)


def run_shortcut_check(
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

    background_manifest = build_background_only_manifest(manifest, cache_dir)
    conditions: list[tuple[str, pd.DataFrame, str]] = [
        ("normal", manifest, "standard"),
        ("background_only", background_manifest, "standard"),
        ("skeleton_rerender", manifest, "skeleton_rerender"),
    ]

    results: list[dict[str, Any]] = []
    for condition_name, condition_manifest, mode in conditions:
        for fold in sorted(condition_manifest["fold"].unique()):
            train_df = condition_manifest[condition_manifest["fold"] != fold].reset_index(drop=True)
            test_df = condition_manifest[condition_manifest["fold"] == fold].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue
            result = train_one_backbone(
                backbone,
                train_df,
                test_df,
                config,
                seed,
                fold,
                max_epochs,
                pretrained,
                preprocess_mode=mode,  # type: ignore[arg-type]
            )
            result["condition"] = condition_name
            results.append(result)
    return results


def summarize_condition(results: list[dict[str, Any]], condition: str) -> dict[str, float]:
    rows = [r for r in results if r["condition"] == condition]
    if not rows:
        return {"mean_auc": float("nan"), "mean_auc_ci_lower": float("nan"), "n_folds": 0}
    return {
        "mean_auc": sum(r["roc_auc"] for r in rows) / len(rows),
        "mean_auc_ci_lower": sum(r["auc_ci_lower"] for r in rows) / len(rows),
        "n_folds": len(rows),
    }


def render_finding(results: list[dict[str, Any]]) -> str:
    normal = summarize_condition(results, "normal")
    background = summarize_condition(results, "background_only")
    skeleton = summarize_condition(results, "skeleton_rerender")

    leak_detected = background["mean_auc_ci_lower"] > LEAK_AUC_LOWER_BOUND
    lines = [
        "# Shortcut check",
        "",
        "Trains on the stroke erased and inpainted with the surrounding",
        "background, so a model can only use background or scanner cues,",
        "never the drawing itself. Compared against normal preprocessing",
        "and the skeleton_rerender mode, which strips stroke-thickness and",
        "pressure cues while keeping the drawing's shape.",
        "",
        f"- normal: mean ROC-AUC {normal['mean_auc']:.3f} "
        f"(mean CI lower bound {normal['mean_auc_ci_lower']:.3f}), {normal['n_folds']} folds",
        f"- background_only: mean ROC-AUC {background['mean_auc']:.3f} "
        f"(mean CI lower bound {background['mean_auc_ci_lower']:.3f}), "
        f"{background['n_folds']} folds",
        f"- skeleton_rerender: mean ROC-AUC {skeleton['mean_auc']:.3f} "
        f"(mean CI lower bound {skeleton['mean_auc_ci_lower']:.3f}), "
        f"{skeleton['n_folds']} folds",
        "",
    ]
    if leak_detected:
        lines += [
            "## Finding: possible shortcut detected",
            "",
            "The background-only model's AUC confidence interval sits",
            "clearly above chance (0.5). Some of what the normal model",
            "learns may come from paper, scanner or dataset-source cues",
            "rather than the drawing itself. Treat reported accuracy on",
            "this data as optimistic; consider the skeleton_rerender",
            "numbers above as a partial mitigation, and note this",
            "limitation in the model card and report.",
        ]
    else:
        lines += [
            "## Finding: no shortcut detected",
            "",
            "The background-only model's AUC confidence interval does",
            "not clearly exceed chance (0.5). No evidence that the normal",
            "model is relying on paper or scanner background cues rather",
            "than the drawing.",
        ]
    return "\n".join(lines)


def plot_shortcut_comparison(results: list[dict[str, Any]]) -> go.Figure:
    conditions = ["normal", "background_only", "skeleton_rerender"]
    summaries = [summarize_condition(results, c) for c in conditions]
    figure = go.Figure()
    figure.add_trace(
        go.Bar(x=conditions, y=[s["mean_auc"] for s in summaries], name="mean ROC-AUC")
    )
    figure.add_hline(y=0.5, line_dash="dash", annotation_text="chance")
    figure.update_layout(title="Shortcut check: mean ROC-AUC by condition", yaxis_title="ROC-AUC")
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

    with tempfile.TemporaryDirectory(prefix="shortcut_check_") as cache_dir:
        results = run_shortcut_check(config, seed, args.fast, Path(cache_dir))

    results_path = reports_dir / "shortcut_check_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")

    finding = render_finding(results)
    finding_path = reports_dir / "shortcut_check_finding.md"
    finding_path.write_text(finding, encoding="utf-8")

    figure = plot_shortcut_comparison(results)
    figure_path = reports_dir / "shortcut_check.html"
    figure.write_html(figure_path, include_plotlyjs="cdn")

    print(finding)
    print(f"\nWrote results to {results_path}")
    print(f"Wrote finding to {finding_path}")
    print(f"Wrote figure to {figure_path}")


if __name__ == "__main__":
    main()
