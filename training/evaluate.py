"""Comparison table, ROC curves and confusion matrices from
training/train.py's real results (never fabricated), saved to reports/.

Usage:
    python -m training.evaluate --results reports/training_results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.metrics import confusion_matrix, roc_curve

SUMMARY_METRIC_COLUMNS = [
    "accuracy",
    "sensitivity",
    "specificity",
    "precision",
    "f1",
    "roc_auc",
    "pr_auc",
    "brier_score",
    "cpu_latency_ms",
]


def load_results(path: str | Path) -> list[dict[str, Any]]:
    with open(path, encoding="utf-8") as handle:
        results: list[dict[str, Any]] = json.load(handle)
    return results


def build_comparison_table(results: list[dict[str, Any]]) -> pd.DataFrame:
    """Mean +/- std of each metric per model, aggregated across folds and
    seeds, plus parameter count (constant per model)."""
    df = pd.DataFrame(results)
    grouped = df.groupby("model")[SUMMARY_METRIC_COLUMNS]
    means = grouped.mean().add_suffix("_mean")
    stds = grouped.std().add_suffix("_std")
    n_parameters = df.groupby("model")["n_parameters"].first()
    n_runs = df.groupby("model").size().rename("n_runs")
    table = pd.concat([means, stds, n_parameters, n_runs], axis=1).reset_index()
    return table.sort_values("roc_auc_mean", ascending=False).reset_index(drop=True)


def save_comparison_table(table: pd.DataFrame, out_dir: Path) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "model_comparison.csv"
    markdown_path = out_dir / "model_comparison.md"
    table.to_csv(csv_path, index=False)
    markdown_path.write_text(table.to_markdown(index=False), encoding="utf-8")
    return csv_path, markdown_path


def pooled_predictions(results: list[dict[str, Any]], model: str) -> tuple[np.ndarray, np.ndarray]:
    """Pool y_true/y_prob across every fold and seed for one model, so the
    ROC curve and confusion matrix reflect all held-out predictions rather
    than a single fold."""
    y_true: list[int] = []
    y_prob: list[float] = []
    for result in results:
        if result["model"] != model:
            continue
        y_true.extend(result["y_true"])
        y_prob.extend(result["y_prob"])
    return np.array(y_true), np.array(y_prob)


def plot_roc_curves(results: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Scatter(
            x=[0, 1], y=[0, 1], mode="lines", name="chance", line={"dash": "dash", "color": "gray"}
        )
    )
    for model in sorted({r["model"] for r in results}):
        y_true, y_prob = pooled_predictions(results, model)
        if len(np.unique(y_true)) < 2:
            continue
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        figure.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model))
    figure.update_layout(
        title="ROC curves (pooled across folds and seeds)",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
    )
    return figure


def plot_confusion_matrices(results: list[dict[str, Any]]) -> go.Figure:
    models = sorted({r["model"] for r in results})
    figure = go.Figure()
    buttons = []
    for i, model in enumerate(models):
        y_true, y_prob = pooled_predictions(results, model)
        thresholds = [r["threshold"] for r in results if r["model"] == model]
        threshold = float(np.mean(thresholds)) if thresholds else 0.5
        y_pred = (y_prob >= threshold).astype(int)
        matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
        figure.add_trace(
            go.Heatmap(
                z=matrix,
                x=["predicted healthy", "predicted parkinson"],
                y=["actual healthy", "actual parkinson"],
                colorscale="Blues",
                text=matrix,
                texttemplate="%{text}",
                visible=(i == 0),
            )
        )
        visibility = [j == i for j in range(len(models))]
        buttons.append({"label": model, "method": "update", "args": [{"visible": visibility}]})
    figure.update_layout(
        title="Confusion matrix (pooled, at each model's mean selected threshold)",
        updatemenus=[{"buttons": buttons, "direction": "down", "x": 1.15, "y": 1.15}],
    )
    return figure


def run_evaluation(results_path: Path, reports_dir: Path) -> dict[str, Path]:
    results = load_results(results_path)
    if not results:
        raise ValueError(f"No results found in {results_path}; run training/train.py first")

    table = build_comparison_table(results)
    csv_path, markdown_path = save_comparison_table(table, reports_dir)

    written = {"comparison_csv": csv_path, "comparison_markdown": markdown_path}

    roc_figure = plot_roc_curves(results)
    roc_path = reports_dir / "roc_curves.html"
    roc_figure.write_html(roc_path, include_plotlyjs="cdn")
    written["roc_curves"] = roc_path

    confusion_figure = plot_confusion_matrices(results)
    confusion_path = reports_dir / "confusion_matrices.html"
    confusion_figure.write_html(confusion_path, include_plotlyjs="cdn")
    written["confusion_matrices"] = confusion_path

    print(table.to_string(index=False))
    for key, path in written.items():
        print(f"Wrote {key} to {path}")
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", default="reports/training_results.json")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()
    run_evaluation(Path(args.results), Path(args.reports_dir))


if __name__ == "__main__":
    main()
