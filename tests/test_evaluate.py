from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from training.evaluate import (
    build_comparison_table,
    load_results,
    plot_confusion_matrices,
    plot_roc_curves,
    pooled_predictions,
    run_evaluation,
    save_comparison_table,
)


def make_fake_results() -> list[dict[str, Any]]:
    results = []
    for model, base_auc in [("resnet50", 0.9), ("hog_svm_baseline", 0.7)]:
        for fold in range(2):
            results.append(
                {
                    "model": model,
                    "fold": fold,
                    "seed": 42,
                    "accuracy": base_auc,
                    "sensitivity": base_auc,
                    "specificity": base_auc,
                    "precision": base_auc,
                    "f1": base_auc,
                    "roc_auc": base_auc,
                    "pr_auc": base_auc,
                    "brier_score": 1 - base_auc,
                    "threshold": 0.5,
                    "n_samples": 10,
                    "auc_ci_lower": base_auc - 0.1,
                    "auc_ci_upper": base_auc + 0.1,
                    "n_parameters": 1000,
                    "cpu_latency_ms": 5.0,
                    "y_true": [0, 0, 1, 1, 0],
                    "y_prob": (
                        [0.1, 0.2, 0.8, 0.9, 0.3]
                        if base_auc > 0.8
                        else [0.4, 0.6, 0.5, 0.7, 0.5]
                    ),
                }
            )
    return results


def test_load_results_round_trip(tmp_path: Path) -> None:
    results = make_fake_results()
    path = tmp_path / "results.json"
    path.write_text(json.dumps(results), encoding="utf-8")
    loaded = load_results(path)
    assert loaded == results


def test_build_comparison_table_has_one_row_per_model() -> None:
    table = build_comparison_table(make_fake_results())
    assert set(table["model"]) == {"resnet50", "hog_svm_baseline"}
    assert len(table) == 2


def test_build_comparison_table_sorted_by_auc_descending() -> None:
    table = build_comparison_table(make_fake_results())
    assert table.iloc[0]["model"] == "resnet50"


def test_build_comparison_table_has_n_runs_per_model() -> None:
    table = build_comparison_table(make_fake_results())
    assert (table["n_runs"] == 2).all()


def test_save_comparison_table_writes_csv_and_markdown(tmp_path: Path) -> None:
    table = build_comparison_table(make_fake_results())
    csv_path, markdown_path = save_comparison_table(table, tmp_path)
    assert csv_path.exists()
    assert markdown_path.exists()
    assert "resnet50" in markdown_path.read_text(encoding="utf-8")


def test_pooled_predictions_concatenates_across_folds() -> None:
    results = make_fake_results()
    y_true, y_prob = pooled_predictions(results, "resnet50")
    assert len(y_true) == 10  # 5 per fold, 2 folds
    assert len(y_prob) == 10


def test_pooled_predictions_empty_for_unknown_model() -> None:
    y_true, y_prob = pooled_predictions(make_fake_results(), "not_a_model")
    assert len(y_true) == 0


def test_plot_roc_curves_returns_figure_with_traces() -> None:
    figure = plot_roc_curves(make_fake_results())
    assert len(figure.data) >= 2


def test_plot_confusion_matrices_returns_figure_with_dropdown() -> None:
    figure = plot_confusion_matrices(make_fake_results())
    assert len(figure.data) == 2
    assert figure.layout.updatemenus


def test_run_evaluation_writes_all_expected_files(tmp_path: Path) -> None:
    results_path = tmp_path / "results.json"
    results_path.write_text(json.dumps(make_fake_results()), encoding="utf-8")
    reports_dir = tmp_path / "reports"
    written = run_evaluation(results_path, reports_dir)
    assert set(written.keys()) == {
        "comparison_csv",
        "comparison_markdown",
        "roc_curves",
        "confusion_matrices",
    }
    for path in written.values():
        assert path.exists()
