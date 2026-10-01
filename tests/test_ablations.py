from __future__ import annotations

import math

import pandas as pd

from training.ablations import (
    fuse_kind_predictions,
    fused_condition_rows,
    plot_ablation_bars,
    plot_kind_fusion,
    summarize_by_condition,
)


def test_summarize_by_condition_groups_and_averages() -> None:
    results = [
        {"ablation": "pretrained_vs_scratch", "condition": "pretrained", "roc_auc": 0.6},
        {"ablation": "pretrained_vs_scratch", "condition": "pretrained", "roc_auc": 0.8},
        {"ablation": "pretrained_vs_scratch", "condition": "scratch", "roc_auc": 0.5},
        {"ablation": "augmentation", "condition": "augmented", "roc_auc": 0.9},
    ]
    summary = summarize_by_condition(results, "pretrained_vs_scratch")
    pretrained_row = summary[summary["condition"] == "pretrained"]
    assert pretrained_row["roc_auc_mean"].iloc[0] == 0.7
    assert pretrained_row["count"].iloc[0] == 2
    assert set(summary["condition"]) == {"pretrained", "scratch"}


def test_summarize_by_condition_empty_for_unknown_ablation() -> None:
    results = [{"ablation": "pretrained_vs_scratch", "condition": "pretrained", "roc_auc": 0.6}]
    summary = summarize_by_condition(results, "learning_curve")
    assert summary.empty


def test_plot_ablation_bars_returns_figure() -> None:
    results = [
        {"ablation": "augmentation", "condition": "augmented", "roc_auc": 0.6},
        {"ablation": "augmentation", "condition": "no_augmentation", "roc_auc": 0.5},
    ]
    summary = summarize_by_condition(results, "augmentation")
    figure = plot_ablation_bars(summary, "test title")
    assert len(figure.data) == 1
    assert len(figure.data[0].x) == 2


def test_fused_condition_rows_filters_correctly() -> None:
    results = [
        {"ablation": "kind_fusion", "condition": "spiral", "model": "resnet50"},
        {"ablation": "kind_fusion", "condition": "fused", "model": "fused"},
        {"ablation": "augmentation", "condition": "augmented"},
    ]
    fused = fused_condition_rows(results)
    assert len(fused) == 1
    assert fused[0]["condition"] == "fused"


def make_kind_result(subject_ids, labels, probs) -> dict:
    return {
        "roc_auc": 0.7,
        "val_subject_id": subject_ids,
        "val_true": labels,
        "val_prob": probs,
        "test_subject_id": subject_ids,
        "y_true": labels,
        "y_prob": probs,
    }


def test_fuse_kind_predictions_with_matched_subjects() -> None:
    spiral_result = make_kind_result(
        ["S1", "S2", "S3", "S4"], [0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]
    )
    wave_result = make_kind_result(
        ["S1", "S2", "S3", "S4"], [0, 0, 1, 1], [0.15, 0.25, 0.75, 0.85]
    )
    fused = fuse_kind_predictions(spiral_result, wave_result, fold=0, seed=0)
    assert fused["n_matched_test_subjects"] == 4
    assert not math.isnan(fused["average_fusion_auc"])


def test_fuse_kind_predictions_no_matched_subjects_returns_nan_not_crash() -> None:
    spiral_result = make_kind_result(["S1", "S2"], [0, 1], [0.1, 0.9])
    wave_result = make_kind_result(["S3", "S4"], [0, 1], [0.2, 0.8])
    fused = fuse_kind_predictions(spiral_result, wave_result, fold=0, seed=0)
    assert fused["n_matched_test_subjects"] == 0
    assert math.isnan(fused["average_fusion_auc"])
    assert fused["fusion_helps_vs_best_single"] is None


def test_fuse_kind_predictions_single_class_test_returns_nan_not_crash() -> None:
    spiral_result = make_kind_result(["S1", "S2"], [0, 0], [0.1, 0.2])
    wave_result = make_kind_result(["S1", "S2"], [0, 0], [0.15, 0.25])
    fused = fuse_kind_predictions(spiral_result, wave_result, fold=0, seed=0)
    assert fused["n_matched_test_subjects"] == 2
    assert math.isnan(fused["average_fusion_auc"])


def test_plot_kind_fusion_returns_figure_with_four_series() -> None:
    fusion_df = pd.DataFrame(
        {
            "fold": [0, 1],
            "spiral_only_auc": [0.6, 0.7],
            "wave_only_auc": [0.5, 0.55],
            "average_fusion_auc": [0.65, 0.68],
            "stacked_fusion_auc": [0.66, 0.7],
        }
    )
    figure = plot_kind_fusion(fusion_df)
    assert len(figure.data) == 4
