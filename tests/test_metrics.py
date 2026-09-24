from __future__ import annotations

import numpy as np

from training.metrics import bootstrap_auc_ci, compute_metrics, select_threshold


def test_compute_metrics_perfect_classifier() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics.accuracy == 1.0
    assert metrics.sensitivity == 1.0
    assert metrics.specificity == 1.0
    assert metrics.precision == 1.0
    assert metrics.f1 == 1.0
    assert metrics.roc_auc == 1.0
    assert metrics.pr_auc == 1.0


def test_compute_metrics_worst_case_classifier() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.9, 0.8, 0.7, 0.3, 0.2, 0.1])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics.accuracy == 0.0
    assert metrics.sensitivity == 0.0
    assert metrics.specificity == 0.0


def test_compute_metrics_threshold_changes_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.2, 0.4, 0.6, 0.8])
    lenient = compute_metrics(y_true, y_prob, threshold=0.1)
    strict = compute_metrics(y_true, y_prob, threshold=0.9)
    assert lenient.sensitivity >= strict.sensitivity
    assert lenient.specificity <= strict.specificity


def test_compute_metrics_brier_score_is_zero_for_perfect_confident_predictions() -> None:
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics.brier_score == 0.0


def test_compute_metrics_n_samples() -> None:
    y_true = np.array([0, 1, 0, 1, 1])
    y_prob = np.array([0.1, 0.9, 0.2, 0.8, 0.6])
    metrics = compute_metrics(y_true, y_prob, threshold=0.5)
    assert metrics.n_samples == 5


def test_select_threshold_reaches_target_sensitivity_on_validation() -> None:
    rng = np.random.default_rng(0)
    y_true = np.array([0] * 50 + [1] * 50)
    y_prob = np.concatenate(
        [rng.uniform(0.0, 0.6, size=50), rng.uniform(0.4, 1.0, size=50)]
    )
    threshold = select_threshold(y_true, y_prob, target_sensitivity=0.9)
    y_pred = (y_prob >= threshold).astype(int)
    achieved_sensitivity = y_pred[y_true == 1].mean()
    assert achieved_sensitivity >= 0.9 - 1e-9


def test_select_threshold_favors_sensitivity_over_specificity() -> None:
    # A classifier with heavy class overlap: a sensitivity-favoring
    # threshold should sit lower (classify more as positive) than 0.5.
    y_true = np.array([0] * 20 + [1] * 20)
    y_prob = np.concatenate([np.linspace(0.3, 0.7, 20), np.linspace(0.3, 0.7, 20)])
    threshold = select_threshold(y_true, y_prob, target_sensitivity=0.95)
    assert threshold <= 0.5


def test_select_threshold_is_deterministic() -> None:
    y_true = np.array([0, 1, 0, 1, 0, 1, 0, 1])
    y_prob = np.array([0.1, 0.9, 0.3, 0.7, 0.2, 0.8, 0.4, 0.6])
    first = select_threshold(y_true, y_prob)
    second = select_threshold(y_true, y_prob)
    assert first == second


def test_bootstrap_auc_ci_contains_point_estimate() -> None:
    rng = np.random.default_rng(1)
    y_true = np.array([0] * 40 + [1] * 40)
    y_prob = np.concatenate(
        [rng.uniform(0.0, 0.5, size=40), rng.uniform(0.5, 1.0, size=40)]
    )
    point, lower, upper = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=200, seed=1)
    assert lower <= point <= upper


def test_bootstrap_auc_ci_narrower_with_more_separable_classes() -> None:
    rng = np.random.default_rng(2)
    y_true = np.array([0] * 40 + [1] * 40)

    overlapping_prob = np.concatenate(
        [rng.uniform(0.3, 0.7, size=40), rng.uniform(0.3, 0.7, size=40)]
    )
    separable_prob = np.concatenate(
        [rng.uniform(0.0, 0.3, size=40), rng.uniform(0.7, 1.0, size=40)]
    )
    _, low_o, high_o = bootstrap_auc_ci(y_true, overlapping_prob, n_bootstrap=300, seed=2)
    _, low_s, high_s = bootstrap_auc_ci(y_true, separable_prob, n_bootstrap=300, seed=2)
    assert (high_s - low_s) < (high_o - low_o)


def test_bootstrap_auc_ci_is_deterministic_for_a_given_seed() -> None:
    y_true = np.array([0, 0, 0, 1, 1, 1])
    y_prob = np.array([0.1, 0.3, 0.4, 0.6, 0.7, 0.9])
    first = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=100, seed=7)
    second = bootstrap_auc_ci(y_true, y_prob, n_bootstrap=100, seed=7)
    assert first == second
