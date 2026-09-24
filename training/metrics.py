"""Classification metrics and threshold selection, shared by every model
(the neural backbones and the HOG+SVM baseline) so comparisons in
training/evaluate.py are apples to apples.

Threshold selection happens on validation data only, never on the test
fold, per CLAUDE.md Phase 3 rule 4: "Never tune anything on the test
fold." select_threshold favors sensitivity (catching true Parkinson's
cases) over specificity, appropriate for a screening aid.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)


@dataclass(frozen=True)
class Metrics:
    accuracy: float
    sensitivity: float
    specificity: float
    precision: float
    f1: float
    roc_auc: float
    pr_auc: float
    brier_score: float
    threshold: float
    n_samples: int

    def as_dict(self) -> dict[str, float]:
        return {
            "accuracy": self.accuracy,
            "sensitivity": self.sensitivity,
            "specificity": self.specificity,
            "precision": self.precision,
            "f1": self.f1,
            "roc_auc": self.roc_auc,
            "pr_auc": self.pr_auc,
            "brier_score": self.brier_score,
            "threshold": self.threshold,
            "n_samples": float(self.n_samples),
        }


def select_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, target_sensitivity: float = 0.9
) -> float:
    """Pick the decision threshold with the highest specificity among
    thresholds that reach at least target_sensitivity, on validation data.
    Falls back to the threshold that gets closest to target_sensitivity if
    none reaches it (a weak validation set, for example in --fast mode).
    """
    fpr, tpr, thresholds = roc_curve(y_true, y_prob)
    specificity = 1 - fpr
    reaches_target = tpr >= target_sensitivity
    if reaches_target.any():
        candidate_specificity = np.where(reaches_target, specificity, -1.0)
        best_index = int(np.argmax(candidate_specificity))
    else:
        best_index = int(np.argmin(np.abs(tpr - target_sensitivity)))
    threshold = thresholds[best_index]
    # roc_curve's first threshold can be inf (the "classify nothing as
    # positive" point); clip to a valid probability.
    return float(np.clip(threshold, 0.0, 1.0))


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> Metrics:
    """Compute the full metric set at a fixed threshold, chosen beforehand
    on validation data. y_true is 0/1, y_prob is the predicted probability
    of the positive (parkinson) class."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()
    accuracy = (tp + tn) / len(y_true)
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) > 0 else float("nan")
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    f1 = (
        2 * precision * sensitivity / (precision + sensitivity)
        if (precision + sensitivity) > 0 and not np.isnan(precision) and not np.isnan(sensitivity)
        else float("nan")
    )

    has_both_classes = len(np.unique(y_true)) == 2
    roc_auc = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")
    pr_auc = float(average_precision_score(y_true, y_prob)) if has_both_classes else float("nan")
    brier = float(brier_score_loss(y_true, y_prob))

    return Metrics(
        accuracy=float(accuracy),
        sensitivity=float(sensitivity),
        specificity=float(specificity),
        precision=float(precision),
        f1=float(f1),
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        brier_score=brier,
        threshold=float(threshold),
        n_samples=len(y_true),
    )


def bootstrap_auc_ci(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bootstrap: int = 1000,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap confidence interval for ROC-AUC: resample (y_true, y_prob)
    pairs with replacement n_bootstrap times and take the percentile
    interval. Returns (point_estimate, lower, upper). NaN bounds when a
    resample has only one class present (AUC undefined) are dropped."""
    y_true = np.asarray(y_true)
    y_prob = np.asarray(y_prob)
    rng = np.random.default_rng(seed)
    n = len(y_true)
    has_both_classes = len(np.unique(y_true)) == 2
    point_estimate = float(roc_auc_score(y_true, y_prob)) if has_both_classes else float("nan")

    scores = []
    for _ in range(n_bootstrap):
        indices = rng.integers(0, n, size=n)
        y_true_sample, y_prob_sample = y_true[indices], y_prob[indices]
        if len(np.unique(y_true_sample)) < 2:
            continue
        scores.append(roc_auc_score(y_true_sample, y_prob_sample))

    if not scores:
        return point_estimate, float("nan"), float("nan")
    alpha = (1 - confidence) / 2
    lower = float(np.percentile(scores, 100 * alpha))
    upper = float(np.percentile(scores, 100 * (1 - alpha)))
    return point_estimate, lower, upper
