"""LightGBM classifier on the tremor features, trained on synthetic
spirals, plus offline SHAP figures.

There is no labeled kinematic dataset in data/ (the HandPD and NewHandPD
images carry no stroke timing, and no recorded-stroke dataset with
diagnoses has been obtained), so per CLAUDE.md Phase 6 this model is
trained only on simulated spirals: label 1 means a tremor was injected,
label 0 means none was. That makes it a demonstration of the feature
pipeline and of per-feature contributions, not a clinical tremor
classifier, and every artifact it writes says so. The backend and UI must
show the provenance below wherever this model's output appears.

The backend serves the booster with `predict(..., pred_contrib=True)`,
which returns exact TreeSHAP values, so it needs neither the `shap`
package nor PyTorch. The `shap` library is used here only to draw the
offline summary and bar figures.

Usage:
    python -m tremor.train_lgbm --config configs/tremor_lgbm.yaml --fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd
import yaml
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

from tremor.features import compute_features
from tremor.synthetic import generate_synthetic_spiral

FEATURE_NAMES = [
    "duration_s",
    "path_length",
    "mean_speed",
    "speed_cv",
    "jerk_rms",
    "pen_lifts",
    "pause_ratio",
    "tremor_band_ratio",
    "peak_freq_hz",
    "spectral_entropy",
    "spiral_deviation",
    "pressure_mean",
    "pressure_sd",
]

PROVENANCE = "Feature pipeline demo, trained on simulated data"
BOOSTER_FILENAME = "tremor_lgbm.txt"
CARD_FILENAME = "tremor_model_card.json"
DISCLAIMER = "Screening aid, not a diagnosis. Possible signs only; not clinically validated."

TREMOR_FREQUENCY_HZ = (3.5, 11.5)
TREMOR_AMPLITUDE_PX = (0.75, 8.0)
MOUSE_PROBABILITY = 0.25

FAST_N_PER_CLASS = 40
FAST_N_FOLDS = 3
FAST_ROUNDS = 40

LGBM_PARAMS: dict[str, Any] = {
    "objective": "binary",
    "learning_rate": 0.1,
    "num_leaves": 8,
    "min_data_in_leaf": 5,
    "feature_fraction": 0.9,
    "verbosity": -1,
    "deterministic": True,
    "force_row_wise": True,
    "num_threads": 1,
}

KNOWN_LIMITS = [
    "Trained only on simulated spirals. The label is whether a tremor was injected, so a "
    "high score means 'looks like the simulator's tremor', not 'looks like a patient'.",
    "The two classes are separable by construction, so a high cross-validated AUC here is "
    "evidence the features respond to tremor, not evidence of clinical accuracy.",
    "Real hand tremor, real devices and real drawing styles differ from the simulator.",
    "Not a diagnostic tool and not clinically validated.",
]


def features_to_vector(features: dict[str, float]) -> np.ndarray:
    """Order a features dict as the model expects, with NaN for anything
    absent (pressure features do not exist for mouse input). LightGBM
    handles NaN natively, so a missing value is not imputed."""
    return np.array([features.get(name, np.nan) for name in FEATURE_NAMES], dtype=np.float64)


def sample_scenario(rng: np.random.Generator, has_tremor: bool) -> dict[str, Any]:
    """Nuisance parameters are drawn from the same distribution for both
    classes, so the label is only whether a tremor was injected and the
    model cannot separate classes through duration, noise level or input
    device."""
    mouse = bool(rng.random() < MOUSE_PROBABILITY)
    scenario: dict[str, Any] = {
        "duration_s": float(rng.uniform(4.0, 8.0)),
        "sample_rate_hz": float(rng.uniform(50.0, 80.0) if mouse else rng.uniform(60.0, 140.0)),
        "turns": float(rng.uniform(3.0, 5.0)),
        "max_radius": float(rng.uniform(80.0, 120.0)),
        "position_noise_std": float(rng.uniform(0.0, 2.5)),
        "sampling_jitter_std_ms": float(rng.uniform(0.0, 6.0)),
        "mouse_noise": mouse,
        "tremor_freq_hz": None,
        "tremor_amplitude": 0.0,
    }
    if has_tremor:
        scenario["tremor_freq_hz"] = float(rng.uniform(*TREMOR_FREQUENCY_HZ))
        scenario["tremor_amplitude"] = float(rng.uniform(*TREMOR_AMPLITUDE_PX))
    return scenario


def generate_dataset(n_per_class: int, seed: int) -> tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(seed)
    rows: list[np.ndarray] = []
    labels: list[int] = []
    for index in range(2 * n_per_class):
        has_tremor = index % 2 == 1
        scenario = sample_scenario(rng, has_tremor)
        points = generate_synthetic_spiral(seed=seed * 100003 + index, **scenario)
        rows.append(features_to_vector(compute_features(points).features))
        labels.append(int(has_tremor))
    return pd.DataFrame(rows, columns=FEATURE_NAMES), np.array(labels)


def cross_validate(
    features: pd.DataFrame,
    labels: np.ndarray,
    n_folds: int,
    num_boost_round: int,
    seed: int,
) -> dict[str, Any]:
    """Grouped, stratified cross-validation. There are no subjects here,
    so every sample is its own group; the splitter is the same one the
    image models use so a real kinematic dataset with subject IDs can
    replace the groups without changing this function."""
    groups = np.arange(len(labels))
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    aucs: list[float] = []
    params = {**LGBM_PARAMS, "seed": seed}
    for train_index, test_index in splitter.split(features, labels, groups):
        booster = lgb.train(
            params,
            lgb.Dataset(features.iloc[train_index], labels[train_index]),
            num_boost_round=num_boost_round,
        )
        scores = booster.predict(features.iloc[test_index])
        aucs.append(float(roc_auc_score(labels[test_index], scores)))
    return {
        "n_folds": n_folds,
        "auc_per_fold": aucs,
        "auc_mean": float(np.mean(aucs)),
        "auc_std": float(np.std(aucs)),
    }


def train_final(
    features: pd.DataFrame, labels: np.ndarray, num_boost_round: int, seed: int
) -> lgb.Booster:
    return lgb.train(
        {**LGBM_PARAMS, "seed": seed},
        lgb.Dataset(features, labels),
        num_boost_round=num_boost_round,
    )


def contribution_residual(booster: lgb.Booster, features: pd.DataFrame) -> float:
    """Largest gap between the per-feature contributions plus the bias
    term and the model's raw score. TreeSHAP is exact, so this should be
    floating-point noise; the backend relies on that to present
    contributions that really add up to the score."""
    contributions = booster.predict(features, pred_contrib=True)
    raw_scores = booster.predict(features, raw_score=True)
    return float(np.abs(np.asarray(contributions).sum(axis=1) - np.asarray(raw_scores)).max())


def save_shap_figures(
    booster: lgb.Booster, features: pd.DataFrame, reports_dir: Path
) -> list[Path]:
    """Summary (beeswarm) and bar figures with the shap library. Imported
    here, not at module level, so the rest of this module and the backend
    never need shap or matplotlib."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import shap

    reports_dir.mkdir(parents=True, exist_ok=True)
    explainer = shap.TreeExplainer(booster)
    values = explainer.shap_values(features)
    if isinstance(values, list):
        values = values[1]

    written = []
    for plot_type, filename in (("dot", "tremor_shap_summary.png"), ("bar", "tremor_shap_bar.png")):
        plt.figure()
        shap.summary_plot(values, features, plot_type=plot_type, show=False)
        plt.tight_layout()
        path = reports_dir / filename
        plt.savefig(path, dpi=110)
        plt.close()
        written.append(path)
    return written


def build_model_card(
    cv: dict[str, Any], n_samples: int, num_boost_round: int, seed: int, fast: bool
) -> dict[str, Any]:
    card: dict[str, Any] = {
        "model_name": "tremor-feature-lightgbm",
        "trained_on": "synthetic",
        "provenance": PROVENANCE,
        "development_model": fast,
        "disclaimer": DISCLAIMER,
        "booster_file": BOOSTER_FILENAME,
        "format": "lightgbm text booster",
        "label_definition": "1 = a tremor was injected into the simulated spiral, 0 = none.",
        "feature_names": FEATURE_NAMES,
        "missing_values": "NaN is accepted natively; pressure features are NaN for mouse input.",
        "n_training_samples": n_samples,
        "num_boost_round": num_boost_round,
        "params": LGBM_PARAMS,
        "seed": seed,
        "cross_validation": cv,
        "contributions": "booster.predict(X, pred_contrib=True) returns exact TreeSHAP values; "
        "the last column is the bias, and each row sums to the raw score.",
        "known_limits": KNOWN_LIMITS,
    }
    if fast:
        card["warning"] = (
            "Development model: trained on a small simulated sample to prove the pipeline runs."
        )
    return card


def run_training(config: dict[str, Any], seed: int, fast: bool) -> dict[str, Any]:
    n_per_class = FAST_N_PER_CLASS if fast else int(config["n_per_class"])
    n_folds = FAST_N_FOLDS if fast else int(config["n_folds"])
    rounds = FAST_ROUNDS if fast else int(config["num_boost_round"])

    features, labels = generate_dataset(n_per_class, seed)
    cv = cross_validate(features, labels, n_folds, rounds, seed)
    booster = train_final(features, labels, rounds, seed)
    residual = contribution_residual(booster, features)

    output_dir = Path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    booster_path = output_dir / BOOSTER_FILENAME
    booster.save_model(str(booster_path))
    card = build_model_card(cv, len(labels), rounds, seed, fast)
    card["contribution_residual_max_abs"] = residual
    card_path = output_dir / CARD_FILENAME
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")

    figures = save_shap_figures(booster, features, Path(config["reports_dir"]))
    return {
        "booster": booster_path,
        "card": card_path,
        "figures": figures,
        "cv": cv,
        "residual": residual,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/tremor_lgbm.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    with open(args.config, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = args.seed if args.seed is not None else int(config["seed"])

    result = run_training(config, seed, args.fast)
    cv = result["cv"]
    print(f"Provenance: {PROVENANCE}")
    print(
        f"Cross-validated AUC: {cv['auc_mean']:.3f} +/- {cv['auc_std']:.3f} "
        f"over {cv['n_folds']} folds"
    )
    print(f"Max |sum(contributions) - raw score|: {result['residual']:.2e}")
    print(f"Wrote {result['booster']}, {result['card']}")
    for figure in result["figures"]:
        print(f"Wrote {figure}")


if __name__ == "__main__":
    main()
