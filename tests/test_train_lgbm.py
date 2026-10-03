from __future__ import annotations

import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pytest

from tremor.train_lgbm import (
    BOOSTER_FILENAME,
    CARD_FILENAME,
    FEATURE_NAMES,
    PROVENANCE,
    build_model_card,
    contribution_residual,
    cross_validate,
    features_to_vector,
    generate_dataset,
    run_training,
    sample_scenario,
    save_shap_figures,
    train_final,
)

N_PER_CLASS = 40
ROUNDS = 40


@pytest.fixture(scope="module")
def dataset():
    return generate_dataset(N_PER_CLASS, seed=3)


@pytest.fixture(scope="module")
def booster(dataset):
    features, labels = dataset
    return train_final(features, labels, ROUNDS, seed=3)


def test_feature_names_cover_every_feature_compute_features_can_emit() -> None:
    from tremor.features import compute_features
    from tremor.synthetic import generate_synthetic_spiral

    emitted = set(compute_features(generate_synthetic_spiral(duration_s=4.0)).features)
    assert emitted <= set(FEATURE_NAMES)
    assert len(FEATURE_NAMES) == len(set(FEATURE_NAMES))


def test_features_to_vector_follows_model_order_and_fills_nan() -> None:
    vector = features_to_vector({"duration_s": 5.0, "path_length": 100.0})
    assert vector.shape == (len(FEATURE_NAMES),)
    assert vector[FEATURE_NAMES.index("duration_s")] == 5.0
    assert vector[FEATURE_NAMES.index("path_length")] == 100.0
    assert np.isnan(vector[FEATURE_NAMES.index("pressure_mean")])


def test_both_classes_share_the_same_nuisance_distribution() -> None:
    quiet = sample_scenario(np.random.default_rng(5), has_tremor=False)
    shaky = sample_scenario(np.random.default_rng(5), has_tremor=True)
    nuisance = [
        "duration_s", "sample_rate_hz", "turns", "max_radius",
        "position_noise_std", "sampling_jitter_std_ms", "mouse_noise",
    ]
    assert {k: quiet[k] for k in nuisance} == {k: shaky[k] for k in nuisance}


def test_only_the_tremor_class_gets_a_tremor() -> None:
    quiet = sample_scenario(np.random.default_rng(1), has_tremor=False)
    shaky = sample_scenario(np.random.default_rng(1), has_tremor=True)
    assert quiet["tremor_freq_hz"] is None and quiet["tremor_amplitude"] == 0.0
    assert 3.5 <= shaky["tremor_freq_hz"] <= 11.5
    assert shaky["tremor_amplitude"] > 0.0


def test_generated_dataset_is_balanced_and_shaped(dataset) -> None:
    features, labels = dataset
    assert features.shape == (2 * N_PER_CLASS, len(FEATURE_NAMES))
    assert list(features.columns) == FEATURE_NAMES
    assert labels.sum() == N_PER_CLASS


def test_generated_dataset_is_deterministic_for_a_seed() -> None:
    first, first_labels = generate_dataset(6, seed=9)
    second, second_labels = generate_dataset(6, seed=9)
    assert first.equals(second)
    assert np.array_equal(first_labels, second_labels)


def test_tremor_class_has_higher_tremor_band_ratio(dataset) -> None:
    features, labels = dataset
    ratio = features["tremor_band_ratio"]
    assert ratio[labels == 1].mean() > ratio[labels == 0].mean()


def test_cross_validation_reports_one_auc_per_fold(dataset) -> None:
    features, labels = dataset
    cv = cross_validate(features, labels, n_folds=3, num_boost_round=ROUNDS, seed=3)
    assert cv["n_folds"] == 3 and len(cv["auc_per_fold"]) == 3
    assert cv["auc_mean"] > 0.9


def test_contributions_sum_to_the_raw_score(booster, dataset) -> None:
    features, _ = dataset
    assert contribution_residual(booster, features) < 1e-9


def test_contribution_matrix_has_one_column_per_feature_plus_a_constant_bias(
    booster, dataset
) -> None:
    features, _ = dataset
    contributions = booster.predict(features, pred_contrib=True)
    assert contributions.shape == (len(features), len(FEATURE_NAMES) + 1)
    assert np.allclose(contributions[:, -1], contributions[0, -1])


def test_probability_is_the_sigmoid_of_the_raw_score(booster, dataset) -> None:
    features, _ = dataset
    raw = booster.predict(features, raw_score=True)
    assert np.allclose(booster.predict(features), 1.0 / (1.0 + np.exp(-raw)))


def test_a_vector_with_missing_pressure_still_predicts(booster) -> None:
    vector = features_to_vector({"duration_s": 5.0, "tremor_band_ratio": 0.4})
    probability = booster.predict(vector.reshape(1, -1))[0]
    assert 0.0 <= probability <= 1.0


def test_saved_booster_reloads_with_identical_predictions(booster, dataset, tmp_path: Path) -> None:
    features, _ = dataset
    path = tmp_path / BOOSTER_FILENAME
    booster.save_model(str(path))
    reloaded = lgb.Booster(model_file=str(path))
    assert np.allclose(reloaded.predict(features), booster.predict(features))


def test_model_card_states_synthetic_provenance_and_feature_order() -> None:
    cv = {"n_folds": 3, "auc_mean": 0.99}
    card = build_model_card(cv, 80, ROUNDS, seed=1, fast=False)
    assert card["trained_on"] == "synthetic"
    assert card["provenance"] == PROVENANCE == "Feature pipeline demo, trained on simulated data"
    assert card["feature_names"] == FEATURE_NAMES
    assert "not a diagnosis" in card["disclaimer"].lower()
    assert "warning" not in card


def test_fast_model_card_carries_a_development_warning() -> None:
    card = build_model_card({}, 80, ROUNDS, seed=1, fast=True)
    assert card["development_model"] is True
    assert "Development model" in card["warning"]


def test_model_card_has_no_em_dashes() -> None:
    text = json.dumps(build_model_card({}, 80, ROUNDS, seed=1, fast=True))
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_shap_figures_are_written_as_pngs(booster, dataset, tmp_path: Path) -> None:
    features, _ = dataset
    paths = save_shap_figures(booster, features, tmp_path)
    assert [p.name for p in paths] == ["tremor_shap_summary.png", "tremor_shap_bar.png"]
    assert all(p.stat().st_size > 1000 for p in paths)


def test_run_training_fast_writes_model_card_and_figures(tmp_path: Path) -> None:
    config = {
        "n_per_class": 10,
        "n_folds": 3,
        "num_boost_round": 10,
        "output_dir": str(tmp_path / "models"),
        "reports_dir": str(tmp_path / "reports"),
    }
    result = run_training(config, seed=2, fast=True)
    assert Path(result["booster"]).exists()
    card = json.loads(Path(result["card"]).read_text(encoding="utf-8"))
    assert card["trained_on"] == "synthetic"
    assert card["contribution_residual_max_abs"] < 1e-9
    assert Path(result["card"]).name == CARD_FILENAME
    assert len(result["figures"]) == 2
