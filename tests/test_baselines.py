from __future__ import annotations

from pathlib import Path

import numpy as np

from common.preprocess import PreprocessConfig
from training.baselines import (
    HogSvmConfig,
    extract_hog_batch,
    extract_hog_features,
    predict_proba,
    train_hog_svm,
)


def make_labeled_fixture(tmp_path: Path, spiral_image_factory, count_per_class: int = 8):
    paths: list[str] = []
    labels: list[int] = []
    for label, turns, noise in [(0, 4.0, 0.0), (1, 4.0, 3.0)]:
        for i in range(count_per_class):
            image = spiral_image_factory(turns=turns, noise=noise, seed=i + label * 100)
            path = tmp_path / f"label{label}_{i}.png"
            image.save(path)
            paths.append(str(path))
            labels.append(label)
    return paths, labels


def test_extract_hog_features_returns_nonempty_vector(tmp_path: Path, spiral_image_factory) -> None:
    path = tmp_path / "spiral.png"
    spiral_image_factory().save(path)
    features = extract_hog_features(str(path), PreprocessConfig(size=128))
    assert features.ndim == 1
    assert features.shape[0] > 0


def test_extract_hog_features_is_deterministic(tmp_path: Path, spiral_image_factory) -> None:
    path = tmp_path / "spiral.png"
    spiral_image_factory().save(path)
    config = PreprocessConfig(size=128)
    first = extract_hog_features(str(path), config)
    second = extract_hog_features(str(path), config)
    assert np.array_equal(first, second)


def test_extract_hog_batch_shape(tmp_path: Path, spiral_image_factory) -> None:
    paths, _ = make_labeled_fixture(tmp_path, spiral_image_factory, count_per_class=2)
    features = extract_hog_batch(paths, PreprocessConfig(size=64))
    assert features.shape[0] == len(paths)


def test_train_hog_svm_fits_and_predicts(tmp_path: Path, spiral_image_factory) -> None:
    paths, labels = make_labeled_fixture(tmp_path, spiral_image_factory, count_per_class=10)
    config = HogSvmConfig(preprocess=PreprocessConfig(size=96), seed=0)
    classifier, train_features = train_hog_svm(paths, labels, config)
    assert train_features.shape[0] == len(paths)
    probabilities = predict_proba(classifier, paths, config.preprocess)
    assert probabilities.shape[0] == len(paths)
    assert (probabilities >= 0).all()
    assert (probabilities <= 1).all()


def test_predict_proba_orders_classes_correctly(tmp_path: Path, spiral_image_factory) -> None:
    paths, labels = make_labeled_fixture(tmp_path, spiral_image_factory, count_per_class=10)
    config = HogSvmConfig(preprocess=PreprocessConfig(size=96), seed=0)
    classifier, _ = train_hog_svm(paths, labels, config)
    assert set(classifier.classes_) == {0, 1}
    probabilities = predict_proba(classifier, paths, config.preprocess)
    mean_positive = probabilities[np.array(labels) == 1].mean()
    mean_negative = probabilities[np.array(labels) == 0].mean()
    assert mean_positive > mean_negative
