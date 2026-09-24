"""HOG + SVM classical baseline, trained on the same folds as the neural
backbones so it is a fair point of comparison in training/evaluate.py.

Uses common/preprocess.py for the same deterministic preprocessing every
other model gets, then extracts Histogram of Oriented Gradients features
(skimage.feature.hog) from the grayscale image, and trains a probability-
calibrated SVM (sklearn.svm.SVC with probability=True) on those features.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from skimage.feature import hog
from sklearn.svm import SVC

from common.preprocess import PreprocessConfig, preprocess_image

HOG_PIXELS_PER_CELL = (16, 16)
HOG_CELLS_PER_BLOCK = (2, 2)
HOG_ORIENTATIONS = 9


@dataclass(frozen=True)
class HogSvmConfig:
    preprocess: PreprocessConfig
    svm_c: float = 1.0
    svm_kernel: str = "rbf"
    seed: int = 42


def extract_hog_features(image_path: str, preprocess_config: PreprocessConfig) -> np.ndarray:
    """Preprocess an image (the shared pipeline) and return its HOG
    feature vector, computed on the grayscale channel."""
    array = preprocess_image(image_path, preprocess_config)
    grayscale = array[..., 0]
    features: np.ndarray = hog(
        grayscale,
        orientations=HOG_ORIENTATIONS,
        pixels_per_cell=HOG_PIXELS_PER_CELL,
        cells_per_block=HOG_CELLS_PER_BLOCK,
        feature_vector=True,
    )
    return features


def extract_hog_batch(paths: list[str], preprocess_config: PreprocessConfig) -> np.ndarray:
    return np.stack([extract_hog_features(path, preprocess_config) for path in paths])


def train_hog_svm(
    train_paths: list[str], train_labels: list[int], config: HogSvmConfig
) -> tuple[SVC, np.ndarray]:
    """Extract HOG features and fit an SVM. Returns the fitted classifier
    and the training feature matrix (callers reuse it rather than
    recomputing for a training-set evaluation)."""
    features = extract_hog_batch(train_paths, config.preprocess)
    classifier = SVC(
        C=config.svm_c,
        kernel=config.svm_kernel,
        probability=True,
        random_state=config.seed,
        class_weight="balanced",
    )
    classifier.fit(features, train_labels)
    return classifier, features


def predict_proba(
    classifier: SVC, paths: list[str], preprocess_config: PreprocessConfig
) -> np.ndarray:
    """Probability of the positive (parkinson) class for each path."""
    features = extract_hog_batch(paths, preprocess_config)
    probabilities: np.ndarray = classifier.predict_proba(features)
    positive_class_index = list(classifier.classes_).index(1)
    return probabilities[:, positive_class_index]
