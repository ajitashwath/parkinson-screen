"""Model loading and inference: ONNX image model with an occlusion map, and
the LightGBM tremor model with exact per-feature contributions.

No PyTorch and no shap here. LightGBM's `pred_contrib=True` returns exact
TreeSHAP values, so the contributions sum to the raw score.
"""

from __future__ import annotations

import base64
import binascii
import json
import logging
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import onnxruntime as ort

from common.preprocess import PreprocessConfig, preprocess_image
from tremor.features import FeatureResult, StrokePoint, compute_features

logger = logging.getLogger("backend.inference")

IMAGE_SIZE = 224
OCCLUSION_FILL = 1.0
TOP_CONTRIBUTIONS = 5
MIN_CONTRIBUTION = 1e-9

FEATURE_EXPLANATIONS = {
    "duration_s": "How long the drawing took.",
    "path_length": "Total distance the pen or finger travelled.",
    "mean_speed": "Average drawing speed.",
    "speed_cv": "How uneven the drawing speed was.",
    "jerk_rms": "How abruptly the movement changed, relative to its speed.",
    "pen_lifts": "How many times the pen left the surface.",
    "pause_ratio": "Share of the drawing spent almost still.",
    "tremor_band_ratio": "Share of movement energy in the 4 to 12 Hz band where tremor sits.",
    "peak_freq_hz": "Frequency of the strongest rhythm between 3 and 12 Hz.",
    "spectral_entropy": "How spread out the movement frequencies are (higher is more spread).",
    "spiral_deviation": "How far the line strays from a smooth spiral.",
    "pressure_mean": "Average pen pressure (pen input only).",
    "pressure_sd": "Variation in pen pressure (pen input only).",
}


class ImageDecodeError(ValueError):
    """The uploaded image could not be decoded."""


@dataclass
class ImagePrediction:
    probability: float
    grid_size: int
    probability_drop: np.ndarray


def decode_image_base64(data: str, max_bytes: int) -> bytes:
    """Decode a base64 string (optionally a data URL) and enforce the size cap."""
    payload = data.split(",", 1)[1] if data.startswith("data:") and "," in data else data
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ImageDecodeError("image_base64 is not valid base64.") from error
    if len(raw) > max_bytes:
        raise ImageDecodeError(f"Image is larger than {max_bytes // (1024 * 1024)} MB.")
    return raw


def to_model_input(image_bytes: bytes) -> np.ndarray:
    """Shared preprocessing, then HWC to CHW with a batch axis of 1."""
    try:
        image = preprocess_image(image_bytes, PreprocessConfig(size=IMAGE_SIZE))
    except Exception as error:
        raise ImageDecodeError("The image could not be read as a PNG or JPEG.") from error
    return np.transpose(image, (2, 0, 1))[None].astype(np.float32)


def effective_grid(requested: int, max_passes: int) -> int:
    """Largest grid not above the request whose forward passes (grid squared
    plus the baseline) stay within the cap."""
    grid = max(1, min(requested, IMAGE_SIZE))
    while grid > 1 and grid * grid + 1 > max_passes:
        grid -= 1
    return grid


def occluded_batch(image: np.ndarray, grid: int) -> np.ndarray:
    """One copy of the image per grid cell with that cell painted white,
    which is the paper background the model was trained on."""
    edges = np.linspace(0, IMAGE_SIZE, grid + 1).round().astype(int)
    batch = np.repeat(image, grid * grid, axis=0)
    index = 0
    for row in range(grid):
        for column in range(grid):
            batch[
                index, :, edges[row] : edges[row + 1], edges[column] : edges[column + 1]
            ] = OCCLUSION_FILL
            index += 1
    return batch


class ImageModel:
    def __init__(self, path: Path, card: dict[str, Any], threshold: float) -> None:
        options = ort.SessionOptions()
        options.intra_op_num_threads = 2
        self.session = ort.InferenceSession(
            str(path), sess_options=options, providers=["CPUExecutionProvider"]
        )
        self.input_name = self.session.get_inputs()[0].name
        self.card = card
        self.threshold = threshold

    @property
    def version(self) -> str:
        return f"{self.card.get('model_name', 'image-model')}@{self.card.get('version', '?')}"

    @property
    def development(self) -> bool:
        return bool(self.card.get("development_model", False))

    def probabilities(self, batch: np.ndarray) -> np.ndarray:
        output = self.session.run(None, {self.input_name: batch})[0]
        return np.asarray(output, dtype=np.float64).reshape(-1)

    def predict_with_occlusion(
        self, image: np.ndarray, requested_grid: int, max_passes: int
    ) -> ImagePrediction:
        grid = effective_grid(requested_grid, max_passes)
        batch = np.concatenate([image, occluded_batch(image, grid)], axis=0)
        probabilities = self.probabilities(batch)
        baseline = float(probabilities[0])
        drop = (baseline - probabilities[1:]).reshape(grid, grid)
        return ImagePrediction(baseline, grid, drop)


def normalize_drop(drop: np.ndarray) -> np.ndarray:
    """Positive drops scaled to [0, 1]; cells that raised the probability
    when covered, or a flat map, become 0."""
    positive = np.clip(drop, 0.0, None)
    peak = float(positive.max()) if positive.size else 0.0
    if peak <= 0.0:
        return np.zeros_like(positive)
    return positive / peak


@dataclass
class TremorOutput:
    features: dict[str, float]
    warnings: list[str]
    score: float | None
    contributions: list[dict[str, Any]]


class TremorModel:
    def __init__(self, path: Path, card: dict[str, Any]) -> None:
        self.booster = lgb.Booster(model_file=str(path))
        self.card = card
        self.feature_names: list[str] = list(card["feature_names"])

    @property
    def provenance(self) -> str:
        return str(self.card.get("provenance", "Tremor model"))

    @property
    def trained_on(self) -> str:
        return str(self.card.get("trained_on", "unknown"))

    def vector(self, features: dict[str, float]) -> np.ndarray:
        values = [features.get(name, math.nan) for name in self.feature_names]
        return np.array([values], dtype=np.float64)

    def score_and_contributions(
        self, features: dict[str, float]
    ) -> tuple[float, list[dict[str, Any]]]:
        row = self.vector(features)
        score = float(self.booster.predict(row)[0])
        contributions = np.asarray(self.booster.predict(row, pred_contrib=True))[0]
        ranked = sorted(
            zip(self.feature_names, contributions[:-1], strict=True),
            key=lambda pair: abs(pair[1]),
            reverse=True,
        )
        top = [
            {
                "feature": name,
                "value": float(value),
                "explanation": FEATURE_EXPLANATIONS.get(name, name),
            }
            for name, value in ranked[:TOP_CONTRIBUTIONS]
            if abs(value) > MIN_CONTRIBUTION
        ]
        return score, top


def finite_features(result: FeatureResult) -> dict[str, float]:
    """Drop NaN and infinite values, which JSON cannot carry."""
    return {k: float(v) for k, v in result.features.items() if math.isfinite(v)}


def analyse_tremor(points: list[StrokePoint], model: TremorModel | None) -> TremorOutput:
    result = compute_features(points)
    features = finite_features(result)
    score: float | None = None
    contributions: list[dict[str, Any]] = []
    if model is not None and features:
        score, contributions = model.score_and_contributions(result.features)
    return TremorOutput(features, list(result.warnings), score, contributions)


@dataclass
class LoadedModels:
    image: ImageModel | None
    tremor: TremorModel | None
    messages: list[str]


def read_json(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def load_models(model_dir: Path, image_model_file: str) -> LoadedModels:
    """Load both models once. A missing or unreadable file leaves that model
    as None and records why, so /health can report it and the prediction
    endpoints can answer 503."""
    messages: list[str] = []
    image: ImageModel | None = None
    tremor: TremorModel | None = None

    try:
        card = read_json(model_dir / "model_card.json")
        threshold = float(read_json(model_dir / "threshold.json")["threshold"])
        image = ImageModel(model_dir / image_model_file, card, threshold)
    except Exception as error:
        logger.warning("image model unavailable: %s", error)
        messages.append(f"Image model unavailable ({type(error).__name__}).")

    try:
        tremor_card = read_json(model_dir / "tremor_model_card.json")
        tremor = TremorModel(model_dir / str(tremor_card["booster_file"]), tremor_card)
    except Exception as error:
        logger.warning("tremor model unavailable: %s", error)
        messages.append(f"Tremor model unavailable ({type(error).__name__}).")

    return LoadedModels(image, tremor, messages)
