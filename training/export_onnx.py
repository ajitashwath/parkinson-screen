"""Export the registered champion to ONNX for the backend, prove the
export matches PyTorch, optionally quantize to INT8, and write the model
card and decision threshold next to the model.

The backend serves this file with onnxruntime and has no PyTorch, so the
export is a contract: an image in [0, 1], shape (batch, 3, H, W) as
produced by common/preprocess.py, goes in; a screening probability per
image comes out, sigmoid already applied. parity_check is the proof that
contract holds, within 1e-4, on a real batch.

Usage:
    python -m training.export_onnx --config configs/export.yaml --fast
"""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
import pandas as pd
import torch
from sklearn.metrics import roc_auc_score
from torch import nn

from common.preprocess import PreprocessConfig, preprocess_image
from training.register import (
    DEV_MODEL_WARNING,
    fit_and_register,
    load_champion,
    load_merged_config,
    tracking_uri_for,
)
from training.train import load_config, sample_fast_subset, set_all_seeds

MODEL_FILENAME = "model.onnx"
INT8_FILENAME = "model_int8.onnx"
CARD_FILENAME = "model_card.json"
THRESHOLD_FILENAME = "threshold.json"
DISCLAIMER = "Screening aid, not a diagnosis. Possible signs only; not clinically validated."
LATENCY_RUNS = 20

KNOWN_LIMITS = [
    "Trained on scanned paper drawings (HandPD and NewHandPD); live canvas drawings are a "
    "different domain, and predictions on them are not validated against clinical labels.",
    "Small dataset (158 subjects across both sources): metrics have wide confidence intervals "
    "and many model comparisons are within noise.",
    "Class imbalance: Parkinson's subjects outnumber healthy subjects in the training data.",
    "The decision threshold favors sensitivity on a validation split, so expect false positives.",
    "Not a diagnostic tool. It must not be used to rule a condition in or out.",
]


class ProbabilityWrapper(nn.Module):
    """The deployed graph: sigmoid on top of the single-logit model, so
    the backend never needs to know the model emits logits."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        probabilities: torch.Tensor = torch.sigmoid(self.model(images))
        return probabilities


@dataclass(frozen=True)
class ParityResult:
    max_abs_diff: float
    atol: float

    @property
    def passed(self) -> bool:
        return self.max_abs_diff <= self.atol


def export_to_onnx(
    model: nn.Module, out_path: Path, image_size: int = 224, opset: int = 17
) -> Path:
    """Export with a dynamic batch axis. The dummy batch has 2 images, not
    1, so the tracer cannot specialize any op to batch size 1."""
    out_path.parent.mkdir(parents=True, exist_ok=True)
    wrapper = ProbabilityWrapper(model).cpu().eval()
    dummy = torch.zeros(2, 3, image_size, image_size)
    torch.onnx.export(
        wrapper,
        (dummy,),
        str(out_path),
        input_names=["image"],
        output_names=["probability"],
        dynamic_axes={"image": {0: "batch"}, "probability": {0: "batch"}},
        opset_version=opset,
        do_constant_folding=True,
    )
    return out_path


def make_session(path: Path) -> ort.InferenceSession:
    return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


def onnx_probabilities(
    session: ort.InferenceSession, images: np.ndarray, batch_size: int = 8
) -> np.ndarray:
    outputs = [
        session.run(None, {"image": images[i : i + batch_size]})[0]
        for i in range(0, len(images), batch_size)
    ]
    return np.concatenate(outputs).astype(np.float64)


def parity_check(
    model: nn.Module, onnx_path: Path, images: np.ndarray, atol: float = 1e-4
) -> ParityResult:
    """PyTorch and ONNX Runtime must agree on the same batch."""
    wrapper = ProbabilityWrapper(model).cpu().eval()
    with torch.no_grad():
        torch_probabilities = wrapper(torch.from_numpy(images)).numpy().astype(np.float64)
    onnx_result = onnx_probabilities(make_session(onnx_path), images)
    return ParityResult(float(np.abs(torch_probabilities - onnx_result).max()), atol)


def quantize_int8(src: Path, dst: Path, sample: np.ndarray) -> dict[str, Any]:
    """Dynamic INT8 quantization. Tries all supported op types first and
    falls back to MatMul/Gemm only if ONNX Runtime cannot execute the
    quantized convolutions on this CPU, recording which one worked."""
    from onnxruntime.quantization import QuantType, quantize_dynamic

    attempts: list[tuple[str, list[str] | None]] = [
        ("all_supported_ops", None),
        ("matmul_gemm_only", ["MatMul", "Gemm"]),
    ]
    errors: list[str] = []
    for label, op_types in attempts:
        try:
            quantize_dynamic(
                str(src),
                str(dst),
                weight_type=QuantType.QUInt8,
                op_types_to_quantize=op_types,
            )
            onnx_probabilities(make_session(dst), sample[:1])
            return {"quantized_ops": label, "path": str(dst)}
        except Exception as error:
            errors.append(f"{label}: {type(error).__name__}: {error}")
            dst.unlink(missing_ok=True)
    raise RuntimeError("INT8 quantization failed for every strategy:\n" + "\n".join(errors))


def median_latency_ms(session: ort.InferenceSession, image: np.ndarray) -> float:
    session.run(None, {"image": image})
    timings = []
    for _ in range(LATENCY_RUNS):
        start = time.perf_counter()
        session.run(None, {"image": image})
        timings.append((time.perf_counter() - start) * 1000.0)
    return float(np.median(timings))


def auc_or_none(labels: np.ndarray, probabilities: np.ndarray) -> float | None:
    if len(np.unique(labels)) < 2:
        return None
    return float(roc_auc_score(labels, probabilities))


def load_evaluation_images(
    config: dict[str, Any], seed: int
) -> tuple[np.ndarray, np.ndarray]:
    """A small labeled sample for the parity check and the INT8 AUC
    comparison. It is only used to compare the two exports against each
    other, never to tune anything."""
    manifest = pd.read_csv(config["folds_path"])
    sample = sample_fast_subset(manifest, max(1, int(config["parity_images"]) // 2), seed)
    preprocess_config = PreprocessConfig(size=config["image_size"])
    images = np.stack(
        [
            preprocess_image(path, preprocess_config).transpose(2, 0, 1)
            for path in sample["path"]
        ]
    ).astype(np.float32)
    labels = np.array([1 if label == "parkinson" else 0 for label in sample["label"]])
    return images, labels


def load_cv_metrics(results_path: str | None, backbone: str) -> dict[str, Any]:
    """Quote cross-validation metrics only from a real results file the
    config points at; otherwise say plainly that none exist yet."""
    if not results_path or not Path(results_path).exists():
        return {
            "status": "TODO(run-on-gpu)",
            "note": "No real cross-validation run has been recorded for this model yet.",
        }
    rows = json.loads(Path(results_path).read_text(encoding="utf-8"))
    frame = pd.DataFrame([r for r in rows if r["model"] == backbone])
    if frame.empty:
        return {"status": "TODO(run-on-gpu)", "note": f"No results for backbone {backbone}."}
    metrics: dict[str, Any] = {"status": "from_cross_validation", "n_runs": int(len(frame))}
    for column in ("roc_auc", "sensitivity", "specificity", "pr_auc", "brier_score"):
        metrics[column] = {
            "mean": float(frame[column].mean()),
            "std": float(frame[column].std()) if len(frame) > 1 else None,
        }
    return metrics


def build_model_card(
    config: dict[str, Any],
    champion_version: str,
    threshold: float,
    fast: bool,
    export_report: dict[str, Any],
    seed: int,
) -> dict[str, Any]:
    size = int(config["image_size"])
    card: dict[str, Any] = {
        "model_name": config["model_name"],
        "backbone": config["backbone"],
        "version": champion_version,
        "format": "onnx",
        "opset": int(config["opset"]),
        "development_model": fast,
        "disclaimer": DISCLAIMER,
        "input_spec": {
            "name": "image",
            "dtype": "float32",
            "shape": ["batch", 3, size, size],
            "value_range": [0.0, 1.0],
            "preprocessing": "common/preprocess.py preprocess_image (standard mode), "
            "then HWC to CHW. No ImageNet mean/std normalization.",
        },
        "output_spec": {
            "name": "probability",
            "dtype": "float32",
            "shape": ["batch"],
            "meaning": "Probability of the Parkinson's class, sigmoid already applied.",
        },
        "threshold": threshold,
        "threshold_selection": (
            f"Highest specificity on a grouped validation split at a target sensitivity of "
            f"{config['target_sensitivity']}. Chosen on validation only."
        ),
        "training_data": {
            "sources": "HandPD and NewHandPD (UNESP), spiral and meander (wave) drawings",
            "refit": "Final model refit on all available images; cross-validation estimates "
            "come from separate fold models.",
            "seed": seed,
        },
        "cv_metrics": load_cv_metrics(config.get("cv_results_path"), config["backbone"]),
        "export": export_report,
        "known_limits": KNOWN_LIMITS,
    }
    if fast:
        card["warning"] = DEV_MODEL_WARNING
    return card


def write_artifacts(output_dir: Path, card: dict[str, Any], threshold: float) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    card_path = output_dir / CARD_FILENAME
    card_path.write_text(json.dumps(card, indent=2), encoding="utf-8")
    threshold_path = output_dir / THRESHOLD_FILENAME
    threshold_path.write_text(json.dumps({"threshold": threshold}, indent=2), encoding="utf-8")
    return {"model_card": card_path, "threshold": threshold_path}


def run_export(config: dict[str, Any], seed: int, fast: bool) -> dict[str, Any]:
    set_all_seeds(seed)
    uri = tracking_uri_for(config["mlruns_dir"])
    if fast:
        fit_and_register(config, seed, fast=True)
    champion = load_champion(config["model_name"], uri)

    output_dir = Path(config["output_dir"])
    size = int(config["image_size"])
    model_path = export_to_onnx(champion.model, output_dir / MODEL_FILENAME, size, config["opset"])
    images, labels = load_evaluation_images(config, seed)

    parity = parity_check(champion.model, model_path, images, float(config["parity_atol"]))
    if not parity.passed:
        raise AssertionError(
            f"ONNX parity failed: max abs difference {parity.max_abs_diff:.2e} exceeds "
            f"{parity.atol:.0e}. Do not ship this export."
        )

    fp32_session = make_session(model_path)
    fp32_probabilities = onnx_probabilities(fp32_session, images)
    report: dict[str, Any] = {
        "parity_max_abs_diff": parity.max_abs_diff,
        "parity_atol": parity.atol,
        "parity_images": int(len(images)),
        "file_size_mb": round(model_path.stat().st_size / 1e6, 2),
        "cpu_latency_ms": median_latency_ms(fp32_session, images[:1]),
        "sample_auc": auc_or_none(labels, fp32_probabilities),
        "int8": None,
    }

    if config.get("quantize_int8"):
        int8_path = output_dir / INT8_FILENAME
        quantization = quantize_int8(model_path, int8_path, images)
        int8_session = make_session(int8_path)
        int8_auc = auc_or_none(labels, onnx_probabilities(int8_session, images))
        fp32_auc = report["sample_auc"]
        report["int8"] = {
            **quantization,
            "file_size_mb": round(int8_path.stat().st_size / 1e6, 2),
            "cpu_latency_ms": median_latency_ms(int8_session, images[:1]),
            "sample_auc": int8_auc,
            "sample_auc_change": (
                None if int8_auc is None or fp32_auc is None else int8_auc - fp32_auc
            ),
            "note": "AUC change is measured on a small sample and only compares the two "
            "exports against each other.",
        }

    card = build_model_card(config, champion.version, champion.threshold, fast, report, seed)
    written = write_artifacts(output_dir, card, champion.threshold)
    return {"model": model_path, "report": report, **written}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config = load_merged_config(args.config)
    seed = args.seed if args.seed is not None else int(load_config(args.config)["seed"])
    result = run_export(config, seed, args.fast)
    print(json.dumps(result["report"], indent=2))
    print(f"Wrote {result['model']}, {result['model_card']}, {result['threshold']}")


if __name__ == "__main__":
    main()
