from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from training.backbones import build_backbone
from training.export_onnx import (
    CARD_FILENAME,
    THRESHOLD_FILENAME,
    ProbabilityWrapper,
    auc_or_none,
    build_model_card,
    export_to_onnx,
    load_cv_metrics,
    make_session,
    onnx_probabilities,
    parity_check,
    quantize_int8,
    write_artifacts,
)


@pytest.fixture(scope="module")
def exported(tmp_path_factory):
    torch.manual_seed(0)
    model = build_backbone("efficientnet_b0", pretrained=False).eval()
    path = export_to_onnx(model, tmp_path_factory.mktemp("onnx") / "model.onnx")
    return model, path


def random_images(count: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).random((count, 3, 224, 224)).astype(np.float32)


def test_export_writes_a_loadable_graph_with_named_io(exported) -> None:
    _, path = exported
    session = make_session(path)
    assert [i.name for i in session.get_inputs()] == ["image"]
    assert [o.name for o in session.get_outputs()] == ["probability"]


@pytest.mark.parametrize("batch_size", [1, 3, 5])
def test_exported_graph_accepts_any_batch_size(exported, batch_size: int) -> None:
    _, path = exported
    probabilities = onnx_probabilities(make_session(path), random_images(batch_size))
    assert probabilities.shape == (batch_size,)


def test_exported_probabilities_are_valid_probabilities(exported) -> None:
    _, path = exported
    probabilities = onnx_probabilities(make_session(path), random_images(4))
    assert ((probabilities >= 0.0) & (probabilities <= 1.0)).all()


def test_onnx_matches_pytorch_within_the_contract_tolerance(exported) -> None:
    model, path = exported
    result = parity_check(model, path, random_images(4), atol=1e-4)
    assert result.passed
    assert result.max_abs_diff <= 1e-4


def test_parity_check_fails_when_the_model_no_longer_matches_the_export(exported) -> None:
    model, path = exported
    changed = build_backbone("efficientnet_b0", pretrained=False).eval()
    changed.load_state_dict(model.state_dict())
    with torch.no_grad():
        for parameter in changed.parameters():
            parameter.add_(0.05)
    result = parity_check(changed, path, random_images(4), atol=1e-4)
    assert not result.passed


def test_probability_wrapper_applies_sigmoid() -> None:
    class Constant(torch.nn.Module):
        def forward(self, images: torch.Tensor) -> torch.Tensor:
            return torch.zeros(images.shape[0])

    output = ProbabilityWrapper(Constant())(torch.zeros(3, 3, 8, 8))
    assert torch.allclose(output, torch.full((3,), 0.5))


def test_int8_quantization_runs_and_is_smaller(exported, tmp_path: Path) -> None:
    _, path = exported
    int8_path = tmp_path / "model_int8.onnx"
    report = quantize_int8(path, int8_path, random_images(2))
    assert report["quantized_ops"] in {"all_supported_ops", "matmul_gemm_only"}
    assert int8_path.stat().st_size < path.stat().st_size
    assert onnx_probabilities(make_session(int8_path), random_images(2)).shape == (2,)


def test_auc_or_none_needs_both_classes() -> None:
    assert auc_or_none(np.array([1, 1, 1]), np.array([0.1, 0.5, 0.9])) is None
    assert auc_or_none(np.array([0, 1]), np.array([0.1, 0.9])) == 1.0


def test_cv_metrics_say_todo_when_no_real_results_exist() -> None:
    assert load_cv_metrics(None, "resnet50")["status"] == "TODO(run-on-gpu)"


def test_cv_metrics_say_todo_when_results_file_is_missing(tmp_path: Path) -> None:
    assert load_cv_metrics(str(tmp_path / "nope.json"), "resnet50")["status"] == "TODO(run-on-gpu)"


def test_cv_metrics_aggregate_a_real_results_file(tmp_path: Path) -> None:
    rows = [
        {"model": "resnet50", "roc_auc": 0.8, "sensitivity": 0.9, "specificity": 0.6,
         "pr_auc": 0.85, "brier_score": 0.2},
        {"model": "resnet50", "roc_auc": 0.6, "sensitivity": 0.7, "specificity": 0.4,
         "pr_auc": 0.65, "brier_score": 0.3},
        {"model": "vit_small", "roc_auc": 0.1, "sensitivity": 0.1, "specificity": 0.1,
         "pr_auc": 0.1, "brier_score": 0.9},
    ]
    path = tmp_path / "results.json"
    path.write_text(json.dumps(rows), encoding="utf-8")
    metrics = load_cv_metrics(str(path), "resnet50")
    assert metrics["status"] == "from_cross_validation"
    assert metrics["n_runs"] == 2
    assert metrics["roc_auc"]["mean"] == pytest.approx(0.7)


def test_cv_metrics_say_todo_for_a_backbone_with_no_results(tmp_path: Path) -> None:
    path = tmp_path / "results.json"
    path.write_text(json.dumps([{"model": "resnet50"}]), encoding="utf-8")
    assert load_cv_metrics(str(path), "vit_small")["status"] == "TODO(run-on-gpu)"


CONFIG = {
    "model_name": "m",
    "backbone": "efficientnet_b0",
    "opset": 17,
    "image_size": 224,
    "target_sensitivity": 0.9,
    "cv_results_path": None,
}


def test_dev_model_card_is_labeled_and_carries_a_warning() -> None:
    card = build_model_card(CONFIG, "3", 0.42, fast=True, export_report={}, seed=1)
    assert card["development_model"] is True
    assert "Not a real screening model" in card["warning"]
    assert card["threshold"] == 0.42
    assert card["version"] == "3"


def test_real_model_card_has_no_dev_warning() -> None:
    card = build_model_card(CONFIG, "1", 0.5, fast=False, export_report={}, seed=1)
    assert card["development_model"] is False
    assert "warning" not in card


def test_model_card_states_scope_and_input_contract() -> None:
    card = build_model_card(CONFIG, "1", 0.5, fast=False, export_report={}, seed=1)
    assert "not a diagnosis" in card["disclaimer"].lower()
    assert card["input_spec"]["shape"] == ["batch", 3, 224, 224]
    assert card["input_spec"]["value_range"] == [0.0, 1.0]
    assert card["cv_metrics"]["status"] == "TODO(run-on-gpu)"
    assert len(card["known_limits"]) >= 4


def test_model_card_is_json_serializable_and_free_of_em_dashes() -> None:
    card = build_model_card(CONFIG, "1", 0.5, fast=True, export_report={"a": 1}, seed=1)
    text = json.dumps(card)
    assert chr(0x2014) not in text and chr(0x2013) not in text


def test_write_artifacts_saves_card_and_threshold_together(tmp_path: Path) -> None:
    card = build_model_card(CONFIG, "1", 0.37, fast=False, export_report={}, seed=1)
    written = write_artifacts(tmp_path / "models", card, 0.37)
    assert written["model_card"].name == CARD_FILENAME
    assert written["threshold"].name == THRESHOLD_FILENAME
    assert json.loads(written["threshold"].read_text(encoding="utf-8")) == {"threshold": 0.37}
    assert json.loads(written["model_card"].read_text(encoding="utf-8"))["threshold"] == 0.37
