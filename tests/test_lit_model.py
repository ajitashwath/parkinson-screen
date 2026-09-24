from __future__ import annotations

import pytest
import torch

from training.lit_model import SpiralWaveClassifier, smooth_binary_labels


def test_smooth_binary_labels_moves_away_from_extremes() -> None:
    labels = torch.tensor([0.0, 1.0])
    smoothed = smooth_binary_labels(labels, smoothing=0.1)
    assert smoothed[0].item() > 0.0
    assert smoothed[1].item() < 1.0
    assert smoothed[0].item() == pytest.approx(0.05)
    assert smoothed[1].item() == pytest.approx(0.95)


def test_smooth_binary_labels_no_smoothing_is_identity() -> None:
    labels = torch.tensor([0.0, 1.0])
    smoothed = smooth_binary_labels(labels, smoothing=0.0)
    assert torch.equal(smoothed, labels)


def test_lit_model_forward_shape() -> None:
    model = SpiralWaveClassifier(backbone_name="efficientnet_b0", pretrained=False, freeze_epochs=0)
    images = torch.zeros(2, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (2,)


def test_lit_model_training_step_returns_scalar_loss() -> None:
    model = SpiralWaveClassifier(backbone_name="efficientnet_b0", pretrained=False, freeze_epochs=0)
    images = torch.zeros(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    loss = model.training_step((images, labels), batch_idx=0)
    assert loss.ndim == 0
    assert not torch.isnan(loss)


def test_lit_model_starts_with_backbone_frozen_when_freeze_epochs_positive() -> None:
    model = SpiralWaveClassifier(backbone_name="efficientnet_b0", pretrained=False, freeze_epochs=2)
    frozen = [p for p in model.model.parameters() if not p.requires_grad]
    assert len(frozen) > 0


def test_lit_model_no_freeze_when_freeze_epochs_zero() -> None:
    model = SpiralWaveClassifier(backbone_name="efficientnet_b0", pretrained=False, freeze_epochs=0)
    assert all(p.requires_grad for p in model.model.parameters())


def test_lit_model_configure_optimizers_returns_optimizer_and_scheduler() -> None:
    model = SpiralWaveClassifier(backbone_name="efficientnet_b0", pretrained=False, freeze_epochs=0)
    result = model.configure_optimizers()
    assert "optimizer" in result
    assert "lr_scheduler" in result
    param_groups = result["optimizer"].param_groups
    assert len(param_groups) == 2
    assert param_groups[0]["lr"] == model.head_lr
    assert param_groups[1]["lr"] == model.backbone_lr


def test_lit_model_predict_step_returns_probabilities() -> None:
    model = SpiralWaveClassifier(backbone_name="efficientnet_b0", pretrained=False, freeze_epochs=0)
    images = torch.zeros(2, 3, 224, 224)
    labels = torch.tensor([0, 1])
    probabilities = model.predict_step((images, labels), batch_idx=0)
    assert probabilities.shape == (2,)
    assert (probabilities >= 0).all() and (probabilities <= 1).all()
