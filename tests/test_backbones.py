from __future__ import annotations

import pytest
import torch

from training.backbones import (
    BackboneName,
    Dinov2Classifier,
    SingleLogitClassifier,
    build_backbone,
    count_parameters,
    freeze_backbone_params,
    get_head_parameters,
    get_param_groups,
    unfreeze_all_params,
)

TIMM_BACKBONES: list[BackboneName] = ["resnet50", "efficientnet_b0", "convnext_tiny", "vit_small"]


@pytest.mark.parametrize("name", TIMM_BACKBONES)
def test_build_backbone_forward_shape(name: BackboneName) -> None:
    model = build_backbone(name, pretrained=False)
    images = torch.zeros(2, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (2,)


def test_build_backbone_dinov2_forward_shape() -> None:
    model = build_backbone("dinov2_small", pretrained=False)
    images = torch.zeros(2, 3, 224, 224)
    logits = model(images)
    assert logits.shape == (2,)


def test_build_backbone_unknown_name_raises() -> None:
    with pytest.raises(ValueError, match="Unknown backbone"):
        build_backbone("not_a_real_backbone", pretrained=False)  # type: ignore[arg-type]


@pytest.mark.parametrize("name", TIMM_BACKBONES)
def test_count_parameters_positive(name: BackboneName) -> None:
    model = build_backbone(name, pretrained=False)
    assert count_parameters(model) > 0


@pytest.mark.parametrize("name", TIMM_BACKBONES)
def test_freeze_backbone_leaves_head_trainable(name: BackboneName) -> None:
    model = build_backbone(name, pretrained=False)
    freeze_backbone_params(model)
    head_ids = {id(p) for p in get_head_parameters(model)}
    for parameter in model.parameters():
        expected = id(parameter) in head_ids
        assert parameter.requires_grad == expected


def test_freeze_backbone_dinov2_freezes_backbone_only() -> None:
    model = build_backbone("dinov2_small", pretrained=False)
    freeze_backbone_params(model)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_unfreeze_all_params_restores_gradients() -> None:
    model = build_backbone("efficientnet_b0", pretrained=False)
    freeze_backbone_params(model)
    unfreeze_all_params(model)
    assert all(p.requires_grad for p in model.parameters())


def test_get_param_groups_covers_every_parameter_once() -> None:
    model = build_backbone("resnet50", pretrained=False)
    groups = get_param_groups(model, head_lr=1e-4, backbone_lr=1e-5)
    total_in_groups = sum(len(g["params"]) for g in groups)
    assert total_in_groups == len(list(model.parameters()))
    assert groups[0]["lr"] == 1e-4
    assert groups[1]["lr"] == 1e-5


def test_dinov2_linear_probe_mode_freezes_backbone_at_construction() -> None:
    model = build_backbone("dinov2_small", pretrained=False, dinov2_mode="linear_probe")
    assert isinstance(model, Dinov2Classifier)
    assert all(not p.requires_grad for p in model.backbone.parameters())
    assert all(p.requires_grad for p in model.head.parameters())


def test_single_logit_classifier_wraps_timm_model() -> None:
    model = build_backbone("resnet50", pretrained=False)
    assert isinstance(model, SingleLogitClassifier)
