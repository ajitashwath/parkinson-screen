"""Backbone factory: a single-logit binary classifier built on top of a
pretrained (by default) image backbone.

A single logit rather than a 2-class softmax head, since the product only
ever needs one probability (screening score); BCEWithLogitsLoss on one
output is equivalent to 2-class cross-entropy for a binary problem and is
simpler to wire through the whole pipeline (metrics, ONNX export, the
backend). See docs/ASSUMPTIONS.md.

resnet50, efficientnet_b0 and convnext_tiny come from timm, which is also
used for vit_small: CLAUDE.md allows either timm or Transformers for
vit_small "whichever loads reliably", and timm was chosen for consistency
with the other three CNN/ViT backbones and simpler pretrained-weight
handling (a single create_model call, no separate image processor).
dinov2_small uses Transformers specifically, per CLAUDE.md, and supports
both a linear-probe mode (backbone frozen) and a full fine-tune mode.
"""

from __future__ import annotations

from typing import Any, Literal

import torch
from torch import nn

BackboneName = Literal["resnet50", "efficientnet_b0", "convnext_tiny", "vit_small", "dinov2_small"]
Dinov2Mode = Literal["linear_probe", "full_finetune"]

TIMM_MODEL_NAMES: dict[str, str] = {
    "resnet50": "resnet50",
    "efficientnet_b0": "efficientnet_b0",
    "convnext_tiny": "convnext_tiny",
    "vit_small": "vit_small_patch16_224",
}

DINOV2_HF_NAME = "facebook/dinov2-small"


class SingleLogitClassifier(nn.Module):
    """Wraps a timm backbone (already built with num_classes=1) so every
    backbone in this module exposes the same forward signature: an image
    batch in, a single logit per image out."""

    def __init__(self, backbone: nn.Module) -> None:
        super().__init__()
        self.backbone = backbone

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.backbone(images)
        return logits.squeeze(-1)


class Dinov2Classifier(nn.Module):
    """DINOv2 backbone (Transformers) plus a linear head on the pooled
    [CLS] token output. linear_probe freezes the backbone so only the head
    trains; full_finetune trains both."""

    def __init__(self, backbone: nn.Module, hidden_size: int, mode: Dinov2Mode) -> None:
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(hidden_size, 1)
        self.mode = mode
        if mode == "linear_probe":
            for parameter in self.backbone.parameters():
                parameter.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        outputs = self.backbone(pixel_values=images)
        pooled = outputs.pooler_output
        logits: torch.Tensor = self.head(pooled)
        return logits.squeeze(-1)


def build_backbone(
    name: BackboneName,
    pretrained: bool = True,
    dinov2_mode: Dinov2Mode = "full_finetune",
) -> nn.Module:
    """Build a single-logit binary classifier for the given backbone name.
    pretrained=False builds the architecture with random weights, used for
    the pretrained-vs-scratch ablation (Phase 4) and for fast CPU smoke
    tests where downloading pretrained weights would be wasteful."""
    if name == "dinov2_small":
        from transformers import Dinov2Config, Dinov2Model

        if pretrained:
            backbone = Dinov2Model.from_pretrained(DINOV2_HF_NAME)
        else:
            config = Dinov2Config.from_pretrained(DINOV2_HF_NAME)
            backbone = Dinov2Model(config)
        hidden_size = backbone.config.hidden_size
        return Dinov2Classifier(backbone, hidden_size, mode=dinov2_mode)

    if name not in TIMM_MODEL_NAMES:
        known = sorted(TIMM_MODEL_NAMES) + ["dinov2_small"]
        raise ValueError(f"Unknown backbone {name!r}; expected one of {known}")

    import timm

    timm_backbone = timm.create_model(TIMM_MODEL_NAMES[name], pretrained=pretrained, num_classes=1)
    return SingleLogitClassifier(timm_backbone)


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def get_head_parameters(model: nn.Module) -> list[nn.Parameter]:
    """The final classifier layer's parameters, regardless of backbone
    type: timm's own get_classifier() for SingleLogitClassifier, the
    linear head for Dinov2Classifier. Used to freeze everything else
    during the initial frozen-backbone epochs and to give the head a
    higher learning rate than the rest of the network."""
    if isinstance(model, Dinov2Classifier):
        return list(model.head.parameters())
    if isinstance(model, SingleLogitClassifier):
        classifier = model.backbone.get_classifier()
        return list(classifier.parameters())
    raise TypeError(f"Unsupported model type for get_head_parameters: {type(model)}")


def freeze_backbone_params(model: nn.Module) -> None:
    """Freeze every parameter except the head, for the initial
    frozen-backbone training epochs (CLAUDE.md Phase 3 rule 2)."""
    head_param_ids = {id(p) for p in get_head_parameters(model)}
    for parameter in model.parameters():
        parameter.requires_grad = id(parameter) in head_param_ids


def unfreeze_all_params(model: nn.Module) -> None:
    for parameter in model.parameters():
        parameter.requires_grad = True


def get_param_groups(model: nn.Module, head_lr: float, backbone_lr: float) -> list[dict[str, Any]]:
    """Two optimizer param groups: the head at head_lr, everything else at
    the lower backbone_lr, per CLAUDE.md Phase 3 rule 2 ("differential
    learning rates")."""
    head_param_ids = {id(p) for p in get_head_parameters(model)}
    head_params = [p for p in model.parameters() if id(p) in head_param_ids]
    backbone_params = [p for p in model.parameters() if id(p) not in head_param_ids]
    return [
        {"params": head_params, "lr": head_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
