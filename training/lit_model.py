"""LightningModule wrapping a single-logit binary classifier backbone.

AdamW with differential learning rates (head higher than backbone), the
backbone frozen for the first freeze_epochs then unfrozen, label
smoothing, class weights via pos_weight, and a cosine learning rate
schedule, per CLAUDE.md Phase 3 rule 2. Early stopping and checkpointing
are Lightning Trainer callbacks configured in training/train.py, not part
of this module.
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as pl
import torch
from torch import nn

from training.backbones import (
    BackboneName,
    Dinov2Mode,
    build_backbone,
    freeze_backbone_params,
    get_param_groups,
    unfreeze_all_params,
)


def smooth_binary_labels(labels: torch.Tensor, smoothing: float) -> torch.Tensor:
    """y=1 becomes 1-smoothing/2, y=0 becomes smoothing/2, so the model is
    never asked to output a perfectly confident 0 or 1."""
    return labels * (1 - smoothing) + 0.5 * smoothing


class SpiralWaveClassifier(pl.LightningModule):
    """Binary classifier LightningModule for one backbone."""

    def __init__(
        self,
        backbone_name: BackboneName,
        pretrained: bool = True,
        dinov2_mode: Dinov2Mode = "full_finetune",
        head_lr: float = 1e-4,
        backbone_lr: float = 1e-5,
        freeze_epochs: int = 2,
        label_smoothing: float = 0.05,
        pos_weight: float | None = None,
        max_epochs: int = 10,
    ) -> None:
        super().__init__()
        self.save_hyperparameters()
        # Plain attributes, not just self.hparams, since mypy cannot know
        # the dynamic AttributeDict's contents statically.
        self.freeze_epochs = freeze_epochs
        self.label_smoothing = label_smoothing
        self.head_lr = head_lr
        self.backbone_lr = backbone_lr
        self.max_epochs = max_epochs
        self.model = build_backbone(backbone_name, pretrained=pretrained, dinov2_mode=dinov2_mode)
        pos_weight_tensor = torch.tensor(pos_weight) if pos_weight is not None else None
        self.loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor)
        if freeze_epochs > 0:
            freeze_backbone_params(self.model)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        result: torch.Tensor = self.model(images)
        return result

    def on_train_epoch_start(self) -> None:
        if self.current_epoch == self.freeze_epochs:
            unfreeze_all_params(self.model)

    def step(self, batch: tuple[torch.Tensor, torch.Tensor], stage: str) -> torch.Tensor:
        images, labels = batch
        labels = labels.float()
        logits = self(images)
        smoothed_labels = smooth_binary_labels(labels, self.label_smoothing)
        loss = self.loss_fn(logits, smoothed_labels)
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, "train")

    def validation_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        return self.step(batch, "val")

    def predict_step(
        self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int
    ) -> torch.Tensor:
        images = batch[0] if isinstance(batch, list | tuple) else batch
        logits = self(images)
        return torch.sigmoid(logits)

    def configure_optimizers(self) -> Any:
        param_groups = get_param_groups(
            self.model, head_lr=self.head_lr, backbone_lr=self.backbone_lr
        )
        optimizer = torch.optim.AdamW(param_groups)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(self.max_epochs, 1)
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
