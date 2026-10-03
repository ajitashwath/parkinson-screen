"""Offline Grad-CAM explanations plus two quantitative sanity checks.

Grad-CAM here is an evaluation tool, run in PyTorch. The deployed backend
has no PyTorch, so it serves an occlusion sensitivity map instead (Phase
7); this module is how the project checks, offline, whether a model
looks at the drawing at all. Per CLAUDE.md Phase 5:

    (a) stroke_mass: the fraction of heatmap mass that lies on stroke
        pixels, reported next to the strokes' share of image area (a
        uniform heatmap would score exactly that share) and next to the
        mass in a border margin, so attention to borders or background is
        stated plainly rather than hidden inside an average.
    (b) deletion test: remove the most important pixels first and watch
        how fast the predicted class's probability falls, against a
        random-order baseline. A faithful map falls faster than random.

Usage:
    python -m training.explain --config configs/train_resnet50.yaml --fast
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torch import nn

from common.preprocess import PreprocessConfig, preprocess_image
from training.backbones import Dinov2Classifier, SingleLogitClassifier, unfreeze_all_params
from training.data import assign_folds
from training.train import (
    evaluate_fitted_backbone,
    fit_backbone,
    load_config,
    sample_fast_subset,
    set_all_seeds,
)

STROKE_THRESHOLD = 0.8
STROKE_DILATION_PX = 3
BORDER_MARGIN_FRACTION = 0.1
DELETION_STEPS = 10
DELETION_MAX_FRACTION = 0.5
RANDOM_DELETION_RUNS = 5
MAX_OVERLAYS_PER_GROUP = 4


def vit_reshape_transform(tensor: torch.Tensor) -> torch.Tensor:
    """Turn a transformer's token sequence (B, 1 + N, C) into a
    convolution-style feature map (B, C, side, side) by dropping the class
    token and folding the patch tokens back into their square grid. Works
    for timm's ViT-S/16 (14 x 14 patches) and DINOv2-S/14 (16 x 16)
    without hardcoding either grid size."""
    patches = tensor[:, 1:, :]
    side = int(round(patches.shape[1] ** 0.5))
    if side * side != patches.shape[1]:
        raise ValueError(
            f"Expected a square patch grid, got {patches.shape[1]} tokens after the class token"
        )
    grid = patches.reshape(patches.shape[0], side, side, patches.shape[2])
    return grid.permute(0, 3, 1, 2)


def find_target_layer(
    model: nn.Module,
) -> tuple[nn.Module, Any]:
    """The layer Grad-CAM reads activations from, and the reshape
    transform it needs (None for convolutional feature maps)."""
    if isinstance(model, Dinov2Classifier):
        return model.backbone.encoder.layer[-1].norm1, vit_reshape_transform
    if isinstance(model, SingleLogitClassifier):
        backbone = model.backbone
        if hasattr(backbone, "cls_token") and hasattr(backbone, "blocks"):
            return backbone.blocks[-1].norm1, vit_reshape_transform
        if hasattr(backbone, "layer4"):
            return backbone.layer4[-1], None
        if hasattr(backbone, "stages"):
            return backbone.stages[-1], None
        if hasattr(backbone, "blocks"):
            return backbone.blocks[-1], None
    raise TypeError(f"No Grad-CAM target layer known for {type(model).__name__}")


class LogitAsClassScores(nn.Module):
    """Grad-CAM expects (batch, classes) scores; this model emits one
    logit per image, so add the trailing class axis."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = self.model(images)
        return logits.unsqueeze(-1)


class SignedLogitTarget:
    """Explain the predicted class: +logit evidence for "parkinson" when
    that is the prediction, -logit evidence for "healthy" otherwise."""

    def __init__(self, sign: float) -> None:
        self.sign = sign

    def __call__(self, model_output: torch.Tensor) -> torch.Tensor:
        return self.sign * model_output[0]


def image_to_chw(path: str, preprocess_config: PreprocessConfig) -> torch.Tensor:
    array = preprocess_image(path, preprocess_config)
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def predict_probability(model: nn.Module, images: torch.Tensor) -> np.ndarray:
    model.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(model(images))
    return np.asarray(probabilities.cpu().numpy(), dtype=np.float64)


def compute_gradcam(model: nn.Module, image: torch.Tensor, predicted_positive: bool) -> np.ndarray:
    """Heatmap in [0, 1] at the input resolution for one (3, H, W) image."""
    unfreeze_all_params(model)
    model.eval()
    layer, reshape_transform = find_target_layer(model)
    wrapped = LogitAsClassScores(model)
    target = SignedLogitTarget(1.0 if predicted_positive else -1.0)
    with GradCAM(
        model=wrapped, target_layers=[layer], reshape_transform=reshape_transform
    ) as cam:
        heatmap = cam(input_tensor=image.unsqueeze(0), targets=[target])[0]
    return np.asarray(heatmap, dtype=np.float64)


def stroke_mask(image: torch.Tensor) -> np.ndarray:
    """Boolean mask of stroke pixels in a (3, H, W) preprocessed image,
    dilated a few pixels because Grad-CAM maps are coarse and a stroke
    one pixel wide would otherwise never score any mass."""
    gray = image[0].numpy()
    mask = (gray < STROKE_THRESHOLD).astype(np.uint8)
    kernel = np.ones((STROKE_DILATION_PX, STROKE_DILATION_PX), np.uint8)
    return cv2.dilate(mask, kernel, iterations=1).astype(bool)


def heatmap_stats(heatmap: np.ndarray, mask: np.ndarray) -> dict[str, float]:
    total = float(heatmap.sum())
    if total <= 0 or not mask.any():
        return {
            "stroke_mass_fraction": float("nan"),
            "stroke_area_fraction": float(mask.mean()),
            "mass_to_area_ratio": float("nan"),
            "border_mass_fraction": float("nan"),
        }
    height, width = heatmap.shape
    margin_y = int(height * BORDER_MARGIN_FRACTION)
    margin_x = int(width * BORDER_MARGIN_FRACTION)
    interior = np.zeros_like(mask)
    interior[margin_y : height - margin_y, margin_x : width - margin_x] = True
    stroke_mass = float(heatmap[mask].sum() / total)
    area = float(mask.mean())
    return {
        "stroke_mass_fraction": stroke_mass,
        "stroke_area_fraction": area,
        "mass_to_area_ratio": stroke_mass / area,
        "border_mass_fraction": float(heatmap[~interior].sum() / total),
    }


def class_probability(model: nn.Module, image: torch.Tensor, predicted_positive: bool) -> float:
    probability = float(predict_probability(model, image.unsqueeze(0))[0])
    return probability if predicted_positive else 1.0 - probability


def deletion_curve(
    model: nn.Module,
    image: torch.Tensor,
    order: np.ndarray,
    predicted_positive: bool,
    steps: int = DELETION_STEPS,
    max_fraction: float = DELETION_MAX_FRACTION,
) -> list[float]:
    """Predicted-class probability as pixels are deleted (set to white
    background) in the given order, from 0 up to max_fraction of all
    pixels in `steps` increments."""
    flat = image.reshape(3, -1)
    n_pixels = flat.shape[1]
    curve = []
    for fraction in np.linspace(0.0, max_fraction, steps + 1):
        n_delete = int(round(fraction * n_pixels))
        edited = flat.clone()
        if n_delete > 0:
            edited[:, torch.from_numpy(order[:n_delete])] = 1.0
        curve.append(class_probability(model, edited.reshape_as(image), predicted_positive))
    return curve


def deletion_scores(
    model: nn.Module,
    image: torch.Tensor,
    heatmap: np.ndarray,
    predicted_positive: bool,
    seed: int = 0,
) -> dict[str, Any]:
    """Compare deleting most-important-first against random order. The
    mean of each curve is its normalized area; lower means the
    probability fell faster, and a faithful heatmap should fall faster
    than random."""
    importance_order = np.argsort(-heatmap.reshape(-1), kind="stable")
    guided = deletion_curve(model, image, importance_order, predicted_positive)

    rng = np.random.default_rng(seed)
    random_curves = [
        deletion_curve(
            model, image, rng.permutation(importance_order.size), predicted_positive
        )
        for _ in range(RANDOM_DELETION_RUNS)
    ]
    random_curve = np.mean(random_curves, axis=0).tolist()
    guided_area = float(np.mean(guided))
    random_area = float(np.mean(random_curve))
    return {
        "deletion_curve": guided,
        "random_deletion_curve": random_curve,
        "deletion_auc": guided_area,
        "random_deletion_auc": random_area,
        "faithful": guided_area < random_area,
    }


def overlay_image(image: torch.Tensor, heatmap: np.ndarray) -> np.ndarray:
    rgb = image.permute(1, 2, 0).numpy().astype(np.float32)
    overlay: np.ndarray = show_cam_on_image(rgb, heatmap.astype(np.float32), use_rgb=True)
    return overlay


def explain_one(
    model: nn.Module, image: torch.Tensor, seed: int
) -> tuple[np.ndarray, dict[str, Any], bool]:
    probability = float(predict_probability(model, image.unsqueeze(0))[0])
    predicted_positive = probability >= 0.5
    heatmap = compute_gradcam(model, image, predicted_positive)
    stats: dict[str, Any] = heatmap_stats(heatmap, stroke_mask(image))
    stats.update(deletion_scores(model, image, heatmap, predicted_positive, seed=seed))
    return heatmap, stats, predicted_positive


def select_examples(
    labels: np.ndarray, probabilities: np.ndarray, threshold: float, per_group: int
) -> dict[str, list[int]]:
    predictions = (probabilities >= threshold).astype(int)
    correct = [i for i in range(len(labels)) if predictions[i] == labels[i]]
    incorrect = [i for i in range(len(labels)) if predictions[i] != labels[i]]
    return {"correct": correct[:per_group], "incorrect": incorrect[:per_group]}


DEV_RUN_NOTE = (
    "Development run (--fast): the model was trained for one epoch from random "
    "initialization on a tiny sample, so the numbers below only prove the pipeline "
    "runs. They say nothing about what a trained model attends to."
)


def render_finding(summary: pd.DataFrame, development: bool = False) -> str:
    if summary.empty:
        return "# Explainability check\n\nNo images were explained.\n"
    mean_ratio = float(summary["mass_to_area_ratio"].mean())
    mean_border = float(summary["border_mass_fraction"].mean())
    faithful_share = float(summary["faithful"].mean())
    lines = ["# Explainability check", ""]
    if development:
        lines += [DEV_RUN_NOTE, ""]
    lines += [
        f"Explained {len(summary)} test images with Grad-CAM.",
        "",
        f"- Mean heatmap mass on stroke pixels relative to the strokes' area share: "
        f"{mean_ratio:.2f} (1.0 means no preference for strokes over blank paper).",
        f"- Mean heatmap mass in the outer {int(BORDER_MARGIN_FRACTION * 100)}% border margin: "
        f"{mean_border:.2f}.",
        f"- Share of heatmaps that fell faster than a random deletion order: "
        f"{faithful_share:.2f}.",
        "",
    ]
    if mean_ratio < 1.0:
        lines.append(
            "Finding: on average the model attends to blank paper more than to the "
            "stroke, which is a warning sign for background or scanner cues."
        )
    else:
        lines.append(
            "Finding: on average the model attends to stroke pixels at least in "
            "proportion to their area."
        )
    if mean_border > BORDER_MARGIN_FRACTION * 4:
        lines.append(
            "Border attention is high: more of the heatmap sits in the outer margin "
            "than the margin's own area would suggest."
        )
    lines.append("")
    return "\n".join(lines)


def plot_deletion_curves(rows: list[dict[str, Any]]) -> go.Figure:
    figure = go.Figure()
    fractions = np.linspace(0.0, DELETION_MAX_FRACTION, DELETION_STEPS + 1)
    for i, row in enumerate(rows):
        figure.add_trace(
            go.Scatter(
                x=fractions,
                y=row["deletion_curve"],
                mode="lines",
                name=f"{row['group']} {i} guided",
                line={"color": "crimson"},
            )
        )
        figure.add_trace(
            go.Scatter(
                x=fractions,
                y=row["random_deletion_curve"],
                mode="lines",
                name=f"{row['group']} {i} random",
                line={"color": "gray", "dash": "dash"},
            )
        )
    figure.update_layout(
        title="Deletion test: predicted-class probability as pixels are removed",
        xaxis_title="Fraction of pixels deleted",
        yaxis_title="Predicted-class probability",
    )
    return figure


def run_explain(
    config: dict[str, Any], seed: int, fast: bool, reports_dir: Path
) -> pd.DataFrame:
    set_all_seeds(seed)
    manifest = pd.read_csv(config["folds_path"])
    if fast:
        manifest = sample_fast_subset(manifest, config.get("fast_sample_per_class", 6), seed)
        manifest = assign_folds(manifest, n_folds=2, seed=seed)
    fold = int(manifest["fold"].min())
    train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
    test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)

    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)
    trainer, fitted, threshold, _ = fit_backbone(
        config["backbone"], train_df, config, seed, max_epochs, pretrained
    )
    evaluation = evaluate_fitted_backbone(trainer, fitted, test_df, config, threshold)
    labels = np.asarray(evaluation["y_true"])
    probabilities = np.asarray(evaluation["y_prob"])

    model = fitted.model.cpu()
    preprocess_config = PreprocessConfig(size=config["image_size"])
    selected = select_examples(labels, probabilities, threshold, MAX_OVERLAYS_PER_GROUP)

    overlay_dir = reports_dir / "explain"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for group, indices in selected.items():
        for index in indices:
            image = image_to_chw(test_df.iloc[index]["path"], preprocess_config)
            heatmap, stats, predicted_positive = explain_one(model, image, seed + index)
            overlay = overlay_image(image, heatmap)
            overlay_path = overlay_dir / f"{group}_{index}.png"
            cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            rows.append(
                {
                    "group": group,
                    "test_index": index,
                    "true_label": int(labels[index]),
                    "probability": float(probabilities[index]),
                    "predicted_positive": predicted_positive,
                    "overlay_path": str(overlay_path),
                    **stats,
                }
            )
    return pd.DataFrame(rows)


def write_explain_outputs(
    summary: pd.DataFrame, reports_dir: Path, development: bool = False
) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    scalar_columns = [
        c for c in summary.columns if c not in ("deletion_curve", "random_deletion_curve")
    ]
    csv_path = reports_dir / "explain_summary.csv"
    summary[scalar_columns].to_csv(csv_path, index=False)

    finding_path = reports_dir / "explain_finding.md"
    finding_path.write_text(render_finding(summary, development), encoding="utf-8")
    written = {"summary_csv": csv_path, "finding": finding_path}

    if not summary.empty:
        figure = plot_deletion_curves(summary.to_dict("records"))
        figure_path = reports_dir / "explain_deletion.html"
        figure.write_html(figure_path, include_plotlyjs="cdn")
        written["deletion_html"] = figure_path
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_resnet50.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--reports-dir", default="reports")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    reports_dir = Path(args.reports_dir)

    summary = run_explain(config, seed, args.fast, reports_dir)
    written = write_explain_outputs(summary, reports_dir, development=args.fast)
    print(render_finding(summary, development=args.fast))
    for key, path in written.items():
        print(f"Wrote {key} to {path}")


if __name__ == "__main__":
    main()
