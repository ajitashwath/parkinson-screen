from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch
from torch import nn

from training.backbones import BackboneName, build_backbone
from training.explain import (
    LogitAsClassScores,
    SignedLogitTarget,
    compute_gradcam,
    deletion_scores,
    find_target_layer,
    heatmap_stats,
    overlay_image,
    plot_deletion_curves,
    render_finding,
    select_examples,
    stroke_mask,
    vit_reshape_transform,
    write_explain_outputs,
)

ALL_BACKBONES: list[BackboneName] = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_small",
    "dinov2_small",
]


class TopLeftBrightnessModel(nn.Module):
    """Dark top-left quadrant means "positive": the only region that
    matters is known exactly, so deletion behavior is checkable."""

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return (0.5 - images[:, 0, :112, :112].mean(dim=(1, 2))) * 20.0


def test_vit_reshape_transform_vit_grid() -> None:
    tokens = torch.zeros(2, 1 + 14 * 14, 384)
    assert vit_reshape_transform(tokens).shape == (2, 384, 14, 14)


def test_vit_reshape_transform_dinov2_grid() -> None:
    tokens = torch.zeros(2, 1 + 16 * 16, 384)
    assert vit_reshape_transform(tokens).shape == (2, 384, 16, 16)


def test_vit_reshape_transform_places_patches_in_row_major_order() -> None:
    tokens = torch.zeros(1, 1 + 4, 1)
    tokens[0, 1:, 0] = torch.tensor([1.0, 2.0, 3.0, 4.0])
    grid = vit_reshape_transform(tokens)[0, 0]
    assert grid.tolist() == [[1.0, 2.0], [3.0, 4.0]]


def test_vit_reshape_transform_rejects_non_square_grid() -> None:
    with pytest.raises(ValueError, match="square patch grid"):
        vit_reshape_transform(torch.zeros(1, 1 + 7, 4))


@pytest.mark.parametrize("name", ALL_BACKBONES)
def test_gradcam_produces_a_normalized_heatmap_for_every_backbone(name: BackboneName) -> None:
    model = build_backbone(name, pretrained=False)
    image = torch.rand(3, 224, 224)
    heatmap = compute_gradcam(model, image, predicted_positive=True)
    assert heatmap.shape == (224, 224)
    assert heatmap.min() >= 0.0
    assert heatmap.max() <= 1.0 + 1e-6
    assert np.isfinite(heatmap).all()


def test_find_target_layer_raises_for_unknown_model() -> None:
    with pytest.raises(TypeError, match="No Grad-CAM target layer"):
        find_target_layer(nn.Linear(2, 1))


def test_logit_as_class_scores_adds_class_axis() -> None:
    wrapped = LogitAsClassScores(TopLeftBrightnessModel())
    assert wrapped(torch.zeros(3, 3, 224, 224)).shape == (3, 1)


def test_signed_logit_target_flips_sign_for_negative_class() -> None:
    output = torch.tensor([2.0])
    assert SignedLogitTarget(1.0)(output).item() == 2.0
    assert SignedLogitTarget(-1.0)(output).item() == -2.0


def test_stroke_mask_marks_dark_pixels_and_dilates() -> None:
    image = torch.ones(3, 20, 20)
    image[:, 10, 10] = 0.0
    mask = stroke_mask(image)
    assert mask[10, 10]
    assert mask.sum() > 1


def test_heatmap_stats_all_mass_on_stroke() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    heatmap = np.zeros((10, 10))
    heatmap[mask] = 1.0
    stats = heatmap_stats(heatmap, mask)
    assert stats["stroke_mass_fraction"] == pytest.approx(1.0)
    assert stats["stroke_area_fraction"] == pytest.approx(0.04)
    assert stats["mass_to_area_ratio"] == pytest.approx(25.0)
    assert stats["border_mass_fraction"] == pytest.approx(0.0)


def test_heatmap_stats_uniform_heatmap_has_ratio_one() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[3:7, 3:7] = True
    stats = heatmap_stats(np.ones((10, 10)), mask)
    assert stats["mass_to_area_ratio"] == pytest.approx(1.0)


def test_heatmap_stats_reports_border_attention() -> None:
    mask = np.zeros((10, 10), dtype=bool)
    mask[4:6, 4:6] = True
    heatmap = np.zeros((10, 10))
    heatmap[0, :] = 1.0
    stats = heatmap_stats(heatmap, mask)
    assert stats["border_mass_fraction"] == pytest.approx(1.0)
    assert stats["stroke_mass_fraction"] == pytest.approx(0.0)


def test_heatmap_stats_blank_inputs_are_nan_not_errors() -> None:
    stats = heatmap_stats(np.zeros((10, 10)), np.zeros((10, 10), dtype=bool))
    assert np.isnan(stats["mass_to_area_ratio"])


def test_deletion_is_faithful_when_heatmap_marks_the_region_that_matters() -> None:
    image = torch.zeros(3, 224, 224)
    heatmap = np.zeros((224, 224))
    heatmap[:112, :112] = 1.0
    scores = deletion_scores(TopLeftBrightnessModel(), image, heatmap, predicted_positive=True)
    assert scores["deletion_curve"][0] > 0.99
    assert scores["deletion_curve"][-1] < 0.01
    assert scores["deletion_auc"] < scores["random_deletion_auc"]
    assert scores["faithful"] is True


def test_deletion_is_unfaithful_when_heatmap_marks_an_irrelevant_region() -> None:
    image = torch.zeros(3, 224, 224)
    heatmap = np.zeros((224, 224))
    heatmap[112:, 112:] = 1.0
    scores = deletion_scores(TopLeftBrightnessModel(), image, heatmap, predicted_positive=True)
    assert scores["deletion_auc"] > scores["random_deletion_auc"]
    assert scores["faithful"] is False


def test_deletion_scores_are_deterministic_for_a_seed() -> None:
    image = torch.zeros(3, 224, 224)
    heatmap = np.random.default_rng(0).random((224, 224))
    first = deletion_scores(TopLeftBrightnessModel(), image, heatmap, True, seed=3)
    second = deletion_scores(TopLeftBrightnessModel(), image, heatmap, True, seed=3)
    assert first == second


def test_overlay_image_is_uint8_rgb_at_input_size() -> None:
    overlay = overlay_image(torch.rand(3, 64, 64), np.random.default_rng(0).random((64, 64)))
    assert overlay.shape == (64, 64, 3)
    assert overlay.dtype == np.uint8


def test_select_examples_splits_correct_and_incorrect_and_caps_each() -> None:
    labels = np.array([1, 1, 0, 0, 1, 0])
    probabilities = np.array([0.9, 0.2, 0.1, 0.8, 0.95, 0.05])
    selected = select_examples(labels, probabilities, threshold=0.5, per_group=2)
    assert selected["correct"] == [0, 2]
    assert selected["incorrect"] == [1, 3]


def make_summary(ratio: float, border: float, faithful: bool) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "group": "correct",
                "mass_to_area_ratio": ratio,
                "border_mass_fraction": border,
                "faithful": faithful,
                "deletion_curve": [1.0, 0.5],
                "random_deletion_curve": [1.0, 0.9],
            }
        ]
    )


def test_render_finding_warns_when_model_prefers_blank_paper() -> None:
    assert "blank paper more than to the stroke" in render_finding(make_summary(0.5, 0.1, True))


def test_render_finding_reports_stroke_attention() -> None:
    assert "attends to stroke pixels" in render_finding(make_summary(2.0, 0.1, True))


def test_render_finding_flags_border_attention() -> None:
    assert "Border attention is high" in render_finding(make_summary(2.0, 0.9, True))


def test_render_finding_handles_no_images() -> None:
    assert "No images were explained" in render_finding(pd.DataFrame())


def test_plot_deletion_curves_has_guided_and_random_trace_per_image() -> None:
    summary = make_summary(1.0, 0.1, True)
    figure = plot_deletion_curves(summary.to_dict("records"))
    assert len(figure.data) == 2


def test_write_explain_outputs_writes_files_without_curve_columns(tmp_path) -> None:
    written = write_explain_outputs(make_summary(1.0, 0.1, True), tmp_path)
    assert set(written) == {"summary_csv", "finding", "deletion_html"}
    csv_columns = pd.read_csv(written["summary_csv"]).columns
    assert "deletion_curve" not in csv_columns
    assert "mass_to_area_ratio" in csv_columns


def test_render_finding_labels_development_runs_as_meaningless() -> None:
    finding = render_finding(make_summary(2.0, 0.1, True), development=True)
    assert "Development run" in finding
    assert "say nothing about what a trained model attends to" in finding


def test_render_finding_omits_the_development_note_for_real_runs() -> None:
    assert "Development run" not in render_finding(make_summary(2.0, 0.1, True))


def test_write_explain_outputs_passes_the_development_flag_through(tmp_path) -> None:
    written = write_explain_outputs(make_summary(2.0, 0.1, True), tmp_path, development=True)
    assert "Development run" in written["finding"].read_text(encoding="utf-8")
