"""Ablation studies and the learning curve, reusing
training/train.py's train_one_backbone for every actual training run so
this file only varies the experimental condition, not the training loop.

Five experiments, per CLAUDE.md Phase 4:
    1. pretrained vs scratch
    2. augmentation vs none
    3. spiral only vs wave only vs fused (probability averaging and a
       logistic-regression stacker), reporting honestly whether fusion
       actually helps
    4. full fine-tune vs frozen backbone vs DINOv2 linear probe
    5. learning curve at 25%, 50% and 100% of the training data

Usage:
    python -m training.ablations --config configs/train_resnet50.yaml --fast
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LogisticRegression

from training.metrics import compute_metrics, select_threshold
from training.train import (
    load_config,
    sample_fast_subset,
    set_all_seeds,
    train_one_backbone,
)


def run_pretrained_vs_scratch(
    manifest: pd.DataFrame, config: dict[str, Any], seed: int, fast: bool
) -> list[dict[str, Any]]:
    backbone = config["backbone"]
    max_epochs = 1 if fast else config["max_epochs"]
    results = []
    for pretrained in (True, False):
        for fold in sorted(manifest["fold"].unique()):
            train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
            test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue
            result = train_one_backbone(
                backbone, train_df, test_df, config, seed, fold, max_epochs, pretrained
            )
            result["ablation"] = "pretrained_vs_scratch"
            result["condition"] = "pretrained" if pretrained else "scratch"
            results.append(result)
    return results


def run_augmentation_ablation(
    manifest: pd.DataFrame, config: dict[str, Any], seed: int, fast: bool
) -> list[dict[str, Any]]:
    backbone = config["backbone"]
    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)
    results = []
    for augment in (True, False):
        for fold in sorted(manifest["fold"].unique()):
            train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
            test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue
            result = train_one_backbone(
                backbone,
                train_df,
                test_df,
                config,
                seed,
                fold,
                max_epochs,
                pretrained,
                augment=augment,
            )
            result["ablation"] = "augmentation"
            result["condition"] = "augmented" if augment else "no_augmentation"
            results.append(result)
    return results


def run_finetune_mode_ablation(
    manifest: pd.DataFrame, config: dict[str, Any], seed: int, fast: bool
) -> list[dict[str, Any]]:
    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)
    conditions: list[tuple[str, str, int | None, str]] = [
        ("full_finetune", config["backbone"], config["freeze_epochs"], "full_finetune"),
        ("frozen_backbone", config["backbone"], max_epochs + 1, "full_finetune"),
        ("dinov2_linear_probe", "dinov2_small", 0, "linear_probe"),
    ]
    results = []
    for condition_name, backbone, freeze_epochs, dinov2_mode in conditions:
        for fold in sorted(manifest["fold"].unique()):
            train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
            test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue
            result = train_one_backbone(
                backbone,  # type: ignore[arg-type]
                train_df,
                test_df,
                config,
                seed,
                fold,
                max_epochs,
                pretrained,
                freeze_epochs=freeze_epochs,
                dinov2_mode=dinov2_mode,  # type: ignore[arg-type]
            )
            result["ablation"] = "finetune_mode"
            result["condition"] = condition_name
            results.append(result)
    return results


def run_kind_fusion_ablation(
    manifest: pd.DataFrame, config: dict[str, Any], seed: int, fast: bool
) -> list[dict[str, Any]]:
    """Train separate spiral-only and wave-only models on the same folds,
    then fuse their held-out predictions for subjects who have both: by
    simple probability averaging, and by a logistic-regression stacker
    fit on the validation predictions (never on test). Reports fused
    AUC alongside each single-kind AUC so whether fusion actually helps
    is visible either way, not assumed."""
    backbone = config["backbone"]
    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)
    results = []
    fusion_rows = []

    for fold in sorted(manifest["fold"].unique()):
        per_kind: dict[str, dict[str, Any]] = {}
        for kind in ("spiral", "wave"):
            kind_manifest = manifest[manifest["kind"] == kind]
            train_df = kind_manifest[kind_manifest["fold"] != fold].reset_index(drop=True)
            test_df = kind_manifest[kind_manifest["fold"] == fold].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue
            result = train_one_backbone(
                backbone, train_df, test_df, config, seed, fold, max_epochs, pretrained
            )
            result["ablation"] = "kind_fusion"
            result["condition"] = kind
            results.append(result)
            per_kind[kind] = result

        if "spiral" not in per_kind or "wave" not in per_kind:
            continue
        fusion_rows.append(fuse_kind_predictions(per_kind["spiral"], per_kind["wave"], fold, seed))

    results.extend(fusion_rows)
    return results


def fuse_kind_predictions(
    spiral_result: dict[str, Any], wave_result: dict[str, Any], fold: int, seed: int
) -> dict[str, Any]:
    val_merged = pd.merge(
        pd.DataFrame(
            {
                "subject_id": spiral_result["val_subject_id"],
                "label": spiral_result["val_true"],
                "spiral_prob": spiral_result["val_prob"],
            }
        ),
        pd.DataFrame(
            {"subject_id": wave_result["val_subject_id"], "wave_prob": wave_result["val_prob"]}
        ),
        on="subject_id",
    )
    test_merged = pd.merge(
        pd.DataFrame(
            {
                "subject_id": spiral_result["test_subject_id"],
                "label": spiral_result["y_true"],
                "spiral_prob": spiral_result["y_prob"],
            }
        ),
        pd.DataFrame(
            {"subject_id": wave_result["test_subject_id"], "wave_prob": wave_result["y_prob"]}
        ),
        on="subject_id",
    )

    average_test_prob = test_merged[["spiral_prob", "wave_prob"]].mean(axis=1).to_numpy()
    average_val_prob = val_merged[["spiral_prob", "wave_prob"]].mean(axis=1).to_numpy()
    average_threshold = select_threshold(val_merged["label"].to_numpy(), average_val_prob)
    average_metrics = compute_metrics(
        test_merged["label"].to_numpy(), average_test_prob, average_threshold
    )

    stacker_metrics = average_metrics
    if val_merged["label"].nunique() == 2 and len(val_merged) >= 4:
        stacker = LogisticRegression()
        stacker.fit(val_merged[["spiral_prob", "wave_prob"]].to_numpy(), val_merged["label"])
        stacked_val_prob = stacker.predict_proba(val_merged[["spiral_prob", "wave_prob"]])[:, 1]
        stacked_test_prob = stacker.predict_proba(test_merged[["spiral_prob", "wave_prob"]])[:, 1]
        stacked_threshold = select_threshold(val_merged["label"].to_numpy(), stacked_val_prob)
        stacker_metrics = compute_metrics(
            test_merged["label"].to_numpy(), stacked_test_prob, stacked_threshold
        )

    return {
        "ablation": "kind_fusion",
        "condition": "fused",
        "model": "fused",
        "fold": fold,
        "seed": seed,
        "n_matched_test_subjects": len(test_merged),
        "spiral_only_auc": spiral_result["roc_auc"],
        "wave_only_auc": wave_result["roc_auc"],
        "average_fusion_auc": average_metrics.roc_auc,
        "stacked_fusion_auc": stacker_metrics.roc_auc,
        "fusion_helps_vs_best_single": max(
            average_metrics.roc_auc, stacker_metrics.roc_auc
        )
        > max(spiral_result["roc_auc"], wave_result["roc_auc"]),
        **average_metrics.as_dict(),
        "y_true": test_merged["label"].tolist(),
        "y_prob": average_test_prob.tolist(),
    }


def run_learning_curve(
    manifest: pd.DataFrame, config: dict[str, Any], seed: int, fast: bool
) -> list[dict[str, Any]]:
    backbone = config["backbone"]
    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)
    fractions = (0.25, 0.5, 1.0)
    results = []
    for fraction in fractions:
        for fold in sorted(manifest["fold"].unique()):
            train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
            test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)
            if train_df.empty or test_df.empty:
                continue
            if fraction < 1.0:
                train_df = train_df.groupby("label", group_keys=False).apply(
                    lambda group, frac=fraction: group.sample(
                        max(1, int(len(group) * frac)), random_state=seed
                    )
                )
                train_df = train_df.reset_index(drop=True)
            result = train_one_backbone(
                backbone, train_df, test_df, config, seed, fold, max_epochs, pretrained
            )
            result["ablation"] = "learning_curve"
            result["condition"] = f"{int(fraction * 100)}pct"
            result["train_fraction"] = fraction
            result["n_train_images"] = len(train_df)
            results.append(result)
    return results


def fused_condition_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        r for r in results if r.get("ablation") == "kind_fusion" and r.get("condition") == "fused"
    ]


def summarize_by_condition(results: list[dict[str, Any]], ablation: str) -> pd.DataFrame:
    df = pd.DataFrame([r for r in results if r["ablation"] == ablation])
    if df.empty:
        return df
    grouped = df.groupby("condition")["roc_auc"].agg(["mean", "std", "count"])
    return grouped.reset_index().rename(columns={"mean": "roc_auc_mean", "std": "roc_auc_std"})


def plot_ablation_bars(summary: pd.DataFrame, title: str) -> go.Figure:
    figure = go.Figure()
    figure.add_trace(
        go.Bar(
            x=summary["condition"],
            y=summary["roc_auc_mean"],
            error_y={"type": "data", "array": summary["roc_auc_std"].fillna(0)},
        )
    )
    figure.update_layout(title=title, yaxis_title="ROC-AUC (mean +/- std across folds)")
    return figure


def run_all_ablations(config: dict[str, Any], seed: int, fast: bool) -> list[dict[str, Any]]:
    set_all_seeds(seed)
    manifest = pd.read_csv(config["folds_path"])
    if fast:
        manifest = sample_fast_subset(manifest, config.get("fast_sample_per_class", 6), seed)
        from training.data import assign_folds

        manifest = assign_folds(manifest, n_folds=2, seed=seed)

    results: list[dict[str, Any]] = []
    results += run_pretrained_vs_scratch(manifest, config, seed, fast)
    results += run_augmentation_ablation(manifest, config, seed, fast)
    results += run_finetune_mode_ablation(manifest, config, seed, fast)
    results += run_kind_fusion_ablation(manifest, config, seed, fast)
    results += run_learning_curve(manifest, config, seed, fast)
    return results


def write_ablation_report(results: list[dict[str, Any]], reports_dir: Path) -> dict[str, Path]:
    reports_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    results_path = reports_dir / "ablation_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    written["results"] = results_path

    for ablation in ("pretrained_vs_scratch", "augmentation", "finetune_mode", "learning_curve"):
        summary = summarize_by_condition(results, ablation)
        if summary.empty:
            continue
        csv_path = reports_dir / f"ablation_{ablation}.csv"
        summary.to_csv(csv_path, index=False)
        written[f"{ablation}_csv"] = csv_path

        figure = plot_ablation_bars(summary, f"Ablation: {ablation}")
        html_path = reports_dir / f"ablation_{ablation}.html"
        figure.write_html(html_path, include_plotlyjs="cdn")
        written[f"{ablation}_html"] = html_path

    fusion_rows = fused_condition_rows(results)
    if fusion_rows:
        fusion_df = pd.DataFrame(fusion_rows)[
            [
                "fold",
                "spiral_only_auc",
                "wave_only_auc",
                "average_fusion_auc",
                "stacked_fusion_auc",
                "fusion_helps_vs_best_single",
                "n_matched_test_subjects",
            ]
        ]
        fusion_csv = reports_dir / "ablation_kind_fusion.csv"
        fusion_df.to_csv(fusion_csv, index=False)
        written["kind_fusion_csv"] = fusion_csv

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

    results = run_all_ablations(config, seed, args.fast)
    written = write_ablation_report(results, Path(args.reports_dir))

    for ablation in ("pretrained_vs_scratch", "augmentation", "finetune_mode", "learning_curve"):
        summary = summarize_by_condition(results, ablation)
        if not summary.empty:
            print(f"\n{ablation}:")
            print(summary.to_string(index=False))

    fusion_rows = fused_condition_rows(results)
    if fusion_rows:
        print("\nkind_fusion:")
        for row in fusion_rows:
            print(
                f"  fold={row['fold']}: spiral={row['spiral_only_auc']:.3f} "
                f"wave={row['wave_only_auc']:.3f} avg_fusion={row['average_fusion_auc']:.3f} "
                f"stacked_fusion={row['stacked_fusion_auc']:.3f} "
                f"fusion_helps={row['fusion_helps_vs_best_single']}"
            )

    for key, path in written.items():
        print(f"Wrote {key} to {path}")


if __name__ == "__main__":
    main()
