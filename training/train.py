"""Training orchestration: for each backbone (and the HOG+SVM baseline)
and each fold, hold out part of the training fold as validation (grouped
the same way as the outer folds), pick the epoch and the decision
threshold on validation only, then evaluate once on the test fold. Never
tunes anything on the test fold, per CLAUDE.md Phase 3 rule 4.

--fast trains every backbone plus the baseline for one epoch on one fold,
sampled down to a small subset: a CPU smoke test that the whole pipeline
runs end to end, not a real training run. It ignores the config's single
`backbone` field and loops over every backbone, since its job is to prove
every model's plumbing works, not to fully train the one model named in
the config; see docs/ASSUMPTIONS.md.

Usage:
    python -m training.train --config configs/train_resnet50.yaml --fast
    python -m training.train --config configs/train_resnet50.yaml --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import tempfile
import time
from pathlib import Path
from typing import Any

import mlflow
import numpy as np
import pandas as pd
import torch
import yaml
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import StratifiedGroupKFold
from torch.utils.data import DataLoader

from common.preprocess import PreprocessConfig, PreprocessMode
from training.backbones import BackboneName, Dinov2Mode, count_parameters
from training.baselines import HogSvmConfig, predict_proba, train_hog_svm
from training.data import SpiralWaveDataset, assign_folds, has_usable_subject_ids
from training.lit_model import SpiralWaveClassifier
from training.metrics import bootstrap_auc_ci, compute_metrics, select_threshold

BACKBONE_NAMES: list[BackboneName] = [
    "resnet50",
    "efficientnet_b0",
    "convnext_tiny",
    "vit_small",
    "dinov2_small",
]


def set_all_seeds(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_config(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as handle:
        return dict(yaml.safe_load(handle))


def split_train_val(
    manifest: pd.DataFrame, val_fraction: float, seed: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a training fold into train and validation, grouped the same
    way the outer folds are (by subject when available, else image hash),
    so validation never leaks the same subject's other images. n_splits is
    capped at the smallest class's count and at the total sample count:
    StratifiedGroupKFold requires both, and --fast mode's per-kind,
    per-fold subsets (training/ablations.py's kind_fusion ablation) can
    be as small as a single row, which no 2-way split can serve. When a
    real split is infeasible, everything goes to train and validation is
    left empty; fit_backbone falls back to a default decision threshold
    in that case rather than crashing on an empty roc_curve input."""
    smallest_class_count = int(manifest["label"].value_counts().min())
    max_feasible_splits = min(len(manifest), smallest_class_count)
    if max_feasible_splits < 2:
        empty = manifest.iloc[0:0].reset_index(drop=True)
        return manifest.reset_index(drop=True), empty
    n_splits = max(2, min(round(1 / val_fraction), max_feasible_splits))
    groups = manifest["subject_id"] if has_usable_subject_ids(manifest) else manifest["image_hash"]
    splitter = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    train_index, val_index = next(splitter.split(manifest, manifest["label"], groups))
    train_df = manifest.iloc[train_index].reset_index(drop=True)
    val_df = manifest.iloc[val_index].reset_index(drop=True)
    return train_df, val_df


def sample_fast_subset(manifest: pd.DataFrame, per_class: int, seed: int) -> pd.DataFrame:
    return (
        manifest.groupby("label", group_keys=False)
        .apply(lambda group: group.sample(min(len(group), per_class), random_state=seed))
        .reset_index(drop=True)
    )


def as_single_fold_dataset(
    df: pd.DataFrame, preprocess_config: PreprocessConfig, train: bool, augment: bool | None = None
) -> SpiralWaveDataset:
    tagged = df.copy()
    tagged["fold"] = 0
    return SpiralWaveDataset(
        tagged, fold_ids=[0], preprocess_config=preprocess_config, train=train, augment=augment
    )


def predict_probabilities(
    trainer: Trainer, model: SpiralWaveClassifier, loader: DataLoader
) -> np.ndarray:
    batches = trainer.predict(model, dataloaders=loader)
    return torch.cat(batches).numpy()  # type: ignore[arg-type]


def measure_cpu_latency_ms(model: torch.nn.Module, image_size: int, n_runs: int = 10) -> float:
    model.eval()
    dummy = torch.zeros(1, 3, image_size, image_size)
    with torch.no_grad():
        model(dummy)  # warm up
        start = time.perf_counter()
        for _ in range(n_runs):
            model(dummy)
        elapsed = time.perf_counter() - start
    return (elapsed / n_runs) * 1000.0


def fit_backbone(
    backbone_name: BackboneName,
    train_df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    max_epochs: int,
    pretrained: bool,
    augment: bool | None = None,
    freeze_epochs: int | None = None,
    dinov2_mode: Dinov2Mode = "full_finetune",
    preprocess_mode: PreprocessMode = "standard",
) -> tuple[Trainer, SpiralWaveClassifier, float, pd.DataFrame]:
    """Train one backbone and pick its decision threshold on a held-out
    validation split, without touching any test set. Returns the trainer,
    the best-epoch model, the selected threshold, and the validation
    dataframe with its true/predicted-probability columns attached (used
    by training/ablations.py's kind-fusion stacker). Split out of
    train_one_backbone so training/robustness.py can fit a model once and
    evaluate it against many corrupted test sets without retraining."""
    set_all_seeds(seed)
    preprocess_config = PreprocessConfig(size=config["image_size"], mode=preprocess_mode)
    fit_df, val_df = split_train_val(train_df, config["val_fraction"], seed)

    train_dataset = as_single_fold_dataset(fit_df, preprocess_config, train=True, augment=augment)
    val_dataset = as_single_fold_dataset(val_df, preprocess_config, train=False)

    batch_size = config["batch_size"]
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    positive_fraction = (fit_df["label"] == "parkinson").mean()
    pos_weight = (1 - positive_fraction) / positive_fraction if 0 < positive_fraction < 1 else None

    model = SpiralWaveClassifier(
        backbone_name=backbone_name,
        pretrained=pretrained,
        dinov2_mode=dinov2_mode,
        head_lr=config["head_lr"],
        backbone_lr=config["backbone_lr"],
        freeze_epochs=freeze_epochs if freeze_epochs is not None else config["freeze_epochs"],
        label_smoothing=config["label_smoothing"],
        pos_weight=pos_weight,
        max_epochs=max_epochs,
    )

    # A --fast per-kind, per-fold subset (training/ablations.py's
    # kind_fusion ablation) can be too small for split_train_val to carve
    # out any validation rows at all. With no validation data there is
    # nothing to monitor a checkpoint or pick a threshold from, so this
    # falls back to the final-epoch model and a neutral 0.5 threshold
    # instead of crashing; real training always has a real validation
    # split, since the full dataset is never this small.
    has_validation = len(val_df) > 0

    # Checkpoints are working state for picking the best epoch, not a
    # deliverable of a --fast smoke run or even of one fold/seed of a real
    # run (Phase 5's export_onnx.py saves the actual champion model), so
    # they live in a scratch directory that is deleted right after use
    # rather than accumulating under the repo.
    with tempfile.TemporaryDirectory(prefix="lightning_ckpt_") as checkpoint_dir:
        callbacks: list[Any] = []
        checkpoint_callback: ModelCheckpoint | None = None
        if has_validation:
            checkpoint_callback = ModelCheckpoint(
                dirpath=checkpoint_dir,
                monitor="val_loss",
                mode="min",
                save_top_k=1,
                save_weights_only=True,
            )
            callbacks.append(checkpoint_callback)
            if max_epochs > 1:
                patience = config.get("patience", 3)
                callbacks.append(EarlyStopping(monitor="val_loss", mode="min", patience=patience))

        trainer = Trainer(
            max_epochs=max_epochs,
            accelerator="cpu",
            enable_progress_bar=False,
            enable_checkpointing=has_validation,
            logger=False,
            callbacks=callbacks,
            num_sanity_val_steps=0,
        )
        if has_validation:
            trainer.fit(model, train_loader, val_loader)
        else:
            trainer.fit(model, train_loader)

        best_model = model
        if checkpoint_callback is not None and checkpoint_callback.best_model_path:
            best_model = SpiralWaveClassifier.load_from_checkpoint(
                checkpoint_callback.best_model_path
            )

        if has_validation:
            val_probabilities = predict_probabilities(trainer, best_model, val_loader)
            val_labels = np.array([1 if label == "parkinson" else 0 for label in val_df["label"]])
            threshold = select_threshold(
                val_labels, val_probabilities, config["target_sensitivity"]
            )
        else:
            val_probabilities = np.array([])
            val_labels = np.array([])
            threshold = 0.5

    val_result = val_df.copy()
    val_result["y_true"] = val_labels
    val_result["y_prob"] = val_probabilities
    return trainer, best_model, threshold, val_result


def evaluate_fitted_backbone(
    trainer: Trainer,
    model: SpiralWaveClassifier,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    threshold: float,
    preprocess_mode: PreprocessMode = "standard",
) -> dict[str, Any]:
    """Evaluate an already-fitted model (from fit_backbone) on a test set,
    at a threshold already chosen on validation. Never re-picks the
    threshold here: that would tune on the test set."""
    preprocess_config = PreprocessConfig(size=config["image_size"], mode=preprocess_mode)
    test_dataset = as_single_fold_dataset(test_df, preprocess_config, train=False)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=False)

    test_probabilities = predict_probabilities(trainer, model, test_loader)
    test_labels = np.array([1 if label == "parkinson" else 0 for label in test_df["label"]])
    metrics = compute_metrics(test_labels, test_probabilities, threshold)
    _, auc_lower, auc_upper = bootstrap_auc_ci(test_labels, test_probabilities)

    return {
        **metrics.as_dict(),
        "auc_ci_lower": auc_lower,
        "auc_ci_upper": auc_upper,
        "y_true": test_labels.tolist(),
        "y_prob": test_probabilities.tolist(),
        "test_subject_id": test_df["subject_id"].tolist(),
    }


def train_one_backbone(
    backbone_name: BackboneName,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    fold: int,
    max_epochs: int,
    pretrained: bool,
    augment: bool | None = None,
    freeze_epochs: int | None = None,
    dinov2_mode: Dinov2Mode = "full_finetune",
    preprocess_mode: PreprocessMode = "standard",
) -> dict[str, Any]:
    """Fit one backbone and evaluate it once on the test fold: the common
    case, used by run_training and every ablation that does not need to
    reuse a fitted model across multiple test sets (see fit_backbone and
    evaluate_fitted_backbone for that case, used by
    training/robustness.py)."""
    trainer, best_model, threshold, val_result = fit_backbone(
        backbone_name,
        train_df,
        config,
        seed,
        max_epochs,
        pretrained,
        augment=augment,
        freeze_epochs=freeze_epochs,
        dinov2_mode=dinov2_mode,
        preprocess_mode=preprocess_mode,
    )
    evaluation = evaluate_fitted_backbone(
        trainer, best_model, test_df, config, threshold, preprocess_mode=preprocess_mode
    )
    return {
        "model": backbone_name,
        "fold": fold,
        "seed": seed,
        **evaluation,
        "n_parameters": count_parameters(best_model.model),
        "cpu_latency_ms": measure_cpu_latency_ms(best_model.model, config["image_size"]),
        "val_true": val_result["y_true"].tolist(),
        "val_prob": val_result["y_prob"].tolist(),
        "val_subject_id": val_result["subject_id"].tolist(),
    }


def train_baseline_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    config: dict[str, Any],
    seed: int,
    fold: int,
) -> dict[str, Any]:
    preprocess_config = PreprocessConfig(size=config["image_size"])
    fit_df, val_df = split_train_val(train_df, config["val_fraction"], seed)

    baseline_config = HogSvmConfig(preprocess=preprocess_config, seed=seed)
    train_labels = [1 if label == "parkinson" else 0 for label in fit_df["label"]]
    classifier, _ = train_hog_svm(list(fit_df["path"]), train_labels, baseline_config)

    val_labels = np.array([1 if label == "parkinson" else 0 for label in val_df["label"]])
    val_probabilities = predict_proba(classifier, list(val_df["path"]), preprocess_config)
    threshold = select_threshold(val_labels, val_probabilities, config["target_sensitivity"])

    test_labels = np.array([1 if label == "parkinson" else 0 for label in test_df["label"]])
    start = time.perf_counter()
    test_probabilities = predict_proba(classifier, list(test_df["path"]), preprocess_config)
    latency_ms = (time.perf_counter() - start) / max(len(test_df), 1) * 1000.0
    metrics = compute_metrics(test_labels, test_probabilities, threshold)
    _, auc_lower, auc_upper = bootstrap_auc_ci(test_labels, test_probabilities)

    return {
        "model": "hog_svm_baseline",
        "fold": fold,
        "seed": seed,
        **metrics.as_dict(),
        "auc_ci_lower": auc_lower,
        "auc_ci_upper": auc_upper,
        "n_parameters": -1,
        "cpu_latency_ms": latency_ms,
        "y_true": test_labels.tolist(),
        "y_prob": test_probabilities.tolist(),
    }


def run_training(config: dict[str, Any], seed: int, fast: bool) -> list[dict[str, Any]]:
    set_all_seeds(seed)
    manifest = pd.read_csv(config["folds_path"])

    if fast:
        backbone_names = BACKBONE_NAMES
        max_epochs = 1
        pretrained = False
        # Sample first, then assign fresh folds on the sampled subset
        # itself (not reused from the full manifest's fold column): a
        # global per-class sample scattered across the original folds can
        # easily leave a single original fold's test set with only one
        # class present, which makes roc_auc undefined. Re-splitting
        # after sampling keeps both classes in the fast-mode test fold.
        manifest = sample_fast_subset(manifest, config.get("fast_sample_per_class", 6), seed)
        manifest = assign_folds(manifest, n_folds=2, seed=seed)
        folds_to_run = [0]
    else:
        backbone_names = [config["backbone"]]
        folds_to_run = sorted(int(f) for f in manifest["fold"].unique())
        max_epochs = config["max_epochs"]
        pretrained = config.get("pretrained", True)

    results: list[dict[str, Any]] = []
    mlflow.set_tracking_uri(f"file:{config.get('mlruns_dir', 'mlruns')}")
    mlflow.set_experiment("parkinsons-screening")

    for fold in folds_to_run:
        train_df = manifest[manifest["fold"] != fold].reset_index(drop=True)
        test_df = manifest[manifest["fold"] == fold].reset_index(drop=True)
        if train_df.empty or test_df.empty:
            continue

        for backbone_name in backbone_names:
            with mlflow.start_run(run_name=f"{backbone_name}_fold{fold}_seed{seed}"):
                mlflow.log_params(
                    {"backbone": backbone_name, "fold": fold, "seed": seed, "fast": fast}
                )
                result = train_one_backbone(
                    backbone_name, train_df, test_df, config, seed, fold, max_epochs, pretrained
                )
                mlflow.log_metrics(
                    {k: v for k, v in result.items() if isinstance(v, int | float)}
                )
                results.append(result)

        with mlflow.start_run(run_name=f"hog_svm_baseline_fold{fold}_seed{seed}"):
            mlflow.log_params(
                {"backbone": "hog_svm_baseline", "fold": fold, "seed": seed, "fast": fast}
            )
            baseline_result = train_baseline_fold(train_df, test_df, config, seed, fold)
            mlflow.log_metrics(
                {k: v for k, v in baseline_result.items() if isinstance(v, int | float)}
            )
            results.append(baseline_result)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/train_resnet50.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    parser.add_argument("--out", default=None, help="override results_path from the config")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]

    results = run_training(config, seed, args.fast)

    out_path = Path(args.out or config.get("results_path", "reports/training_results.json"))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Ran {len(results)} model/fold/seed combinations. Wrote results to {out_path}")
    for result in results:
        print(
            f"  {result['model']} fold={result['fold']}: "
            f"roc_auc={result['roc_auc']:.3f} sensitivity={result['sensitivity']:.3f} "
            f"specificity={result['specificity']:.3f}"
        )


if __name__ == "__main__":
    main()
