"""Refit the champion backbone on all data and register it in MLflow with
the alias `champion`, with its decision threshold stored as a version tag
so training/export_onnx.py can ship the model and its threshold together.

Cross-validation (training/train.py) answers "how well does this recipe
generalize"; it deliberately keeps no model, since every fold's model saw
only part of the data and its checkpoint is scratch. The model that
actually ships is this refit: the chosen backbone trained on every
available image, with a grouped internal validation split used only to
pick the epoch and the sensitivity-favoring threshold, never a test fold.

Usage:
    python -m training.register --config configs/export.yaml --fast
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import pandas as pd
from mlflow import MlflowClient
from torch import nn

from training.data import assign_folds
from training.train import fit_backbone, load_config, sample_fast_subset, set_all_seeds

CHAMPION_ALIAS = "champion"
DEV_MODEL_WARNING = (
    "Development model: trained for one epoch on a tiny sample with random "
    "initialization to prove the pipeline runs. Not a real screening model."
)


@dataclass(frozen=True)
class Champion:
    model: nn.Module
    threshold: float
    version: str
    tags: dict[str, str]


def tracking_uri_for(mlruns_dir: str | Path) -> str:
    return Path(mlruns_dir).resolve().as_uri()


def load_merged_config(path: str | Path) -> dict[str, Any]:
    """An export config names a training config and overrides parts of
    it, so the architecture and hyperparameters have one source of truth."""
    config = load_config(path)
    train_config_path = config.get("train_config")
    if train_config_path:
        merged = load_config(train_config_path)
        merged.update({k: v for k, v in config.items() if k != "train_config"})
        return merged
    return config


def register_champion(
    model: nn.Module,
    name: str,
    threshold: float,
    tracking_uri: str,
    tags: dict[str, str],
) -> dict[str, str]:
    """Log the model, register it, tag the version with its threshold and
    provenance, and point the `champion` alias at it."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)

    with mlflow.start_run(run_name="champion_refit") as run:
        mlflow.log_params({k: v for k, v in tags.items() if k != "warning"})
        mlflow.log_metric("threshold", threshold)
        mlflow.pytorch.log_model(
            model,
            artifact_path="model",
            registered_model_name=name,
            pip_requirements=["torch", "timm", "transformers"],
        )
        run_id = run.info.run_id

    versions = client.search_model_versions(f"name='{name}'")
    version = str(max(int(v.version) for v in versions))
    client.set_model_version_tag(name, version, "threshold", repr(float(threshold)))
    for key, value in tags.items():
        client.set_model_version_tag(name, version, key, value)
    client.set_registered_model_alias(name, CHAMPION_ALIAS, version)
    return {"name": name, "version": version, "run_id": run_id}


def load_champion(name: str, tracking_uri: str) -> Champion:
    """Load the model the `champion` alias points at, with its threshold."""
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_registry_uri(tracking_uri)
    client = MlflowClient(tracking_uri=tracking_uri, registry_uri=tracking_uri)
    try:
        version = client.get_model_version_by_alias(name, CHAMPION_ALIAS)
    except Exception as error:
        raise LookupError(
            f"No registered model {name!r} with alias {CHAMPION_ALIAS!r} in {tracking_uri}. "
            "Run `python -m training.register --config configs/export.yaml` first "
            "(add --fast for a development model)."
        ) from error
    model = mlflow.pytorch.load_model(f"models:/{name}@{CHAMPION_ALIAS}")
    tags = dict(version.tags)
    return Champion(
        model=model, threshold=float(tags["threshold"]), version=str(version.version), tags=tags
    )


def fit_and_register(config: dict[str, Any], seed: int, fast: bool) -> dict[str, str]:
    set_all_seeds(seed)
    manifest = pd.read_csv(config["folds_path"])
    if fast:
        manifest = sample_fast_subset(manifest, config.get("fast_sample_per_class", 6), seed)
        manifest = assign_folds(manifest, n_folds=2, seed=seed)

    max_epochs = 1 if fast else config["max_epochs"]
    pretrained = False if fast else config.get("pretrained", True)
    _, fitted, threshold, _ = fit_backbone(
        config["backbone"], manifest.reset_index(drop=True), config, seed, max_epochs, pretrained
    )

    tags = {
        "backbone": str(config["backbone"]),
        "pretrained": str(pretrained),
        "fast_mode": str(fast),
        "n_train_images": str(len(manifest)),
        "datasets": ",".join(sorted(manifest["dataset"].unique())),
        "seed": str(seed),
    }
    if fast:
        tags["warning"] = DEV_MODEL_WARNING
    return register_champion(
        fitted.model.cpu(),
        config["model_name"],
        threshold,
        tracking_uri_for(config["mlruns_dir"]),
        tags,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/export.yaml")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--fast", action="store_true")
    args = parser.parse_args()

    config = load_merged_config(args.config)
    seed = args.seed if args.seed is not None else config["seed"]
    registered = fit_and_register(config, seed, args.fast)
    print(
        f"Registered {registered['name']} version {registered['version']} "
        f"(alias {CHAMPION_ALIAS}) from run {registered['run_id']}"
    )


if __name__ == "__main__":
    main()
