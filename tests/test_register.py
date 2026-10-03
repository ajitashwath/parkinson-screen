from __future__ import annotations

from pathlib import Path

import pytest
import torch
import yaml
from torch import nn

from training.register import (
    CHAMPION_ALIAS,
    load_champion,
    load_merged_config,
    register_champion,
    tracking_uri_for,
)

NAME = "test-champion"


def tiny_model(seed: int = 0) -> nn.Module:
    torch.manual_seed(seed)
    return nn.Sequential(nn.Flatten(), nn.Linear(12, 1))


@pytest.fixture
def uri(tmp_path: Path) -> str:
    return tracking_uri_for(tmp_path / "mlruns")


def test_tracking_uri_is_an_absolute_file_uri(tmp_path: Path) -> None:
    result = tracking_uri_for(tmp_path / "mlruns")
    assert result.startswith("file:///")


def test_register_then_load_round_trips_model_and_threshold(uri: str) -> None:
    model = tiny_model()
    registered = register_champion(model, NAME, 0.37, uri, {"backbone": "tiny"})
    assert registered["version"] == "1"

    champion = load_champion(NAME, uri)
    assert champion.threshold == pytest.approx(0.37)
    assert champion.version == "1"
    assert champion.tags["backbone"] == "tiny"
    sample = torch.rand(2, 3, 2, 2)
    assert torch.allclose(champion.model(sample), model(sample))


def test_registering_again_moves_the_champion_alias_to_the_new_version(uri: str) -> None:
    register_champion(tiny_model(0), NAME, 0.3, uri, {})
    register_champion(tiny_model(1), NAME, 0.6, uri, {})
    champion = load_champion(NAME, uri)
    assert champion.version == "2"
    assert champion.threshold == pytest.approx(0.6)


def test_loading_a_missing_champion_explains_how_to_create_one(uri: str) -> None:
    with pytest.raises(LookupError, match="training.register"):
        load_champion("never-registered", uri)


def test_alias_name_is_champion() -> None:
    assert CHAMPION_ALIAS == "champion"


def test_merged_config_lets_the_export_config_override_the_train_config(tmp_path: Path) -> None:
    train = tmp_path / "train.yaml"
    train.write_text(yaml.safe_dump({"backbone": "resnet50", "batch_size": 16}), encoding="utf-8")
    export = tmp_path / "export.yaml"
    export.write_text(
        yaml.safe_dump({"train_config": str(train), "backbone": "efficientnet_b0"}),
        encoding="utf-8",
    )
    merged = load_merged_config(export)
    assert merged["backbone"] == "efficientnet_b0"
    assert merged["batch_size"] == 16
    assert "train_config" not in merged


def test_config_without_train_config_is_returned_as_is(tmp_path: Path) -> None:
    path = tmp_path / "plain.yaml"
    path.write_text(yaml.safe_dump({"a": 1}), encoding="utf-8")
    assert load_merged_config(path) == {"a": 1}


def test_real_export_config_merges_cleanly() -> None:
    merged = load_merged_config("configs/export.yaml")
    assert merged["backbone"] == "efficientnet_b0"
    assert merged["model_name"]
    assert merged["image_size"] == 224
    assert merged["parity_atol"] == 1e-4
