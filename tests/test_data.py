from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

import training.data as data_module
from training.data import (
    assign_folds,
    compute_image_hash,
    discover_images,
    extract_subject_id,
    has_usable_subject_ids,
    load_subject_metadata,
    save_folds_csv,
    summarize,
)


def test_discover_images_raises_helpful_error_when_missing(tmp_path: Path) -> None:
    missing = tmp_path / "does_not_exist"
    with pytest.raises(FileNotFoundError, match="data/README.md"):
        discover_images(missing)


def test_discover_images_raises_when_empty(tmp_path: Path) -> None:
    empty_dir = tmp_path / "raw"
    empty_dir.mkdir()
    with pytest.raises(ValueError, match="No images found"):
        discover_images(empty_dir)


def test_discover_images_counts_match_fixture(synthetic_raw_dataset: Path) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    # 4 healthy + 4 parkinson spiral (plus 1 duplicate) + 3 + 3 wave
    assert len(manifest) == 4 + 4 + 1 + 3 + 3
    assert set(manifest["kind"]) == {"spiral", "wave"}
    assert set(manifest["label"]) == {"healthy", "parkinson"}
    assert set(manifest["dataset"]) == {"fixture_set"}


def test_discover_images_flags_duplicate_hash(synthetic_raw_dataset: Path) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    spiral_healthy = manifest[(manifest["kind"] == "spiral") & (manifest["label"] == "healthy")]
    duplicate_rows = spiral_healthy[spiral_healthy["path"].str.contains("healthy_00")]
    assert len(duplicate_rows) == 2
    assert duplicate_rows["image_hash"].nunique() == 1


def test_extract_subject_id_returns_none_without_pattern() -> None:
    assert extract_subject_id("P01_spiral_02.png", None) is None


def test_extract_subject_id_uses_supplied_pattern() -> None:
    subject_id = extract_subject_id("P01_spiral_02.png", pattern=r"^(P\d+)_")
    assert subject_id == "P01"


def test_extract_subject_id_returns_none_on_no_match() -> None:
    assert extract_subject_id("unrelated_name.png", pattern=r"^(P\d+)_") is None


def test_load_subject_metadata_returns_empty_dict_when_absent(tmp_path: Path) -> None:
    assert load_subject_metadata(tmp_path) == {}


def test_load_subject_metadata_reads_kind_image_subject_columns(tmp_path: Path) -> None:
    (tmp_path / "metadata.csv").write_text(
        "kind,image_name,subject_id,extra_column\nspiral,healthy_00.png,S001,ignored\n",
        encoding="utf-8",
    )
    lookup = load_subject_metadata(tmp_path)
    assert lookup == {("spiral", "healthy_00.png"): "S001"}


def test_load_subject_metadata_raises_on_missing_columns(tmp_path: Path) -> None:
    (tmp_path / "metadata.csv").write_text(
        "kind,image_name\nspiral,healthy_00.png\n", encoding="utf-8"
    )
    with pytest.raises(ValueError, match="missing required columns"):
        load_subject_metadata(tmp_path)


def test_discover_images_prefers_metadata_csv_over_regex(tmp_path: Path) -> None:
    raw_dir = tmp_path / "raw"
    dataset_dir = raw_dir / "handpd_like" / "spiral" / "healthy"
    dataset_dir.mkdir(parents=True)
    (dataset_dir / "S001_01.png").write_bytes(b"\x89PNG\r\n\x1a\n")
    (raw_dir / "handpd_like" / "metadata.csv").write_text(
        "kind,image_name,subject_id\nspiral,S001_01.png,FROM_METADATA\n", encoding="utf-8"
    )
    # a regex is also supplied, but the metadata.csv value must win
    with patch.object(data_module, "compute_image_hash", return_value="fakehash"):
        manifest = discover_images(raw_dir, subject_id_regex=r"^(S\d+)_")
    assert manifest.loc[0, "subject_id"] == "FROM_METADATA"


def test_has_usable_subject_ids_false_when_all_none() -> None:
    manifest = pd.DataFrame({"subject_id": [None, None, None]})
    assert has_usable_subject_ids(manifest) is False


def test_has_usable_subject_ids_false_when_only_one_subject() -> None:
    manifest = pd.DataFrame({"subject_id": ["A", "A", "A"]})
    assert has_usable_subject_ids(manifest) is False


def test_has_usable_subject_ids_true_when_fully_populated() -> None:
    manifest = pd.DataFrame({"subject_id": ["A", "B", "C"]})
    assert has_usable_subject_ids(manifest) is True


def test_assign_folds_falls_back_to_hash_grouping_without_subjects(
    synthetic_raw_dataset: Path,
) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    folded = assign_folds(manifest, n_folds=3, seed=0)
    assert folded.attrs["fold_method"] == "grouped_by_image_hash_fallback"
    assert (folded["fold"] >= 0).all()


def test_assign_folds_keeps_duplicate_hash_in_one_fold(synthetic_raw_dataset: Path) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    folded = assign_folds(manifest, n_folds=3, seed=0)
    for _, group in folded.groupby("image_hash"):
        assert group["fold"].nunique() == 1


def test_assign_folds_groups_by_subject_id_when_available() -> None:
    manifest = pd.DataFrame(
        {
            "path": [f"img_{i}.png" for i in range(12)],
            "dataset": ["fixture"] * 12,
            "kind": ["spiral"] * 12,
            "label": (["healthy"] * 6) + (["parkinson"] * 6),
            "subject_id": [f"S{i // 2}" for i in range(12)],
            "image_hash": [f"hash_{i}" for i in range(12)],
        }
    )
    folded = assign_folds(manifest, n_folds=3, seed=0)
    assert folded.attrs["fold_method"] == "grouped_by_subject_id"
    for _, group in folded.groupby("subject_id"):
        assert group["fold"].nunique() == 1


def test_assign_folds_is_deterministic_for_a_given_seed(synthetic_raw_dataset: Path) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    first = assign_folds(manifest, n_folds=3, seed=7)
    second = assign_folds(manifest, n_folds=3, seed=7)
    assert list(first["fold"]) == list(second["fold"])


def test_save_folds_csv_round_trip(tmp_path: Path, synthetic_raw_dataset: Path) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    folded = assign_folds(manifest, n_folds=3, seed=0)
    out_path = tmp_path / "nested" / "folds.csv"
    save_folds_csv(folded, out_path)
    reloaded = pd.read_csv(out_path)
    assert len(reloaded) == len(folded)
    assert "fold" in reloaded.columns


def test_summarize_reports_expected_fields(synthetic_raw_dataset: Path) -> None:
    manifest = discover_images(synthetic_raw_dataset)
    summary = summarize(manifest)
    assert summary["total_images"] == len(manifest)
    assert summary["subject_ids_available"] is False
    assert summary["duplicate_group_count"] == 1
    assert summary["duplicate_image_count"] == 2
    assert summary["counts_by_kind_label"]["spiral/healthy"] == 5


def test_compute_image_hash_matches_for_identical_files(synthetic_raw_dataset: Path) -> None:
    original = synthetic_raw_dataset / "fixture_set" / "spiral" / "healthy" / "healthy_00.png"
    duplicate = synthetic_raw_dataset / "fixture_set" / "spiral" / "healthy" / "healthy_00_copy.png"
    assert compute_image_hash(original) == compute_image_hash(duplicate)


def test_spiral_wave_dataset_returns_tensor_and_label(synthetic_raw_dataset: Path) -> None:
    torch = pytest.importorskip("torch")
    torchvision = pytest.importorskip("torchvision")
    del torchvision

    from training.data import SpiralWaveDataset

    manifest = discover_images(synthetic_raw_dataset)
    folded = assign_folds(manifest, n_folds=3, seed=0)

    eval_dataset = SpiralWaveDataset(folded, fold_ids=[0], train=False)
    assert len(eval_dataset) > 0
    tensor, label = eval_dataset[0]
    assert isinstance(tensor, torch.Tensor)
    assert tensor.shape == (3, 224, 224)
    assert label in (0, 1)
    tensor_again, _ = eval_dataset[0]
    assert torch.equal(tensor, tensor_again)

    train_dataset = SpiralWaveDataset(folded, fold_ids=[0], train=True)
    train_tensor, train_label = train_dataset[0]
    assert isinstance(train_tensor, torch.Tensor)
    assert train_tensor.shape == (3, 224, 224)
    assert train_label in (0, 1)
