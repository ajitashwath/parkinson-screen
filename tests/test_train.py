from __future__ import annotations

import pandas as pd

from training.train import sample_fast_subset, split_train_val


def make_manifest(n_subjects: int = 20, labels_per_subject: int = 4) -> pd.DataFrame:
    rows = []
    for i in range(n_subjects):
        label = "healthy" if i % 2 == 0 else "parkinson"
        for j in range(labels_per_subject):
            rows.append(
                {
                    "path": f"img_{i}_{j}.png",
                    "dataset": "fixture",
                    "kind": "spiral",
                    "label": label,
                    "subject_id": f"S{i}",
                    "image_hash": f"hash_{i}_{j}",
                }
            )
    return pd.DataFrame(rows)


def test_split_train_val_covers_all_rows() -> None:
    manifest = make_manifest()
    train_df, val_df = split_train_val(manifest, val_fraction=0.2, seed=0)
    assert len(train_df) + len(val_df) == len(manifest)


def test_split_train_val_no_subject_overlap() -> None:
    manifest = make_manifest()
    train_df, val_df = split_train_val(manifest, val_fraction=0.2, seed=0)
    assert set(train_df["subject_id"]).isdisjoint(set(val_df["subject_id"]))


def test_split_train_val_is_deterministic_for_a_given_seed() -> None:
    manifest = make_manifest()
    first_train, first_val = split_train_val(manifest, val_fraction=0.2, seed=3)
    second_train, second_val = split_train_val(manifest, val_fraction=0.2, seed=3)
    assert list(first_train["path"]) == list(second_train["path"])
    assert list(first_val["path"]) == list(second_val["path"])


def test_split_train_val_both_splits_nonempty() -> None:
    manifest = make_manifest()
    train_df, val_df = split_train_val(manifest, val_fraction=0.2, seed=0)
    assert len(train_df) > 0
    assert len(val_df) > 0


def test_sample_fast_subset_respects_per_class_limit() -> None:
    manifest = make_manifest(n_subjects=20, labels_per_subject=4)
    subset = sample_fast_subset(manifest, per_class=3, seed=0)
    counts = subset["label"].value_counts()
    assert (counts <= 3).all()


def test_sample_fast_subset_keeps_both_classes_when_available() -> None:
    manifest = make_manifest(n_subjects=20, labels_per_subject=4)
    subset = sample_fast_subset(manifest, per_class=5, seed=0)
    assert set(subset["label"]) == {"healthy", "parkinson"}


def test_sample_fast_subset_is_deterministic() -> None:
    manifest = make_manifest()
    first = sample_fast_subset(manifest, per_class=4, seed=1)
    second = sample_fast_subset(manifest, per_class=4, seed=1)
    assert list(first["path"]) == list(second["path"])
