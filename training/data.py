"""Dataset manifest, leakage-aware fold assignment, and train-only
augmentation for spiral and wave drawings.

Reads whatever dataset layout is present under data/raw/, in the form
data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.png, per CLAUDE.md
Phase 1. Subject IDs come from one of two places, in priority order: an
optional data/raw/<dataset>/metadata.csv (kind, image_name, subject_id)
sourced from the dataset's own published metadata when one exists (see
training/prepare_handpd.py for how HandPD's original CSVs are normalized
into this form), or a caller-supplied regex on the filename
(configs/data.yaml subject_id_regex) for datasets that encode the subject
in the filename itself. Neither is guessed: a dataset with no metadata.csv
and no working regex yields subject_id = None, per CLAUDE.md rule 2.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import imagehash
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold

from common.preprocess import PreprocessConfig, preprocess_image

try:
    from torch.utils.data import Dataset as _TorchDataset
except ImportError:  # training/data.py's manifest and fold functions do not need torch
    _TorchDataset = object  # type: ignore[assignment, misc]

IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")
DRAWING_KINDS = ("spiral", "wave")
LABELS = ("healthy", "parkinson")
MANIFEST_COLUMNS = ["path", "dataset", "kind", "label", "subject_id", "image_hash"]
METADATA_FILENAME = "metadata.csv"
METADATA_REQUIRED_COLUMNS = {"kind", "image_name", "subject_id"}


@dataclass(frozen=True)
class DataConfig:
    raw_dir: Path
    subject_id_regex: str | None = None
    n_folds: int = 5
    seed: int = 42
    phash_size: int = 8


def extract_subject_id(filename: str, pattern: str | None) -> str | None:
    """Best-effort subject ID extraction from a filename, using a regex the
    caller supplies. Returns None (unknown) when no pattern is given or it
    does not match, rather than guessing a convention."""
    if not pattern:
        return None
    match = re.search(pattern, filename)
    if not match:
        return None
    if match.groups():
        return match.group(1)
    return match.group(0)


def compute_image_hash(path: Path, hash_size: int = 8) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("L"), hash_size=hash_size))


def load_subject_metadata(dataset_dir: Path) -> dict[tuple[str, str], str]:
    """Load an optional dataset/metadata.csv (columns: kind, image_name,
    subject_id) giving exact subject IDs sourced from the dataset's own
    published metadata, keyed by (kind, image filename). Returns an empty
    dict, not an error, when the dataset has no metadata.csv: most datasets
    will not have one, and that is the normal case handled by the
    subject_id_regex fallback in discover_images."""
    metadata_path = dataset_dir / METADATA_FILENAME
    if not metadata_path.exists():
        return {}
    metadata = pd.read_csv(metadata_path, dtype=str)
    missing = METADATA_REQUIRED_COLUMNS - set(metadata.columns)
    if missing:
        raise ValueError(f"{metadata_path} is missing required columns: {sorted(missing)}")
    return {
        (row["kind"], row["image_name"]): row["subject_id"]
        for row in metadata.to_dict("records")
    }


def discover_images(
    raw_dir: str | Path,
    subject_id_regex: str | None = None,
    phash_size: int = 8,
) -> pd.DataFrame:
    """Walk data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/* and build
    a manifest. Fails loudly with a helpful message if the directory is
    missing or nothing matches the expected layout, per CLAUDE.md rule 2."""
    raw_path = Path(raw_dir)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"{raw_path} does not exist. Download a dataset into data/raw/ "
            "first; see data/README.md for instructions and the expected "
            "layout data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.png"
        )
    records: list[dict[str, Any]] = []
    for dataset_dir in sorted(p for p in raw_path.iterdir() if p.is_dir()):
        metadata_lookup = load_subject_metadata(dataset_dir)
        for kind in DRAWING_KINDS:
            kind_dir = dataset_dir / kind
            if not kind_dir.exists():
                continue
            for label in LABELS:
                label_dir = kind_dir / label
                if not label_dir.exists():
                    continue
                for image_path in sorted(label_dir.iterdir()):
                    if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                        continue
                    subject_id = metadata_lookup.get((kind, image_path.name))
                    if subject_id is None:
                        subject_id = extract_subject_id(image_path.name, subject_id_regex)
                    records.append(
                        {
                            "path": str(image_path),
                            "dataset": dataset_dir.name,
                            "kind": kind,
                            "label": label,
                            "subject_id": subject_id,
                            "image_hash": compute_image_hash(image_path, phash_size),
                        }
                    )
    if not records:
        raise ValueError(
            f"No images found under {raw_path} matching "
            "<dataset>/<spiral|wave>/<healthy|parkinson>/*.png. "
            "See data/README.md for the expected layout."
        )
    return pd.DataFrame.from_records(records, columns=MANIFEST_COLUMNS)


def has_usable_subject_ids(manifest: pd.DataFrame) -> bool:
    """Subject IDs are usable for grouping only if every row has one and
    there is more than one distinct subject; a partially-known column would
    silently under-protect the rows with no ID, so it is treated as unknown."""
    return bool(manifest["subject_id"].notna().all() and manifest["subject_id"].nunique() > 1)


def summarize(manifest: pd.DataFrame) -> dict[str, Any]:
    """Counts and leakage-relevant facts for the datasheet: images per
    class and drawing kind, image size distribution, duplicate groups, and
    whether subject IDs are available."""
    duplicate_counts = manifest["image_hash"].value_counts()
    duplicate_groups = duplicate_counts[duplicate_counts > 1]
    sizes: set[tuple[int, int]] = set()
    for path in manifest["path"]:
        with Image.open(path) as image:
            sizes.add(image.size)
    counts_by_kind_label = {
        f"{kind}/{label}": int(count)
        for (kind, label), count in manifest.groupby(["kind", "label"]).size().items()
    }
    return {
        "total_images": int(len(manifest)),
        "counts_by_kind_label": counts_by_kind_label,
        "counts_by_dataset": {k: int(v) for k, v in manifest["dataset"].value_counts().items()},
        "image_sizes": sorted(sizes),
        "duplicate_group_count": int(len(duplicate_groups)),
        "duplicate_image_count": int(duplicate_counts[duplicate_counts > 1].sum()),
        "subject_ids_available": has_usable_subject_ids(manifest),
    }


def assign_folds(manifest: pd.DataFrame, n_folds: int = 5, seed: int = 42) -> pd.DataFrame:
    """Grouped stratified k-fold split. Groups by subject_id when every row
    has one (leakage-safe by subject); otherwise falls back to grouping by
    image_hash, so duplicate or near-duplicate images never split across
    train and test. The fold method used is recorded in the returned
    DataFrame's attrs, for the datasheet to report the fallback loudly."""
    result = manifest.reset_index(drop=True).copy()
    if has_usable_subject_ids(result):
        groups = result["subject_id"]
        method = "grouped_by_subject_id"
    else:
        groups = result["image_hash"]
        method = "grouped_by_image_hash_fallback"
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    result["fold"] = -1
    for fold_index, (_, test_index) in enumerate(splitter.split(result, result["label"], groups)):
        result.loc[result.index[test_index], "fold"] = fold_index
    result.attrs["fold_method"] = method
    return result


def save_folds_csv(manifest: pd.DataFrame, out_path: str | Path) -> None:
    out_file = Path(out_path)
    out_file.parent.mkdir(parents=True, exist_ok=True)
    manifest.to_csv(out_file, index=False)


def build_train_augmentation() -> Any:
    """Torchvision transform used only on training folds, applied on the
    fly and never saved to disk. Small rotation, slight scale and
    translation, brightness/contrast, random erasing. No elastic warps:
    they would distort the tremor pattern the model and the tremor
    analyzer both rely on. Imports torchvision lazily so the rest of this
    module stays importable without the heavy training stack installed."""
    import torchvision.transforms as T

    return T.Compose(
        [
            T.RandomAffine(degrees=10, translate=(0.05, 0.05), scale=(0.95, 1.05), fill=255),
            T.ColorJitter(brightness=0.15, contrast=0.15),
            T.ToTensor(),
            T.RandomErasing(p=0.25, scale=(0.02, 0.08)),
        ]
    )


class SpiralWaveDataset(_TorchDataset):
    """torch.utils.data.Dataset over a fold-assigned manifest. Applies
    common.preprocess.preprocess_image, then, when augmentation is
    enabled, the augmentation from build_train_augmentation. Imports
    torch lazily so training/data.py stays importable without the heavy
    training stack installed; instantiating this class does require it."""

    def __init__(
        self,
        manifest: pd.DataFrame,
        fold_ids: list[int],
        preprocess_config: PreprocessConfig | None = None,
        train: bool = False,
        augment: bool | None = None,
    ) -> None:
        """train controls which rows are treated as a training split (only
        relevant to callers that branch on it); augment controls whether
        the on-the-fly augmentation pipeline runs, and defaults to train
        so the common case (train=True means "augment") does not need
        both arguments. The augmentation ablation in training/ablations.py
        passes augment=False with train=True to get a training split with
        no augmentation."""
        self.manifest = manifest[manifest["fold"].isin(fold_ids)].reset_index(drop=True)
        self.preprocess_config = preprocess_config or PreprocessConfig()
        self.train = train
        self.augment = train if augment is None else augment
        self.augmentation = build_train_augmentation() if self.augment else None

    def __len__(self) -> int:
        return len(self.manifest)

    def __getitem__(self, index: int) -> tuple[Any, int]:
        import torch

        row = self.manifest.iloc[index]
        array = preprocess_image(row["path"], self.preprocess_config)
        if self.augment and self.augmentation is not None:
            image = Image.fromarray((array * 255).astype(np.uint8))
            tensor = self.augmentation(image)
        else:
            tensor = torch.from_numpy(array).permute(2, 0, 1).contiguous()
        label = 1 if row["label"] == "parkinson" else 0
        return tensor, label
