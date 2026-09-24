"""One-time ETL for the HandPD and NewHandPD datasets (UNESP, Botucatu
Medical School): unzips the archives downloaded per data/README.md and
reorganizes them into this project's canonical
data/raw/<dataset>/<spiral|wave>/<healthy|parkinson>/*.jpg layout, plus a
metadata.csv (kind, image_name, subject_id) that training/data.py reads for
leakage-safe, subject-grouped folds.

Subject IDs come from each dataset's own filename convention, not from the
published CSVs: HandPD's original Spiral_HandPD.csv/Meander_HandPD.csv turn
out to be plain LibOPF feature exports with no image_name or patient_id
column at all, and NewHandPD's NewSpiral.csv has a publishing bug where
every patient row's IMAGE_NAME points at a meander file regardless of which
CSV it is in (confirmed by joining it against the real archive contents:
zero of the 124 real patient spiral filenames appear in NewSpiral.csv, and
NewSpiral.csv's ID_PATIENT overlaps between the healthy and patient classes,
which a real per-class patient code should not). Both archives instead
encode a real, dataset-native subject code directly in every filename:
    HandPD:    0068-1.jpg   -> subject "0068" (the part before the first "-")
    NewHandPD: sp1-H9.jpg   -> subject "H9"   (the part before ".jpg")
               mea3-P12.jpg -> subject "P12"
This was verified against the actual downloaded archives: every filename
matches its pattern with no exceptions, the same subject code appears in
both the spiral and the meander archive for a given person, and healthy vs
patient codes never collide (H-prefixed vs P-prefixed for NewHandPD, and
disjoint 4-digit codes for HandPD). See docs/ASSUMPTIONS.md.

Usage:
    python -m training.prepare_handpd --downloads-dir data/raw/_downloads --out-dir data/raw
"""

from __future__ import annotations

import argparse
import csv
import re
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

JUNK_NAME_PREFIXES = ("._",)
JUNK_NAME_EXACT = (".DS_Store",)
JUNK_PATH_COMPONENTS = ("__MACOSX",)

HANDPD_FILENAME_PATTERN = re.compile(r"^(\d+)-\d+\.jpg$", re.IGNORECASE)
NEWHANDPD_FILENAME_PATTERN = re.compile(r"^[a-z]+\d+-([A-Za-z]+\d+)\.jpg$", re.IGNORECASE)

DrawingKind = Literal["spiral", "wave"]
Label = Literal["healthy", "parkinson"]


@dataclass(frozen=True)
class ArchiveSpec:
    zip_path: Path
    kind: DrawingKind
    label: Label
    subject_pattern: re.Pattern[str]


def is_junk_member(member_name: str) -> bool:
    parts = Path(member_name).parts
    if any(part in JUNK_PATH_COMPONENTS for part in parts):
        return True
    basename = Path(member_name).name
    if basename in JUNK_NAME_EXACT or basename == "":
        return True
    return any(basename.startswith(prefix) for prefix in JUNK_NAME_PREFIXES)


def extract_subject_id(filename: str, pattern: re.Pattern[str]) -> str:
    match = pattern.match(filename)
    if not match:
        raise ValueError(
            f"Filename {filename!r} does not match the expected pattern "
            f"{pattern.pattern!r}; the archive layout may have changed."
        )
    return match.group(1)


def process_archive(spec: ArchiveSpec, dataset_out_dir: Path) -> list[tuple[str, str, str]]:
    """Extract real (non-junk) jpg members of one archive into
    dataset_out_dir/<kind>/<label>/, returning (kind, image_name,
    subject_id) rows for the metadata.csv."""
    target_dir = dataset_out_dir / spec.kind / spec.label
    target_dir.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, str, str]] = []
    with zipfile.ZipFile(spec.zip_path) as archive:
        for member in archive.infolist():
            if member.is_dir() or is_junk_member(member.filename):
                continue
            image_name = Path(member.filename).name
            if not image_name.lower().endswith(".jpg"):
                continue
            subject_id = extract_subject_id(image_name, spec.subject_pattern)
            with archive.open(member) as source:
                (target_dir / image_name).write_bytes(source.read())
            rows.append((spec.kind, image_name, subject_id))
    return rows


def write_metadata_csv(rows: list[tuple[str, str, str]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["kind", "image_name", "subject_id"])
        writer.writerows(rows)


def prepare_handpd(downloads_dir: Path, out_dir: Path) -> dict[str, int]:
    """HandPD ships both classes inside one zip each (SpiralControl +
    SpiralPatients, MeanderControl + MeanderPatients), so the label is
    decided per archive member path rather than per archive, unlike
    NewHandPD's already-split zips handled by process_archive."""
    dataset_dir = out_dir / "handpd"
    archives = (
        (downloads_dir / "Spiral_HandPD.zip", "spiral"),
        (downloads_dir / "Meander_HandPD.zip", "wave"),
    )
    rows: list[tuple[str, str, str]] = []
    for zip_path, kind in archives:
        with zipfile.ZipFile(zip_path) as archive:
            for member in archive.infolist():
                if member.is_dir() or is_junk_member(member.filename):
                    continue
                image_name = Path(member.filename).name
                if not image_name.lower().endswith(".jpg"):
                    continue
                label = "healthy" if "Control" in member.filename else "parkinson"
                subject_id = extract_subject_id(image_name, HANDPD_FILENAME_PATTERN)
                target_dir = dataset_dir / kind / label
                target_dir.mkdir(parents=True, exist_ok=True)
                with archive.open(member) as source:
                    (target_dir / image_name).write_bytes(source.read())
                rows.append((kind, image_name, subject_id))
    write_metadata_csv(rows, dataset_dir / "metadata.csv")
    return {"handpd_images": len(rows)}


def prepare_newhandpd(downloads_dir: Path, out_dir: Path) -> dict[str, int]:
    dataset_dir = out_dir / "newhandpd"
    pattern = NEWHANDPD_FILENAME_PATTERN
    healthy_dir = downloads_dir / "NewHealthy"
    patient_dir = downloads_dir / "NewPatients"
    specs = [
        ArchiveSpec(healthy_dir / "HealthySpiral.zip", "spiral", "healthy", pattern),
        ArchiveSpec(healthy_dir / "HealthyMeander.zip", "wave", "healthy", pattern),
        ArchiveSpec(patient_dir / "PatientSpiral.zip", "spiral", "parkinson", pattern),
        ArchiveSpec(patient_dir / "PatientMeander.zip", "wave", "parkinson", pattern),
    ]
    rows: list[tuple[str, str, str]] = []
    for spec in specs:
        rows.extend(process_archive(spec, dataset_dir))
    write_metadata_csv(rows, dataset_dir / "metadata.csv")
    return {"newhandpd_images": len(rows)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--downloads-dir", default="data/raw/_downloads")
    parser.add_argument("--out-dir", default="data/raw")
    args = parser.parse_args()

    downloads_dir = Path(args.downloads_dir)
    out_dir = Path(args.out_dir)

    handpd_stats = prepare_handpd(downloads_dir, out_dir)
    newhandpd_stats = prepare_newhandpd(downloads_dir, out_dir)

    print(f"HandPD: wrote {handpd_stats['handpd_images']} images to {out_dir / 'handpd'}")
    print(
        f"NewHandPD: wrote {newhandpd_stats['newhandpd_images']} images "
        f"to {out_dir / 'newhandpd'}"
    )
    print("Next: python -m training.audit_data --config configs/data.yaml")


if __name__ == "__main__":
    main()
