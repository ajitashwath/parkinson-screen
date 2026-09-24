from __future__ import annotations

import zipfile
from pathlib import Path

import pandas as pd
import pytest

from training.prepare_handpd import (
    HANDPD_FILENAME_PATTERN,
    NEWHANDPD_FILENAME_PATTERN,
    ArchiveSpec,
    extract_subject_id,
    is_junk_member,
    prepare_handpd,
    prepare_newhandpd,
    process_archive,
    write_metadata_csv,
)


def test_is_junk_member_flags_apple_double_files() -> None:
    assert is_junk_member("SpiralControl/._0068-1.jpg") is True


def test_is_junk_member_flags_ds_store() -> None:
    assert is_junk_member("SpiralControl/.DS_Store") is True


def test_is_junk_member_flags_macosx_folder() -> None:
    assert is_junk_member("__MACOSX/SpiralControl/._0068-1.jpg") is True


def test_is_junk_member_false_for_real_image() -> None:
    assert is_junk_member("SpiralControl/0068-1.jpg") is False


def test_extract_subject_id_handpd_pattern() -> None:
    assert extract_subject_id("0068-1.jpg", HANDPD_FILENAME_PATTERN) == "0068"


def test_extract_subject_id_newhandpd_spiral_pattern() -> None:
    assert extract_subject_id("sp1-H9.jpg", NEWHANDPD_FILENAME_PATTERN) == "H9"


def test_extract_subject_id_newhandpd_meander_pattern() -> None:
    assert extract_subject_id("mea3-P12.jpg", NEWHANDPD_FILENAME_PATTERN) == "P12"


def test_extract_subject_id_raises_on_unexpected_filename() -> None:
    with pytest.raises(ValueError, match="does not match"):
        extract_subject_id("unexpected_name.jpg", HANDPD_FILENAME_PATTERN)


def test_write_metadata_csv_round_trip(tmp_path: Path) -> None:
    out_path = tmp_path / "nested" / "metadata.csv"
    write_metadata_csv([("spiral", "0068-1.jpg", "0068")], out_path)
    reloaded = pd.read_csv(out_path)
    assert list(reloaded.columns) == ["kind", "image_name", "subject_id"]
    assert reloaded.iloc[0]["image_name"] == "0068-1.jpg"


def make_zip(path: Path, members: dict[str, bytes]) -> None:
    with zipfile.ZipFile(path, "w") as archive:
        for name, content in members.items():
            archive.writestr(name, content)


def test_process_archive_extracts_real_files_and_skips_junk(tmp_path: Path) -> None:
    zip_path = tmp_path / "HealthySpiral.zip"
    make_zip(
        zip_path,
        {
            "HealthySpiral/sp1-H1.jpg": b"fake-jpg-bytes-1",
            "HealthySpiral/._sp1-H1.jpg": b"apple-double-junk",
            "HealthySpiral/.DS_Store": b"ds-store-junk",
            "HealthySpiral/sp1-H2.jpg": b"fake-jpg-bytes-2",
        },
    )
    spec = ArchiveSpec(zip_path, "spiral", "healthy", NEWHANDPD_FILENAME_PATTERN)
    out_dir = tmp_path / "out"
    rows = process_archive(spec, out_dir)

    assert sorted(rows) == [
        ("spiral", "sp1-H1.jpg", "H1"),
        ("spiral", "sp1-H2.jpg", "H2"),
    ]
    written = sorted(p.name for p in (out_dir / "spiral" / "healthy").iterdir())
    assert written == ["sp1-H1.jpg", "sp1-H2.jpg"]
    assert (out_dir / "spiral" / "healthy" / "sp1-H1.jpg").read_bytes() == b"fake-jpg-bytes-1"


def test_prepare_handpd_splits_control_and_patients_by_path(tmp_path: Path) -> None:
    downloads_dir = tmp_path / "downloads"
    downloads_dir.mkdir()
    make_zip(
        downloads_dir / "Spiral_HandPD.zip",
        {
            "SpiralControl/0068-1.jpg": b"control-bytes",
            "SpiralControl/._0068-1.jpg": b"junk",
            "SpiralPatients/0002-1.jpg": b"patient-bytes",
            "__MACOSX/SpiralPatients/._0002-1.jpg": b"junk",
        },
    )
    make_zip(
        downloads_dir / "Meander_HandPD.zip",
        {
            "MeanderControl/0068-5.jpg": b"control-meander-bytes",
            "MeanderPatients/0002-5.jpg": b"patient-meander-bytes",
        },
    )

    out_dir = tmp_path / "out"
    stats = prepare_handpd(downloads_dir, out_dir)

    assert stats == {"handpd_images": 4}
    control_spiral = out_dir / "handpd" / "spiral" / "healthy" / "0068-1.jpg"
    assert control_spiral.read_bytes() == b"control-bytes"
    assert (out_dir / "handpd" / "spiral" / "parkinson" / "0002-1.jpg").exists()
    assert (out_dir / "handpd" / "wave" / "healthy" / "0068-5.jpg").exists()
    assert (out_dir / "handpd" / "wave" / "parkinson" / "0002-5.jpg").exists()

    metadata = pd.read_csv(out_dir / "handpd" / "metadata.csv", dtype=str)
    assert len(metadata) == 4
    assert set(metadata["subject_id"]) == {"0068", "0002"}


def test_prepare_newhandpd_splits_by_archive(tmp_path: Path) -> None:
    downloads_dir = tmp_path / "downloads"
    (downloads_dir / "NewHealthy").mkdir(parents=True)
    (downloads_dir / "NewPatients").mkdir(parents=True)
    healthy_dir = downloads_dir / "NewHealthy"
    patient_dir = downloads_dir / "NewPatients"
    make_zip(healthy_dir / "HealthySpiral.zip", {"HealthySpiral/sp1-H1.jpg": b"a"})
    make_zip(healthy_dir / "HealthyMeander.zip", {"HealthyMeander/mea1-H1.jpg": b"b"})
    make_zip(patient_dir / "PatientSpiral.zip", {"PatientSpiral/sp1-P1.jpg": b"c"})
    make_zip(patient_dir / "PatientMeander.zip", {"PatientMeander/mea1-P1.jpg": b"d"})

    out_dir = tmp_path / "out"
    stats = prepare_newhandpd(downloads_dir, out_dir)

    assert stats == {"newhandpd_images": 4}
    assert (out_dir / "newhandpd" / "spiral" / "healthy" / "sp1-H1.jpg").exists()
    assert (out_dir / "newhandpd" / "wave" / "healthy" / "mea1-H1.jpg").exists()
    assert (out_dir / "newhandpd" / "spiral" / "parkinson" / "sp1-P1.jpg").exists()
    assert (out_dir / "newhandpd" / "wave" / "parkinson" / "mea1-P1.jpg").exists()
