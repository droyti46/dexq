"""Одно изображение — одна CSV-строка, даже при ошибке или превышении размера исследования."""

import csv
from io import BytesIO, StringIO
from zipfile import ZipFile

import pytest
from test_analysis_contract import dicom

from app.batch import analyze_items, read_archive, render_csv
from app.schemas import AnalysisResult, AnatomicalRegion


class StudyAnalyzer:
    def analyze(self, content: bytes, filename: str, region: AnatomicalRegion) -> AnalysisResult:
        from app.imaging import safe_dicom_fields

        uid, image_uid, _ = safe_dicom_fields(content, ".dcm")
        return AnalysisResult(
            analysis_id="an_test",
            filename=filename,
            study_uid=uid,
            image_uid=image_uid,
            anatomical_region=AnatomicalRegion.LUMBAR_SPINE,
            region_source="model",
            quality_class=1,
            violation_types=["spine_positioning", "spine_axis"],
            processing_status="Success",
            time_of_processing=0.25,
            checks=[],
            preview_data_url="",
        )


def _with_study(uid: str) -> bytes:
    from io import BytesIO

    import pydicom

    dataset = pydicom.dcmread(BytesIO(dicom()))
    dataset.StudyInstanceUID = uid
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_fourth_image_of_study_is_failure_and_other_study_continues() -> None:
    first = dicom()
    second = _with_study("1.2.4")
    items = [(f"case/a{i}.dcm", first) for i in range(3)] + [
        ("case/b1.dcm", second),
        ("case/a4.dcm", first),
        ("case/b2.dcm", second),
        ("case/b3.dcm", second),
    ]
    result = analyze_items(StudyAnalyzer(), items, AnatomicalRegion.AUTO)
    assert len(result.items) == 7
    assert result.successful == 6
    assert result.failed == 1
    assert result.items[4].filename == "case/a4.dcm"
    assert "трёх" in (result.items[4].error or "")
    assert result.items[-1].result is not None


def test_invalid_dicom_keeps_path_and_empty_uid_and_class() -> None:
    batch = analyze_items(StudyAnalyzer(), [("incoming/broken.dcm", b"bad")], AnatomicalRegion.AUTO)
    rows = list(csv.DictReader(StringIO(render_csv(batch).lstrip("﻿"))))
    assert rows[0]["path_to_study"] == "incoming/broken.dcm"
    assert rows[0]["study_uid"] == ""
    assert rows[0]["quality_class"] == ""
    assert rows[0]["processing_status"] == "Failure"
    assert len(rows[0]) == 8


def test_csv_codes_are_sorted_and_semicolon_separated() -> None:
    batch = analyze_items(StudyAnalyzer(), [("incoming/study.dcm", dicom())], AnatomicalRegion.AUTO)
    rows = list(csv.DictReader(StringIO(render_csv(batch).lstrip("﻿"))))
    assert rows[0]["violation_type"] == "spine_axis;spine_positioning"
    assert rows[0]["time_of_processing"] == "0.25"


@pytest.mark.parametrize("bad", ["../escape.dcm", "/absolute.dcm", "CON.dcm"])
def test_archive_rejects_unsafe_names(bad: str) -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr(bad, dicom())
    with pytest.raises(ValueError, match="путь"):
        read_archive(buffer.getvalue(), 1024 * 1024)


def test_archive_rejects_duplicate_image_name() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w") as archive:
        archive.writestr("study/file.dcm", dicom())
        archive.writestr("study/file.dcm", dicom())
    with pytest.raises(ValueError, match="повтор"):
        read_archive(buffer.getvalue(), 1024 * 1024)
