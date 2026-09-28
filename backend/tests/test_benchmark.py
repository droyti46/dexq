"""Проверка агрегированного benchmark по целым исследованиям."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from app.schemas import AnalysisResult, AnatomicalRegion

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from benchmark import benchmark_studies  # noqa: E402


class FakeAnalyzer:
    def analyze(
        self,
        content: bytes,
        filename: str,
        requested_region: AnatomicalRegion = AnatomicalRegion.AUTO,
    ) -> AnalysisResult:
        return AnalysisResult(
            analysis_id="test",
            filename=filename,
            study_uid=None,
            image_uid=None,
            anatomical_region=AnatomicalRegion.LUMBAR_SPINE,
            region_source="model",
            quality_class=0,
            violation_types=[],
            processing_status="Success",
            time_of_processing=0,
            checks=[],
            preview_data_url="",
            annotated_data_url=None,
            geometry=None,
            needs_review=False,
        )


def test_benchmark_times_read_infer_and_csv_without_private_rows(tmp_path: Path) -> None:
    file_a = tmp_path / "a.dcm"
    file_b = tmp_path / "b.dcm"
    file_a.write_bytes(b"secret 1")
    file_b.write_bytes(b"secret 2")
    report = benchmark_studies(
        {"study-1": [file_a, file_b]},
        FakeAnalyzer(),
        limit_seconds=180,
        clock=iter([0.0, 2.0]).__next__,
    )
    assert report["studies"] == 1
    assert report["images"] == 2
    assert report["successful"] == 2
    assert report["max_study_seconds"] == 2
    assert report["within_limit"] is True
    assert "study-1" not in str(report)
    assert "secret" not in str(report)


def test_benchmark_refuses_slow_or_oversized_studies(tmp_path: Path) -> None:
    file = tmp_path / "sample.dcm"
    file.write_bytes(b"input")
    result = benchmark_studies(
        {"private": [file]}, FakeAnalyzer(), limit_seconds=180, clock=iter([0.0, 181.0]).__next__
    )
    assert result["within_limit"] is False
    with pytest.raises(ValueError, match="трёх"):
        benchmark_studies({"private": [file] * 4}, FakeAnalyzer())
