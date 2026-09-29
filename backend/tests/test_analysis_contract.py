"""Контракт отчёта без выдуманных клинических классов и приватных тегов."""

from __future__ import annotations

import warnings
from io import BytesIO
from pathlib import Path

import numpy as np
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage

from app.analyzer import Analyzer
from app.schemas import AnatomicalRegion


class FakeRuntime:
    def __init__(self, result: dict) -> None:
        self.result = result
        self.calls = 0

    def ready(self) -> bool:
        return True

    def infer(self, content: bytes, suffix: str, region: AnatomicalRegion) -> dict:
        self.calls += 1
        return self.result


def dicom(view: str | None = None) -> bytes:
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = "1.2.3.4"
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    ds = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    ds.SOPClassUID = meta.MediaStorageSOPClassUID
    ds.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    ds.StudyInstanceUID = "1.2.3"
    ds.PatientName = "SECRET_PATIENT"
    ds.Rows, ds.Columns = 32, 32
    ds.SamplesPerPixel = 1
    ds.PhotometricInterpretation = "MONOCHROME2"
    ds.BitsAllocated = ds.BitsStored = 8
    ds.HighBit = 7
    ds.PixelRepresentation = 0
    ds.PixelData = np.arange(1024, dtype=np.uint8).tobytes()
    if view is not None:
        ds.ViewPosition = view
    buffer = BytesIO()
    ds.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def spine_result(axis: bool | None = False) -> dict:
    return {
        "region": "spine",
        "region_source": "model",
        "labels": {"spine_positioning": False, "spine_axis": axis, "spine_artifacts": False},
        "scores": {"spine_positioning": 0.1, "spine_axis": 2.5, "spine_artifacts": 0.1},
        "any_violation": axis,
        "geometry": {
            "coordinate_system": "native_pixels_x_right_y_down",
            "image_width": 32,
            "image_height": 32,
            "axis_line": None
            if axis is None
            else {"top_xy": [17, 3], "bottom_xy": [16, 30], "angle_deg": 2.12},
            "vertebral_candidates": [],
            "gap_lines": [],
            "internal_metadata": "SECRET_INTERNAL",
        },
        "annotated_image": {"encoding": "base64_png", "data": "YQ==", "width": 32, "height": 32},
        "routing": {"metadata": {"patient": "SECRET_INTERNAL"}},
    }


def test_spine_success_is_binary_and_has_native_geometry() -> None:
    runtime = FakeRuntime(spine_result())
    result = Analyzer(runtime).analyze(dicom(), "study.dcm")
    assert runtime.calls == 1
    assert result.processing_status == "Success"
    assert result.quality_class == 0
    assert result.projection == "unknown"
    assert result.projection_source == "not_determined"
    assert result.study_uid == "1.2.3"
    assert result.geometry is not None and result.geometry.axis_line is not None
    assert result.preview_data_url.startswith("data:image/png;base64,")
    assert "SECRET" not in result.model_dump_json()


def test_hip_combined_type_is_not_split_into_false_claims() -> None:
    output = {
        "region": "hip",
        "region_source": "model",
        "laterality": "right_hip",
        "labels": {"hip_positioning_rotation": True, "hip_roi": False},
        "scores": {"hip_positioning_rotation": 0.8, "hip_roi": 0.1},
        "any_violation": True,
        "geometry": {
            "coordinate_system": "native_pixels_x_right_y_down",
            "image_width": 32,
            "image_height": 32,
            "foreground_bbox": [3, 3, 26, 28],
            "interpretation": "SECRET_INTERNAL",
        },
    }
    result = Analyzer(FakeRuntime(output)).analyze(dicom(), "hip.dcm")
    assert result.processing_status == "Success"
    assert result.quality_class == 1
    assert result.violation_types == ["hip_positioning_rotation"]
    assert "hip_rotation" not in result.violation_types
    assert result.geometry is not None and result.geometry.axis_line is None


def test_missing_required_axis_decision_is_failure() -> None:
    result = Analyzer(FakeRuntime(spine_result(None))).analyze(dicom(), "spine.dcm")
    assert result.processing_status == "Failure"
    assert result.quality_class is None
    assert result.geometry is not None and result.geometry.axis_line is None


def test_explicit_dicom_view_is_reported_without_changing_quality() -> None:
    for view, expected in [("AP", "AP"), ("PA", "PA")]:
        runtime = FakeRuntime(spine_result())
        result = Analyzer(runtime).analyze(dicom(view), "study.dcm")
        assert runtime.calls == 1
        assert result.processing_status == "Success"
        assert result.quality_class == 0
        assert result.projection == expected
        assert result.projection_source == "dicom_view_position"
        assert "SECRET" not in result.model_dump_json()


def test_unknown_dicom_view_is_not_guessed_from_anatomy() -> None:
    result = Analyzer(FakeRuntime(spine_result())).analyze(dicom(), "spine.dcm")
    assert result.projection == "unknown"
    assert result.projection_source == "not_determined"


def test_unsupported_view_never_reaches_model() -> None:
    runtime = FakeRuntime(spine_result())
    result = Analyzer(runtime).analyze(dicom("LATERAL"), "study.dcm")
    assert result.processing_status == "Failure"
    assert result.quality_class is None
    assert runtime.calls == 0
    assert result.projection == "unknown"


def test_invalid_uid_warning_never_contains_identifier() -> None:
    payload = dicom().replace(b"1.2.3.4", b"1.2.3X4")
    with warnings.catch_warnings(record=True) as logged:
        Analyzer(FakeRuntime(spine_result())).analyze(payload, "study.dcm")
    assert all("1.2.3X4" not in str(item.message) for item in logged)


def test_filename_is_sanitized_and_error_details_never_leak(tmp_path: Path) -> None:
    class BrokenRuntime(FakeRuntime):
        def infer(self, content: bytes, suffix: str, region: AnatomicalRegion) -> dict:
            raise ValueError(f"broken {tmp_path}/SECRET_PATIENT")

    result = Analyzer(BrokenRuntime({})).analyze(dicom(), "C:/patients/SECRET_PATIENT/study.dcm")
    assert "C:/patients" not in result.model_dump_json()
    assert "broken" not in result.model_dump_json()
    assert "SECRET_PATIENT" not in result.model_dump_json()
    assert result.processing_status == "Failure"
