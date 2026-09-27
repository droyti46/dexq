"""Контрактные тесты API на синтетическом обезличенном DICOM."""

from io import BytesIO
from zipfile import ZIP_DEFLATED, ZipFile

import numpy as np
from fastapi.testclient import TestClient
from pydicom.dataset import FileDataset, FileMetaDataset
from pydicom.uid import ExplicitVRLittleEndian, SecondaryCaptureImageStorage, generate_uid

from app.checks.spine import measure_tilt
from app.main import app

client = TestClient(app)


def make_dicom() -> bytes:
    """Создаёт минимальный DICOM без персональных данных."""
    meta = FileMetaDataset()
    meta.MediaStorageSOPClassUID = SecondaryCaptureImageStorage
    meta.MediaStorageSOPInstanceUID = generate_uid()
    meta.TransferSyntaxUID = ExplicitVRLittleEndian
    dataset = FileDataset(None, {}, file_meta=meta, preamble=b"\0" * 128)
    dataset.SOPClassUID = meta.MediaStorageSOPClassUID
    dataset.SOPInstanceUID = meta.MediaStorageSOPInstanceUID
    dataset.StudyInstanceUID = generate_uid()
    dataset.BodyPartExamined = "LUMBAR SPINE"
    dataset.Rows = 256
    dataset.Columns = 192
    dataset.SamplesPerPixel = 1
    dataset.PhotometricInterpretation = "MONOCHROME2"
    dataset.BitsAllocated = 16
    dataset.BitsStored = 12
    dataset.HighBit = 11
    dataset.PixelRepresentation = 0
    y, x = np.mgrid[:256, :192]
    pixels = (2800 * np.exp(-((x - 96) ** 2) / 500 - ((y - 128) ** 2) / 9000)).astype(
        np.uint16
    )
    dataset.PixelData = pixels.tobytes()
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_health_and_checks() -> None:
    assert client.get("/api/v1/health").json()["status"] == "ok"
    checks = client.get("/api/v1/checks").json()
    assert {item["check_id"] for item in checks} == {
        "spine_coverage",
        "spine_tilt",
        "artifact",
        "hip_coverage",
        "hip_rotation",
        "hip_roi",
    }


def test_analyze_dicom() -> None:
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("study.dcm", make_dicom(), "application/dicom")},
        data={"anatomical_region": "auto"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["anatomical_region"] == "lumbar_spine"
    assert payload["study_uid"]
    assert payload["preview_data_url"].startswith("data:image/png;base64,")
    assert {item["check_id"] for item in payload["checks"]} == {
        "spine_coverage",
        "spine_tilt",
        "artifact",
    }


def test_batch_csv_keeps_failures() -> None:
    response = client.post(
        "/api/v1/analyses/batch.csv",
        files=[
            ("files", ("valid.dcm", make_dicom(), "application/dicom")),
            ("files", ("broken.dcm", b"not dicom", "application/dicom")),
        ],
        data={"anatomical_region": "auto"},
    )
    assert response.status_code == 200
    assert "Success" in response.text
    assert "Failure" in response.text


def test_archive_csv_processes_files_without_extracting() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("study/valid_ПОП.dcm", make_dicom())
        archive.writestr("study/broken.dcm", b"not dicom")
        archive.writestr("notes/readme.txt", "ignored")
    response = client.post(
        "/api/v1/analyses/archive.csv",
        files={"archive": ("studies.zip", buffer.getvalue(), "application/zip")},
        data={"anatomical_region": "auto"},
    )
    assert response.status_code == 200
    assert "study/valid_ПОП.dcm" in response.text
    assert "study/broken.dcm" in response.text
    assert "Success" in response.text
    assert "Failure" in response.text


def test_tilt_uses_strict_five_degree_threshold() -> None:
    straight = measure_tilt([100, 10], [100, 110])
    assert straight["violation"] is False
    tilted = measure_tilt([112, 10], [100, 110])
    assert tilted["angle_deg"] > 5
    assert tilted["violation"] is True
