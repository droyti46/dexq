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
    dataset.BitsAllocated = 8
    dataset.BitsStored = 8
    dataset.HighBit = 7
    dataset.PixelRepresentation = 0
    y, x = np.mgrid[:256, :192]
    pixels = (255 * np.exp(-((x - 96) ** 2) / 500 - ((y - 128) ** 2) / 9000)).astype(np.uint8)
    dataset.PixelData = pixels.tobytes()
    buffer = BytesIO()
    dataset.save_as(buffer, enforce_file_format=True)
    return buffer.getvalue()


def test_health_and_checks() -> None:
    assert client.get("/api/v1/health").json()["status"] == "ready"
    checks = client.get("/api/v1/checks").json()
    assert {item["check_id"] for item in checks} == {
        "spine_positioning",
        "spine_axis",
        "spine_artifacts",
        "hip_positioning_rotation",
        "hip_roi",
    }


def test_unclassifiable_axis_returns_controlled_failure() -> None:
    response = client.post(
        "/api/v1/analyses",
        files={"file": ("study.dcm", make_dicom(), "application/dicom")},
        data={"anatomical_region": "lumbar_spine"},
    )
    assert response.status_code == 422
    assert "SECRET" not in response.text


def test_batch_csv_keeps_failures() -> None:
    response = client.post(
        "/api/v1/analyses/batch.csv",
        files=[
            ("files", ("valid.dcm", make_dicom(), "application/dicom")),
            ("files", ("broken.dcm", b"not dicom", "application/dicom")),
        ],
        data={"anatomical_region": "lumbar_spine"},
    )
    assert response.status_code == 200
    assert "Success" not in response.text
    assert response.text.count("Failure") == 2


def test_archive_csv_processes_files_without_extracting() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("study/valid_ПОП.dcm", make_dicom())
        archive.writestr("study/broken.dcm", b"not dicom")
        archive.writestr("notes/readme.txt", "ignored")
    response = client.post(
        "/api/v1/analyses/archive.csv",
        files={"archive": ("studies.zip", buffer.getvalue(), "application/zip")},
        data={"anatomical_region": "lumbar_spine"},
    )
    assert response.status_code == 200
    assert "study/valid_ПОП.dcm" in response.text
    assert "study/broken.dcm" in response.text
    assert "Success" not in response.text
    assert response.text.count("Failure") == 2


def test_batch_keeps_independent_files_in_one_study() -> None:
    payload = make_dicom()
    response = client.post(
        "/api/v1/analyses/batch",
        files=[("files", (f"image{i}.dcm", payload, "application/dicom")) for i in range(3)],
        data={"anatomical_region": "lumbar_spine"},
    )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 3
    assert response.json()["successful"] == 0
    assert response.json()["failed"] == 3


def test_four_uploads_are_processed_without_silent_truncation() -> None:
    payload = make_dicom()
    response = client.post(
        "/api/v1/analyses/batch",
        files=[("files", (f"image{i}.dcm", payload, "application/dicom")) for i in range(4)],
    )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 4
    assert response.json()["failed"] == 4  # Синтетический кадр не даёт двух точек.


def test_batch_keeps_uploaded_order_when_one_image_is_too_large() -> None:
    old_limit = app.state.max_upload_bytes
    app.state.max_upload_bytes = 100_000
    try:
        response = client.post(
            "/api/v1/analyses/batch",
            files=[
                ("files", ("first.dcm", make_dicom(), "application/dicom")),
                ("files", ("too-large.dcm", b"x" * 100_001, "application/dicom")),
                ("files", ("last.dcm", b"invalid", "application/dicom")),
            ],
        )
    finally:
        app.state.max_upload_bytes = old_limit
    assert response.status_code == 200
    assert [item["filename"] for item in response.json()["items"]] == [
        "first.dcm", "too-large.dcm", "last.dcm"
    ]


def test_archive_json_keeps_results_for_browser() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        archive.writestr("study/broken.dcm", b"not dicom")
    response = client.post(
        "/api/v1/analyses/archive",
        files={"archive": ("study.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 200
    assert len(response.json()["items"]) == 1
    assert response.json()["failed"] == 1


def test_archive_keeps_two_hundred_failure_rows() -> None:
    buffer = BytesIO()
    with ZipFile(buffer, "w", ZIP_DEFLATED) as archive:
        for index in range(200):
            archive.writestr(f"study/case_{index:04d}.dcm", b"invalid")
    response = client.post(
        "/api/v1/analyses/archive.csv",
        files={"archive": ("studies.zip", buffer.getvalue(), "application/zip")},
    )
    assert response.status_code == 200
    assert response.text.count("Failure") == 200
    assert len(response.text.splitlines()) == 201


def test_more_than_two_hundred_uploads_are_rejected() -> None:
    response = client.post(
        "/api/v1/analyses/batch",
        files=[("files", (f"image{i}.dcm", b"invalid", "application/dicom")) for i in range(201)],
    )
    assert response.status_code == 422
    assert "200" in response.text


def test_tilt_uses_strict_five_degree_threshold() -> None:
    straight = measure_tilt([100, 10], [100, 110])
    assert straight["violation"] is False
    tilted = measure_tilt([112, 10], [100, 110])
    assert tilted["angle_deg"] > 5
    assert tilted["violation"] is True
