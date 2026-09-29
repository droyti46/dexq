"""Комплект передачи не должен содержать медицинские данные из истории Git."""

from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from zipfile import ZipFile

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))
from package_submission import package_submission  # noqa: E402


def test_submission_contains_only_source_models_and_local_images(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "source"
    (root / "backend/app").mkdir(parents=True)
    (root / "data").mkdir()
    (root / "backend/app/main.py").write_text("# public", encoding="utf-8")
    (root / "data/patient_SECRET.dcm").write_bytes(b"DICOM")
    (root / "backend/models").mkdir()
    (root / "backend/models/secret.onnx").write_bytes(b"wrong directory")
    models = tmp_path / "models"
    models.mkdir()
    file = models / "router/checkpoints/runtime/model.onnx"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"private weights")
    (models / "installed-manifest.json").write_text(
        json.dumps(
            {
                "files": {
                    "router/checkpoints/runtime/model.onnx": hashlib.sha256(
                        file.read_bytes()
                    ).hexdigest()
                }
            }
        ),
        encoding="utf-8",
    )
    images = tmp_path / "images.tar"
    images.write_bytes(b"local image export")
    output = tmp_path / "submission.zip"
    monkeypatch.setattr("package_submission.verify_models", lambda _: None)
    package_submission(output, root, models, images, source_paths=[Path("backend/app/main.py")])
    with ZipFile(output) as archive:
        assert archive.namelist() == [
            "source/backend/app/main.py",
            "models/reference/router/checkpoints/runtime/model.onnx",
            "models/reference/installed-manifest.json",
            "images/dexq-images.tar",
        ]
        assert (
            archive.read("models/reference/router/checkpoints/runtime/model.onnx")
            == b"private weights"
        )
        assert "patient_SECRET" not in " ".join(archive.namelist())


def test_submission_requires_offline_image_export(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="образ"):
        package_submission(
            tmp_path / "out.zip", tmp_path, tmp_path, tmp_path / "missing.tar", source_paths=[]
        )
    assert not (tmp_path / "out.zip").exists()
