"""Проверка локальной поставки эталонного runtime без медицинских данных."""

import hashlib
import stat
from pathlib import Path
from zipfile import ZipFile, ZipInfo

import pytest

from app.reference_assets import prepare_reference, verify_models


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _archive(path: Path, entries: list[tuple[str, bytes]]) -> None:
    with ZipFile(path, "w") as archive:
        for name, payload in entries:
            archive.writestr(name, payload)


def test_rejects_mismatched_archive_digest(tmp_path: Path) -> None:
    path = tmp_path / "input.zip"
    _archive(path, [("dxa_qc_code/combined_qc/common.py", b"pass\n")])
    with pytest.raises(ValueError, match="SHA-256"):
        prepare_reference(path, tmp_path / "source", tmp_path / "models", expected_zip_sha="0" * 64)
    assert not (tmp_path / "source").exists()


def test_ignores_unrelated_archive_entries(tmp_path: Path) -> None:
    path = tmp_path / "input.zip"
    _archive(
        path,
        [
            ("unrelated/README.md", b"ignored"),
            ("dxa_qc_code/combined_qc/common.py", b"# source\n"),
            ("dxa_qc_code/combined_qc/router/checkpoints/runtime/model.onnx", b"weights"),
        ],
    )
    prepare_reference(path, tmp_path / "source", tmp_path / "models", expected_zip_sha=_sha(path))
    assert (tmp_path / "source/combined_qc/common.py").read_bytes() == b"# source\n"


def test_rejects_traversal(tmp_path: Path) -> None:
    path = tmp_path / "input.zip"
    _archive(path, [("dxa_qc_code/combined_qc/../escape.py", b"bad")])
    with pytest.raises(ValueError, match="path"):
        prepare_reference(
            path, tmp_path / "source", tmp_path / "models", expected_zip_sha=_sha(path)
        )
    assert not (tmp_path / "escape.py").exists()


def test_rejects_duplicate_member(tmp_path: Path) -> None:
    path = tmp_path / "input.zip"
    with ZipFile(path, "w") as archive:
        archive.writestr("dxa_qc_code/combined_qc/common.py", b"first")
        archive.writestr("dxa_qc_code/combined_qc/common.py", b"second")
    with pytest.raises(ValueError, match="duplicate"):
        prepare_reference(
            path, tmp_path / "source", tmp_path / "models", expected_zip_sha=_sha(path)
        )


def test_rejects_symlink_member(tmp_path: Path) -> None:
    path = tmp_path / "input.zip"
    member = ZipInfo("dxa_qc_code/combined_qc/common.py")
    member.create_system = 3
    member.external_attr = (stat.S_IFLNK | 0o777) << 16
    with ZipFile(path, "w") as archive:
        archive.writestr(member, b"outside")
    with pytest.raises(ValueError, match="symlink"):
        prepare_reference(
            path, tmp_path / "source", tmp_path / "models", expected_zip_sha=_sha(path)
        )


def test_copies_source_and_verifies_model_digest(tmp_path: Path) -> None:
    path = tmp_path / "input.zip"
    model = b"tiny-model"
    source = b"# original bytes\r\n"
    model_name = "dxa_qc_code/combined_qc/router/checkpoints/runtime/resnet18_features.onnx"
    _archive(
        path,
        [
            ("dxa_qc_code/combined_qc/common.py", source),
            (model_name, model),
        ],
    )
    models = tmp_path / "models"
    provenance = prepare_reference(path, tmp_path / "source", models, expected_zip_sha=_sha(path))
    assert (tmp_path / "source/combined_qc/common.py").read_bytes() == source
    installed = models / "router/checkpoints/runtime/resnet18_features.onnx"
    assert installed.read_bytes() == model
    assert (
        provenance["router/checkpoints/runtime/resnet18_features.onnx"]
        == hashlib.sha256(model).hexdigest()
    )
    verify_models(models)
    installed.write_bytes(b"tampered")
    with pytest.raises(ValueError, match="SHA-256"):
        verify_models(models)
