"""Проверки офлайн-запуска без обращения к Docker daemon и медицинским данным."""

from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

from app.reference_assets import REFERENCE_SHA, verify_models

ROOT = Path(__file__).resolve().parents[2]


def _models(root: Path, *, source_sha: str = REFERENCE_SHA) -> Path:
    models = root / "models"
    file = models / "router/checkpoints/runtime/tiny.onnx"
    file.parent.mkdir(parents=True)
    file.write_bytes(b"fixture, not a real model")
    digest = hashlib.sha256(file.read_bytes()).hexdigest()
    (models / "installed-manifest.json").write_text(
        json.dumps(
            {
                "reference_sha256": source_sha,
                "files": {
                    "router/checkpoints/runtime/tiny.onnx": digest
                },
            }
        ),
        encoding="utf-8",
    )
    return models


def _run(tmp_path: Path, model_dir: Path) -> subprocess.CompletedProcess[str]:
    binary = tmp_path / "bin"
    binary.mkdir(exist_ok=True)
    docker = binary / "docker"
    docker.write_text('#!/bin/sh\nprintf "%s\\n" "$*" > "$DEXQ_DOCKER_CALLS"\n', encoding="utf-8")
    docker.chmod(0o755)
    calls = tmp_path / "docker.calls"
    return subprocess.run(
        ["sh", str(ROOT / "scripts/run.sh")],
        capture_output=True,
        text=True,
        cwd=tmp_path,
        env={
            **os.environ,
            "PATH": str(binary) + os.pathsep + os.environ["PATH"],
            "PYTHON": sys.executable,
            "DEXQ_MODELS_DIR": str(model_dir),
            "DEXQ_DOCKER_CALLS": str(calls),
        },
        check=False,
    )


def test_missing_models_stop_before_docker(tmp_path: Path) -> None:
    result = _run(tmp_path, tmp_path / "absent")
    assert result.returncode != 0
    assert not (tmp_path / "docker.calls").exists()


def test_corrupt_model_stops_before_docker(tmp_path: Path) -> None:
    models = _models(tmp_path)
    (models / "router/checkpoints/runtime/tiny.onnx").write_bytes(b"tampered")
    result = _run(tmp_path, models)
    assert result.returncode != 0
    assert not (tmp_path / "docker.calls").exists()


def test_foreign_model_bundle_is_not_accepted(tmp_path: Path) -> None:
    models = _models(tmp_path, source_sha="0" * 64)
    with pytest.raises(ValueError, match="reference"):
        verify_models(models)
    result = _run(tmp_path, models)
    assert result.returncode != 0
    assert not (tmp_path / "docker.calls").exists()


def test_real_manifest_cannot_omit_a_model(tmp_path: Path) -> None:
    original = ROOT / "backend/models/reference/installed-manifest.json"
    if not original.is_file():
        pytest.skip("Локальный комплект эталона недоступен")
    manifest = json.loads(original.read_text(encoding="utf-8"))
    manifest["files"].pop(next(iter(manifest["files"])))
    folder = tmp_path / "reference"
    folder.mkdir()
    (folder / "installed-manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(ValueError, match="inventory"):
        verify_models(folder)


def test_verified_bundle_has_read_only_mount(tmp_path: Path) -> None:
    source = ROOT / "backend/models/reference"
    if not source.is_dir():
        pytest.skip("Локальный комплект эталона недоступен")
    result = _run(tmp_path, source)
    assert result.returncode == 0, result.stderr
    assert (tmp_path / "docker.calls").read_text(encoding="utf-8").strip() == "compose up --build"
    compose = (ROOT / "docker-compose.yml").read_text(encoding="utf-8")
    assert "${DEXQ_MODELS_DIR}:/models/reference:ro" in compose
    assert "DEXQ_MODELS_DIR: /models/reference" in compose
    assert "reference_sha256" in (source / "installed-manifest.json").read_text(encoding="utf-8")
