"""Локальная установка неизменённого inference-кода и проверенных весов DEXQ."""

from __future__ import annotations

import argparse
import hashlib
import json
import stat
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile, ZipInfo

REFERENCE_SHA = "2cd75f3777a4a7735d8e36cca371d4637b54053fb9283863c64c7214535a8867"
PREFIX = "dxa_qc_code/combined_qc/"
MODULES = ("router", "hip", "spine")
MAX_MEMBER_BYTES = 1024 * 1024 * 1024
MAX_RUNTIME_BYTES = 2 * 1024 * 1024 * 1024


def sha256(path: Path) -> str:
    """Считает контрольную сумму файла без загрузки целиком в память.

    Args:
        path: Путь к файлу.

    Returns:
        SHA-256 в шестнадцатеричной записи.
    """
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _member_path(info: ZipInfo) -> PurePosixPath:
    name = info.filename
    parts = name.split("/")
    if (
        "\\" in name
        or "\x00" in name
        or ":" in name
        or any(part in {"", ".", ".."} for part in parts)
        or PurePosixPath(name).is_absolute()
    ):
        raise ValueError("Unsafe archive path")
    if info.create_system == 3 and stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK:
        raise ValueError("Archive symlink is not supported")
    if info.file_size > MAX_MEMBER_BYTES:
        raise ValueError("Archive member exceeds size limit")
    return PurePosixPath(name[len(PREFIX) :]) if name.startswith(PREFIX) else PurePosixPath(name)


def _destination(relative: PurePosixPath, source_root: Path, models_root: Path) -> Path | None:
    parts = relative.parts
    if relative.suffix == ".py" and "checkpoints" not in parts and "data" not in parts:
        return source_root.joinpath("combined_qc", *parts)
    if (
        len(parts) >= 4
        and parts[0] in MODULES
        and parts[1:3] == ("checkpoints", "runtime")
        and relative.suffix in {".onnx", ".npz", ".json"}
    ):
        return models_root.joinpath(*parts)
    return None


def prepare_reference(
    zip_path: Path,
    source_root: Path,
    models_root: Path,
    *,
    expected_zip_sha: str = REFERENCE_SHA,
) -> dict[str, str]:
    """Устанавливает только исходный inference-код и runtime-веса из эталона.

    Args:
        zip_path: Исходный архив эталонного решения.
        source_root: Корень локального Python-пакета.
        models_root: Корень весов вне отслеживаемого Git.
        expected_zip_sha: Ожидаемый SHA-256 полного архива.

    Returns:
        Контрольные суммы установленных файлов моделей.

    Raises:
        ValueError: Если архив подменён, небезопасен или установка неполна.
    """
    if sha256(zip_path) != expected_zip_sha:
        raise ValueError("Reference archive SHA-256 mismatch")
    selected: list[tuple[ZipInfo, Path, str]] = []
    seen: set[str] = set()
    total_size = 0
    try:
        with ZipFile(zip_path) as archive:
            for info in archive.infolist():
                if info.is_dir():
                    continue
                relative = _member_path(info)
                if not info.filename.startswith(PREFIX):
                    continue
                normalized = str(relative).casefold()
                if normalized in seen:
                    raise ValueError("Archive contains duplicate path")
                seen.add(normalized)
                destination = _destination(relative, source_root, models_root)
                if destination is None:
                    continue
                total_size += info.file_size
                if total_size > MAX_RUNTIME_BYTES:
                    raise ValueError("Reference runtime exceeds size limit")
                selected.append((info, destination, relative.as_posix()))
            if not selected:
                raise ValueError("Reference archive has no inference files")
            manifest: dict[str, str] = {}
            for info, destination, relative in selected:
                payload = archive.read(info)
                digest = hashlib.sha256(payload).hexdigest()
                if destination.exists():
                    if sha256(destination) != digest:
                        raise ValueError(f"Existing file SHA-256 mismatch: {relative}")
                else:
                    destination.parent.mkdir(parents=True, exist_ok=True)
                    destination.write_bytes(payload)
                if relative.split("/", 1)[0] in MODULES and "/checkpoints/runtime/" in relative:
                    manifest[relative] = digest
    except BadZipFile as error:
        raise ValueError("Invalid reference ZIP") from error
    if not manifest:
        raise ValueError("Reference archive has no runtime models")
    models_root.mkdir(parents=True, exist_ok=True)
    manifest_path = models_root / "installed-manifest.json"
    encoded = json.dumps({"reference_sha256": expected_zip_sha, "files": manifest}, indent=2)
    if manifest_path.exists() and manifest_path.read_text(encoding="utf-8") != encoded:
        raise ValueError("Existing model manifest differs from reference")
    manifest_path.write_text(encoded, encoding="utf-8")
    return manifest


def verify_models(models_root: Path) -> None:
    """Проверяет каждый установленный runtime-файл до начала инференса.

    Args:
        models_root: Корень локально установленных весов.

    Raises:
        ValueError: Если манифест отсутствует, повреждён или файл подменён.
    """
    manifest_path = models_root / "installed-manifest.json"
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ValueError("Model manifest is unavailable") from error
    entries = manifest.get("files")
    if not isinstance(entries, dict) or not entries:
        raise ValueError("Model manifest has no files")
    for name, digest in entries.items():
        relative = PurePosixPath(name)
        if (
            not isinstance(name, str)
            or not isinstance(digest, str)
            or len(digest) != 64
            or relative.is_absolute()
            or any(part in {"", ".", ".."} for part in relative.parts)
        ):
            raise ValueError("Unsafe model manifest path")
        file = models_root.joinpath(*relative.parts)
        if not file.is_file() or sha256(file) != digest:
            raise ValueError(f"Model SHA-256 mismatch: {name}")


def main() -> None:
    """Устанавливает проверенный эталон из явно переданного локального ZIP."""
    parser = argparse.ArgumentParser(description="Локальная установка эталонного DXA-QC")
    parser.add_argument("archive", type=Path)
    parser.add_argument(
        "--source", type=Path, default=Path(__file__).resolve().parents[1] / "reference"
    )
    parser.add_argument(
        "--models", type=Path, default=Path(__file__).resolve().parents[1] / "models" / "reference"
    )
    options = parser.parse_args()
    installed = prepare_reference(options.archive, options.source, options.models)
    verify_models(options.models)
    print(f"Проверено {len(installed)} файлов runtime; веса остаются локальными.")


if __name__ == "__main__":
    main()
