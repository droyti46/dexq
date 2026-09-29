"""Локальная сборка проверенной поставки без DICOM и сетевой отправки."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from zipfile import ZIP_DEFLATED, ZIP_STORED, ZipFile

from app.reference_assets import verify_models


def package_submission(
    output: Path,
    root: Path,
    models: Path,
    images: Path,
    *,
    source_paths: list[Path] | None = None,
) -> None:
    """Создаёт локальный ZIP из разрешённого исходного кода, весов и готовых образов.

    Args:
        output: Путь под игнорируемым каталогом `submission/`.
        root: Корень проекта.
        models: Проверенный каталог ONNX/NPZ.
        images: Ранее созданный локальный `docker save` с двумя образами.
        source_paths: Белый список файлов для тестов; иначе берётся `git ls-files`.

    Raises:
        ValueError: Если образы или веса отсутствуют, список небезопасен.
    """
    if not images.is_file() or images.stat().st_size == 0:
        raise ValueError("Локальный экспорт Docker-образов отсутствует")
    verify_models(models)
    manifest_path = models / "installed-manifest.json"
    entries = json.loads(manifest_path.read_text(encoding="utf-8"))["files"]
    if source_paths is None:
        result = subprocess.run(
            ["git", "ls-files", "-z"], cwd=root, capture_output=True, check=True
        )
        source_paths = [Path(name.decode("utf-8")) for name in result.stdout.split(b"\0") if name]
    excluded = {"data", "submission", "local-test-images", ".venv", "node_modules"}
    extensions = {
        ".dcm",
        ".dicom",
        ".png",
        ".jpg",
        ".jpeg",
        ".onnx",
        ".npz",
        ".pt",
        ".pth",
        ".zip",
        ".env",
    }
    safe = []
    for relative in source_paths:
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or any(part in excluded for part in relative.parts)
            or relative.suffix.lower() in extensions
        ):
            continue
        source = root / relative
        if not source.is_file():
            raise ValueError("Отсутствует файл отслеживаемого исходного кода")
        safe.append(relative)
    if not safe:
        raise ValueError("Не найден безопасный исходный код для комплекта")
    if output.exists():
        raise ValueError("Файл поставки уже существует")
    output.parent.mkdir(parents=True, exist_ok=True)
    try:
        with ZipFile(output, "x") as archive:
            for relative in sorted(safe, key=lambda path: path.as_posix()):
                archive.write(
                    root / relative, f"source/{relative.as_posix()}", compress_type=ZIP_DEFLATED
                )
            for name, digest in sorted(entries.items()):
                file = models / name
                if hashlib.sha256(file.read_bytes()).hexdigest() != digest:
                    raise ValueError("Контрольная сумма файла модели изменилась")
                archive.write(file, f"models/reference/{name}", compress_type=ZIP_STORED)
            archive.write(manifest_path, "models/reference/installed-manifest.json")
            archive.write(images, "images/dexq-images.tar", compress_type=ZIP_STORED)
    except BaseException:
        output.unlink(missing_ok=True)
        raise


def main() -> None:
    """Собирает локальную поставку после сохранения проверенных Docker-образов."""
    parser = argparse.ArgumentParser(description="Локальный комплект DXA-QC без медицинских данных")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--images", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.output.resolve().is_relative_to((root / "submission").resolve()):
        parser.error("Комплект допускается создавать только внутри игнорируемой submission/")
    package_submission(args.output, root, args.models, args.images)
    print("Локальный комплект создан; не загружайте его в GitHub или внешние сервисы.")


if __name__ == "__main__":
    main()
