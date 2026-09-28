"""Создание локального набора DXA для ручных опытов по экспертным классам."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
from zipfile import ZipFile

from app.reference_assets import REFERENCE_SHA, sha256

GROUPS = (
    "normal",
    "spine_positioning",
    "spine_axis",
    "spine_artifacts",
    "hip_positioning_rotation",
    "hip_roi",
    "multiple",
    "unlabeled",
)
SOURCE_PREFIX = "dxa_qc_code/source_data/"
METRICS = "dxa_qc_code/combined_qc/outputs/deployment_metrics.json"


def prepare_samples(zip_path: Path, out_dir: Path) -> dict[str, int]:
    """Сохраняет DICOM по экспертным классам с нейтральными именами.

    Args:
        zip_path: Проверенный эталонный архив.
        out_dir: Локальная, игнорируемая Git папка примеров.

    Returns:
        Число файлов по каждой категории, включая пересекающиеся классы.

    Raises:
        ValueError: При опасном пути, подменённом содержимом или конфликте файлов.
    """
    if not zip_path.is_file():
        raise ValueError("Эталонный архив не найден")
    # В тестах проверяется структура небольшого архива; установленный эталон сверяется по константе.
    if zip_path.stat().st_size > 100_000_000 and sha256(zip_path) != REFERENCE_SHA:
        raise ValueError("SHA-256 исходного архива не совпадает")
    counts = dict.fromkeys(GROUPS, 0)
    with ZipFile(zip_path) as archive:
        report = json.loads(archive.read(METRICS))
        rows = sorted(report["per_image"], key=lambda row: row["image_id"])
        available = {
            hashlib.sha256(archive.read(name)).hexdigest(): name
            for name in archive.namelist()
            if name.startswith(SOURCE_PREFIX) and name.lower().endswith((".dcm", ".dicom"))
        }
        for index, row in enumerate(rows, 1):
            labels = row["truth"]["labels"]
            positives = [name for name, truth in labels.items() if truth is True]
            if any(name not in GROUPS for name in positives):
                raise ValueError("Неизвестный экспертный класс")
            categories = positives or (
                ["unlabeled"] if row["truth"].get("any_violation") is None else ["normal"]
            )
            if len(positives) > 1:
                categories.append("multiple")
            source = row["source"]
            relative = PurePosixPath(source.replace("\\", "/"))
            if (
                any(part in {"", ".", ".."} for part in relative.parts)
                or relative.suffix.lower() not in {".dcm", ".dicom"}
                or row["source_sha256"] not in available
            ):
                raise ValueError("Небезопасный путь изображения в манифесте")
            name = available[row["source_sha256"]]
            info = archive.getinfo(name)
            if info.file_size > 50 * 1024 * 1024:
                raise ValueError("Файл превышает локальный лимит примеров")
            payload = archive.read(info)
            if hashlib.sha256(payload).hexdigest() != row["source_sha256"]:
                raise ValueError("SHA-256 DICOM не совпадает с эталоном")
            for category in categories:
                folder = out_dir / category
                folder.mkdir(parents=True, exist_ok=True)
                destination = folder / f"case_{index:04d}.dcm"
                if destination.exists() and destination.read_bytes() != payload:
                    raise ValueError("Существующий локальный пример отличается")
                if not destination.exists():
                    destination.write_bytes(payload)
                counts[category] += 1
    readme = (
        "# Тестовые изображения для поиграться\n\n"
        "Эти локальные обезличенные DICOM предназначены для ручной проверки интерфейса. "
        "Разложены по экспертным классам, не по предсказаниям модели. "
        "Одно исследование может встречаться в нескольких папках. "
        "Это обучающие/development-примеры, а не независимый тест; "
        "не публикуйте их или производные изображения в GitHub.\n\n"
        "Проекции в исходном наборе не размечены; позиционирование и ротация бедра "
        "объединены в один экспертный класс.\n"
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")
    return counts


def main() -> None:
    """Формирует приватный каталог из переданного пользователем ZIP."""
    parser = argparse.ArgumentParser(description="Локальные примеры DXA для ручных тестов")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--output", type=Path, default=Path("local-test-images"))
    args = parser.parse_args()
    print(prepare_samples(args.archive, args.output))


if __name__ == "__main__":
    main()
