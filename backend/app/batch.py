"""Пакетная обработка и табличный формат результатов."""

from __future__ import annotations

import csv
import stat
from io import BytesIO, StringIO
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

from app.analyzer import Analyzer
from app.schemas import AnatomicalRegion, BatchItem, BatchResult

SUPPORTED_SUFFIXES = {".dcm", ".dicom", ".png", ".jpg", ".jpeg"}
MAX_ARCHIVE_FILES = 200
MAX_UNCOMPRESSED_BYTES = 2 * 1024 * 1024 * 1024


def read_archive(content: bytes, max_file_bytes: int) -> list[tuple[str, bytes]]:
    """Читает подходящие файлы ZIP в памяти с защитой от zip bomb.

    Архив никогда не распаковывается на диск, поэтому пути внутри него не могут
    выйти за рабочую директорию приложения.

    Args:
        content: Полное содержимое ZIP-архива.
        max_file_bytes: Максимальный допустимый размер одного файла после распаковки.

    Returns:
        Список пар «путь внутри архива — содержимое».

    Raises:
        ValueError: Если архив повреждён, зашифрован или превышает ограничения.
    """
    try:
        with ZipFile(BytesIO(content)) as archive:
            candidates = [
                info
                for info in archive.infolist()
                if not info.is_dir()
                and PurePosixPath(info.filename.replace("\\", "/")).suffix.lower()
                in SUPPORTED_SUFFIXES
            ]
            if not candidates:
                raise ValueError("В архиве нет поддерживаемых медицинских изображений")
            if len(candidates) > MAX_ARCHIVE_FILES:
                raise ValueError(f"В архиве больше {MAX_ARCHIVE_FILES} изображений")
            seen: set[str] = set()
            for info in candidates:
                raw = info.filename
                parts = raw.split("/")
                if (
                    "\\" in raw
                    or ":" in raw
                    or raw.startswith("/")
                    or any(
                        part in {"", ".", ".."}
                        or part.split(".")[0].upper()
                        in {
                            "CON",
                            "PRN",
                            "AUX",
                            "NUL",
                            *(f"COM{i}" for i in range(1, 10)),
                            *(f"LPT{i}" for i in range(1, 10)),
                        }
                        for part in parts
                    )
                    or info.create_system == 3
                    and stat.S_IFMT(info.external_attr >> 16) == stat.S_IFLNK
                ):
                    raise ValueError("Небезопасный путь внутри ZIP-архива")
                folded = raw.casefold()
                if folded in seen:
                    raise ValueError("повтор имени изображения в ZIP-архиве")
                seen.add(folded)
            if any(info.flag_bits & 0x1 for info in candidates):
                raise ValueError("Зашифрованные ZIP-архивы не поддерживаются")
            if any(info.file_size > max_file_bytes for info in candidates):
                raise ValueError(
                    f"Файл внутри архива превышает {max_file_bytes // (1024 * 1024)} МБ"
                )
            if sum(info.file_size for info in candidates) > MAX_UNCOMPRESSED_BYTES:
                raise ValueError("Распакованный архив превышает ограничение 2 ГБ")
            return [(info.filename, archive.read(info)) for info in candidates]
    except BadZipFile as error:
        raise ValueError("Не удалось прочитать ZIP-архив") from error


def analyze_items(
    analyzer: Analyzer,
    items: list[tuple[str, bytes]],
    region: AnatomicalRegion,
) -> BatchResult:
    """Обрабатывает элементы независимо и сохраняет ошибки в отчёте.

    Args:
        analyzer: Настроенный оркестратор проверок.
        items: Имена и байты входных изображений.
        region: Автоопределение или выбор оператора для всего пакета.

    Returns:
        Пакетный результат с количеством успешных и ошибочных файлов.
    """
    results: list[BatchItem] = []
    for filename, content in items:
        try:
            result = analyzer.analyze(content, filename, region)
            if result.processing_status == "Failure":
                results.append(
                    BatchItem(
                        filename=filename, result=result, error=result.error or "Анализ не завершён"
                    )
                )
            else:
                results.append(BatchItem(filename=filename, result=result))
        except (ValueError, OSError, RuntimeError):
            results.append(
                BatchItem(filename=filename, error="Не удалось безопасно обработать файл")
            )
    successful = sum(
        item.result is not None and item.result.processing_status == "Success" for item in results
    )
    return BatchResult(items=results, successful=successful, failed=len(results) - successful)


def render_csv(batch: BatchResult) -> str:
    """Формирует CSV с колонками, заданными организатором.

    Args:
        batch: Результат пакетной обработки.

    Returns:
        CSV-текст с BOM для корректного открытия в Excel.
    """
    output = StringIO(newline="")
    fieldnames = [
        "path_to_study",
        "study_uid",
        "image_uid",
        "anatomical_region",
        "quality_class",
        "violation_type",
        "processing_status",
        "time_of_processing",
    ]
    writer = csv.DictWriter(output, fieldnames=fieldnames, lineterminator="\n")
    writer.writeheader()
    for item in batch.items:
        result = item.result
        writer.writerow(
            {
                "path_to_study": item.filename,
                "study_uid": result.study_uid if result else "",
                "image_uid": result.image_uid if result else "",
                "anatomical_region": result.anatomical_region if result else "unknown",
                "quality_class": result.quality_class
                if result and result.processing_status == "Success"
                else "",
                "violation_type": ";".join(sorted(result.violation_types))
                if result and result.processing_status == "Success"
                else "",
                "processing_status": result.processing_status if result else "Failure",
                "time_of_processing": result.time_of_processing if result else 0,
            }
        )
    return "\ufeff" + output.getvalue()


def collect_local_files(path: Path, max_file_bytes: int) -> list[tuple[str, bytes]]:
    """Собирает изображения из локальной директории или ZIP для CLI.

    Args:
        path: ZIP-файл, отдельное изображение или директория.
        max_file_bytes: Ограничение размера одного изображения.

    Returns:
        Список входов в стабильном лексикографическом порядке.

    Raises:
        ValueError: Если путь или входные файлы некорректны.
    """
    path = path.resolve()
    if not path.exists():
        raise ValueError(f"Путь не существует: {path}")
    if path.is_file() and path.suffix.lower() == ".zip":
        return read_archive(path.read_bytes(), max_file_bytes)
    if path.is_file():
        if path.suffix.lower() not in SUPPORTED_SUFFIXES:
            raise ValueError("Неподдерживаемый формат входного файла")
        files = [path]
    else:
        files = sorted(
            item
            for item in path.rglob("*")
            if item.is_file() and item.suffix.lower() in SUPPORTED_SUFFIXES
        )
    if not files:
        raise ValueError("Поддерживаемые изображения не найдены")
    if len(files) > MAX_ARCHIVE_FILES:
        raise ValueError(f"Найдено больше {MAX_ARCHIVE_FILES} изображений")
    oversized = next((item for item in files if item.stat().st_size > max_file_bytes), None)
    if oversized:
        raise ValueError(f"Файл превышает ограничение: {oversized}")
    root = path if path.is_dir() else path.parent
    return [(item.relative_to(root).as_posix(), item.read_bytes()) for item in files]
