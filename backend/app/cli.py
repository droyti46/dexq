"""Командный интерфейс пакетного локального инференса."""

import argparse
from pathlib import Path

from app.analyzer import Analyzer
from app.batch import analyze_items, collect_local_files, render_csv
from app.schemas import AnatomicalRegion


def main() -> None:
    """Обрабатывает файл, директорию или ZIP и записывает итоговый CSV."""
    parser = argparse.ArgumentParser(description="Пакетный контроль качества DXA")
    parser.add_argument("input", type=Path, help="DICOM, директория или ZIP-архив")
    parser.add_argument("output", type=Path, help="Путь итогового CSV")
    parser.add_argument(
        "--region",
        choices=["auto", "lumbar_spine", "proximal_femur"],
        default="auto",
        help="Анатомическая область; по умолчанию определяется автоматически",
    )
    parser.add_argument(
        "--max-file-mb",
        type=int,
        default=50,
        help="Максимальный размер одного изображения",
    )
    arguments = parser.parse_args()
    try:
        items = collect_local_files(arguments.input, arguments.max_file_mb * 1024 * 1024)
        batch = analyze_items(Analyzer(), items, AnatomicalRegion(arguments.region))
    except ValueError as error:
        parser.error(str(error))
    arguments.output.parent.mkdir(parents=True, exist_ok=True)
    arguments.output.write_text(render_csv(batch), encoding="utf-8", newline="")
    print(f"Обработано: {batch.successful}; ошибок: {batch.failed}; CSV: {arguments.output}")


if __name__ == "__main__":
    main()
