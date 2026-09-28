"""Локальный end-to-end замер DXA без публикации снимков и UID."""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path

from app.analyzer import Analyzer
from app.batch import analyze_items, render_csv
from app.imaging import safe_dicom_fields
from app.model_runtime import ReferenceRuntime
from app.schemas import AnatomicalRegion


def benchmark_studies(
    studies: dict[str, list[Path]],
    analyzer: Analyzer,
    *,
    limit_seconds: float = 180,
    clock: Callable[[], float] = time.perf_counter,
) -> dict[str, int | float | bool | str]:
    """Измеряет чтение, декодирование, инференс, сборку DTO и CSV по исследованиям.

    Args:
        studies: Приватные списки файлов, сгруппированные по Study UID.
        analyzer: Разделяемый прогретый анализатор эталонной модели.
        limit_seconds: Максимальное время одного исследования.
        clock: Монотонный источник времени для воспроизводимых тестов.

    Returns:
        Только агрегаты и предел скорости, без UID, имён и изображений.

    Raises:
        ValueError: Если исследование пусто или содержит более трёх кадров.
    """
    if not studies or any(not 1 <= len(paths) <= 3 for paths in studies.values()):
        raise ValueError("Каждое исследование должно содержать от одного до трёх изображений")
    durations = []
    total = successes = 0
    for paths in studies.values():
        started = clock()
        items = [(file.name, file.read_bytes()) for file in paths]
        batch = analyze_items(analyzer, items, AnatomicalRegion.AUTO)
        render_csv(batch)
        durations.append(clock() - started)
        total += len(paths)
        successes += batch.successful
    return {
        "studies": len(studies),
        "images": total,
        "successful": successes,
        "failed": total - successes,
        "success_fraction": successes / total,
        "max_study_seconds": round(max(durations), 4),
        "mean_study_seconds": round(sum(durations) / len(durations), 4),
        "limit_seconds": limit_seconds,
        "within_limit": all(duration <= limit_seconds for duration in durations),
        "boundary": "disk read + DICOM decode + reference inference + DTO + CSV; warmed pipeline",
    }


def main() -> None:
    """Измеряет приватный набор и сохраняет агрегаты вне Git."""
    parser = argparse.ArgumentParser(description="Локальный benchmark DXA-QC")
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--models", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("local-test-images/benchmark.json"))
    args = parser.parse_args()
    if not args.source.is_dir():
        parser.error("Не найдена приватная директория снимков")
    source = args.source.resolve()
    if not args.output.resolve().is_relative_to(
        source.parent
    ) and not args.output.resolve().is_relative_to(
        (Path(__file__).resolve().parents[1] / "local-test-images").resolve()
    ):
        parser.error("Отчёт допускается сохранять только рядом с приватными снимками")
    groups: dict[str, list[Path]] = defaultdict(list)
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in {".dcm", ".dicom"}:
            continue
        uid, _, _ = safe_dicom_fields(path.read_bytes(), path.suffix.lower())
        groups[uid or f"unknown-{len(groups)}"].append(path)
    runtime = ReferenceRuntime(args.models)
    started = time.perf_counter()
    if not runtime.ready():
        parser.error("Локальная модель не загрузилась")
    cold_load = time.perf_counter() - started
    result = benchmark_studies(groups, Analyzer(runtime))
    result["cold_load_seconds"] = round(cold_load, 4)
    result["platform"] = platform.platform()
    result["python"] = sys.version.split()[0]
    result["provider"] = "CPUExecutionProvider"
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["within_limit"] or result["failed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
