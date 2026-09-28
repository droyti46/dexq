"""Агрегированные QC-метрики и интервалы на уровне исследований."""

from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from zipfile import ZipFile

from app.reference_assets import REFERENCE_SHA, sha256

SOURCE = "dxa_qc_code/combined_qc/outputs/deployment_metrics.json"


def _auc(truth: list[bool], scores: list[float]) -> float | None:
    positives = sum(truth)
    negatives = len(truth) - positives
    if not positives or not negatives:
        return None
    # U-статистика с половинным вкладом совпадающих значений.
    result = sum(
        1 if scores[i] > scores[j] else 0.5 if scores[i] == scores[j] else 0
        for i, value in enumerate(truth)
        if value
        for j, other in enumerate(truth)
        if not other
    )
    return result / (positives * negatives)


def evaluate_rows(rows: list[dict]) -> dict[str, float | int | None]:
    """Вычисляет метрики по бинарным меткам без подбора порога.

    Args:
        rows: Строки с `truth`, `prediction` и необязательным `score`.

    Returns:
        Число классов, F1, macro-F1, чувствительность, специфичность и AUC.
    """
    tp = sum(row["truth"] is True and row["prediction"] is True for row in rows)
    tn = sum(row["truth"] is False and row["prediction"] is False for row in rows)
    fp = sum(row["truth"] is False and row["prediction"] is True for row in rows)
    fn = sum(row["truth"] is True and row["prediction"] is False for row in rows)
    n = tp + tn + fp + fn
    sensitivity = tp / (tp + fn) if tp + fn else None
    specificity = tn / (tn + fp) if tn + fp else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    f1_negative = 2 * tn / (2 * tn + fp + fn) if 2 * tn + fp + fn else None
    scored = [row for row in rows if row.get("score") is not None]
    auc = _auc([row["truth"] for row in scored], [float(row["score"]) for row in scored])
    return {
        "n": n,
        "positives": tp + fn,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "accuracy": (tp + tn) / n if n else None,
        "sensitivity": sensitivity,
        "specificity": specificity,
        "f1": f1,
        "macro_f1": (f1 + f1_negative) / 2 if f1 is not None and f1_negative is not None else None,
        "balanced_accuracy": (sensitivity + specificity) / 2
        if sensitivity is not None and specificity is not None
        else None,
        "roc_auc": auc,
    }


def bootstrap_ci(rows: list[dict], *, seed: int = 20260928, repeats: int = 2000) -> dict:
    """Строит 95% percentile CI с ресэмплингом целых исследований.

    Args:
        rows: Размеченные решения с `study_id`.
        seed: Фиксированный источник случайности.
        repeats: Число повторов ресэмплинга.

    Returns:
        Интервалы, числа определённых и вырожденных повторений.
    """
    by_study: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_study[row["study_id"]].append(row)
    if not by_study or repeats < 1:
        raise ValueError("Необходимы исследования и положительное число повторов")
    ids = sorted(by_study)
    rng = random.Random(seed)
    names = ("f1", "macro_f1", "sensitivity", "specificity", "balanced_accuracy", "roc_auc")
    values: dict[str, list[float]] = {key: [] for key in names}
    for _ in range(repeats):
        sampled = [row for _ in ids for row in by_study[rng.choice(ids)]]
        result = evaluate_rows(sampled)
        for name in names:
            value = result[name]
            if value is not None and math.isfinite(value):
                values[name].append(float(value))
    output = {}
    for name, measurements in values.items():
        measurements.sort()
        valid = len(measurements)
        output[name] = {
            "lower": measurements[max(0, math.floor((valid - 1) * 0.025))] if valid else None,
            "upper": measurements[min(valid - 1, math.ceil((valid - 1) * 0.975))]
            if valid
            else None,
            "valid": valid,
            "invalid": repeats - valid,
        }
    return output


def aggregate_metrics(zip_path: Path, *, seed: int = 20260928, repeats: int = 2000) -> dict:
    """Вычисляет только агрегаты по сохранённым решениям эталона.

    Args:
        zip_path: Локальный эталонный архив.
        seed: Начальное состояние bootstrap.
        repeats: Число study-level повторений.

    Returns:
        Агрегаты с явным источником и неопределёнными метриками.

    Raises:
        ValueError: При несовпадении эталонного архива.
    """
    if sha256(zip_path) != REFERENCE_SHA:
        raise ValueError("SHA-256 эталонного архива не совпадает")
    with ZipFile(zip_path) as archive:
        report = json.loads(archive.read(SOURCE))
    entries = report["per_image"]
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in entries:
        region = row["expected_region"]
        truth = row["truth"]
        if truth["any_violation"] is not None and row["any_violation"] is not None:
            item = {
                "study_id": row["study_id"],
                "truth": truth["any_violation"],
                "prediction": row["any_violation"],
            }
            groups["overall_or"].append(item)
            groups[f"{region}_or"].append(item)
        for name, label in truth["labels"].items():
            prediction = row["labels"].get(name)
            if label is None or prediction is None:
                continue
            groups[name].append(
                {
                    "study_id": row["study_id"],
                    "truth": label,
                    "prediction": prediction,
                    "score": row["scores"].get(name),
                }
            )
    return {
        "source": "эталонный полный QCPipeline.infer; обучение/development, не независимая оценка",
        "n_processed": len(entries),
        "n_labeled": len(groups["overall_or"]),
        "seed": seed,
        "repeats": repeats,
        "cluster": "study_id",
        "groups": {
            name: {
                "metrics": evaluate_rows(rows),
                "ci95": bootstrap_ci(rows, seed=seed, repeats=repeats),
            }
            for name, rows in sorted(groups.items())
        },
    }


def main() -> None:
    """Печатает только суммарные метрики без строк пациентов."""
    parser = argparse.ArgumentParser(description="Метрики и интервалы DXA-QC")
    parser.add_argument("archive", type=Path)
    parser.add_argument("--repeats", type=int, default=2000)
    args = parser.parse_args()
    print(
        json.dumps(
            aggregate_metrics(args.archive, repeats=args.repeats), ensure_ascii=False, indent=2
        )
    )


if __name__ == "__main__":
    main()
