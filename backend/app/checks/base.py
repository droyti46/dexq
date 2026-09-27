"""Общий контракт модулей контроля качества."""

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from app.schemas import AnatomicalRegion, CheckResult


@dataclass(frozen=True, slots=True)
class AnalysisContext:
    """Входные данные, общие для всех проверок."""

    pixels: np.ndarray
    region: AnatomicalRegion


class QualityCheck(Protocol):
    """Минимальный интерфейс независимой проверки."""

    check_id: str
    title: str
    regions: tuple[AnatomicalRegion, ...]
    model_status: str

    def run(self, context: AnalysisContext) -> CheckResult:
        """Выполняет проверку для подходящей анатомической области."""
        ...

