"""Публичные модели API DEXQ."""

from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class AnatomicalRegion(StrEnum):
    """Поддерживаемые анатомические области."""

    AUTO = "auto"
    LUMBAR_SPINE = "lumbar_spine"
    PROXIMAL_FEMUR = "proximal_femur"
    UNKNOWN = "unknown"


class CheckStatus(StrEnum):
    """Состояние отдельной проверки качества."""

    PASSED = "passed"
    FAILED = "failed"
    NOT_EVALUATED = "not_evaluated"
    ERROR = "error"


class CheckResult(BaseModel):
    """Нейтральный результат одного модуля проверки."""

    check_id: str
    title: str
    status: CheckStatus
    violation: bool | None
    summary: str
    confidence: float | None = Field(default=None, ge=0, le=1)
    details: dict[str, Any] = Field(default_factory=dict)
    method: str
    model_status: str = "prototype"


class AnalysisResult(BaseModel):
    """Итог анализа одного изображения."""

    analysis_id: str
    filename: str
    study_uid: str | None
    image_uid: str | None
    anatomical_region: AnatomicalRegion
    region_source: str
    quality_class: int | None = Field(default=None, ge=0, le=1)
    violation_types: list[str]
    processing_status: str
    time_of_processing: float = Field(ge=0)
    checks: list[CheckResult]
    preview_data_url: str


class BatchItem(BaseModel):
    """Результат или управляемая ошибка одного файла пакета."""

    filename: str
    result: AnalysisResult | None = None
    error: str | None = None


class BatchResult(BaseModel):
    """Ответ пакетного анализа."""

    items: list[BatchItem]
    successful: int
    failed: int


class CheckInfo(BaseModel):
    """Краткое описание зарегистрированной проверки."""

    check_id: str
    title: str
    regions: list[AnatomicalRegion]
    model_status: str

