"""Публичные модели API DEXQ."""

from enum import StrEnum
from typing import Any, Literal

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


class AxisLine(BaseModel):
    """Редактируемые точки оси в нативных пикселях исходного кадра."""

    top_xy: tuple[float, float]
    bottom_xy: tuple[float, float]
    angle_deg: float | None = None


class ImageGeometry(BaseModel):
    """Только геометрия, необходимая для честной визуализации."""

    image_width: int = Field(gt=0)
    image_height: int = Field(gt=0)
    coordinate_system: Literal["native_pixels_x_right_y_down"]
    axis_line: AxisLine | None = None
    vertebral_candidates: list[dict[str, Any]] = Field(default_factory=list)
    gap_lines: list[dict[str, Any]] = Field(default_factory=list)
    foreground_bbox: tuple[float, float, float, float] | None = None


class AnalysisResult(BaseModel):
    """Итог анализа одного изображения."""

    analysis_id: str
    filename: str
    study_uid: str | None
    image_uid: str | None
    anatomical_region: AnatomicalRegion
    region_source: str
    projection: Literal["AP", "PA", "unknown"] = "unknown"
    projection_source: Literal["dicom_view_position", "not_determined"] = "not_determined"
    needs_review: bool = False
    quality_class: int | None = Field(default=None, ge=0, le=1)
    violation_types: list[str]
    processing_status: Literal["Success", "Failure"]
    time_of_processing: float = Field(ge=0)
    checks: list[CheckResult]
    preview_data_url: str
    annotated_data_url: str | None = None
    geometry: ImageGeometry | None = None
    error: str | None = None


class BatchItem(BaseModel):
    """Результат или управляемая ошибка одного файла пакета."""

    filename: str
    result: AnalysisResult | None = None
    error: str | None = None
    input_position: int | None = None
    input_path: str | None = Field(default=None, exclude=True)


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

