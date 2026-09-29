"""Оркестрация единственного эталонного DXA-QC без эвристического fallback."""

from __future__ import annotations

import base64
import time
from pathlib import PurePath
from typing import Any

from app.checks.reference import REFERENCE_CHECKS
from app.imaging import content_id, safe_dicom_fields
from app.model_runtime import ReferenceRuntime
from app.schemas import AnalysisResult, AnatomicalRegion, CheckInfo, ImageGeometry


class Analyzer:
    """Вызывает эталон и формирует безопасный отчёт для API.

    Args:
        runtime: Один разделяемый экземпляр оригинального ONNX-пайплайна.
    """

    def __init__(self, runtime: ReferenceRuntime | None = None) -> None:
        self.runtime = runtime or ReferenceRuntime()
        self.checks = REFERENCE_CHECKS

    def describe_checks(self) -> list[CheckInfo]:
        """Возвращает независимые исходные выходы проверки качества.

        Returns:
            Пять применимых проверок эталона без разделения общего hip-флага.
        """
        return [
            CheckInfo(
                check_id=check.check_id,
                title=check.title,
                regions=list(check.regions),
                model_status=check.model_status,
            )
            for check in self.checks
        ]

    def analyze(
        self,
        content: bytes,
        filename: str,
        requested_region: AnatomicalRegion = AnatomicalRegion.AUTO,
    ) -> AnalysisResult:
        """Анализирует оригинальные байты, не заменяя preprocessing модели.

        Args:
            content: Исходный DICOM или нативный PNG.
            filename: Отображаемое имя; не используется моделью.
            requested_region: Автоматический маршрут или ручной выбор области.

        Returns:
            Машинный бинарный класс либо управляемую строку Failure.
        """
        started = time.perf_counter()
        filename = PurePath(filename.replace("\\", "/")).name
        if not filename or filename in {".", ".."}:
            filename = "study.dcm"
        suffix = PurePath(filename).suffix.lower()
        study_uid: str | None = None
        image_uid: str | None = None
        region = AnatomicalRegion.UNKNOWN
        region_source = "not_determined"
        preview = ""
        annotated: str | None = None
        geometry: ImageGeometry | None = None
        checks = []
        needs_review = False
        violation_types: list[str] = []
        quality_class: int | None = None
        processing_status: str = "Failure"
        error_message: str | None = None
        try:
            study_uid, image_uid, unsupported_view = safe_dicom_fields(content, suffix)
            if unsupported_view:
                raise ValueError("Проекция вне подтверждённого профиля эталонной модели")
            if suffix not in {".dcm", ".dicom", ".png"}:
                raise ValueError("Неподдерживаемый формат для эталонной модели")
            raw = self.runtime.infer(content, suffix, requested_region)
            region = {
                "spine": AnatomicalRegion.LUMBAR_SPINE,
                "hip": AnatomicalRegion.PROXIMAL_FEMUR,
            }.get(raw.get("region"), AnatomicalRegion.UNKNOWN)
            region_source = raw.get("region_source", "model")
            needs_review = bool(raw.get("needs_review", False))
            checks = [check.run(raw) for check in self.checks if region in check.regions]
            violations = [item.check_id for item in checks if item.violation is True]
            violation_types = sorted(violations)
            if checks and all(item.violation is not None for item in checks):
                quality_class = int(bool(violations))
                if raw.get("any_violation") != bool(quality_class):
                    raise ValueError("Решение модели противоречит независимым проверкам")
                processing_status = "Success"
            else:
                error_message = "Не удалось определить все обязательные проверки качества"
            source_image = raw.get("image") or raw.get("raw", {}).get("image")
            if source_image is None and region == AnatomicalRegion.LUMBAR_SPINE:
                source_image = self._spine_source(content, suffix)
            if source_image and source_image.get("encoding") == "base64_png":
                preview = "data:image/png;base64," + source_image["data"]
            overlay = raw.get("annotated_image")
            if overlay and overlay.get("encoding") == "base64_png":
                annotated = "data:image/png;base64," + overlay["data"]
            native = raw.get("geometry")
            if native and native.get("coordinate_system") == "native_pixels_x_right_y_down":
                geometry = ImageGeometry(
                    image_width=native["image_width"],
                    image_height=native["image_height"],
                    coordinate_system="native_pixels_x_right_y_down",
                    axis_line=native.get("axis_line"),
                    vertebral_candidates=[
                        {"center_xy": row["center_xy"]}
                        for row in native.get("vertebral_candidates", [])
                        if isinstance(row, dict) and "center_xy" in row
                    ],
                    gap_lines=[
                        {"endpoints_xy": row["endpoints_xy"]}
                        for row in native.get("gap_lines", [])
                        if isinstance(row, dict) and "endpoints_xy" in row
                    ],
                    foreground_bbox=native.get("foreground_bbox"),
                )
        except (ValueError, OSError, KeyError, TypeError, RuntimeError):
            error_message = "Не удалось безопасно обработать файл или модель недоступна"
            quality_class = None
            processing_status = "Failure"
            violation_types = []
        return AnalysisResult(
            analysis_id=content_id(content),
            filename=filename,
            study_uid=study_uid,
            image_uid=image_uid,
            anatomical_region=region,
            region_source=region_source,
            quality_class=quality_class,
            violation_types=violation_types,
            processing_status=processing_status,
            time_of_processing=round(time.perf_counter() - started, 4),
            checks=checks,
            preview_data_url=preview,
            annotated_data_url=annotated,
            geometry=geometry,
            needs_review=needs_review,
            error=error_message,
        )

    @staticmethod
    def _spine_source(content: bytes, suffix: str) -> dict[str, Any]:
        """Читает нативное изображение для наложения, не меняя вход модели.

        Args:
            content: Исходные байты снимка.
            suffix: Исходный формат.

        Returns:
            Встроенное нативное PNG без DICOM-тегов.
        """
        from io import BytesIO

        import numpy as np
        import pydicom
        from PIL import Image

        if suffix == ".png":
            with Image.open(BytesIO(content)) as image:
                png = image.convert("L")
                size = png.size
                buffer = BytesIO()
                png.save(buffer, format="PNG")
        else:
            with pydicom.config.disable_value_validation():
                ds = pydicom.dcmread(BytesIO(content))
                array = np.asarray(ds.pixel_array, dtype=np.uint8)
            png = Image.fromarray(array)
            size = png.size
            buffer = BytesIO()
            png.save(buffer, format="PNG")
        return {
            "encoding": "base64_png",
            "width": size[0],
            "height": size[1],
            "data": base64.b64encode(buffer.getvalue()).decode("ascii"),
        }
