"""Оркестрация независимых проверок качества."""

from __future__ import annotations

import time

from app.checks import (
    ArtifactCheck,
    HipCoverageCheck,
    HipROICheck,
    HipRotationCheck,
    SpineCoverageCheck,
    SpineTiltCheck,
)
from app.checks.base import AnalysisContext, QualityCheck
from app.imaging import content_id, load_medical_image
from app.schemas import (
    AnalysisResult,
    AnatomicalRegion,
    CheckInfo,
    CheckResult,
    CheckStatus,
)


class Analyzer:
    """Запускает подходящие проверки и собирает единый отчёт."""

    def __init__(self) -> None:
        self.checks: list[QualityCheck] = [
            SpineCoverageCheck(),
            SpineTiltCheck(),
            ArtifactCheck(),
            HipCoverageCheck(),
            HipRotationCheck(),
            HipROICheck(),
        ]

    def describe_checks(self) -> list[CheckInfo]:
        """Возвращает безопасное описание реестра проверок."""
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
        """Анализирует один файл без сохранения на диск.

        Args:
            content: Байты DICOM или демонстрационного изображения.
            filename: Исходное имя файла.
            requested_region: Автоопределение или явный выбор оператора.

        Returns:
            Структурированный результат всех применимых проверок.

        Raises:
            ValueError: Если изображение невозможно безопасно декодировать.
        """
        started = time.perf_counter()
        image = load_medical_image(content, filename)
        region = image.region if requested_region == AnatomicalRegion.AUTO else requested_region
        region_source = (
            image.region_source if requested_region == AnatomicalRegion.AUTO else "operator"
        )
        context = AnalysisContext(pixels=image.pixels, region=region)
        results: list[CheckResult] = []
        for check in self.checks:
            if region not in check.regions:
                continue
            try:
                results.append(check.run(context))
            except Exception as error:  # Модуль не должен прерывать остальные проверки.
                results.append(
                    CheckResult(
                        check_id=check.check_id,
                        title=check.title,
                        status=CheckStatus.ERROR,
                        violation=None,
                        summary="Проверка завершилась управляемой ошибкой.",
                        details={"error_type": type(error).__name__},
                        method="failed_before_result",
                        model_status=check.model_status,
                    )
                )

        violations = [item.check_id for item in results if item.violation is True]
        has_unknown = not results or any(item.violation is None for item in results)
        quality_class = 1 if violations else None if has_unknown else 0
        return AnalysisResult(
            analysis_id=content_id(content),
            filename=filename,
            study_uid=image.study_uid,
            image_uid=image.image_uid,
            anatomical_region=region,
            region_source=region_source,
            quality_class=quality_class,
            violation_types=violations,
            processing_status="Success",
            time_of_processing=round(time.perf_counter() - started, 4),
            checks=results,
            preview_data_url=image.preview_data_url,
        )
