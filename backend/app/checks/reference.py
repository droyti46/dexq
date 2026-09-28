"""Нейтральные проверки качества поверх результата неизменённой модели."""

from __future__ import annotations

from typing import Any

from app.schemas import AnatomicalRegion, CheckResult, CheckStatus


class ReferenceCheck:
    """Сопоставляет один флаг эталонного решения с единым контрактом.

    Args:
        check_id: Имя машинного бинарного выхода.
        title: Понятное оператору название.
        regions: Область применения.
    """

    model_status = "reference_onnx_v13"

    def __init__(self, check_id: str, title: str, regions: tuple[AnatomicalRegion, ...]) -> None:
        self.check_id = check_id
        self.title = title
        self.regions = regions

    def run(self, context: dict[str, Any]) -> CheckResult:
        """Возвращает вердикт эталона без самостоятельного preprocessing.

        Args:
            context: Уже вычисленный результат эталонного инференса.

        Returns:
            Нейтральное описание статуса и числового скора.
        """
        violation = context.get("labels", {}).get(self.check_id)
        score = context.get("scores", {}).get(self.check_id)
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=(
                CheckStatus.NOT_EVALUATED
                if violation is None
                else CheckStatus.FAILED
                if violation
                else CheckStatus.PASSED
            ),
            violation=violation,
            summary=(
                "Проверка не дала надёжного решения."
                if violation is None
                else "Модель отметила нарушение качества."
                if violation
                else "Модель не отметила нарушение качества."
            ),
            details={"score": score} if score is not None else {},
            method="reference_qc",
            model_status=self.model_status,
        )


REFERENCE_CHECKS = [
    ReferenceCheck("spine_positioning", "Охват позвоночника", (AnatomicalRegion.LUMBAR_SPINE,)),
    ReferenceCheck("spine_axis", "Наклон оси позвоночника", (AnatomicalRegion.LUMBAR_SPINE,)),
    ReferenceCheck("spine_artifacts", "Артефакты позвоночника", (AnatomicalRegion.LUMBAR_SPINE,)),
    ReferenceCheck(
        "hip_positioning_rotation",
        "Позиционирование / ротация бедра",
        (AnatomicalRegion.PROXIMAL_FEMUR,),
    ),
    ReferenceCheck("hip_roi", "Охват ROI бедра", (AnatomicalRegion.PROXIMAL_FEMUR,)),
]
