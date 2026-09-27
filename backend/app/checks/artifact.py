"""Общая проверка выраженных ярких артефактов."""

import numpy as np

from app.checks.base import AnalysisContext
from app.schemas import AnatomicalRegion, CheckResult, CheckStatus


class ArtifactCheck:
    """Ищет сочетание насыщенных областей и резких границ."""

    check_id = "artifact"
    title = "Посторонние предметы и артефакты"
    regions = (AnatomicalRegion.LUMBAR_SPINE, AnatomicalRegion.PROXIMAL_FEMUR)
    model_status = "prototype_heuristic"

    def run(self, context: AnalysisContext) -> CheckResult:
        """Выполняет детерминированную проверку насыщения и градиента.

        Args:
            context: Нормализованное DXA-изображение.

        Returns:
            Предварительная оценка наличия выраженного артефакта.
        """
        values = context.pixels.astype(np.float32) / 255
        height, width = values.shape
        margin_y, margin_x = max(1, height // 20), max(1, width // 20)
        interior = values[margin_y:-margin_y, margin_x:-margin_x]
        gradient_y, gradient_x = np.gradient(interior)
        gradient = np.hypot(gradient_x, gradient_y)
        saturated_fraction = float(np.mean(interior >= 0.995))
        sharp_fraction = float(np.mean(gradient >= 0.42))
        violation = saturated_fraction > 0.055 and sharp_fraction > 0.02
        confidence = min(0.75, 0.45 + saturated_fraction * 2 + sharp_fraction)
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=CheckStatus.FAILED if violation else CheckStatus.PASSED,
            violation=violation,
            summary=(
                "Обнаружена яркая высококонтрастная область — требуется визуальная проверка."
                if violation
                else "Выраженные высококонтрастные артефакты не обнаружены."
            ),
            confidence=confidence,
            details={
                "saturated_fraction": round(saturated_fraction, 5),
                "sharp_gradient_fraction": round(sharp_fraction, 5),
            },
            method="saturation_and_gradient_screening_v1",
            model_status=self.model_status,
        )

