"""Контракты проверок проксимального отдела бедра."""

from app.checks.base import AnalysisContext
from app.schemas import AnatomicalRegion, CheckResult, CheckStatus


class _PendingHipCheck:
    regions = (AnatomicalRegion.PROXIMAL_FEMUR,)
    model_status = "model_pending"
    check_id = ""
    title = ""
    pending_summary = ""

    def run(self, context: AnalysisContext) -> CheckResult:
        """Возвращает честный статус до подключения специализированной модели.

        Args:
            context: Нормализованное изображение проксимального отдела бедра.

        Returns:
            Результат `not_evaluated`, не влияющий на ложное решение о качестве.
        """
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=CheckStatus.NOT_EVALUATED,
            violation=None,
            summary=self.pending_summary,
            details={"next_step": "connect_specialized_hip_model"},
            method="pending",
            model_status=self.model_status,
        )


class HipCoverageCheck(_PendingHipCheck):
    """Проверка наличия ключевых структур проксимального отдела бедра."""

    check_id = "hip_coverage"
    title = "Позиционирование бедра"
    pending_summary = "Модель видимости большого вертела, шейки и седалищной кости подключается."


class HipRotationCheck(_PendingHipCheck):
    """Проверка ротации по форме малого вертела."""

    check_id = "hip_rotation"
    title = "Ротация бедра"
    pending_summary = "Модель оценки малого вертела подключается."


class HipROICheck(_PendingHipCheck):
    """Проверка полей вокруг области интереса."""

    check_id = "hip_roi"
    title = "Область интереса"
    pending_summary = (
        "Проверка отступов ROI подключается после локализации анатомических ориентиров."
    )
