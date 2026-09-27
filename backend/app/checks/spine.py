"""Проверки охвата и оси поясничного отдела."""

import math

import numpy as np

from app.checks.base import AnalysisContext
from app.schemas import AnatomicalRegion, CheckResult, CheckStatus


class SpineCoverageCheck:
    """Оценивает наличие костного сигнала в нижних боковых областях."""

    check_id = "spine_coverage"
    title = "Охват поясничного отдела"
    regions = (AnatomicalRegion.LUMBAR_SPINE,)
    model_status = "notebook_rule_partial"

    def run(self, context: AnalysisContext) -> CheckResult:
        """Запускает нижнюю ветку правила охвата из notebook.

        Args:
            context: Нормализованное DXA-изображение поясничного отдела.

        Returns:
            Результат проверки с отношением бокового и общего сигнала.
        """
        pixels = context.pixels.astype(np.float32) / 255
        height, width = pixels.shape
        left = pixels[round(0.68 * height) :, : round(0.34 * width)]
        right = pixels[round(0.68 * height) :, round(0.66 * width) :]
        whole = pixels[
            round(0.05 * height) : round(0.95 * height),
            round(0.05 * width) : round(0.95 * width),
        ]
        feature = float(np.mean(np.concatenate([left.ravel(), right.ravel()]))) / (
            float(np.mean(whole)) + 1e-6
        )
        score = -feature
        threshold = -0.01941480441018939
        violation = score >= threshold
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=CheckStatus.FAILED if violation else CheckStatus.PASSED,
            violation=violation,
            summary=(
                "Недостаточный нижний боковой охват — проверьте визуализацию подвздошных костей."
                if violation
                else "Нижний охват достаточен по пиксельному критерию."
            ),
            confidence=min(0.95, 0.55 + abs(score - threshold) * 2),
            details={
                "lower_lateral_ratio": round(feature, 5),
                "violation_score": round(score, 5),
                "threshold": threshold,
                "upper_t12_branch": "pending_model_integration",
            },
            method="notebook_lower_lateral_sum_full_fold",
            model_status=self.model_status,
        )


class SpineTiltCheck:
    """Оценивает наклон оси позвоночника относительно вертикали."""

    check_id = "spine_tilt"
    title = "Ось позвоночника"
    regions = (AnatomicalRegion.LUMBAR_SPINE,)
    model_status = "prototype_landmarks"

    def run(self, context: AnalysisContext) -> CheckResult:
        """Строит временную ось по центрам яркого сигнала и применяет порог 5°.

        Args:
            context: Нормализованное DXA-изображение поясничного отдела.

        Returns:
            Угол, точки оси и решение о нарушении.
        """
        endpoints = self._estimate_endpoints(context.pixels)
        if endpoints is None:
            return CheckResult(
                check_id=self.check_id,
                title=self.title,
                status=CheckStatus.NOT_EVALUATED,
                violation=None,
                summary="Не удалось устойчиво определить ось позвоночника.",
                details={},
                method="intensity_centerline_with_notebook_geometry",
                model_status=self.model_status,
            )
        top, bottom, span_fraction = endpoints
        measurement = measure_tilt(top, bottom, threshold_deg=5.0)
        violation = measurement["violation"]
        return CheckResult(
            check_id=self.check_id,
            title=self.title,
            status=CheckStatus.FAILED if violation else CheckStatus.PASSED,
            violation=violation,
            summary=(
                f"Наклон {measurement['angle_deg']:.1f}° превышает допустимые 5°."
                if violation
                else f"Наклон {measurement['angle_deg']:.1f}° находится в пределах 5°."
            ),
            confidence=min(0.85, 0.5 + span_fraction * 0.35),
            details={**measurement, "endpoint_span_fraction": round(span_fraction, 4)},
            method="intensity_centerline_with_notebook_geometry",
            model_status=self.model_status,
        )

    @staticmethod
    def _estimate_endpoints(pixels: np.ndarray) -> tuple[list[float], list[float], float] | None:
        height, width = pixels.shape
        values = pixels.astype(np.float32) / 255
        x0, x1 = round(width * 0.2), round(width * 0.8)
        central = values[:, x0:x1]
        threshold = max(0.18, float(np.quantile(central, 0.68)))
        weights = np.maximum(central - threshold, 0)
        row_mass = weights.sum(axis=1)
        useful_rows = np.flatnonzero(row_mass > np.quantile(row_mass, 0.35))
        if len(useful_rows) < max(16, height * 0.2):
            return None

        top_y = int(np.quantile(useful_rows, 0.16))
        bottom_y = int(np.quantile(useful_rows, 0.84))
        band = max(4, round(height * 0.06))
        points: list[list[float]] = []
        for center_y in (top_y, bottom_y):
            start, stop = max(0, center_y - band), min(height, center_y + band + 1)
            band_weights = weights[start:stop]
            mass = float(band_weights.sum())
            if mass <= 1e-6:
                return None
            columns = np.arange(x0, x1, dtype=np.float32)
            rows = np.arange(start, stop, dtype=np.float32)
            x = float(band_weights.sum(axis=0) @ columns / mass)
            y = float(band_weights.sum(axis=1) @ rows / mass)
            points.append([x, y])
        span_fraction = (points[1][1] - points[0][1]) / height
        return (points[0], points[1], span_fraction) if span_fraction >= 0.3 else None


def measure_tilt(
    top_xy: list[float], bottom_xy: list[float], *, threshold_deg: float = 5.0
) -> dict[str, float | bool | list[float]]:
    """Вычисляет угол от нижней точки к верхней относительно вертикали.

    Формула перенесена из `spine_pipeline.ipynb`: X растёт вправо, Y — вниз,
    положительный знак означает смещение верхней точки вправо.

    Args:
        top_xy: Координаты верхней точки в нативных пикселях.
        bottom_xy: Координаты нижней точки в нативных пикселях.
        threshold_deg: Допустимый абсолютный угол.

    Returns:
        Измерение угла и бинарное решение.

    Raises:
        ValueError: Если точки или порог некорректны.
    """
    top = np.asarray(top_xy, dtype=float)
    bottom = np.asarray(bottom_xy, dtype=float)
    if top.shape != (2,) or bottom.shape != (2,) or not np.isfinite([top, bottom]).all():
        raise ValueError("Ожидаются две конечные точки (x, y)")
    if not math.isfinite(threshold_deg) or not 0 <= threshold_deg < 90:
        raise ValueError("Порог должен находиться в диапазоне [0, 90)")
    dx = float(top[0] - bottom[0])
    dy = float(bottom[1] - top[1])
    if dy <= 0:
        raise ValueError("Верхняя точка должна находиться строго выше нижней")
    signed = math.degrees(math.atan2(dx, dy))
    angle = abs(signed)
    violation = angle > threshold_deg and not math.isclose(
        angle, threshold_deg, abs_tol=1e-10, rel_tol=0
    )
    return {
        "angle_deg": round(angle, 4),
        "signed_angle_deg": round(signed, 4),
        "threshold_deg": threshold_deg,
        "violation": bool(violation),
        "top_xy": top.tolist(),
        "bottom_xy": bottom.tolist(),
    }

