"""Visual review of the existing three-criterion spine pipeline.

The overlays show model proposals, never verified vertebral levels or masks.
Nothing in this module changes predictions, centers, thresholds, or geometry.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch
import numpy as np
from PIL import Image


__all__ = ["plot_case"]

_INK = "#182b3a"
_MUTED = "#536574"
_GREEN = "#167b63"
_RED = "#c8474b"
_GOLD = "#ffc55c"


def _decision(value):
    if value is None:
        return "нет оценки", _MUTED
    return ("нарушение", _RED) if value else ("без нарушения", _GREEN)


def _truth_line(truth, key):
    value = truth.get(key) if truth else None
    label = "неизвестно" if value is None else ("нарушение" if value else "без нарушения")
    return f"Разметка: {label}"


def plot_case(image_path, result, truth=None, title=None):
    """Return a three-panel matplotlib Figure; do not show or save it.

    Parameters
    ----------
    image_path : str | pathlib.Path
        The native PNG that corresponds to the result (no letterbox or resize).
    result : dict
        ``compose_spine_result`` output, or its saved record with a ``result`` key.
    truth : dict | None
        Optional target values (0, 1, None) under ``spine_positioning``,
        ``spine_axis`` and ``spine_artifacts``. A manifest record containing
        ``targets`` is also accepted. Labels are displayed, never used to draw
        different predictions or select endpoints.
    title : str | None
        Optional case caption; defaults to the image ID.
    """
    wrapper = result
    result = result.get("result", result)
    if truth and "targets" in truth:
        truth = truth["targets"]
    with Image.open(Path(image_path)) as source:
        pixels = np.asarray(source.convert("L"))
    landmarks = result["shared_landmarks"]
    height, width = pixels.shape
    if (width, height) != (landmarks["image_width"], landmarks["image_height"]):
        raise ValueError("Use the matching native image: its dimensions differ from landmark coordinates")

    with plt.rc_context({"font.family": "DejaVu Sans", "font.size": 10}):
        fig = plt.figure(figsize=(14, 7.4), facecolor="#f5f7fa")
        grid = fig.add_gridspec(1, 3, width_ratios=(1, 1, 1.27),
                               left=.025, right=.985, bottom=.20, top=.86,
                               wspace=.095)
        original, geometry, scores = [fig.add_subplot(grid[0, i]) for i in range(3)]
        caption = title or wrapper.get("legacy_view_id") or result.get("image_id", "Снимок")
        fig.text(.025, .947, f"Позвоночник · {caption}", size=18, weight="bold", color=_INK)
        fig.text(.025, .905, "Три независимые проверки · исходные пороги · координаты в пикселях снимка",
                 size=10.5, color=_MUTED)

        for ax, heading in ((original, "Охват · найденные щели"), (geometry, "Наклон · два центра SpineNet")):
            ax.imshow(pixels, cmap="gray", vmin=0, vmax=255, origin="upper",
                      interpolation="nearest", aspect="equal")
            ax.set_xlim(-.5, width-.5)
            ax.set_ylim(height-.5, -.5)
            ax.set_title(heading, loc="left", fontsize=12, color=_INK, pad=12, weight="bold")
            ax.set_axis_off()

        upper = (result["coverage"].get("components") or {}).get("upper_gap_sequence") or {}
        gaps = upper.get("observed_gaps") or []
        for index, gap in enumerate(gaps, start=1):
            x, y = float(gap["x"]), float(gap["y"])
            half_width = .115 * width
            original.plot([max(0, x-half_width), min(width-1, x+half_width)], [y, y],
                          color=_GOLD, lw=1.8, zorder=3)
            original.text(min(width-8, x+half_width+4), y, str(index), color=_GOLD,
                          fontsize=8, va="center", weight="bold", zorder=4)
        if upper.get("status") == "measured" and upper.get("visible_units") is not None:
            coverage_note = (f"{len(gaps)} щелей · индекс охвата {upper['visible_units']:.2f} "
                             f"/ порог {-upper['threshold']:.2f}")
        else:
            coverage_note = "Нет надёжной цепочки · верхняя ветка воздержалась"
        original.text(.02, -.052, coverage_note, transform=original.transAxes,
                      color=_INK, size=8.2, weight="bold", va="top")

        axis = result["axis"]
        top, bottom = axis.get("top_xy"), axis.get("bottom_xy")
        if top is not None and bottom is not None and axis.get("angle_deg") is not None:
            top, bottom = np.asarray(top, float), np.asarray(bottom, float)
            angle = float(axis["angle_deg"])
            threshold = float(axis.get("threshold_deg", 5.0))
            color = "#ff7481" if angle > threshold else "#5df0b4"
            # This is the only angle line: its endpoints are exactly those used by measure_tilt.
            geometry.plot([bottom[0], top[0]], [bottom[1], top[1]],
                          color=color, lw=2.6, zorder=4, label="angle_segment")
            geometry.scatter([top[0], bottom[0]], [top[1], bottom[1]], s=67,
                             facecolors=color, edgecolors="#162731", linewidths=1.2, zorder=7)
            for point, label in ((top, "верхняя точка"), (bottom, "нижняя точка")):
                geometry.annotate(label, point, xytext=(10, 0), textcoords="offset points",
                                  color="white", fontsize=8, va="center",
                                  bbox=dict(boxstyle="round,pad=.25", fc="#152636", ec="none", alpha=.82))
        else:
            geometry.text(.5, .04, "Угол недоступен", transform=geometry.transAxes,
                          ha="center", color="white", bbox=dict(fc="#152636", ec="none", pad=5))

        scores.set(xlim=(0, 1), ylim=(0, 1))
        scores.set_axis_off()
        scores.set_title("Решения и разметка", loc="left", fontsize=12,
                         color=_INK, pad=12, weight="bold")
        values = [result[key].get("violation") for key in ("coverage", "axis", "artifacts")]
        overall = True if any(v is True for v in values) else (None if any(v is None for v in values) else False)
        label, color = _decision(overall)
        scores.add_patch(FancyBboxPatch((0, .866), 1, .112, boxstyle="round,pad=.008,rounding_size=.02",
                                       facecolor="white", edgecolor="#dce3ea"))
        scores.text(.035, .943, "ИТОГ · хотя бы одна проверка", size=9, color=_MUTED, va="center")
        scores.text(.035, .895, label, size=14, color=color, weight="bold", va="center")

        for key, heading, target, y in (
            ("coverage", "01  Укладка и охват", "spine_positioning", .79),
            ("artifacts", "02  Посторонние предметы", "spine_artifacts", .54),
        ):
            criterion = result[key]
            score = criterion.get("score")
            threshold = float(criterion.get("threshold", .5))
            label, color = _decision(criterion.get("violation"))
            scores.text(0, y, heading, fontsize=11, color=_INK, weight="bold", va="center")
            scores.text(0, y-.046, label, color=color, size=10, va="center")
            left, span, bar_y = .02, .94, y-.103
            scores.barh(bar_y, span, left=left, height=.025, color="#dce3ea", zorder=1)
            if criterion.get("method") == "normalized_lower_lateral_sum_or_anonymous_gap_sequence":
                center = left + span / 2
                if score is not None:
                    extent = .5 * span * np.tanh(abs(float(score)))
                    start = center if float(score) >= 0 else center - extent
                    scores.barh(bar_y, extent, left=start, height=.025, color=color, zorder=2)
                scores.plot([center]*2, [bar_y-.025, bar_y+.025], color=_INK, lw=1.2)
            else:
                if score is not None:
                    scores.barh(bar_y, span * float(score), left=left, height=.025, color=color, zorder=2)
                scores.plot([left + threshold * span]*2, [bar_y-.025, bar_y+.025], color=_INK, lw=1.2)
            score_text = "нет оценки" if score is None else f"score {float(score):.3f}"
            scores.text(left, bar_y-.037, score_text, size=9, color=_MUTED, va="top")
            scores.text(left+span, bar_y-.037, f"порог {threshold:g}", size=9,
                        color=_MUTED, va="top", ha="right")
            if truth is not None:
                scores.text(0, y-.19, _truth_line(truth, target), size=9.5, color=_MUTED, va="center")

        label, color = _decision(axis.get("violation"))
        scores.text(0, .298, "03  Выравнивание оси", fontsize=11, color=_INK, weight="bold", va="center")
        scores.text(0, .254, f"Итог: {label}", color=color, fontsize=11.5,
                    weight="bold", va="center")
        scores.add_patch(FancyBboxPatch((0, .084), .99, .143,
                                       boxstyle="round,pad=.005,rounding_size=.015",
                                       facecolor="white", edgecolor="#dce3ea", zorder=0))
        angle_text = "нет оценки" if axis.get("angle_deg") is None else f"{axis['angle_deg']:.2f}°"
        scores.text(.025, .205, "УГОЛ МЕЖДУ КРАЙНИМИ ЦЕНТРАМИ", size=8.3,
                    color=_MUTED, weight="bold", va="center")
        scores.text(.025, .150, angle_text, size=17, color=_INK, weight="bold", va="center")
        scores.text(.96, .150, "нарушение при > 5°", size=10,
                    color=_MUTED, va="center", ha="right")
        scores.text(0, .047, "Прямой resize 512×512 → SpineNet → угол", color=_MUTED,
                    size=9.2, va="center")
        if truth is not None:
            scores.text(0, .012, _truth_line(truth, "spine_axis"), color=_MUTED,
                        size=9, va="center")

        handles = [Line2D([], [], color=_GOLD, lw=2, label="охват: наблюдаемая щель"),
                   Line2D([], [], color=_MUTED, lw=2.6, label="угол: два крайних найденных центра")]
        fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(.027, .118),
                   ncol=2, frameon=False, fontsize=9, labelcolor=_INK)
        fig.text(.025, .081,
                 "Угол измеряется в исходных координатах снимка; ровно 5° допустимо.",
                 color=_MUTED, fontsize=9)
        fig.text(.025, .050,
                 "Центры — предложения модели. Маркеры охвата — щели, не номера позвонков; Th12 не распознаётся.",
                 color=_MUTED, fontsize=9)
    return fig
