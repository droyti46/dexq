"""Минималистичный промо-ролик DEXQ в motion-графике.

Рисует мультяшную анимацию сценария «нажать кнопку → перетащить снимки →
получить проверку → экспортировать CSV» и кодирует её в MP4. Снимки в ролике —
нарисованные схемы, а не медицинские данные; интерфейс условный.

Зависимости ставятся в отдельное окружение, чтобы не трогать backend:

    python -m venv .venv-motion
    .venv-motion/Scripts/python -m pip install skia-python imageio-ffmpeg numpy fonttools
    .venv-motion/Scripts/python scripts/motion_demo.py
    .venv-motion/Scripts/python scripts/motion_demo.py --still 12.4 30

"""

from __future__ import annotations

import argparse
import math
import tempfile
from collections.abc import Callable
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np
import skia

ROOT = Path(__file__).resolve().parents[1]
FONTS = ROOT / "frontend" / "public" / "fonts"
OUTPUT = ROOT / "frontend" / "public" / "media" / "dexq-demo.mp4"

W, H = 1920, 1080
FPS = 60

BLUE = 0x0202F1
BLUE_DARK = 0x0000C8
INK = 0x12142C
MUTED = 0x71758D
LAV = 0xEEF0FF
LINE = 0xDFE2EE
WHITE = 0xFFFFFF
YELLOW = 0xFFD43B
MINT = 0x20D8A0
CORAL = 0xFF5266
NAVY = 0x0B1433
BONE = 0xDCE5FF


# ---------------------------------------------------------------- математика


def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return lo if x < lo else hi if x > hi else x


def lerp(a: float, b: float, u: float) -> float:
    return a + (b - a) * u


def prog(t: float, start: float, dur: float) -> float:
    """Доля прохождения отрезка [start, start + dur], ограниченная 0..1."""
    return clamp((t - start) / dur) if dur > 0 else float(t >= start)


def ease_out_cubic(u: float) -> float:
    return 1 - (1 - u) ** 3


def ease_in_cubic(u: float) -> float:
    return u**3


def ease_in_out(u: float) -> float:
    return 4 * u**3 if u < 0.5 else 1 - (-2 * u + 2) ** 3 / 2


def ease_in_out_quint(u: float) -> float:
    return 16 * u**5 if u < 0.5 else 1 - (-2 * u + 2) ** 5 / 2


def ease_out_back(u: float, s: float = 1.9) -> float:
    return 1 + (s + 1) * (u - 1) ** 3 + s * (u - 1) ** 2


def spring(u: float, bounces: float = 2.5, damping: float = 5.5) -> float:
    """Затухающая пружина 0 → 1 с перелётом; u — нормированное время."""
    if u <= 0:
        return 0.0
    if u >= 1:
        return 1.0
    return 1 - math.exp(-damping * u) * math.cos(bounces * math.tau * u)


def pop(t: float, start: float, dur: float = 0.55) -> float:
    """Масштаб появления с перелётом: 0 до start, затем пружина к 1."""
    return spring(prog(t, start, dur), 1.6, 6.0)


def keyframes(
    t: float, keys: list[tuple[float, tuple[float, ...]]], ease=ease_in_out_quint
) -> tuple[float, ...]:
    """Интерполирует кортежи значений между ключами времени с плавной кривой."""
    if t <= keys[0][0]:
        return keys[0][1]
    for (t0, v0), (t1, v1) in zip(keys, keys[1:], strict=False):
        if t <= t1:
            u = ease(prog(t, t0, t1 - t0))
            return tuple(lerp(a, b, u) for a, b in zip(v0, v1, strict=False))
    return keys[-1][1]


def bezier(p0, p1, p2, p3, u: float) -> tuple[float, float]:
    m = 1 - u
    return (
        m**3 * p0[0] + 3 * m * m * u * p1[0] + 3 * m * u * u * p2[0] + u**3 * p3[0],
        m**3 * p0[1] + 3 * m * m * u * p1[1] + 3 * m * u * u * p2[1] + u**3 * p3[1],
    )


def glide(
    t: float, start: float, dur: float, a, b, bend: tuple[float, float] = (0, -160)
) -> tuple[float, float]:
    """Движение курсора по дуге с разгоном и торможением."""
    u = ease_in_out(prog(t, start, dur))
    c1 = (lerp(a[0], b[0], 0.3) + bend[0], lerp(a[1], b[1], 0.3) + bend[1])
    c2 = (lerp(a[0], b[0], 0.75) + bend[0] * 0.4, lerp(a[1], b[1], 0.75) + bend[1] * 0.4)
    return bezier(a, c1, c2, b, u)


# ---------------------------------------------------------------- рисование


def color(hex_rgb: int, alpha: float = 1.0) -> int:
    return skia.ColorSetARGB(
        int(clamp(alpha) * 255), hex_rgb >> 16 & 255, hex_rgb >> 8 & 255, hex_rgb & 255
    )


def fill(hex_rgb: int, alpha: float = 1.0, blur: float = 0.0) -> skia.Paint:
    paint = skia.Paint(Color=color(hex_rgb, alpha), AntiAlias=True)
    if blur:
        paint.setMaskFilter(skia.MaskFilter.MakeBlur(skia.kNormal_BlurStyle, blur))
    return paint


def stroke(hex_rgb: int, width: float, alpha: float = 1.0) -> skia.Paint:
    return skia.Paint(
        Color=color(hex_rgb, alpha),
        AntiAlias=True,
        Style=skia.Paint.kStroke_Style,
        StrokeWidth=width,
        StrokeCap=skia.Paint.kRound_Cap,
        StrokeJoin=skia.Paint.kRound_Join,
    )


def rrect(cx: float, cy: float, w: float, h: float, r: float) -> skia.RRect:
    return skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(cx - w / 2, cy - h / 2, w, h), r, r)


def soft_card(
    c: skia.Canvas,
    cx: float,
    cy: float,
    w: float,
    h: float,
    r: float,
    lift: float = 1.0,
    fill_hex: int = WHITE,
) -> None:
    """Карточка с мягкой тенью; lift усиливает тень при «подъёме»."""
    c.drawRRect(
        rrect(cx, cy + 14 * lift, w, h, r), fill(BLUE_DARK, 0.10 + 0.06 * lift, 22 * lift + 6)
    )
    c.drawRRect(rrect(cx, cy, w, h, r), fill(fill_hex))


@cache
def typeface(bold: bool) -> skia.Typeface:
    """Evolventa в WOFF Skia не читает, поэтому один раз конвертируем в TTF."""
    from fontTools.ttLib import TTFont

    name = "Evolventa-Bold" if bold else "Evolventa-Regular"
    ttf = Path(tempfile.gettempdir()) / f"dexq-{name}.ttf"
    if not ttf.exists():
        font = TTFont(FONTS / f"{name}.woff")
        font.flavor = None
        font.save(ttf)
    face = skia.Typeface.MakeFromFile(str(ttf))
    if face is None:
        raise RuntimeError(f"Не удалось загрузить шрифт {ttf}")
    return face


@cache
def font(size: float, bold: bool = True) -> skia.Font:
    f = skia.Font(typeface(bold), size)
    f.setEdging(skia.Font.Edging.kSubpixelAntiAlias)
    f.setSubpixel(True)
    return f


def text(
    c: skia.Canvas,
    s: str,
    x: float,
    y: float,
    size: float,
    hex_rgb: int = INK,
    alpha: float = 1.0,
    bold: bool = True,
    align: str = "center",
) -> float:
    """Рисует строку с вертикальным центром в y. Возвращает ширину."""
    f = font(size, bold)
    width = f.measureText(s)
    m = f.getMetrics()
    dx = {"center": -width / 2, "left": 0.0, "right": -width}[align]
    c.drawString(s, x + dx, y - (m.fAscent + m.fDescent) / 2, f, fill(hex_rgb, alpha))
    return width


class Group:
    """Контекст: перенос, масштаб, поворот вокруг точки и общая прозрачность."""

    def __init__(
        self,
        c: skia.Canvas,
        x: float = 0,
        y: float = 0,
        scale: float = 1,
        rot: float = 0,
        alpha: float = 1,
        sx: float = 1,
        sy: float = 1,
    ):
        self.c, self.args = c, (x, y, scale, rot, alpha, sx, sy)

    def __enter__(self) -> skia.Canvas:
        x, y, scale, rot, alpha, sx, sy = self.args
        if alpha < 1:
            self.c.saveLayerAlpha(None, int(clamp(alpha) * 255))
        else:
            self.c.save()
        self.c.translate(x, y)
        if rot:
            self.c.rotate(rot)
        self.c.scale(scale * sx, scale * sy)
        return self.c

    def __exit__(self, *_) -> None:
        self.c.restore()


def ripple(
    c: skia.Canvas,
    t: float,
    start: float,
    x: float,
    y: float,
    hex_rgb: int = BLUE,
    size: float = 110,
) -> None:
    """Два расходящихся кольца клика."""
    for k in range(2):
        u = prog(t, start + k * 0.12, 0.7)
        if 0 < u < 1:
            e = ease_out_cubic(u)
            c.drawCircle(x, y, 12 + size * e, stroke(hex_rgb, 10 * (1 - e) + 1, 1 - u))


def cursor(c: skia.Canvas, x: float, y: float, scale: float = 1.0, alpha: float = 1.0) -> None:
    """Мультяшная стрелка: белая заливка, толстый контур, жёсткая тень."""
    if alpha <= 0:
        return
    pts = [(0, 0), (0, 36), (9, 28), (15.5, 42), (22, 39), (16, 25.5), (28, 25.5)]
    path = skia.Path()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    path.close()
    with Group(c, x, y, 1.7 * scale, alpha=alpha):
        with Group(c, 3, 4):
            c.drawPath(path, fill(INK, 0.28))
        c.drawPath(path, fill(WHITE))
        c.drawPath(path, stroke(INK, 3.2))


def press_scale(t: float, at: float) -> float:
    """Сжатие курсора/кнопки при нажатии и пружинный возврат."""
    if t < at:
        return 1.0 - 0.14 * ease_out_cubic(prog(t, at - 0.12, 0.12))
    return 0.86 + 0.14 * spring(prog(t, at, 0.5), 1.4, 5)


def pill_button(
    c: skia.Canvas,
    t: float,
    cx: float,
    cy: float,
    label: str,
    *,
    w: float,
    h: float = 84,
    hex_rgb: int = BLUE,
    text_hex: int = WHITE,
    hover: float = 0.0,
    press_at: float | None = None,
    appear: float = 1.0,
) -> None:
    """Кнопка-таблетка с «мультяшной» жёсткой тенью, реагирующая на hover и клик."""
    if appear <= 0:
        return
    s = appear * (press_scale(t, press_at) if press_at is not None else 1.0)
    lift = 6 + 6 * hover
    if press_at is not None:
        lift *= 0.35 + 0.65 * (1 - bump(t, press_at - 0.12, 0.45))
    with Group(c, cx, cy - lift + 6, s * (1 + 0.04 * hover)):
        c.drawRRect(rrect(0, lift, w, h, h / 2), fill(INK, 0.9 if hex_rgb != WHITE else 0.25))
        c.drawRRect(rrect(0, 0, w, h, h / 2), fill(hex_rgb))
        text(c, label, 0, 1, h * 0.38, text_hex)


def bump(t: float, start: float, dur: float) -> float:
    """Колокол 0 → 1 → 0 на отрезке."""
    u = prog(t, start, dur)
    return math.sin(math.pi * u) if 0 < u < 1 else 0.0


# ---------------------------------------------------------------- снимки


def xray(
    c: skia.Canvas,
    kind: str,
    w: float,
    h: float,
    *,
    frame: float = 10,
    artifact: bool = False,
    glow: float = 0.0,
) -> None:
    """Схематичный рентген-снимок в белой рамке (центр в 0,0).

    Это рисунок-иллюстрация, а не изображение пациента.
    """
    r = w * 0.09
    c.drawRRect(rrect(0, 0, w, h, r), fill(WHITE))
    iw, ih = w - frame * 2, h - frame * 2
    inner = rrect(0, 0, iw, ih, r * 0.7)
    c.drawRRect(inner, fill(NAVY))
    c.save()
    c.clipRRect(inner, True)
    bone = fill(BONE, 0.9)
    if kind == "spine":
        # Таз внизу и пять позвонков с лёгким изгибом.
        c.drawOval(
            skia.Rect.MakeXYWH(-iw * 0.62, ih * 0.30, iw * 0.62, ih * 0.34), fill(BONE, 0.35)
        )
        c.drawOval(skia.Rect.MakeXYWH(0, ih * 0.30, iw * 0.62, ih * 0.34), fill(BONE, 0.35))
        for k in range(5):
            y = -ih * 0.34 + k * ih * 0.15
            x = math.sin(k * 0.8) * iw * 0.025
            c.drawRRect(rrect(x, y, iw * (0.34 + k * 0.02), ih * 0.105, ih * 0.035), bone)
            c.drawRRect(
                rrect(x, y + ih * 0.075, iw * 0.22, ih * 0.03, ih * 0.015), fill(BONE, 0.25)
            )
    else:
        # Головка бедра, шейка и диафиз.
        c.drawOval(
            skia.Rect.MakeXYWH(-iw * 0.75, -ih * 0.55, iw * 0.95, ih * 0.55), fill(BONE, 0.3)
        )
        shaft = skia.Path()
        shaft.moveTo(-iw * 0.08, -ih * 0.12)
        shaft.lineTo(iw * 0.16, -ih * 0.02)
        shaft.lineTo(iw * 0.24, ih * 0.6)
        shaft.lineTo(-iw * 0.02, ih * 0.6)
        shaft.lineTo(-iw * 0.02, ih * 0.08)
        shaft.close()
        c.drawPath(shaft, bone)
        c.drawCircle(-iw * 0.12, -ih * 0.14, iw * 0.17, bone)
        c.drawOval(skia.Rect.MakeXYWH(iw * 0.1, -ih * 0.08, iw * 0.2, ih * 0.2), fill(BONE, 0.9))
    if artifact:
        ax, ay = iw * 0.2, ih * 0.02
        c.drawRRect(rrect(ax, ay, iw * 0.07, ih * 0.12, iw * 0.02), fill(WHITE))
        c.drawCircle(ax, ay, iw * 0.09, fill(WHITE, 0.25 + 0.5 * glow, iw * 0.05))
    c.restore()


# ---------------------------------------------------------------- печатный текст

Segment = tuple  # (строка, цвет) или (строка, цвет, подчеркнуть)


def typed_text(
    c: skia.Canvas,
    t: float,
    lines: list[list[Segment]],
    *,
    size: float = 150,
    cps: float = 17,
    fall_at: float | None = None,
    cy: float = H / 2,
    seed: int = 1,
) -> None:
    """Печатает строки по буквам на весь экран.

    Каждая буква выпрыгивает с перелётом, каретка скользит за текстом, выделенные
    слова подчёркиваются «маркерным» штрихом. После fall_at буквы осыпаются.
    """
    f = font(size, True)
    m = f.getMetrics()
    gap = size * 1.08
    chars: list[tuple[str, float, float, float, int]] = []
    underlines: list[tuple[float, float, float, int]] = []
    for li, line in enumerate(lines):
        width = sum(f.measureText(ch) for seg in line for ch in seg[0])
        x = W / 2 - width / 2
        y = cy + (li - (len(lines) - 1) / 2) * gap
        for seg in line:
            start_x = x
            for ch in seg[0]:
                adv = f.measureText(ch)
                chars.append((ch, x, adv, y, seg[1]))
                x += adv
            if len(seg) > 2 and seg[2]:
                underlines.append((start_x, x, y + size * 0.5, len(chars) - 1))

    rng = np.random.default_rng(seed)
    falls = rng.uniform(-1, 1, (len(chars), 3))
    typed = t * cps
    falling = fall_at is not None and t > fall_at

    for x0, x1, y, last in underlines:
        u = ease_out_cubic(prog(t, (last + 1) / cps + 0.05, 0.35))
        if u > 0:
            fade = 1 - prog(t, fall_at, 0.25) if fall_at is not None else 1
            path = skia.Path()
            path.moveTo(x0, y)
            path.cubicTo(
                lerp(x0, x1, 0.3),
                y + size * 0.12,
                lerp(x0, x1, 0.6),
                y - size * 0.08,
                x1,
                y + size * 0.02,
            )
            seg = skia.Path()
            measure = skia.PathMeasure(path, False)
            measure.getSegment(0, measure.getLength() * u, seg, True)
            c.drawPath(seg, stroke(YELLOW, size * 0.1, fade))

    for i, (ch, x, adv, y, hex_rgb) in enumerate(chars):
        u = prog(t, i / cps, 0.24)
        if u <= 0 or ch == " ":
            continue
        s = ease_out_back(u, 2.4)
        dy = (1 - ease_out_cubic(u)) * size * 0.45
        rot = (1 - ease_out_cubic(u)) * 14 * falls[i, 0]
        alpha = 1.0
        if falling:
            te = max(0.0, t - fall_at - i * 0.01 - (falls[i, 1] + 1) * 0.06)
            dy += -500 * te + 0.5 * 5200 * te * te
            x += falls[i, 0] * 380 * te
            rot += falls[i, 2] * 420 * te
            alpha = 1 - prog(te, 0.25, 0.35)
        with Group(c, x + adv / 2, y + dy, s, rot, alpha) as g:
            g.drawString(ch, -adv / 2, -(m.fAscent + m.fDescent) / 2, f, fill(hex_rgb))

    if chars and not falling and typed > 0:
        done = typed >= len(chars)
        k = min(int(typed), len(chars) - 1)
        frac = clamp(typed - int(typed)) if not done else 1.0
        ch, x, adv, y, _ = chars[k]
        # Каретка догоняет последнюю букву с небольшим сглаживанием.
        cx = x + adv * ease_out_cubic(frac) + size * 0.06
        blink = 1.0 if not done or (t * 2.2) % 1 < 0.6 else 0.0
        c.drawRRect(rrect(cx, y, size * 0.075, size * 0.86, size * 0.03), fill(YELLOW, blink))


def text_scene(
    lines: list[list[Segment]],
    *,
    size: float = 150,
    cps: float = 17,
    delay: float = 0.55,
    fall_at: float | None = None,
    bg: int = BLUE,
):
    """Сцена с печатным текстом на весь экран."""

    def draw(c: skia.Canvas, t: float) -> None:
        c.clear(color(bg))
        typed_text(
            c,
            t - delay,
            lines,
            size=size,
            cps=cps,
            fall_at=None if fall_at is None else fall_at - delay,
        )

    return draw


def logo(
    c: skia.Canvas,
    t: float,
    cx: float,
    cy: float,
    size: float,
    start: float = 0.0,
    exit_at: float | None = None,
) -> None:
    """Буквы DEXQ выпрыгивают по очереди; при выходе улетают вверх."""
    f = font(size, True)
    m = f.getMetrics()
    letters = "DEXQ"
    widths = [f.measureText(ch) for ch in letters]
    x = cx - (sum(widths) + size * 0.02 * 3) / 2
    for i, (ch, adv) in enumerate(zip(letters, widths, strict=False)):
        u = prog(t, start + i * 0.09, 0.8)
        s = spring(u, 1.7, 5.5)
        rot = (1 - spring(u, 1.2, 5)) * (-18 if i % 2 else 18)
        dy = 0.0
        if exit_at is not None:
            e = prog(t, exit_at + i * 0.06, 0.45)
            dy = (
                -ease_in_cubic(e) * H * 0.9 + bump(t, exit_at + i * 0.06 - 0.12, 0.24) * size * 0.08
            )
            s *= 1 - 0.3 * e
        # Сплющивание при приземлении — мультяшный squash & stretch.
        squash = 1 + 0.18 * bump(t, start + i * 0.09 + 0.1, 0.35)
        with Group(c, x + adv / 2, cy + dy, s, rot, sx=squash, sy=1 / squash) as g:
            g.drawString(ch, -adv / 2, -(m.fAscent + m.fDescent) / 2, f, fill(WHITE))
        x += adv + size * 0.02
    # Жёлтая точка — «пиксель качества» — отскакивает над Q.
    u = prog(t, start + 0.5, 0.9)
    if u > 0:
        qx = x - widths[-1] / 2 - size * 0.02
        bounce = abs(math.cos(u * math.pi * 2.5)) * (1 - u) ** 1.5
        y = cy - size * 0.62 - bounce * size * 0.9
        if exit_at is not None:
            y -= ease_in_cubic(prog(t, exit_at + 0.2, 0.45)) * H
        c.drawCircle(qx + size * 0.34, y, size * 0.075 * spring(u, 1, 6), fill(YELLOW))


def intro(c: skia.Canvas, t: float) -> None:
    """Логотип, подпись и первая печатная фраза."""
    c.clear(color(BLUE))
    # Жёлтая капля падает и раскрывается кольцом — старт ролика.
    drop = prog(t, 0.0, 0.55)
    if drop < 1:
        y = lerp(-80, H / 2, ease_in_cubic(drop))
        c.drawCircle(W / 2, y, 36, fill(YELLOW))
    ring = prog(t, 0.55, 0.7)
    if 0 < ring < 1:
        c.drawCircle(
            W / 2, H / 2, 40 + 900 * ease_out_cubic(ring), stroke(YELLOW, 40 * (1 - ring) + 1)
        )
    logo(c, t, W / 2, H / 2 - 30, 300, start=0.55, exit_at=3.0)
    tag = "контроль качества DXA"
    shown = int(clamp((t - 1.5) * 22, 0, len(tag)))
    with Group(c, 0, -ease_in_cubic(prog(t, 3.05, 0.4)) * 700, alpha=1 - prog(t, 3.05, 0.3)):
        text(c, tag[:shown], W / 2, H / 2 + 150, 54, WHITE, 0.85, bold=False)
    typed_text(
        c,
        t - 3.55,
        [
            [("Снимок сделан.", WHITE)],
            [("Но ", WHITE), ("годится", YELLOW, True), (" ли он?", WHITE)],
        ],
        size=160,
        cps=16,
        fall_at=7.7 - 3.55,
    )


# ---------------------------------------------------------------- камера


class Camera:
    """Камера: центр кадра в мировых координатах и приближение."""

    def __init__(self, cx: float, cy: float, zoom: float):
        self.cx, self.cy, self.zoom = cx, cy, zoom

    def apply(self, c: skia.Canvas) -> None:
        c.translate(W / 2, H / 2)
        c.scale(self.zoom, self.zoom)
        c.translate(-self.cx, -self.cy)

    def to_screen(self, x: float, y: float) -> tuple[float, float]:
        return (x - self.cx) * self.zoom + W / 2, (y - self.cy) * self.zoom + H / 2


def screen_cursor(
    c: skia.Canvas, cam: Camera, x: float, y: float, scale: float = 1.0, alpha: float = 1.0
) -> None:
    """Курсор поверх сцены: растёт при наезде камеры, но мягче самой сцены."""
    sx, sy = cam.to_screen(x, y)
    cursor(c, sx, sy, scale * cam.zoom**0.55, alpha)


def catmull(points: list[tuple[float, float]], u: float) -> tuple[float, float]:
    """Гладкая кривая Катмулла — Рома через все точки, u в 0..1."""
    n = len(points) - 1
    f = clamp(u) * n
    i = min(int(f), n - 1)
    s = f - i
    p0, p1, p2, p3 = points[max(i - 1, 0)], points[i], points[i + 1], points[min(i + 2, n)]
    return tuple(
        0.5
        * (
            2 * b
            + (-a + c_) * s
            + (2 * a - 5 * b + 4 * c_ - d) * s * s
            + (-a + 3 * b - 3 * c_ + d) * s**3
        )
        for a, b, c_, d in zip(p0, p1, p2, p3, strict=False)
    )


# ---------------------------------------------------------------- загрузка

CARD = (1180, 560, 1040, 800)
DROP = (1180, 505, 920, 480)
PICK_BTN = (900, 868, 380)
CHECK_BTN = (1450, 868, 330)
PILE = (330, 560)
THUMB = (170, 215)
PILE_KINDS = ["hip", "spine", "hip", "spine", "spine", "spine"]
PILE_REST = [(-26, 14, -9), (22, -8, 7), (-8, 6, -4), (26, 16, 10), (-18, -12, -6), (0, 0, 2)]
GRAB, RELEASE = 4.95, 8.75
DRAG_PATH = [(330, 500), (500, 300), (720, 640), (960, 330), (1120, 560), (1190, 400)]


def drag_u(t: float) -> float:
    """Доля пройденного пути перетаскивания: мягкий разгон и торможение."""
    return 0.5 - 0.5 * math.cos(math.pi * prog(t, GRAB + 0.2, RELEASE - GRAB - 0.3))


def upload_cursor(t: float) -> tuple[float, float]:
    start = (2080, 1240)
    pick = (PICK_BTN[0] + 70, PICK_BTN[1] + 24)
    grab = DRAG_PATH[0]
    check = (CHECK_BTN[0] + 70, CHECK_BTN[1] + 24)
    if t < 3.5:
        return glide(t, 1.3, 1.1, start, pick, (60, -260))
    if t < GRAB:
        return glide(t, 3.6, 1.15, pick, grab, (-120, -220))
    if t < RELEASE:
        return catmull(DRAG_PATH, drag_u(t))
    return glide(t, 9.9, 1.0, DRAG_PATH[-1], check, (160, -120))


@cache
def pile_sim() -> np.ndarray:
    """Физика пачки при перетаскивании: пружины с запаздыванием и раскачка.

    Возвращает массив [кадр, карточка, (x, y, угол)] от GRAB до RELEASE.
    """
    sub = 4
    dt = 1 / (FPS * sub)
    n = len(PILE_REST)
    grip = np.array([6.0, 70.0])
    pos = np.array([[PILE[0] + dx, PILE[1] + dy] for dx, dy, _ in PILE_REST], dtype=float)
    vel = np.zeros((n, 2))
    theta = np.zeros(n)
    omega = np.zeros(n)
    frames = []
    steps = int((RELEASE - GRAB) * FPS) + 2
    for f in range(steps):
        for s in range(sub):
            t = GRAB + (f * sub + s) * dt
            cur = np.array(upload_cursor(t))
            for k in range(n):
                dx, dy, _ = PILE_REST[k]
                target = cur + grip + np.array([dx, dy]) * 0.55
                stiff = 38 + 52 * k
                acc = stiff * (target - pos[k]) - 2 * math.sqrt(stiff) * 0.32 * vel[k]
                vel[k] += acc * dt
                pos[k] += vel[k] * dt
                # Маятник: ускорение раскачивает карточку, затем она успокаивается.
                alpha = -110 * theta[k] - 7 * omega[k] + acc[0] * 0.035
                omega[k] += alpha * dt
                theta[k] += omega[k] * dt
        frames.append(
            [
                (
                    pos[k, 0],
                    pos[k, 1],
                    PILE_REST[k][2] * 0.6 + theta[k] + clamp(vel[k, 0] * 0.02, -26, 26),
                )
                for k in range(n)
            ]
        )
    return np.array(frames)


def grid_slot(k: int) -> tuple[float, float]:
    col, row = k % 3, k // 3
    return DROP[0] + (col - 1) * 230, DROP[1] + (row - 0.5) * 232


def pile_card(t: float, k: int) -> tuple[float, float, float, float, float] | None:
    """Положение карточки k: (x, y, угол, масштаб, подъём тени)."""
    appear = pop(t, 3.35 + k * 0.07, 0.6)
    if appear <= 0:
        return None
    dx, dy, rot = PILE_REST[k]
    if t < GRAB:
        return PILE[0] + dx, PILE[1] + dy, rot, appear, 0.6
    lift = 0.6 + 1.2 * ease_out_cubic(prog(t, GRAB, 0.25))
    sim = pile_sim()
    if t < RELEASE:
        x, y, a = sim[min(int((t - GRAB) * FPS), len(sim) - 1), k]
        return x, y, a, 1 + 0.07 * ease_out_back(prog(t, GRAB, 0.3)), lift
    x0, y0, a0 = sim[-1, k]
    gx, gy = grid_slot(5 - k)
    u = prog(t, RELEASE + (5 - k) * 0.06, 0.6)
    e = ease_out_back(u, 1.4)
    land = bump(t, RELEASE + (5 - k) * 0.06 + 0.35, 0.3)
    return (
        lerp(x0, gx, e),
        lerp(y0, gy, e) - 60 * bump(t, RELEASE + (5 - k) * 0.06, 0.6),
        lerp(a0, 0, ease_out_cubic(u)),
        lerp(1.07, 1.0, u) * (1 - 0.05 * land),
        lerp(lift, 0.3, u),
    )


def dropzone(c: skia.Canvas, t: float, hover: float, filled: float) -> None:
    cx, cy, w, h = DROP
    s = pop(t, 0.7) * (1 + 0.025 * hover)
    if s <= 0:
        return
    with Group(c, cx, cy, s):
        c.drawRRect(rrect(0, 0, w, h, 34), fill(lerp_hex(LAV, 0xDCE1FF, hover)))
        border = stroke(lerp_hex(0xB9C0F5, BLUE, hover), 5)
        border.setPathEffect(skia.DashPathEffect.Make([22, 16], -t * 60))
        c.drawRRect(rrect(0, 0, w - 4, h - 4, 32), border)
        alpha = 1 - filled
        if alpha > 0:
            bob = math.sin(t * 3.2) * 8
            with Group(c, 0, -60 + bob, pop(t, 0.9), alpha=alpha * (1 - hover)):
                c.drawCircle(0, 0, 58, fill(BLUE))
                arrow = skia.Path()
                arrow.moveTo(0, 26)
                arrow.lineTo(0, -22)
                arrow.moveTo(-20, -2)
                arrow.lineTo(0, -24)
                arrow.lineTo(20, -2)
                c.drawPath(arrow, stroke(WHITE, 9))
            label = "Отпустите" if hover > 0.5 else "Перетащите снимки"
            with Group(
                c,
                0,
                70 + 100 * ease_out_back(hover),
                pop(t, 1.0) * (1 + 0.1 * bump(hover, 0.3, 0.4)),
                alpha=alpha,
            ):
                text(c, label, 0, 0, 46, BLUE if hover > 0.5 else INK)
                text(c, "DICOM · PNG · ZIP", 0, 58, 28, MUTED, 1 - hover, bold=False)


def lerp_hex(a: int, b: int, u: float) -> int:
    u = clamp(u)
    return sum(int(lerp(a >> s & 255, b >> s & 255, u)) << s for s in (16, 8, 0))


def upload_camera(t: float) -> Camera:
    keys = [
        (0.0, (1180, 560, 0.94)),
        (1.6, (1150, 580, 1.0)),
        (2.5, (PICK_BTN[0], PICK_BTN[1] - 10, 2.1)),
        (3.3, (PICK_BTN[0], PICK_BTN[1] - 10, 2.1)),
        (4.3, (900, 560, 1.0)),
        (5.1, (470, 520, 1.45)),
        (8.3, (1100, 500, 1.2)),
        (9.7, (1150, 540, 1.02)),
        (10.9, (CHECK_BTN[0], CHECK_BTN[1] - 10, 2.25)),
        (12.0, (CHECK_BTN[0], CHECK_BTN[1] - 10, 2.45)),
    ]
    return Camera(*keyframes(t, keys))


def upload(c: skia.Canvas, t: float) -> None:
    """Выбор файлов, перетаскивание пачки снимков и запуск проверки."""
    c.clear(color(LAV))
    cam = upload_camera(t)
    c.save()
    cam.apply(c)
    cx, cy, w, h = CARD
    s = pop(t, 0.3, 0.8)
    if s > 0:
        with Group(c, cx, cy, s):
            soft_card(c, 0, 0, w, h, 44)
        with Group(c, cx - w / 2 + 64, cy - h / 2 + 50, pop(t, 0.55)):
            text(c, "DEXQ", 0, 0, 40, BLUE, align="left")
            c.drawCircle(font(40).measureText("DEXQ") + 14, -14, 7, fill(YELLOW))
    # Зона подсвечивается на последнем отрезке пути, когда пачка уже над ней.
    hover = clamp((drag_u(t) - 0.8) / 0.08) * (1 - prog(t, RELEASE, 0.3))
    dropzone(c, t, hover, prog(t, RELEASE, 0.2))
    pill_button(
        c,
        t,
        *PICK_BTN[:2],
        "Выбрать файлы",
        w=PICK_BTN[2],
        hover=bump(t, 2.35, 1.0),
        press_at=2.9,
        appear=pop(t, 1.0),
    )
    ready = pop(t, 9.55, 0.6)
    pill_button(
        c,
        t,
        *CHECK_BTN[:2],
        "Проверить",
        w=CHECK_BTN[2],
        hex_rgb=lerp_hex(LINE, BLUE, ready),
        text_hex=lerp_hex(MUTED, WHITE, ready),
        hover=prog(t, 10.9, 0.3),
        press_at=11.45,
        appear=pop(t, 1.1) * (1 + 0.12 * bump(t, 9.55, 0.4)),
    )
    ripple(c, t, 2.9, PICK_BTN[0], PICK_BTN[1], BLUE)
    ripple(c, t, 11.45, CHECK_BTN[0], CHECK_BTN[1], BLUE)

    # Снимки: нижние карточки рисуются первыми.
    for k in range(len(PILE_REST)):
        state = pile_card(t, k)
        if state is None:
            continue
        x, y, rot, scale, lift = state
        with Group(c, x, y, scale, rot) as g:
            g.drawRRect(
                rrect(8 * lift, 16 * lift, *THUMB, 18), fill(BLUE_DARK, 0.18, 6 + 12 * lift)
            )
            xray(g, PILE_KINDS[k], *THUMB, frame=9)
    badge = pop(t, RELEASE + 0.7)
    if badge > 0:
        bx, by = DROP[0] + DROP[2] / 2 - 18, DROP[1] - DROP[3] / 2 + 18
        with Group(c, bx, by, badge):
            c.drawCircle(0, 0, 44, fill(YELLOW))
            text(c, "6", 0, 2, 46, INK)
    c.restore()

    press = 2.9 if t < 3.5 else GRAB if t < 9 else 11.45
    scale = press_scale(t, press) if abs(t - press) < 0.6 else 1.0
    if GRAB < t < RELEASE:
        scale *= 0.9
    screen_cursor(c, cam, *upload_cursor(t), scale, alpha=prog(t, 1.3, 0.2))


# ---------------------------------------------------------------- проверка

XR = (600, 545, 540, 760)
XR_FRAME = 14
ROWS_X, ROW_W, ROW_H = 1370, 620, 116
ROWS = [
    ("Укладка", 330, 5.0, True),
    ("Ось позвоночника", 470, 5.7, True),
    ("Артефакты", 610, 6.4, False),
]
EXPORT_BTN = (1370, 895, 420)
EXPORT_CLICK = 12.8


def partial(c: skia.Canvas, path: skia.Path, u: float, paint: skia.Paint) -> None:
    """Рисует начальную долю u контура — для «прорисовки» линий."""
    if u <= 0:
        return
    measure = skia.PathMeasure(path, False)
    seg = skia.Path()
    measure.getSegment(0, measure.getLength() * clamp(u), seg, True)
    c.drawPath(seg, paint)


def line_path(*pts: tuple[float, float]) -> skia.Path:
    path = skia.Path()
    path.moveTo(*pts[0])
    for p in pts[1:]:
        path.lineTo(*p)
    return path


def status_icon(
    c: skia.Canvas, t: float, x: float, y: float, r: float, done_at: float, ok: bool
) -> None:
    """Спиннер, который превращается в галочку или крестик."""
    if t < done_at:
        c.drawCircle(x, y, r, fill(LAV))
        arc = stroke(BLUE, r * 0.22)
        c.drawArc(
            skia.Rect.MakeXYWH(x - r * 0.6, y - r * 0.6, r * 1.2, r * 1.2),
            t * 420 % 360,
            250,
            False,
            arc,
        )
        return
    s = pop(t, done_at, 0.5)
    hex_rgb = MINT if ok else CORAL
    with Group(c, x, y, s) as g:
        g.drawCircle(0, 0, r, fill(hex_rgb))
        u = ease_out_cubic(prog(t, done_at + 0.08, 0.3))
        mark = stroke(WHITE, r * 0.2)
        if ok:
            partial(
                g,
                line_path((-0.42 * r, 0.02 * r), (-0.1 * r, 0.34 * r), (0.46 * r, -0.3 * r)),
                u,
                mark,
            )
        else:
            k = 0.32 * r
            partial(g, line_path((-k, -k), (k, k)), u * 2, mark)
            partial(g, line_path((k, -k), (-k, k)), u * 2 - 1, mark)


def vertebra_center(k: int) -> tuple[float, float]:
    iw, ih = XR[2] - 2 * XR_FRAME, XR[3] - 2 * XR_FRAME
    return XR[0] + math.sin(k * 0.8) * iw * 0.025, XR[1] - ih * 0.34 + k * ih * 0.15


def artifact_point() -> tuple[float, float]:
    iw, ih = XR[2] - 2 * XR_FRAME, XR[3] - 2 * XR_FRAME
    return XR[0] + iw * 0.2, XR[1] + ih * 0.02


def analysis_camera(t: float) -> Camera:
    ax, ay = artifact_point()
    keys = [
        (0.0, (600, 570, 1.45)),
        (3.4, (600, 540, 1.25)),
        (4.4, (980, 560, 1.0)),
        (6.7, (1000, 560, 1.03)),
        (7.7, (ax + 60, ay, 2.5)),
        (9.4, (ax + 70, ay, 2.6)),
        (10.4, (980, 560, 1.0)),
        (11.5, (1000, 580, 1.02)),
        (12.4, (EXPORT_BTN[0], EXPORT_BTN[1] - 10, 2.2)),
        (13.0, (EXPORT_BTN[0], EXPORT_BTN[1] - 10, 2.3)),
        (14.0, (960, 540, 1.0)),
        (18.0, (960, 540, 1.06)),
    ]
    return Camera(*keyframes(t, keys))


def csv_document(c: skia.Canvas, t: float) -> None:
    """Файл CSV вылетает из кнопки экспорта и заполняется строками."""
    u = prog(t, EXPORT_CLICK + 0.15, 0.85)
    if u <= 0:
        return
    x, y = bezier(
        (EXPORT_BTN[0], EXPORT_BTN[1]), (1500, 300), (1000, 200), (960, 540), ease_out_cubic(u)
    )
    land = bump(t, EXPORT_CLICK + 0.95, 0.35)
    rot = -24 * (1 - spring(u, 1.2, 4.5))
    with Group(
        c, x, y, lerp(0.15, 1.0, ease_out_back(u, 1.2)), rot, sx=1 + 0.08 * land, sy=1 - 0.08 * land
    ) as g:
        soft_card(g, 0, 0, 560, 680, 36, lift=1.4)
        corner = line_path((200, -340), (280, -260), (200, -260))
        corner.close()
        g.drawPath(corner, fill(LINE))
        text(g, "results.csv", -220, -262, 42, INK, align="left")
        with Group(g, 140, -262, pop(t, EXPORT_CLICK + 0.9)):
            g.drawRRect(rrect(0, 0, 96, 50, 25), fill(MINT))
            text(g, "CSV", 0, 1, 26, WHITE)
        for k in range(7):
            ry = -165 + k * 64
            r = ease_out_cubic(prog(t, EXPORT_CLICK + 1.1 + k * 0.12, 0.4))
            if r <= 0:
                continue
            g.drawCircle(-220, ry, 13 * ease_out_back(r), fill(CORAL if k == 2 else MINT))
            g.drawRRect(rrect(-185 + 80 * r, ry, 160 * r, 22, 11), fill(LINE))
            bar = 110 + (k * 53) % 120
            g.drawRRect(rrect(5 + bar / 2 * r, ry, bar * r, 22, 11), fill(0xC9CDF5))
        stamp = pop(t, EXPORT_CLICK + 2.2, 0.6)
        if stamp > 0:
            with Group(g, 230, 300, stamp, -12 * (1 - stamp)):
                status_icon(g, t, 0, 0, 62, EXPORT_CLICK + 2.2, True)


def analysis(c: skia.Canvas, t: float) -> None:
    """Скан снимка, найденные точки оси, три проверки, артефакт и экспорт CSV."""
    c.clear(color(LAV))
    cam = analysis_camera(t)
    c.save()
    cam.apply(c)
    xs = pop(t, 0.25, 0.8)
    glow = 0.5 + 0.5 * math.sin(t * 7) if t > 6.4 else 0.0
    with Group(c, XR[0], XR[1], xs, -6 * (1 - xs)) as g:
        g.drawRRect(rrect(10, 22, XR[2], XR[3], 50), fill(BLUE_DARK, 0.16, 26))
        xray(g, "spine", XR[2], XR[3], frame=XR_FRAME, artifact=True, glow=glow)

    # Сканирующая линия: вниз, затем вверх.
    iw, ih = XR[2] - 2 * XR_FRAME, XR[3] - 2 * XR_FRAME
    top = XR[1] - ih / 2
    for start, dur, down in ((1.0, 1.2, True), (2.2, 0.9, False)):
        u = prog(t, start, dur)
        if 0 < u < 1:
            e = ease_in_out(u)
            y = top + ih * (e if down else 1 - e)
            c.save()
            c.clipRRect(rrect(XR[0], XR[1], iw, ih, 30), True)
            trail = skia.Paint(AntiAlias=True)
            y0 = y - 180 if down else y + 180
            trail.setShader(
                skia.GradientShader.MakeLinear(
                    [skia.Point(0, y0), skia.Point(0, y)], [color(MINT, 0), color(MINT, 0.45)]
                )
            )
            c.drawRect(
                skia.Rect.MakeLTRB(XR[0] - iw / 2, min(y0, y), XR[0] + iw / 2, max(y0, y)), trail
            )
            c.drawRect(skia.Rect.MakeXYWH(XR[0] - iw / 2, y - 3, iw, 6), fill(WHITE))
            c.restore()

    # Точки позвонков и ось относительно вертикали.
    p0, p4 = vertebra_center(0), vertebra_center(4)
    ref = stroke(WHITE, 5, 0.55)
    ref.setPathEffect(skia.DashPathEffect.Make([14, 12], 0))
    partial(c, line_path(p4, (p4[0], p0[1] - 40)), ease_out_cubic(prog(t, 2.9, 0.5)), ref)
    partial(
        c,
        line_path(p4, (p0[0] + 16, p0[1] - 40)),
        ease_out_cubic(prog(t, 3.0, 0.6)),
        stroke(YELLOW, 8),
    )
    for k in range(5):
        s = pop(t, 1.5 + k * 0.22, 0.5)
        if s > 0:
            x, y = vertebra_center(k)
            with Group(c, x, y, s):
                c.drawCircle(0, 0, 16, fill(WHITE))
                c.drawCircle(0, 0, 10, fill(MINT))
    angle = pop(t, 3.5)
    if angle > 0:
        with Group(c, p0[0] + 120, p0[1] - 20, angle):
            c.drawRRect(rrect(0, 0, 130, 62, 31), fill(YELLOW))
            text(c, "1.8°", 0, 2, 34, INK)

    # Артефакт: пульсирующие кольца и выноска.
    ax, ay = artifact_point()
    if t > 6.5:
        for k in range(2):
            u = ((t - 6.5) * 0.9 + k * 0.5) % 1
            c.drawCircle(ax, ay, 28 + 60 * ease_out_cubic(u), stroke(CORAL, 6 * (1 - u) + 1, 1 - u))
        c.drawCircle(ax, ay, 34 * pop(t, 6.5), stroke(CORAL, 6))
    callout = pop(t, 7.7, 0.6) * (1 - prog(t, 9.6, 0.3))
    if callout > 0:
        with Group(c, ax + 58, ay - 58, callout, alpha=callout) as g:
            bubble = rrect(125, -18, 250, 64, 32)
            g.drawRRect(bubble, fill(CORAL))
            text(g, "Посторонний предмет", 125, -16, 21, WHITE)

    # Панель проверок.
    title = pop(t, 3.8)
    if title > 0:
        with Group(c, ROWS_X - ROW_W / 2, 215, title):
            text(c, "Позвоночник", 0, 0, 54, INK, align="left")
    for i, (label, y, done_at, ok) in enumerate(ROWS):
        s = pop(t, 3.9 + i * 0.18, 0.6)
        if s <= 0:
            continue
        shake = (
            math.sin(t * 70) * 14 * math.exp(-6 * (t - done_at)) if not ok and t > done_at else 0.0
        )
        slide = (1 - ease_out_cubic(prog(t, 3.9 + i * 0.18, 0.5))) * 160
        with Group(c, ROWS_X + slide + shake, y, s) as g:
            soft_card(g, 0, 0, ROW_W, ROW_H, 34, 0.6)
            status_icon(g, t, -ROW_W / 2 + 70, 0, 34, done_at, ok)
            text(g, label, -ROW_W / 2 + 128, 0, 40, INK, align="left")
    verdict = pop(t, 10.4, 0.6)
    if verdict > 0:
        with Group(c, ROWS_X, 752, verdict) as g:
            g.drawRRect(rrect(0, 8, ROW_W, 100, 50), fill(INK, 0.9))
            g.drawRRect(rrect(0, 0, ROW_W, 100, 50), fill(CORAL))
            g.drawCircle(-ROW_W / 2 + 60, 0, 30, fill(WHITE))
            text(g, "!", -ROW_W / 2 + 60, 1, 40, CORAL)
            text(g, "Найдено нарушение", 30, 0, 42, WHITE)
    pill_button(
        c,
        t,
        *EXPORT_BTN[:2],
        "Экспорт CSV",
        w=EXPORT_BTN[2],
        hover=prog(t, 12.2, 0.3) * (1 - prog(t, 13.2, 0.3)),
        press_at=EXPORT_CLICK,
        appear=pop(t, 10.8),
    )
    ripple(c, t, EXPORT_CLICK, EXPORT_BTN[0], EXPORT_BTN[1], BLUE, 160)

    dim = prog(t, EXPORT_CLICK + 0.2, 0.5) * 0.82
    if dim > 0:
        c.drawRect(skia.Rect.MakeXYWH(-2000, -2000, 6000, 6000), fill(LAV, dim))
    csv_document(c, t)
    c.restore()

    target = (EXPORT_BTN[0] + 80, EXPORT_BTN[1] + 26)
    pos = glide(t, 11.0, 1.2, (2100, 1250), target, (-80, -240))
    if t > EXPORT_CLICK + 0.4:
        pos = glide(t, EXPORT_CLICK + 0.4, 0.7, target, (2300, 1300), (0, 0))
    press = press_scale(t, EXPORT_CLICK) if abs(t - EXPORT_CLICK) < 0.6 else 1.0
    screen_cursor(c, cam, *pos, press, alpha=prog(t, 11.0, 0.2))


# ---------------------------------------------------------------- пакет и финал

BATCH_COLS, BATCH_ROWS = 8, 3
BATCH_THUMB = (150, 190)
BATCH_FLAGGED = {3, 10, 17, 20}
BURST = 1.35


def batch_slot(i: int) -> tuple[float, float]:
    col, row = i % BATCH_COLS, i // BATCH_COLS
    return 960 + (col - (BATCH_COLS - 1) / 2) * 178, 600 + (row - 1) * 218


def zip_icon(c: skia.Canvas, s: float, rot: float) -> None:
    with Group(c, 960, 560, s, rot) as g:
        g.drawRRect(rrect(0, 12, 240, 300, 36), fill(INK, 0.9))
        g.drawRRect(rrect(0, 0, 240, 300, 36), fill(YELLOW))
        for k in range(6):
            g.drawRect(skia.Rect.MakeXYWH(-14 if k % 2 else 0, -150 + k * 22, 14, 22), fill(INK))
        g.drawRRect(rrect(0, 0, 44, 58, 12), fill(INK))
        text(g, "ZIP", 0, 90, 64, INK)


def batch(c: skia.Canvas, t: float) -> None:
    """Архив раскрывается в сетку снимков, по сетке проходит волна проверок."""
    c.clear(color(LAV))
    cam = Camera(
        *keyframes(
            t,
            [
                (0.0, (960, 560, 2.0)),
                (BURST, (960, 560, 1.8)),
                (2.4, (960, 540, 1.0)),
                (6.5, (960, 540, 1.07)),
            ],
        )
    )
    c.save()
    cam.apply(c)
    zs = pop(t, 0.2, 0.7) * (1 - ease_in_cubic(prog(t, BURST, 0.25)))
    if zs > 0:
        shake = math.sin(t * 50) * 8 * bump(t, 0.75, BURST - 0.75)
        zip_icon(c, zs * (1 + 0.1 * bump(t, BURST - 0.25, 0.3)), shake)
    flash = prog(t, BURST, 0.6)
    if 0 < flash < 1:
        c.drawCircle(
            960, 560, 60 + 900 * ease_out_cubic(flash), stroke(YELLOW, 50 * (1 - flash) + 1)
        )

    rng = np.random.default_rng(7)
    spin = rng.uniform(-40, 40, BATCH_COLS * BATCH_ROWS)
    checked = 0
    for i in range(BATCH_COLS * BATCH_ROWS):
        gx, gy = batch_slot(i)
        delay = BURST + math.hypot(gx - 960, gy - 560) / 1400 * 0.4
        u = prog(t, delay, 0.75)
        if u <= 0:
            continue
        e = ease_out_back(u, 1.3)
        x, y = lerp(960, gx, e), lerp(560, gy, e) - 160 * bump(t, delay, 0.75)
        with Group(
            c, x, y, lerp(0.3, 1, ease_out_cubic(u)), spin[i] * (1 - ease_out_cubic(u))
        ) as g:
            g.drawRRect(rrect(6, 12, *BATCH_THUMB, 16), fill(BLUE_DARK, 0.15, 10))
            xray(g, "spine" if i % 3 else "hip", *BATCH_THUMB, frame=8)
        col, row = i % BATCH_COLS, i // BATCH_COLS
        done_at = 2.6 + (col + row) * 0.1
        if t > done_at:
            checked += 1
            status_icon(c, t, gx + 62, gy - 82, 26, done_at, i not in BATCH_FLAGGED)

    head = pop(t, 2.3)
    if head > 0:
        with Group(c, 960, 255, head):
            width = 1400
            c.drawRRect(rrect(0, 0, width, 18, 9), fill(LINE))
            done = ease_in_out(prog(t, 2.5, 2.1))
            if done > 0:
                c.drawRRect(
                    rrect(-width / 2 + width * done / 2, 0, width * done, 18, 9), fill(BLUE)
                )
            text(
                c,
                f"Проверено {checked} из {BATCH_COLS * BATCH_ROWS}",
                -width / 2,
                -52,
                44,
                INK,
                align="left",
            )
            flagged = sum(
                1 for i in BATCH_FLAGGED if t > 2.6 + (i % BATCH_COLS + i // BATCH_COLS) * 0.1
            )
            if flagged:
                text(c, f"нарушений: {flagged}", width / 2, -52, 40, CORAL, align="right")
    c.restore()


def outro(c: skia.Canvas, t: float) -> None:
    """Логотип, подпись и кнопка демо с финальным кликом."""
    c.clear(color(BLUE))
    logo(c, t, W / 2, 400, 240, start=0.1)
    tag = "Контроль качества DXA-снимков"
    shown = int(clamp((t - 0.8) * 30, 0, len(tag)))
    text(c, tag[:shown], W / 2, 580, 54, WHITE, 0.9, bold=False)
    click = 2.75
    for k in range(2):
        u = ((t - 1.9) * 0.8 + k * 0.5) % 1 if t > 1.9 else 0
        if u > 0:
            c.drawRRect(
                rrect(W / 2, 745, 470 + 120 * u, 100 + 120 * u, 50 + 60 * u),
                stroke(WHITE, 4 * (1 - u), 1 - u),
            )
    pill_button(
        c,
        t,
        W / 2,
        745,
        "Попробовать демо",
        w=470,
        h=100,
        hex_rgb=WHITE,
        text_hex=BLUE,
        hover=prog(t, 2.4, 0.2),
        press_at=click,
        appear=pop(t, 1.3),
    )
    ripple(c, t, click, W / 2, 745, YELLOW, 200)
    text(
        c,
        "Исследовательский прототип: контроль качества, не диагноз",
        W / 2,
        1015,
        26,
        WHITE,
        0.6 * prog(t, 1.6, 0.5),
        bold=False,
    )
    pos = glide(t, 1.7, 0.85, (1560, 1200), (W / 2 + 90, 772), (80, -160))
    press = press_scale(t, click) if abs(t - click) < 0.6 else 1.0
    cursor(c, *pos, press, alpha=prog(t, 1.7, 0.15))


# ---------------------------------------------------------------- монтаж

TRANSITION = 0.85


@dataclass(frozen=True)
class Scene:
    """Сцена: начало на общей шкале, отрисовка и точка, из которой она раскрывается."""

    start: float
    draw: Callable[[skia.Canvas, float], None]
    origin: tuple[float, float] = (W / 2, H / 2)
    rings: tuple[int, int] = (YELLOW, WHITE)


def reveal(c: skia.Canvas, u: float, scene: Scene, t_local: float) -> None:
    """Раскрытие новой сцены тремя концентрическими кругами из точки."""
    ox, oy = scene.origin
    far = max(math.hypot(ox - x, oy - y) for x in (0, W) for y in (0, H)) + 40
    radius = [far * ease_in_out_quint(clamp((u - k * 0.13) / 0.74)) for k in range(3)]
    for r, hex_rgb in zip(radius, scene.rings, strict=False):
        if r > 0:
            c.drawCircle(ox, oy, r, fill(hex_rgb))
    if radius[2] > 0:
        c.save()
        clip = skia.Path()
        clip.addCircle(ox, oy, radius[2])
        c.clipPath(clip, skia.ClipOp.kIntersect, True)
        scene.draw(c, t_local)
        c.restore()


SCENES = [
    Scene(0.0, intro),
    Scene(8.5, upload, rings=(YELLOW, BLUE)),
    Scene(
        20.25,
        text_scene(
            [[("Укладка.", WHITE)], [("Ось.", WHITE)], [("Артефакты.", YELLOW, True)]],
            size=170,
            cps=15,
            fall_at=3.55,
        ),
        origin=upload_camera(11.75).to_screen(*CHECK_BTN[:2]),
    ),
    Scene(24.5, analysis, rings=(YELLOW, BLUE)),
    Scene(
        42.5,
        text_scene(
            [
                [("Один снимок —", WHITE)],
                [("или ", WHITE), ("целый архив", YELLOW, True), (".", WHITE)],
            ],
            size=150,
            cps=17,
        ),
    ),
    Scene(46.0, batch, rings=(YELLOW, BLUE)),
    Scene(
        52.5,
        text_scene(
            [
                [("Проверьте качество,", WHITE)],
                [("пока пациент ", WHITE), ("рядом", YELLOW, True), (".", WHITE)],
            ],
            size=140,
            cps=19,
            fall_at=3.3,
        ),
    ),
    Scene(56.3, outro, rings=(YELLOW, WHITE)),
]
DURATION = 60.0


def render(c: skia.Canvas, t: float) -> None:
    i = max(k for k, s in enumerate(SCENES) if s.start <= t)
    scene = SCENES[i]
    u = prog(t, scene.start, TRANSITION)
    if i == 0 or u >= 1:
        scene.draw(c, t - scene.start)
        return
    prev = SCENES[i - 1]
    prev.draw(c, t - prev.start)
    reveal(c, u, scene, t - scene.start)


def frame(surface: skia.Surface, t: float) -> np.ndarray:
    c = surface.getCanvas()
    c.clear(color(WHITE))
    render(c, t)
    return surface.makeImageSnapshot().toarray(colorType=skia.kRGBA_8888_ColorType)


_worker_surface: skia.Surface | None = None


def frame_bytes(n: int) -> bytes:
    """Кадр n для пула процессов: у каждого процесса своя поверхность."""
    global _worker_surface
    if _worker_surface is None:
        _worker_surface = skia.Surface(W, H)
    return frame(_worker_surface, n / FPS).tobytes()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=OUTPUT)
    parser.add_argument(
        "--still", type=float, nargs="*", help="сохранить PNG кадры на указанных секундах"
    )
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--end", type=float, default=DURATION)
    args = parser.parse_args()

    surface = skia.Surface(W, H)
    if args.still:
        folder = Path(tempfile.gettempdir()) / "dexq-stills"
        folder.mkdir(exist_ok=True)
        for t in args.still:
            path = folder / f"still-{t:05.2f}.png"
            frame(surface, t)
            surface.makeImageSnapshot().save(str(path), skia.kPNG)
            print(path)
        return

    import os
    from multiprocessing import Pool

    import imageio_ffmpeg

    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = imageio_ffmpeg.write_frames(
        str(args.out),
        (W, H),
        pix_fmt_in="rgba",
        fps=FPS,
        codec="libx264",
        macro_block_size=8,
        output_params=["-crf", "18", "-preset", "slow", "-movflags", "+faststart"],
    )
    writer.send(None)
    first, last = round(args.start * FPS), round(args.end * FPS)
    # Размытые тени дороги на CPU, поэтому кадры рисуются параллельно, а
    # кодируются строго по порядку.
    with Pool(max(1, (os.cpu_count() or 2) - 1)) as pool:
        for n, data in enumerate(pool.imap(frame_bytes, range(first, last), chunksize=4), first):
            writer.send(data)
            if n % FPS == 0:
                print(f"{n / FPS:5.1f} с", flush=True)
    writer.close()
    print(args.out)


if __name__ == "__main__":
    main()
