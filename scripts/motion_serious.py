"""Строгий промо-ролик DEXQ: спокойный tech-стиль и упрощённое рабочее пространство.

Вторая версия к motion_demo.py: без перелётов и сплющиваний, медленные плавные
кривые, графитовая тема workspace и светлые «бумажные» кадры с печатным текстом.
Снимки процедурные, а не изображения пациентов.

    .venv-motion/Scripts/python scripts/motion_serious.py
    .venv-motion/Scripts/python scripts/motion_serious.py --still 5 20

"""

from __future__ import annotations

import math
from functools import cache

import numpy as np
import skia
from motion_demo import (
    FPS,
    ROOT,
    Camera,
    Group,
    H,
    W,
    bezier,
    catmull,
    clamp,
    color,
    ease_in_out,
    ease_in_out_quint,
    ease_out_cubic,
    fill,
    font,
    keyframes,
    lerp,
    line_path,
    partial,
    prog,
    rrect,
    run,
    stroke,
    text,
)

OUTPUT = ROOT / "frontend" / "public" / "media" / "dexq-demo-serious.mp4"
DURATION = 60.0

PAPER = 0xF1EFEA
INK = 0x16171A
GRAY = 0x74777E
BRAND = 0x0202F1
# Графитовая тема из спецификации workspace.
BG = 0x191B1E
PANEL = 0x222529
RAISED = 0x2C3035
BORDER = 0x3B4047
TEXT = 0xEDF0F2
TEXT2 = 0xA8ADB5
ACCENT = 0x92ACCB
GREEN = 0x6CC59A
RED = 0xE4717A
VIEWER = 0x0E0F11


def ease_out_expo(u: float) -> float:
    return 1.0 if u >= 1 else 1 - 2 ** (-10 * u)


def fade(t: float, start: float, dur: float = 0.7) -> float:
    """Плавное появление без перелёта."""
    return ease_out_cubic(prog(t, start, dur))


# ---------------------------------------------------------------- процедурный снимок

IMG_W, IMG_H = 600, 820
VERTEBRAE = [(300 + 4 * k, 118 + k * 128) for k in range(5)]
ARTIFACT = (392, 372)


@cache
def study_image(kind: str = "spine", artifact: bool = True) -> skia.Image:
    """Схематичный DXA-подобный снимок: мягкие кости, шум и металлический предмет.

    Args:
        kind: "spine" — поясничный отдел, "hip" — проксимальный отдел бедра.
        artifact: Добавить яркий посторонний предмет (только для позвоночника).

    Returns:
        Изображение Skia размером IMG_W × IMG_H.
    """
    y, x = np.mgrid[0:IMG_H, 0:IMG_W].astype(np.float32)
    img = 0.05 + 0.03 * (y / IMG_H)
    img += 0.13 * np.exp(-(((x - 300) / 230) ** 2))

    def blob(cx, cy, a, b, power=4.0, level=0.5, rim=0.18):
        d = (np.abs(x - cx) / a) ** power + (np.abs(y - cy) / b) ** power
        body = level / (1 + np.exp(np.minimum((d - 1) * 7, 60)))
        return body + rim * np.exp(-((d - 1) ** 2) / 0.03)

    if kind == "spine":
        for k, (cx, cy) in enumerate(VERTEBRAE):
            img += blob(cx, cy, 86 + 3 * k, 50, level=0.4, rim=0.1)
            img += blob(cx, cy - 8, 150 + 4 * k, 11, 2, 0.14, 0.03)
            for side in (-1, 1):
                img += blob(cx + side * 48, cy - 14, 17, 21, 2, 0.1, 0.16)
            img += blob(cx, cy + 34, 16, 34, 2, 0.12, 0.04)
        for side in (-1, 1):
            img += blob(300 + side * 215, 830, 190, 150, 2, 0.12, 0.05)
        if artifact:
            ax, ay = ARTIFACT
            metal = (np.abs(x - ax) < 9) & (np.abs(y - ay) < 24)
            img = np.where(metal, 1.05, img)
            img += 0.9 * np.exp(-(((x - ax) / 5) ** 2)) * ((y > ay + 20) & (y < ay + 92))
    else:
        img += blob(130, 150, 300, 200, 2, 0.14, 0.05)
        img += blob(300, 300, 84, 84, 2, 0.38, 0.12)
        along = (x - 300) * 0.64 + (y - 300) * 0.77
        across = np.abs((x - 300) * 0.77 - (y - 300) * 0.64)
        soft = lambda v: 1 / (1 + np.exp(np.clip(v, -60, 60)))  # noqa: E731
        img += 0.3 * soft((across - 44) / 5) * soft((20 - along) / 10) * soft((along - 205) / 10)
        img += blob(450, 420, 58, 64, 2, 0.3, 0.08)
        img += blob(420, 680, 66, 270, 2.2, 0.36, 0.12)
    rng = np.random.default_rng(3)
    img += rng.normal(0, 0.03, img.shape)
    # Лёгкое размытие по двум осям — снимок не выглядит векторным.
    kernel = np.array([1, 4, 6, 4, 1], np.float32) / 16
    for _ in range(2):
        img = np.apply_along_axis(lambda r: np.convolve(r, kernel, "same"), 1, img)
        img = np.apply_along_axis(lambda r: np.convolve(r, kernel, "same"), 0, img)
    gray = (np.clip(img, 0, 1) ** 0.9 * 255).astype(np.uint8)
    rgba = np.dstack(
        [
            gray,
            gray,
            np.clip(gray.astype(int) + 6, 0, 255).astype(np.uint8),
            np.full_like(gray, 255),
        ]
    )
    return skia.Image.fromarray(np.ascontiguousarray(rgba), colorType=skia.kRGBA_8888_ColorType)


def draw_image(c: skia.Canvas, image: skia.Image, rect: skia.Rect, alpha: float = 1.0) -> None:
    paint = skia.Paint(AntiAlias=True, Alphaf=clamp(alpha))
    c.drawImageRect(
        image, rect, skia.SamplingOptions(skia.FilterMode.kLinear, skia.MipmapMode.kLinear), paint
    )


# ---------------------------------------------------------------- курсор и текст


def cursor(c: skia.Canvas, x: float, y: float, scale: float = 1.0, alpha: float = 1.0) -> None:
    """Системная стрелка: чёрная заливка, белый контур, мягкая тень."""
    if alpha <= 0:
        return
    pts = [(0, 0), (0, 34), (8.5, 26.5), (14.5, 40), (20, 37.5), (14.2, 24.5), (25, 24.5)]
    path = line_path(*pts)
    path.close()
    with Group(c, x, y, 1.05 * scale, alpha=alpha) as g:
        with Group(g, 1, 3):
            g.drawPath(path, fill(0x000000, 0.35, 3))
        g.drawPath(path, fill(0x111111))
        g.drawPath(path, stroke(0xFFFFFF, 2.2))


def click_ring(
    c: skia.Canvas, t: float, at: float, x: float, y: float, hex_rgb: int = ACCENT
) -> None:
    u = prog(t, at, 0.6)
    if 0 < u < 1:
        c.drawCircle(x, y, 6 + 34 * ease_out_cubic(u), stroke(hex_rgb, 1.6, 1 - u))


def click_scale(t: float, at: float) -> float:
    return 1 - 0.1 * math.sin(math.pi * prog(t, at - 0.06, 0.22))


def typed(
    c: skia.Canvas,
    t: float,
    lines: list[tuple[str, int]],
    size: float = 92,
    cps: float = 24,
    cy: float = H / 2,
    caret: int = INK,
    bold: bool = False,
) -> None:
    """Спокойная печать: буква проявляется и слегка поднимается, тонкая каретка."""
    f = font(size, bold)
    m = f.getMetrics()
    gap = size * 1.22
    n = 0
    caret_at = None
    typed_chars = t * cps
    for li, (line, hex_rgb) in enumerate(lines):
        width = f.measureText(line)
        x = W / 2 - width / 2
        y = cy + (li - (len(lines) - 1) / 2) * gap
        base = y - (m.fAscent + m.fDescent) / 2
        for ch in line:
            u = prog(t, n / cps, 0.16)
            adv = f.measureText(ch)
            if u > 0:
                c.drawString(ch, x, base + 8 * (1 - ease_out_cubic(u)), f, fill(hex_rgb, u))
                caret_at = (x + adv, y)
            n += 1
            x += adv
    if caret_at and t > 0:
        done = typed_chars >= n
        on = not done or (t * 1.6) % 1 < 0.55
        c.drawRect(
            skia.Rect.MakeXYWH(
                caret_at[0] + size * 0.06, caret_at[1] - size * 0.42, 3, size * 0.84
            ),
            fill(caret, 0.9 if on else 0),
        )


def wordmark(
    c: skia.Canvas, x: float, y: float, size: float, hex_rgb: int = INK, alpha: float = 1.0
) -> None:
    """DEXQ и небольшая точка бренда."""
    width = font(size, True).measureText("DEXQ")
    text(c, "DEXQ", x, y, size, hex_rgb, alpha)
    c.drawCircle(x + width / 2 + size * 0.16, y - size * 0.3, size * 0.075, fill(BRAND, alpha))


def statement(lines: list[tuple[str, int]], delay: float = 0.5, size: float = 92):
    """Кадр на светлом фоне с печатным заявлением."""

    def draw(c: skia.Canvas, t: float) -> None:
        c.clear(color(PAPER))
        typed(c, t - delay, lines, size)

    return draw


def opening(c: skia.Canvas, t: float) -> None:
    c.clear(color(PAPER))
    a = fade(t, 0.3, 1.0) * (1 - prog(t, 2.0, 0.6))
    rise = 18 * (1 - fade(t, 0.3, 1.2)) - 24 * ease_in_out(prog(t, 2.0, 0.6))
    wordmark(c, W / 2, H / 2 + rise, 96, INK, a)
    typed(c, t - 2.7, [("Снимок готов.", INK), ("Но пригоден ли он для оценки?", GRAY)], 88)


def closing(c: skia.Canvas, t: float) -> None:
    c.clear(color(PAPER))
    a = fade(t, 0.4, 1.2)
    wordmark(c, W / 2, H / 2 - 40 + 16 * (1 - a), 132, INK, a)
    text(
        c,
        "Контроль качества DXA-исследований",
        W / 2,
        H / 2 + 70,
        34,
        GRAY,
        fade(t, 1.2, 1.0),
        bold=False,
    )
    text(
        c,
        "Исследовательский прототип, не диагностическое средство",
        W / 2,
        H - 80,
        20,
        GRAY,
        0.8 * fade(t, 2.2, 1.0),
        bold=False,
    )


# ---------------------------------------------------------------- рабочее пространство

WIN = (160, 90, 1760, 990)
TOP, SIDE_R, INSP_L, STATUS_T = 142, 460, 1420, 926
VIEW = (SIDE_R, TOP, INSP_L, STATUS_T)
IMG_SCALE = 700 / IMG_H
IMG_L, IMG_T = 940 - IMG_W * IMG_SCALE / 2, 184

FILES = 24
FLAGGED = {0, 7, 15}
ROW0, ROW_H, VISIBLE_ROWS = 330, 52, 10
MENU = ["Новый проект", "Добавить снимки", "Экспорт CSV", "Экспорт Excel", "", "Выйти к проектам"]
MENU_X, MENU_Y, MENU_ITEM = 252, 146, 40
EXPORT_POINT = (392, MENU_Y + 6 + MENU_ITEM * 2.5)

STACK = (40, 560)
STACK_REST = [(-12, 10, -3.5), (10, -6, 2.5), (-5, 5, -1.5), (8, 12, 3.0), (0, 0, 0.6)]
CARD_W, CARD_H = 130, 168
STACK_IN, GRAB, RELEASE = 1.4, 3.0, 5.9
DRAG = [(50, 505), (260, 450), (520, 560), (720, 650)]
PROC0, FIRST_DONE, STEP = 7.6, 10.0, 0.5
MENU_CLICK, EXPORT_CLICK = 27.9, 29.3
AXIS_DRAG = (22.8, 1.4)


def kind(k: int) -> str:
    return "hip" if k % 4 == 2 else "spine"


@cache
def thumb(kind_name: str) -> skia.Image:
    return study_image(kind_name, kind_name == "spine").resize(150, 205)


def img_point(px: float, py: float) -> tuple[float, float]:
    return IMG_L + px * IMG_SCALE, IMG_T + py * IMG_SCALE


def schedule(k: int) -> tuple[float, float]:
    """Очередь последовательная, как в workspace: (начало анализа, готовность)."""
    if k == 0:
        return PROC0, FIRST_DONE
    start = FIRST_DONE + (k - 1) * STEP
    return start, start + STEP


def listed_at(k: int) -> float:
    return RELEASE + 0.3 + 0.05 * min(k, VISIBLE_ROWS)


def row_y(k: int) -> float:
    return ROW0 + k * ROW_H


def drag_cursor(wt: float) -> tuple[float, float]:
    return catmull(DRAG, ease_in_out(prog(wt, GRAB + 0.15, RELEASE - GRAB - 0.35)))


@cache
def stack_sim() -> np.ndarray:
    """Пачка файлов следует за курсором на пружинах с сильным демпфированием.

    Returns:
        Массив [кадр, карточка, (x, y, угол)] от GRAB до RELEASE.
    """
    sub, n = 4, len(STACK_REST)
    dt = 1 / (FPS * sub)
    grip = np.array([28.0, 52.0])
    pos = np.array([[STACK[0] + dx, STACK[1] + dy] for dx, dy, _ in STACK_REST], dtype=float)
    vel = np.zeros((n, 2))
    frames = []
    for f in range(int((RELEASE - GRAB) * FPS) + 2):
        for s in range(sub):
            cur = np.array(drag_cursor(GRAB + (f * sub + s) * dt))
            for j in range(n):
                dx, dy, _ = STACK_REST[j]
                stiff = 90 + 45 * j
                acc = (
                    stiff * (cur + grip + np.array([dx, dy]) * 0.6 - pos[j])
                    - 2 * math.sqrt(stiff) * 0.8 * vel[j]
                )
                vel[j] += acc * dt
                pos[j] += vel[j] * dt
        frames.append(
            [
                (pos[j, 0], pos[j, 1], STACK_REST[j][2] * 0.6 + clamp(vel[j, 0] * 0.005, -5, 5))
                for j in range(n)
            ]
        )
    return np.array(frames)


def stack_card(wt: float, j: int) -> tuple[float, float, float, float, float] | None:
    """Карточка j пачки (0 — нижняя): (x, y, угол, масштаб, прозрачность)."""
    a = fade(wt, STACK_IN + j * 0.07, 0.6)
    if a <= 0:
        return None
    dx, dy, rot = STACK_REST[j]
    if wt < GRAB:
        return STACK[0] + dx, STACK[1] + dy + 20 * (1 - a), rot, 1.0, a
    sim = stack_sim()
    lift = 1 + 0.04 * fade(wt, GRAB, 0.3)
    if wt < RELEASE:
        x, y, r = sim[min(int((wt - GRAB) * FPS), len(sim) - 1), j]
        return x, y, r, lift, 1.0
    # После отпускания карточка улетает в строку списка и уменьшается до миниатюры.
    k = len(STACK_REST) - 1 - j
    u = ease_in_out_quint(prog(wt, RELEASE + 0.05 * k, 0.8))
    if u >= 1:
        return None
    x0, y0, r0 = sim[-1, j]
    x, y = bezier((x0, y0), (x0 - 60, y0 - 40), (260, row_y(k) - 80), (211, row_y(k)), u)
    return x, y, lerp(r0, 0, u), lerp(lift, 30 / CARD_W, u), 1.0


def status_glyph(c: skia.Canvas, wt: float, k: int, x: float, y: float) -> None:
    start, done = schedule(k)
    if wt < start:
        c.drawCircle(x, y, 7, stroke(TEXT2, 1.5, 0.7))
        c.drawPath(line_path((x, y - 4), (x, y), (x + 3, y + 2)), stroke(TEXT2, 1.5, 0.7))
    elif wt < done:
        c.drawArc(
            skia.Rect.MakeXYWH(x - 7, y - 7, 14, 14), wt * 400 % 360, 260, False, stroke(ACCENT, 2)
        )
    elif k in FLAGGED:
        c.drawCircle(x, y, 7.5, fill(RED, fade(wt, done, 0.3)))
        c.drawRect(skia.Rect.MakeXYWH(x - 1, y - 4.5, 2, 5), fill(VIEWER))
        c.drawRect(skia.Rect.MakeXYWH(x - 1, y + 2, 2, 2), fill(VIEWER))
    else:
        partial(
            c,
            line_path((x - 6, y), (x - 2, y + 4), (x + 6, y - 4)),
            fade(wt, done, 0.3),
            stroke(GREEN, 2),
        )


def sidebar(c: skia.Canvas, wt: float) -> None:
    text(c, "‹  Все проекты", 184, 170, 15, TEXT2, align="left", bold=False)
    c.drawRRect(rrect(310, 228, 268, 64, 10), fill(RAISED))
    text(c, "Текущий проект", 196, 214, 12, TEXT2, align="left", bold=False)
    text(c, "Контроль · сентябрь", 196, 240, 17, TEXT, align="left")
    listed = sum(1 for k in range(FILES) if wt > listed_at(k))
    text(c, "Снимки", 184, 292, 13, TEXT2, align="left", bold=False)
    if listed:
        text(
            c,
            str(FILES if listed >= VISIBLE_ROWS else listed),
            444,
            292,
            13,
            TEXT2,
            align="right",
            bold=False,
        )
    for k in range(VISIBLE_ROWS):
        a = fade(wt, listed_at(k), 0.5)
        if a <= 0:
            continue
        y = row_y(k)
        with Group(c, 12 * (1 - a), 0, alpha=a):
            if k == 0 and wt > PROC0:
                sel = fade(wt, PROC0, 0.4)
                c.drawRRect(rrect(310, y, 268, 46, 8), fill(RAISED, sel))
                c.drawRRect(rrect(179, y, 3, 26, 1.5), fill(ACCENT, sel))
            landed = RELEASE + 0.05 * k + 0.8 if k < len(STACK_REST) else 0
            if wt >= landed:
                clip = rrect(211, y, 30, 38, 4)
                c.save()
                c.clipRRect(clip, True)
                draw_image(c, thumb(kind(k)), clip.rect())
                c.restore()
            text(c, f"dxa_{k + 1:03d}.dcm", 236, y, 15, TEXT, align="left", bold=False)
            status_glyph(c, wt, k, 426, y)
    c.drawRRect(rrect(310, 890, 268, 38, 8), stroke(BORDER, 1))
    text(c, "+  Добавить снимки", 310, 890, 14, TEXT2, bold=False)


def inspector(c: skia.Canvas, wt: float) -> None:
    a = fade(wt, 0.9, 0.8)
    x0, x1 = INSP_L + 24, WIN[2] - 24
    text(c, "Проверки качества", x0, 176, 13, TEXT2, a, align="left", bold=False)
    rows = [
        ("Укладка", "Норма", GREEN, 10.2),
        ("Ось позвоночника", "1.8°", GREEN, 10.45),
        ("Артефакты", "Нарушение", RED, 10.7),
    ]
    for i, (name, value, hex_rgb, at) in enumerate(rows):
        y = 216 + i * 46
        done = fade(wt, at, 0.4)
        if hex_rgb == RED and done:
            c.drawRRect(rrect((x0 + x1) / 2, y, x1 - x0 + 20, 38, 8), fill(RED, 0.09 * done))
        c.drawCircle(x0 + 5, y, 4, fill(hex_rgb if done else BORDER, max(done, a * 0.8)))
        text(c, name, x0 + 20, y, 16, TEXT, a, align="left", bold=False)
        text(c, "—", x1, y, 16, TEXT2, a * (1 - done), align="right", bold=False)
        text(c, value, x1, y, 16, hex_rgb, done, align="right")
    c.drawRect(skia.Rect.MakeXYWH(x0, 364, x1 - x0, 1), fill(BORDER, a))
    text(c, "Параметры снимка", x0, 394, 13, TEXT2, a, align="left", bold=False)
    for i, (k, v) in enumerate([("Область", "Поясничный отдел"), ("Проекция", "Не определена")]):
        shown = fade(wt, 10.0 + i * 0.1, 0.4)
        text(c, k, x0, 432 + i * 36, 15, TEXT2, a, align="left", bold=False)
        text(c, v, x1, 432 + i * 36, 15, TEXT, shown, align="right", bold=False)
    c.drawRect(skia.Rect.MakeXYWH(x0, 494, x1 - x0, 1), fill(BORDER, a))
    for i, line in enumerate(["Исследовательский контроль качества,", "не диагностика."]):
        text(c, line, x0, 524 + i * 22, 13, TEXT2, a * 0.9, align="left", bold=False)


def status_bar(c: skia.Canvas, wt: float) -> None:
    progress = sum(clamp((wt - s) / (d - s)) for s, d in map(schedule, range(FILES))) / FILES
    if wt < RELEASE:
        progress = 0.0
    y = (STATUS_T + WIN[3]) / 2
    c.drawRRect(rrect(634, y, 300, 4, 2), fill(BORDER))
    if progress > 0:
        c.drawRRect(rrect(484 + 150 * progress, y, 300 * progress, 4, 2), fill(ACCENT))
    text(c, f"{round(progress * 100)}%", 802, y, 15, TEXT, align="left")
    ready = sum(1 for k in range(FILES) if wt > schedule(k)[1])
    busy = sum(1 for k in range(FILES) if wt > listed_at(k)) - ready if wt > RELEASE else 0
    x = 880
    for value, label, hex_rgb in (
        (ready, "готово", GREEN),
        (busy, "в обработке", TEXT2),
        (0, "ошибок", RED),
    ):
        c.drawCircle(x, y, 4, fill(hex_rgb))
        w = text(c, str(value), x + 14, y, 15, TEXT, align="left")
        w2 = text(c, label, x + 20 + w, y, 15, TEXT2, align="left", bold=False)
        x += 40 + w + w2


def top_bar(c: skia.Canvas, wt: float) -> None:
    y = (WIN[1] + TOP) / 2
    wordmark_w = font(20, True).measureText("DEXQ")
    text(c, "DEXQ", 184, y, 20, TEXT, align="left")
    c.drawCircle(184 + wordmark_w + 5, y - 7, 2.5, fill(ACCENT))
    open_ = fade(wt, MENU_CLICK, 0.2) * (1 - fade(wt, EXPORT_CLICK + 0.1, 0.25))
    if open_ > 0:
        c.drawRRect(rrect(284, y, 62, 32, 7), fill(RAISED, open_))
    for label, x in (("Файл", 262), ("Правка", 322), ("Редактор", 398)):
        text(
            c,
            label,
            x,
            y,
            15,
            TEXT if label == "Файл" and open_ > 0.5 else TEXT2,
            align="left",
            bold=False,
        )


def file_menu(c: skia.Canvas, wt: float, hover_y: float) -> None:
    a = fade(wt, MENU_CLICK + 0.05, 0.25) * (1 - fade(wt, EXPORT_CLICK + 0.15, 0.25))
    if a <= 0:
        return
    h = MENU_ITEM * len(MENU) + 12
    with Group(c, 0, -6 * (1 - a), alpha=a):
        panel = skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(MENU_X, MENU_Y, 250, h), 10, 10)
        c.drawRRect(panel, fill(0x000000, 0.35, 16))
        c.drawRRect(panel, fill(RAISED))
        c.drawRRect(panel, stroke(BORDER, 1))
        for i, item in enumerate(MENU):
            y = MENU_Y + 6 + MENU_ITEM * (i + 0.5)
            if not item:
                c.drawRect(skia.Rect.MakeXYWH(MENU_X + 12, y, 226, 1), fill(BORDER))
                continue
            hot = clamp(1 - abs(hover_y - y) / MENU_ITEM)
            if hot > 0:
                c.drawRRect(rrect(MENU_X + 125, y, 238, 34, 6), fill(ACCENT, 0.18 * hot))
            text(c, item, MENU_X + 16, y, 15, TEXT, align="left", bold=False)


def download_chip(c: skia.Canvas, wt: float) -> None:
    a = fade(wt, EXPORT_CLICK + 0.4, 0.6)
    if a <= 0:
        return
    with Group(c, 1250, 866 + 14 * (1 - a), alpha=a):
        c.drawRRect(rrect(0, 0, 310, 70, 12), fill(0x000000, 0.3, 14))
        c.drawRRect(rrect(0, 0, 310, 70, 12), fill(RAISED))
        c.drawRRect(rrect(0, 0, 310, 70, 12), stroke(BORDER, 1))
        c.drawCircle(-118, 0, 19, fill(ACCENT, 0.16))
        c.drawPath(line_path((-118, -8), (-118, 6)), stroke(ACCENT, 2))
        c.drawPath(line_path((-124, 1), (-118, 7), (-112, 1)), stroke(ACCENT, 2))
        text(c, "results.csv", -86, -11, 16, TEXT, align="left")
        text(c, "24 строки · 8 столбцов", -86, 13, 13, TEXT2, align="left", bold=False)


def toolbar(c: skia.Canvas, a: float) -> None:
    with Group(c, 940, 176, alpha=a):
        c.drawRRect(rrect(0, 0, 318, 40, 20), fill(PANEL))
        c.drawRRect(rrect(0, 0, 318, 40, 20), stroke(BORDER, 1))
        arrow = line_path(
            (-128, -8), (-128, 7), (-124, 3), (-121, 9), (-119, 8), (-122, 2), (-117, 2)
        )
        arrow.close()
        c.drawPath(arrow, fill(ACCENT))
        c.drawCircle(-92, 0, 7, stroke(TEXT2, 1.5))
        c.drawRect(skia.Rect.MakeXYWH(-68, -10, 1, 20), fill(BORDER))
        c.drawRect(skia.Rect.MakeXYWH(-50, -0.75, 12, 1.5), fill(TEXT2))
        text(c, "100%", 0, 0, 14, TEXT, bold=False)
        c.drawRect(skia.Rect.MakeXYWH(38, -0.75, 12, 1.5), fill(TEXT2))
        c.drawRect(skia.Rect.MakeXYWH(43.25, -6, 1.5, 12), fill(TEXT2))
        c.drawRect(skia.Rect.MakeXYWH(68, -10, 1, 20), fill(BORDER))
        c.drawRect(skia.Rect.MakeXYWH(86, -7, 14, 14), stroke(TEXT2, 1.5))
        c.drawPath(line_path((118, -5), (122, 0), (118, 5)), stroke(TEXT2, 1.5))


def axis_points(wt: float) -> tuple[tuple[float, float], tuple[float, float], float]:
    """Верхняя и нижняя точки оси в мире и ручной сдвиг нижней точки (px снимка)."""
    shift = 14 * ease_in_out(prog(wt, *AXIS_DRAG))
    (x0, y0), (x3, y3) = VERTEBRAE[0], VERTEBRAE[3]
    return img_point(x0, y0), img_point(x3 + shift, y3), shift


def angle_of(shift: float) -> float:
    (x0, y0), (x3, y3) = VERTEBRAE[0], VERTEBRAE[3]
    return math.degrees(math.atan2(x3 + shift - x0, y3 - y0))


def label(
    c: skia.Canvas, s: str, x: float, y: float, hex_rgb: int, a: float, size: float = 14
) -> None:
    """Небольшая подпись-плашка поверх снимка, левый край в x."""
    w = font(size, False).measureText(s) + 24
    c.drawRRect(
        skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y - 15, w, 30), 8, 8), fill(PANEL, 0.92 * a)
    )
    c.drawRRect(
        skia.RRect.MakeRectXY(skia.Rect.MakeXYWH(x, y - 15, w, 30), 8, 8),
        stroke(hex_rgb, 1, 0.5 * a),
    )
    text(c, s, x + 12, y, size, hex_rgb, a, align="left", bold=False)


def viewer(c: skia.Canvas, wt: float) -> None:
    c.drawRect(skia.Rect.MakeLTRB(*VIEW), fill(VIEWER))
    cx = (VIEW[0] + VIEW[2]) / 2
    hover = fade(wt, 4.7, 0.3) * (1 - fade(wt, RELEASE, 0.35))
    empty = fade(wt, 0.9, 0.8) * (1 - fade(wt, 4.6, 0.3))
    if empty > 0:
        c.drawCircle(cx, 470, 34, stroke(BORDER, 1.5, empty))
        c.drawPath(line_path((cx, 484), (cx, 456)), stroke(TEXT2, 2, empty))
        c.drawPath(line_path((cx - 10, 466), (cx, 456), (cx + 10, 466)), stroke(TEXT2, 2, empty))
        text(c, "Добавьте снимки в проект", cx, 540, 22, TEXT, empty)
        text(c, "DICOM, PNG, JPEG · ZIP с DICOM или PNG", cx, 574, 15, TEXT2, empty, bold=False)
        text(c, "Выбрать файлы", cx, 616, 15, ACCENT, empty, bold=False)
    if hover > 0:
        zone = skia.RRect.MakeRectXY(
            skia.Rect.MakeLTRB(VIEW[0] + 18, TOP + 18, VIEW[2] - 18, STATUS_T - 18), 14, 14
        )
        c.drawRRect(zone, fill(ACCENT, 0.06 * hover))
        dash = stroke(ACCENT, 1.5, hover)
        dash.setPathEffect(skia.DashPathEffect.Make([10, 8], 0))
        c.drawRRect(zone, dash)
        text(c, "Добавить в проект", cx, 520, 24, TEXT, hover)
        text(c, "Текущие снимки останутся", cx, 556, 15, TEXT2, hover, bold=False)

    ia = fade(wt, RELEASE + 0.5, 1.2)
    if ia > 0:
        s = 1 + 0.015 * (1 - ia)
        w, h = IMG_W * IMG_SCALE * s, IMG_H * IMG_SCALE * s
        rect = skia.Rect.MakeXYWH(cx - w / 2, 534 - h / 2, w, h)
        draw_image(c, study_image("spine"), rect, ia)
        # Скан во время анализа: тонкая линия с коротким шлейфом.
        env = fade(wt, 7.9, 0.3) * (1 - fade(wt, 9.7, 0.3))
        if env > 0:
            u = ease_in_out(((wt - 7.9) / 0.9) % 1)
            y = lerp(rect.top(), rect.bottom(), u)
            trail = skia.Paint(AntiAlias=True)
            trail.setShader(
                skia.GradientShader.MakeLinear(
                    [skia.Point(0, y - 70), skia.Point(0, y)],
                    [color(ACCENT, 0), color(ACCENT, 0.18 * env)],
                )
            )
            c.drawRect(skia.Rect.MakeLTRB(rect.left(), y - 70, rect.right(), y), trail)
            c.drawRect(
                skia.Rect.MakeLTRB(rect.left(), y - 0.75, rect.right(), y + 0.75), fill(ACCENT, env)
            )
    busy = fade(wt, PROC0, 0.4) * (1 - fade(wt, FIRST_DONE, 0.4))
    if busy > 0:
        c.drawRRect(rrect(620, 230, 262, 36, 18), fill(PANEL, 0.9 * busy))
        c.drawArc(
            skia.Rect.MakeXYWH(502, 223, 14, 14),
            wt * 400 % 360,
            260,
            False,
            stroke(ACCENT, 2, busy),
        )
        text(c, "Анализируем изображение", 528, 230, 14, TEXT, busy, align="left", bold=False)

    # Ориентиры: две точки оси, вертикаль и угол.
    (tx, ty), (bx, by), shift = axis_points(wt)
    auto_bottom = img_point(*VERTEBRAE[3])
    guide = stroke(TEXT2, 1.2, 0.55 * fade(wt, 10.0, 0.5))
    guide.setPathEffect(skia.DashPathEffect.Make([6, 6], 0))
    partial(c, line_path(auto_bottom, (auto_bottom[0], ty - 30)), fade(wt, 10.0, 0.8), guide)
    manual = shift > 0.05
    auto_line = stroke(ACCENT, 1.2 if manual else 2.4, 0.45 if manual else 1)
    partial(c, line_path(auto_bottom, (tx, ty)), ease_in_out(prog(wt, 10.2, 0.8)), auto_line)
    if manual:
        c.drawPath(line_path((bx, by), (tx, ty)), stroke(ACCENT, 2.4))
    for i, (px, py) in enumerate(((tx, ty), (bx, by))):
        a = fade(wt, 10.0 + i * 0.15, 0.4)
        if a > 0:
            c.drawCircle(px, py, 8, stroke(ACCENT, 2.2, a))
            c.drawCircle(px, py, 3, fill(ACCENT, a))
    angle = fade(wt, 11.0, 0.5)
    if angle > 0:
        label(c, f"{angle_of(0):.1f}°", tx + 20, ty - 26, TEXT, angle)
    ml = fade(wt, AXIS_DRAG[0] + 0.3, 0.5)
    if ml > 0:
        label(
            c,
            f"Ручная ось {angle_of(shift):.1f}° · авто {angle_of(0):.1f}°",
            bx + 20,
            by + 30,
            ACCENT,
            ml,
            12,
        )

    # Посторонний предмет: тонкое кольцо и подпись при приближении.
    ax, ay = img_point(*ARTIFACT)
    ring = skia.Path()
    ring.addCircle(ax, ay + 12, 34)
    partial(c, ring, ease_in_out(prog(wt, 10.8, 0.8)), stroke(RED, 1.8))
    tag = fade(wt, 19.6, 0.6)
    if tag > 0:
        label(c, "Посторонний предмет", ax + 44, ay - 14, RED, tag, 12)
    toolbar(c, fade(wt, 1.0, 0.8))


def ws_camera(wt: float) -> Camera:
    ax, ay = img_point(*ARTIFACT)
    bx, by = img_point(*VERTEBRAE[3])
    keys = [
        (0.0, (960, 540, 0.8)),
        (1.6, (900, 540, 0.84)),
        (3.0, (820, 540, 0.86)),
        (5.9, (860, 530, 1.0)),
        (7.4, (760, 520, 1.1)),
        (9.8, (930, 520, 1.3)),
        (11.4, (940, 525, 1.33)),
        (12.6, (1590, 300, 2.05)),
        (14.2, (1590, 305, 2.1)),
        (15.8, (960, 540, 1.0)),
        (18.5, (960, 540, 1.03)),
        (19.8, (ax + 50, ay, 2.7)),
        (21.4, (ax + 50, ay, 2.75)),
        (22.4, (bx + 40, by - 40, 2.2)),
        (24.4, (bx + 50, by - 40, 2.25)),
        (25.6, (960, 540, 1.0)),
        (26.4, (960, 540, 1.0)),
        (27.4, (420, 250, 1.9)),
        (29.5, (420, 262, 1.9)),
        (30.6, (1300, 820, 1.75)),
        (32.2, (1300, 820, 1.78)),
        (33.8, (960, 540, 0.92)),
        (36.5, (960, 540, 0.9)),
    ]
    return Camera(*keyframes(t=wt, keys=keys))


def ws_cursor(wt: float) -> tuple[float, float, float, float | None]:
    """Положение курсора в мире, прозрачность и ближайший клик."""
    bx, by = img_point(*VERTEBRAE[3])
    menu_file = (272, 124)
    if wt < GRAB:
        x, y = bezier(
            (1250, 1100), (900, 700), (300, 480), DRAG[0], ease_in_out(prog(wt, 2.0, 0.95))
        )
        return x, y, fade(wt, 2.0, 0.3), None
    if wt < RELEASE:
        return (*drag_cursor(wt), 1.0, None)
    if wt < 15:
        x, y = bezier(
            DRAG[-1], (900, 600), (980, 700), (1000, 760), ease_in_out(prog(wt, 6.3, 1.0))
        )
        return x, y, 1 - fade(wt, 6.9, 0.5), None
    if wt < 26:
        shift = 14 * ease_in_out(prog(wt, *AXIS_DRAG)) * IMG_SCALE
        if wt < AXIS_DRAG[0]:
            x, y = bezier(
                (bx + 260, by + 200),
                (bx + 150, by + 150),
                (bx + 40, by + 40),
                (bx, by),
                ease_in_out(prog(wt, 21.8, 0.8)),
            )
        elif wt < 24.4:
            x, y = bx + shift, by
        else:
            x, y = bezier(
                (bx + shift, by),
                (bx + 80, by + 40),
                (bx + 160, by + 120),
                (bx + 260, by + 200),
                ease_in_out(prog(wt, 24.4, 0.8)),
            )
        return x, y, fade(wt, 21.8, 0.3) * (1 - fade(wt, 24.9, 0.4)), None
    if wt < EXPORT_CLICK:
        if wt < MENU_CLICK:
            x, y = bezier(
                (760, 620), (600, 420), (350, 200), menu_file, ease_in_out(prog(wt, 26.6, 1.2))
            )
        else:
            x, y = bezier(
                menu_file,
                (274, 170),
                (330, EXPORT_POINT[1] - 40),
                EXPORT_POINT,
                ease_in_out(prog(wt, 28.3, 0.8)),
            )
        return x, y, fade(wt, 26.6, 0.3), MENU_CLICK
    x, y = bezier(
        EXPORT_POINT,
        (420, EXPORT_POINT[1] + 60),
        (560, 500),
        (680, 620),
        ease_in_out(prog(wt, 29.7, 0.9)),
    )
    return x, y, 1 - fade(wt, 30.1, 0.5), EXPORT_CLICK


def workspace(c: skia.Canvas, wt: float) -> None:
    """Упрощённое графитовое рабочее пространство на светлом фоне."""
    c.clear(color(PAPER))
    cam = ws_camera(wt)
    c.save()
    cam.apply(c)
    wa = fade(wt, 0.05, 1.4)
    with Group(c, 0, 40 * (1 - wa), alpha=wa):
        frame_rr = skia.RRect.MakeRectXY(skia.Rect.MakeLTRB(*WIN), 16, 16)
        c.drawRRect(frame_rr.makeOffset(0, 24), fill(0x1A1C22, 0.22, 40))
        c.save()
        c.clipRRect(frame_rr, True)
        c.drawRect(skia.Rect.MakeLTRB(*WIN), fill(BG))
        c.drawRect(skia.Rect.MakeLTRB(WIN[0], TOP, SIDE_R, WIN[3]), fill(PANEL))
        c.drawRect(skia.Rect.MakeLTRB(INSP_L, TOP, WIN[2], WIN[3]), fill(PANEL))
        viewer(c, wt)
        c.drawRect(skia.Rect.MakeLTRB(SIDE_R, STATUS_T, INSP_L, WIN[3]), fill(PANEL))
        for x0, y0, x1, y1 in (
            (WIN[0], TOP, WIN[2], TOP + 1),
            (SIDE_R, TOP, SIDE_R + 1, WIN[3]),
            (INSP_L, TOP, INSP_L + 1, WIN[3]),
            (SIDE_R, STATUS_T, INSP_L, STATUS_T + 1),
        ):
            c.drawRect(skia.Rect.MakeLTRB(x0, y0, x1, y1), fill(BORDER))
        top_bar(c, wt)
        sidebar(c, wt)
        inspector(c, wt)
        status_bar(c, wt)
        c.restore()
        c.drawRRect(frame_rr, stroke(0x000000, 1, 0.25))

    x, y, alpha, click = ws_cursor(wt)
    file_menu(c, wt, y)
    download_chip(c, wt)
    for j in range(len(STACK_REST)):
        card = stack_card(wt, j)
        if card is None:
            continue
        cx, cy, rot, s, a = card
        with Group(c, cx, cy, s, rot, alpha=a) as g:
            g.drawRRect(rrect(0, 10, CARD_W, CARD_H, 10), fill(0x000000, 0.22, 12))
            clip = rrect(0, 0, CARD_W, CARD_H, 10)
            g.save()
            g.clipRRect(clip, True)
            draw_image(g, thumb(kind(len(STACK_REST) - 1 - j)), clip.rect())
            g.restore()
            g.drawRRect(clip, stroke(0xFFFFFF, 1.5, 0.35))
    badge = fade(wt, STACK_IN + 0.5, 0.5) * (1 - fade(wt, RELEASE, 0.3))
    if badge > 0:
        top = stack_card(wt, len(STACK_REST) - 1)
        if top:
            with Group(c, top[0] + CARD_W / 2 - 4, top[1] - CARD_H / 2 + 4, alpha=badge):
                c.drawRRect(rrect(0, 0, 50, 28, 14), fill(ACCENT))
                text(c, str(FILES), 0, 0, 15, BG)
    for at, px, py in ((MENU_CLICK, 272, 124), (EXPORT_CLICK, *EXPORT_POINT)):
        click_ring(c, wt, at, px, py)
    c.restore()

    press = click_scale(wt, click) if click is not None else 1.0
    if GRAB - 0.05 < wt < RELEASE:
        press *= 0.94
    sx, sy = cam.to_screen(x, y)
    cursor(c, sx, sy, press * cam.zoom**0.4, alpha)


# ---------------------------------------------------------------- монтаж

WS_A, WS_B, WS_B_OFFSET = 6.5, 29.5, 18.5
SCENES = [
    (0.0, opening),
    (WS_A, lambda c, t: workspace(c, t)),
    (25.0, statement([("Укладка, ось, артефакты —", INK), ("проверяются автоматически.", GRAY)])),
    (WS_B, lambda c, t: workspace(c, t + WS_B_OFFSET)),
    (
        47.5,
        statement([("Всё работает локально.", INK), ("Решение остаётся за специалистом.", GRAY)]),
    ),
    (53.0, closing),
]
CROSSFADE = 0.9


def render(c: skia.Canvas, t: float) -> None:
    """Кадр по общей шкале: сцены сменяются мягким наплывом."""
    i = max(k for k, (start, _) in enumerate(SCENES) if start <= t)
    start, draw = SCENES[i]
    u = prog(t, start, CROSSFADE)
    if i == 0 or u >= 1:
        draw(c, t - start)
        return
    prev_start, prev = SCENES[i - 1]
    prev(c, t - prev_start)
    e = ease_in_out(u)
    c.saveLayerAlpha(None, int(e * 255))
    c.translate(W / 2, H / 2)
    c.scale(1.02 - 0.02 * e, 1.02 - 0.02 * e)
    c.translate(-W / 2, -H / 2)
    draw(c, t - start)
    c.restore()


if __name__ == "__main__":
    run(render, DURATION, OUTPUT, __doc__.splitlines()[0])
