#!/usr/bin/env python3
"""frame_rect.py — нахождение бело-серых рамок на листе (по одной иконке).

Что было не так с сеткой
------------------------
Рамки нарисованы для каждой иконки отдельно. В одном ряду листа
`RPG Potion Icons 01 (83)` левые линии стоят на 1271, а у одного слота —
на 1284; правые 1423 и 1436. Шаг «сетки» гуляет на 13 px, поэтому подгонка
под общую решётку ошибается на этих слотах.

Что общего у всех рамок
-----------------------
Рамка — ТОНКАЯ серая линия (1-3 px) на чёрном фоне. Толщина у всех
одинаковая, яркость разная: на одном листе встречаются линии 40, 52, 71,
88. Поэтому берём не яркость, а «тонкость»: морфологический top-hat с
горизонтальным ядром 7 px даёт на вертикальной линии 40-76, а на фоне и
внутри рисунка — 0-3. Это и есть признак рамки.

Порядок
-------
1. `tophat()` — карты отклика: `V` (вертикальные линии), `H` (горизонтальные).
2. `lines()` — профиль отклика вдоль стороны слота, пики = линии рамки.
   Профиль берём как 75-й перцентиль, поэтому линия остаётся видна, даже
   если иконка наехала на неё на половину стороны.
3. `rect_for()` — из найденных линий собираем прямоугольник вокруг слота:
   ближайшая линия к каждой стороне + проверка, что получился квадрат
   (145-162 px). Если не вышло — перебор вариантов.
4. `crop_rect()` — вырез: от линии внутрь до первого чёрного пикселя, чтобы
   ни сама линия, ни её сглаженный сосед не попали в кадр.

`annotate()` рисует найденные рамки красным — для проверки глазами.
"""
from __future__ import annotations

import cv2
import numpy as np

TH_KERNEL = 7         # ядро top-hat: шире любой линии (1-3 px), уже рисунка
LINE_MIN = 10.0       # отклик top-hat: линия даёт 40-76, фон 0-3
# размер рамки задаётся размером бокса: у разных наборов листы разного
# масштаба (150-160 px на одном, 135 px на другом), поэтому допуск считаем
# от бокса, а не от абсолютного числа
SIDE_LO, SIDE_HI = 0.82, 1.20
SQUARE_TOL = 12
SEARCH = 22           # окно поиска линии вокруг края слота
BLACK = 15            # «чёрный» пиксель после линии


def brightness(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32).mean(axis=2)


def tophat(img: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Карты отклика на тонкие линии: (горизонтальные, вертикальные)."""
    br = brightness(img)
    h = cv2.morphologyEx(br, cv2.MORPH_TOPHAT, np.ones((TH_KERNEL, 1), np.uint8))
    v = cv2.morphologyEx(br, cv2.MORPH_TOPHAT, np.ones((1, TH_KERNEL), np.uint8))
    return h, v


def _peaks(prof: np.ndarray, thr: float = LINE_MIN) -> list[tuple[int, float]]:
    """Пики профиля -> [(координата, значение)]; соседние 1-3 px склеены."""
    idx = [i for i, v in enumerate(prof) if v >= thr]
    groups: list[list[int]] = []
    for i in idx:
        if groups and i - groups[-1][-1] <= 2:
            groups[-1].append(i)
        else:
            groups.append([i])
    out = []
    for g in groups:
        vals = prof[g]
        out.append((int(g[int(np.argmax(vals))]), float(vals.max())))
    return out


def _profile(resp: np.ndarray, box, axis: str) -> tuple[np.ndarray, int]:
    """Профиль отклика вдоль одной стороны слота (75-й перцентиль)."""
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = resp.shape
    if axis == "v":            # ищем вертикальные линии -> срез по строкам
        lo, hi = max(0, x0 - SEARCH), min(W, x1 + SEARCH + 1)
        return np.percentile(resp[y0:y1 + 1, lo:hi], 75, axis=0), lo
    lo, hi = max(0, y0 - SEARCH), min(H, y1 + SEARCH + 1)
    return np.percentile(resp[lo:hi, x0:x1 + 1], 75, axis=1), lo


def lines(img: np.ndarray, box, tophats=None) -> tuple[list[int], list[int]]:
    """Все линии вокруг слота: (x-линии, y-линии)."""
    h, v = tophats if tophats is not None else tophat(img)
    prof_v, off_x = _profile(v, box, "v")
    prof_h, off_y = _profile(h, box, "h")
    xs = [p + off_x for p, _ in _peaks(prof_v)]
    ys = [p + off_y for p, _ in _peaks(prof_h)]
    return xs, ys


def rect_for(img: np.ndarray, box,
             tophats=None) -> tuple[int, int, int, int] | None:
    """Прямоугольник рамки (по линиям) вокруг слота."""
    xs, ys = lines(img, box, tophats)
    if not xs or not ys:
        return None
    x0, y0, x1, y1 = (int(v) for v in box)
    lefts = [c for c in xs if c <= x0 + SEARCH and c >= x0 - SEARCH]
    rights = [c for c in xs if c >= x1 - SEARCH and c <= x1 + SEARCH]
    tops = [c for c in ys if c <= y0 + SEARCH and c >= y0 - SEARCH]
    bots = [c for c in ys if c >= y1 - SEARCH and c <= y1 + SEARCH]
    if not (lefts and rights and tops and bots):
        return None
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    lo_w, hi_w = SIDE_LO * bw, SIDE_HI * bw
    lo_h, hi_h = SIDE_LO * bh, SIDE_HI * bh
    best = None
    for l in lefts:
        for r in rights:
            w = r - l
            if not (lo_w <= w <= hi_w):
                continue
            for t in tops:
                for b in bots:
                    hh = b - t
                    if not (lo_h <= hh <= hi_h):
                        continue
                    if abs(w - hh) > max(SQUARE_TOL, 0.08 * max(w, hh)):
                        continue
                    score = abs(w - hh) + 0.4 * (
                        abs(l - x0) + abs(r - x1) + abs(t - y0) + abs(b - y1))
                    if best is None or score < best[0]:
                        best = (score, l, t, r, b)
    if best is None:
        return None
    return best[1:]


def _crop_side(val: np.ndarray, cov: np.ndarray, pos: int, inward: int,
              max_len: int = 4) -> int:
    """Куда резать со стороны pos: сразу за последним пикселем линии.

    Идём от линии внутрь слота, пока пиксель ещё «линейный» (ярче 18 и
    покрывает сторону не менее чем на 85%). Как только пошёл тёмный разрыв —
    это и есть кромка линии, режем по ней.

    Если же линия «не кончается» (рисунок вплотную к рамке, светлый на всю
    ширину), уходить на его счёт нельзя: останавливаемся на одном пикселе
    внутрь от пика линии.
    """
    step = 0
    cur = pos
    while step < max_len:
        nxt = cur + inward
        if not (0 <= nxt < len(val)):
            break
        if val[nxt] < 18 or cov[nxt] < 0.85:
            return cur            # cur — последний пиксель линии
        cur = nxt
        step += 1
    return pos                    # линия слилась с рисунком — режем по линии


def crop_rect(img: np.ndarray, rect) -> tuple[int, int, int, int]:
    """Вырез по найденной рамке: сразу за линией, пиксель в пиксель.

    Фиксированный отступ в 2 px то оставлял в кадре хвост толстой линии, то
    срезал край рисунка. Поэтому кромку линии ищем по её собственным
    пикселям (`_crop_side`), а не по заданному числу.
    """
    br = brightness(img)
    H, W = br.shape
    cov = br >= 18
    l, t, r, b = rect
    val_v = np.median(br[t:b + 1, :], axis=0)      # яркость по столбцам
    val_h = np.median(br[:, l:r + 1], axis=1)      # по строкам
    cov_v = cov[t:b + 1, :].mean(axis=0)
    cov_h = cov[:, l:r + 1].mean(axis=1)
    left = _crop_side(val_v, cov_v, l, +1)
    right = _crop_side(val_v, cov_v, r, -1)
    top = _crop_side(val_h, cov_h, t, +1)
    bottom = _crop_side(val_h, cov_h, b, -1)
    return (max(0, left + 1), max(0, top + 1),
            min(W - 1, right - 1), min(H - 1, bottom - 1))


def detect(img: np.ndarray, boxes) -> list[tuple[int, int, int, int] | None]:
    """Рамка для каждого слота: (л, в, п, н) — включительно по линиям."""
    th = tophat(img)
    return [rect_for(img, b, th) for b in boxes]


def annotate(img: np.ndarray, boxes, path: str, thick: int = 2) -> int:
    """Нарисовать найденные рамки красным и сохранить. Возвращает число рамок."""
    out = img.copy()
    th = tophat(img)
    n = 0
    for b in boxes:
        r = rect_for(img, b, th)
        if r is None:
            continue
        n += 1
        cv2.rectangle(out, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), thick)
    cv2.imwrite(path, out)
    return n
