#!/usr/bin/env python3
"""frame_none.py — листы БЕЗ рамок: режем по габаритам самого рисунка.

На 8 листах коллекции рамок нет вообще (`RPG Loot Icons 39/40`,
`RPG Weapon Icons 09`, `RPG Weapon and Armor 15`): рисунки лежат прямо на
чёрном фоне, и искать там белую линию нечего. Детектор рамки такие листы
честно отдаёт как «рамки не найдены» — 400 иконок выпадали из прогона.

Как режем здесь
---------------
1. `sheet_boxes()` — сетка слотов по содержимому: маска «не чёрное»
   (яркость > CONTENT_MIN), проекции на оси, полосы через промежутки.
   Даёт те же 10 x 5 боксов, что и детектор на обычных листах.
2. `content_box()` — внутри бокса берём габарит самого рисунка: строки и
   столбцы, где есть хоть сколько-то содержимого. Тень/свечение входят в
   габарит — они часть рисунка.
3. `crop_icon()` — вырез: рисунок вписывается в квадрат `target` пиксель в
   пиксель. Если рисунок больше — уменьшаем (это не мылит); если меньше —
   добираем родным тёмным фоном листа по краям, БЕЗ растягивания, чтобы
   мелкие иконки не размывались.
"""
from __future__ import annotations

import cv2
import numpy as np

CONTENT_MIN = 10      # яркость выше — уже рисунок, не чёрный фон
GAP = 6               # промежуток между слотами, px
MIN_COVER = 0.02      # доля строки/столбца с содержимым, чтобы считать её «слотом»


def _bands(cover: np.ndarray, thr: float = MIN_COVER) -> list[tuple[int, int]]:
    """Полосы, где cover >= thr (склеиваем разрывы до GAP)."""
    idx = np.flatnonzero(cover >= thr)
    if idx.size == 0:
        return []
    breaks = np.flatnonzero(np.diff(idx) > GAP)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    return [(int(idx[s]), int(idx[e])) for s, e in zip(starts, ends)]


def sheet_boxes(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Боксы слотов по содержимому листа (порядок: строки сверху вниз)."""
    br = img.astype(np.float32).mean(axis=2)
    mask = (br > CONTENT_MIN).astype(np.float32)
    if mask.mean() < 0.01:                      # пустой лист
        return []
    rows = _bands(mask.mean(axis=1))
    cols = _bands(mask.mean(axis=0))
    boxes = []
    for y0, y1 in rows:
        for x0, x1 in cols:
            if (y1 - y0) < 20 or (x1 - x0) < 20:
                continue
            boxes.append((x0, y0, x1, y1))
    return boxes


def content_box(img: np.ndarray, box, thr: float = CONTENT_MIN):
    """Габариты рисунка внутри бокса: (x0, y0, x1, y1) или None."""
    bx0, by0, bx1, by1 = (int(v) for v in box)
    br = img.astype(np.float32).mean(axis=2)
    cell = br[by0:by1 + 1, bx0:bx1 + 1]
    if cell.size == 0:
        return None
    m = cell > thr
    ys = np.flatnonzero(m.any(axis=1))
    xs = np.flatnonzero(m.any(axis=0))
    if ys.size == 0 or xs.size == 0:
        return None
    return (bx0 + int(xs[0]), by0 + int(ys[0]),
            bx0 + int(xs[-1]), by0 + int(ys[-1]))


def _bg_color(img: np.ndarray, box) -> tuple[int, int, int]:
    """Цвет фона листа: медиана по углам вокруг слота."""
    bx0, by0, bx1, by1 = (int(v) for v in box)
    H, W = img.shape[:2]
    pads = []
    for ys, xs in (((by0 - 6, by0), (bx0, bx1)), ((by1 + 1, by1 + 7), (bx0, bx1))):
        y0, y1 = max(0, ys[0]), min(H, ys[1])
        if y1 > y0:
            pads.append(img[y0:y1, max(0, xs[0]):min(W, xs[1])].reshape(-1, 3))
    if not pads:
        return (0, 0, 0)
    arr = np.concatenate(pads, axis=0).astype(np.float32)
    med = np.median(arr, axis=0)
    return tuple(int(round(v)) for v in med)


def crop_icon(img: np.ndarray, box, target: int = 0) -> tuple[np.ndarray, tuple]:
    """Вырез по габаритам рисунка.

    target <= 0 -> как есть, пиксель в пиксель (только добить до квадрата).
    target  > 0 -> квадрат target x target: большой рисунок уменьшаем,
                   маленький добираем фоном (без растягивания).
    """
    cb = content_box(img, box)
    if cb is None:
        return None, None
    x0, y0, x1, y1 = cb
    crop = img[y0:y1 + 1, x0:x1 + 1]
    h, w = crop.shape[:2]
    side = target if target > 0 else max(h, w)
    if side <= 0:
        return None, cb
    # уменьшать можно, растягивать нельзя: иначе мелкие иконки мылятся
    scale = min(1.0, side / max(h, w))
    if scale < 1.0:
        crop = cv2.resize(crop, (max(1, int(round(w * scale))),
                                 max(1, int(round(h * scale)))),
                          interpolation=cv2.INTER_AREA)
        h, w = crop.shape[:2]
    bg = _bg_color(img, box)
    out = np.zeros((side, side, 3), np.uint8)
    out[:, :] = bg
    oy, ox = (side - h) // 2, (side - w) // 2
    out[oy:oy + h, ox:ox + w] = crop
    return out, cb


def detect(img: np.ndarray) -> list[tuple[int, int, int, int]]:
    return sheet_boxes(img)
