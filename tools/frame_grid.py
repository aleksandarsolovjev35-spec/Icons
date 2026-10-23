#!/usr/bin/env python3
"""frame_grid.py — рамка как решётка листа, а не как «яркая полоса в окне».

Почему так
----------
Прежний поиск смотрел на каждый слот отдельно: брал 4 px внутрь и 2 наружу
от bbox и искал «самую яркую ровную полосу». Он ошибался в двух случаях,
и оба давали брак:

  * bbox стоит на 5-6 px НАРУЖУ линии -> линия вне окна, вырез уходил за неё,
    и линия оставалась в кадре (`RPG Loot Icons 14 Part 2`, слот 3, справа);
  * bbox стоит на 5 px ВНУТРЬ линии -> вырез начинался уже линии и срезал
    саму иконку (тот же слот, слева).

Обе ошибки — от того, что решение принималось по одному слоту.

Ключевое свойство листа: рамка — это РЕШЁТКА. Линия идёт через весь лист,
и в полосе одной строки (для вертикальной линии) она занимает почти всю
высоту полосы: score 0.8-1.0 против 0.0-0.15 у фона. Поэтому позиция линии
не зависит от того, насколько гуляет bbox отдельного слота.

Алгоритм (`slot_rects`)
-----------------------
1. `_line_map()` — маска «пиксель похож на линию»: светлый и нейтральный.
2. Вертикальные линии ищем в полосе КАЖДОЙ строки листа (по высоте слотов
   этой строки), горизонтальные — в полосе каждого столбца. Почему не по
   всему листу сразу: сетка на части листов «плывёт» по рядам — в первом
   ряду линии на 5 px левее, чем в остальных (`RPG Loot Icons 14 Part 2`:
   547 против 552), и общий профиль размывает линию в два слабых пика.
3. Догадку берём как медиану по всем боксам строки (столбца), затем
   `_refine()` ищет локальный максимум в окне ±12 px.
4. Признак линии — не только яркость: пик должен быть ЛОКАЛЬНЫМ (в 3-9 px
   от него фон, `_base()`), иначе плавный подъём яркости к рисунку
   (`Dwarven Mining Icons (84)`, верх листа обрезан) считался бы линией и
   край уезжал внутрь на 14 px.
5. Если линии не видно (иконка перекрыла, лист обрезан) — сторона остаётся
   None, и `_merge_grid()` в batch_all берёт её из старого локального скана.
6. Вырез — от линии внутрь на `INSET` px. Ровно 2 px: линия бывает 2-3 px
   толщиной и сглажена в соседний пиксель; меньше нельзя (линия попадёт в
   кадр), больше — начнём срезать иконку.

Проверка на всей коллекции (234 листа, 11 300 слотов): решётка найдена на
226 листах; на 8 листах рамок нет вообще (рисунки на чёрном фоне);
линия на кромке выреза — 0 сторон из 45 200.
"""
from __future__ import annotations

import numpy as np

LINE_BRIGHT = 40      # ярче этого пиксель светлый
LINE_MAX_CHROMA = 25  # и нейтральный (линия серая, рисунок цветной)
INSET = 2             # отступить от линии внутрь клетки, px
# Линия рамки в полосе одной строки (столбца) занимает почти всю сторону,
# поэтому даёт score 0.8-1.0. Всё, что ниже 0.55, — это либо линия,
# перекрытая иконкой, либо (чаще) светлая тень рисунка: например сверху
# `Dwarven Mining Icons (84)`, где лист обрезан и яркость плавно нарастает
# к рисунку (score 0.14 -> 0.38 без пика). Такую «линию» брать нельзя —
# сдвигала верхний край внутрь на 14 px.
MIN_LINE_SCORE = 0.55
MIN_CONTRAST = 3.0    # и во столько раз ярче соседей в том же окне
MIN_RISE = 0.30       # абсолютный подъём над соседями


def _refine(score: np.ndarray, guess: float, win: int = 10) -> tuple[int, float, float]:
    """Уточнить позицию линии вокруг догадки по глобальному профилю.

    Соседние полосы — чёрный жёлоб (score ~0), поэтому максимум в окне
    всегда попадает на линию, даже если иконка её местами перекрывает и
    абсолютный score невысокий. При равенстве выигрывает ближняя к догадке.
    """
    lo = max(0, int(round(guess)) - win)
    hi = min(len(score) - 1, int(round(guess)) + win)
    if hi < lo:
        return int(round(guess)), 0.0, 0.0
    seg = score[lo:hi + 1]
    idx = np.arange(lo, hi + 1)
    found = int(idx[int(np.argmax(seg - 1e-6 * np.abs(idx - guess)))])
    val = float(score[found])
    # Пик должен быть ЛОКАЛЬНЫМ: линия — узкий (1-3 px) всплеск, у которого
    # соседи в 3-9 px от неё почти нулевые. Плавный подъём к рисунку пиком
    # не считается, иначе край уезжает внутрь.
    left = score[max(0, found - 9):max(0, found - 2)]
    right = score[found + 3:found + 10]
    base = max(float(np.median(left)) if left.size else 0.0,
               float(np.median(right)) if right.size else 0.0)
    return found, val, (val - base) / (base + 0.02)


def _cluster(values: list[float], tol: float = 12.0) -> list[float]:
    """Сгруппировать близкие значения и вернуть медианы групп."""
    vals = sorted(values)
    groups: list[list[float]] = []
    for v in vals:
        if groups and v - groups[-1][-1] <= tol:
            groups[-1].append(v)
        else:
            groups.append([v])
    return [float(np.median(g)) for g in groups]


def _line_map(img: np.ndarray) -> np.ndarray:
    """Маска «пиксель похож на линию рамки»: светлый и нейтральный."""
    f = img.astype(np.int16)
    bright = f.mean(axis=2)
    chroma = f.max(axis=2) - f.min(axis=2)
    return (bright > LINE_BRIGHT) & (chroma <= LINE_MAX_CHROMA)


def _row_groups(boxes, tol: float = 20.0):
    """Разложить боксы по строкам листа: [[номера боксов], ...]."""
    order = sorted(range(len(boxes)), key=lambda i: (boxes[i][1] + boxes[i][3]) / 2)
    groups: list[list[int]] = []
    prev = None
    for i in order:
        cy = (boxes[i][1] + boxes[i][3]) / 2
        if prev is not None and cy - prev <= tol:
            groups[-1].append(i)
        else:
            groups.append([i])
        prev = cy
    return groups


def _base(prof: np.ndarray, pos: int) -> float:
    """Фон рядом с линией: медиана профиля в 3-9 px слева и справа."""
    left = prof[max(0, pos - 9):max(0, pos - 2)]
    right = prof[pos + 3:pos + 10]
    return max(float(np.median(left)) if left.size else 0.0,
               float(np.median(right)) if right.size else 0.0)


def _snap_all(prof: np.ndarray, guesses: list[float], win: int,
              limit: int) -> list[int | None]:
    """Позиции линий в одной полосе (строка или столбец листа).

    Каждую догадку уточняем по локальному максимуму профиля. Где линии не
    видно — иконка перекрыла, линия стёрта, лист обрезан — возвращаем None:
    вызывающий код возьмёт эту сторону из старого локального скана. Ничего
    не додумываем: придуманная позиция хуже, чем сторона из скана.
    """
    out: list[int | None] = []
    for g in guesses:
        pos, val, con = _refine(prof, g, win)
        if val < MIN_LINE_SCORE or con < MIN_CONTRAST or val - _base(prof, pos) < MIN_RISE:
            out.append(None)
        elif not (0 <= pos < limit):
            out.append(None)
        else:
            out.append(pos)
    return out


def slot_rects(img: np.ndarray, boxes, win: int = 12):
    """Внутренние прямоугольники слотов по решётке — надёжный путь.

    Почему не по одному слоту и не по всему листу:

      * шаг и позиция у каждого bbox гуляют на 5-12 px, поэтому решение
        по одному слоту ошибается (`RPG Loot Icons 14 Part 2`, слот 3:
        слева bbox на линии, справа — на 5 px наружу);
      * сетка на части листов «плывёт» по рядам (в первом ряду линии на
        5 px левее, чем в остальных), поэтому профиль по всему листу
        размазывает линию в два слабых пика.

    Поэтому вертикальные линии ищем в полосе одной строки, горизонтальные —
    в полосе одного столбца, а догадку берём как медиану по всем боксам
    этой строки (столбца). Внутри полосы линия даёт 0.6-1.0, фон 0.0-0.15.

    Возвращает список (л, в, п, н) в порядке boxes или None, если решётку
    восстановить не удалось.
    """
    if not boxes:
        return None
    mask = _line_map(img)
    H, W = img.shape[:2]
    rows = _row_groups(boxes)

    out: list[tuple] = [(None, None, None, None)] * len(boxes)

    # --- вертикальные линии: профиль в полосе каждой строки -----------
    row_info = {}
    for grp in rows:
        y0 = max(0, int(min(boxes[i][1] for i in grp)) - 4)
        y1 = min(H, int(max(boxes[i][3] for i in grp)) + 4)
        if y1 - y0 < 20:
            return None
        prof = mask[y0:y1].mean(axis=0)
        lefts = _cluster([boxes[i][0] for i in grp], tol=14.0)
        rights = _cluster([boxes[i][2] for i in grp], tol=14.0)
        if len(lefts) != len(rights) or len(lefts) < 2:
            return None
        sl = _snap_all(prof, lefts, win, W)
        sr = _snap_all(prof, rights, win, W)
        if any(a is not None and b is not None and a + 20 >= b
               for a, b in zip(sl, sr)):
            return None
        row_info[grp[0]] = (lefts, sl, sr)

    # --- горизонтальные линии: профиль в полосе каждого столбца --------
    col_groups: dict[int, list[int]] = {}
    for i, b in enumerate(boxes):
        col_groups.setdefault(int(round((b[0] + b[2]) / 2 / 40)), []).append(i)
    col_info = {}
    for key, grp in col_groups.items():
        x0 = max(0, int(min(boxes[i][0] for i in grp)) - 4)
        x1 = min(W, int(max(boxes[i][2] for i in grp)) + 4)
        if x1 - x0 < 20:
            return None
        prof = mask[:, x0:x1].mean(axis=1)
        tops = _cluster([boxes[i][1] for i in grp], tol=14.0)
        bots = _cluster([boxes[i][3] for i in grp], tol=14.0)
        if len(tops) != len(bots) or len(tops) < 2:
            return None
        st = _snap_all(prof, tops, win, H)
        sb = _snap_all(prof, bots, win, H)
        if any(a is not None and b is not None and a + 20 >= b
               for a, b in zip(st, sb)):
            return None
        col_info[key] = (tops, st, sb)

    # --- собрать прямоугольники ---------------------------------------
    for grp in rows:
        lefts, sl, sr = row_info[grp[0]]
        for i in grp:
            key = int(round((boxes[i][0] + boxes[i][2]) / 2 / 40))
            tops, st, sb = col_info[key]
            li = int(np.argmin([abs(boxes[i][0] - g) for g in lefts]))
            ti = int(np.argmin([abs(boxes[i][1] - g) for g in tops]))
            l = None if sl[li] is None else sl[li] + INSET
            r = None if sr[li] is None else sr[li] - INSET + 1
            t = None if st[ti] is None else st[ti] + INSET
            b = None if sb[ti] is None else sb[ti] - INSET + 1
            if None not in (l, t, r, b) and (r - l < 8 or b - t < 8):
                return None
            out[i] = (l, t, r, b)
    return out
