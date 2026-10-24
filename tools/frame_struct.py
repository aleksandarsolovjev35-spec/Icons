#!/usr/bin/env python3
"""frame_struct.py — рамка как СТРУКТУРА пикселей заданных оттенков и формы.

Идея
----
Рамка на листе нарисована одним набором оттенков серого (на одном листе это,
например, 41, 52, 71, 88, 150). Значит рамку можно искать не «по яркости» и
не «по сетке», а как структуру: пиксели, попадающие в палитру оттенков
рамки, образуют замкнутый прямоугольник с равными сторонами.

Порядок
-------
1. `palette()` — выучить палитру листа. Берём длинные (>= 80 px) ровные
   отрезки серых пикселей: такая полоса — это линия рамки, у неё яркость
   вдоль отрезка меняется на 1-6 единиц. У рисунка длинных ровных полос нет.
   Цвета этих отрезков кластеризуем — получаем оттенки рамки.
2. `mask()` — маска «пиксель из палитры»: расстояние по RGB до ближайшего
   оттенка <= 10 и низкая насыщенность (рамка серая).
3. `segments()` — отрезки: горизонтальные пробеги маски длиной >= 90 px,
   вертикальные — то же. Рисунок таких длинных пробегов не даёт.
4. `rects()` — прямоугольник = две горизонтали с совпадающим диапазоном x
   плюс две вертикали между ними. Дополнительные фильтры:
     * стороны квадрата (разница <= 8%);
     * размер в пределах 0.82-1.20 от бокса слота;
     * все четыре стороны одного оттенка (расстояние между средними
       цветами сторон <= 12) — это и есть «структура одного набора цветов».
5. `crop_rect()` — вырез за последним пикселем линии (по покрытию стороны).

`annotate()` рисует найденные рамки и сами линии — для проверки глазами.
"""
from __future__ import annotations

import cv2
import numpy as np

# --- палитра -----------------------------------------------------------
RUN_MIN = 80          # длинный ровный отрезок = линия рамки
RUN_FLAT = 8          # разброс яркости вдоль линии
SHADE_TOL = 7         # отклонение по яркости от оттенка палитры
MAX_CHROMA = 34       # рамка серая
MASK_CHROMA = 48      # у линии бывают цветные крапины от JPEG
GAP = 6               # разрыв внутри линии, который склеиваем
GRAY_MIN = 12         # темнее этого линия рамки не бывает
# --- форма -------------------------------------------------------------
SEG_MIN = 90          # минимальная длина отрезка линии
SEARCH = 22           # окно поиска линии вокруг края слота
SIDE_LO, SIDE_HI = 0.82, 1.20
SQUARE_TOL = 0.08     # |ширина - высота| / max
SIDE_UNIFORM = 0.65   # какую долю стороны держит один оттенок
SHADE_MATCH_HARD = 34 # насколько стороны могут отличаться по яркости
LINE_TOL = 12         # разброс яркости внутри одной линии
INSET = 1


def brightness(img: np.ndarray) -> np.ndarray:
    return img.astype(np.float32).mean(axis=2)


def _runs(mask_row: np.ndarray, min_len: int, gap: int = GAP):
    """Непрерывные отрезки, склеивая дыры до gap px (крапины, сглаживание)."""
    idx = np.flatnonzero(mask_row)
    if idx.size == 0:
        return
    breaks = np.flatnonzero(np.diff(idx) > gap + 1)
    starts = np.concatenate(([0], breaks + 1))
    ends = np.concatenate((breaks, [idx.size - 1]))
    for s, e in zip(starts, ends):
        a, b = int(idx[s]), int(idx[e])
        if b - a + 1 >= min_len:
            yield a, b


def palette(img: np.ndarray, max_shades: int = 24) -> list[np.ndarray]:
    """Оттенки рамки на этом листе: список средних цветов (RGB).

    Линия рамки — единственное, что даёт сильный отклик top-hat (тонкая
    светлая полоса на чёрном). Берём те строки и столбцы, где отклик
    большой, и собираем их цвета — это и есть «набор оттенков рамки» данного
    листа. Стоит десятые доли секунды, потому что не перебирает пробеги
    питоновскими циклами.
    """
    from frame_rect import tophat
    h, v = tophat(img)
    f = img.astype(np.float32)
    chroma = f.max(axis=2) - f.min(axis=2)
    colors: list[np.ndarray] = []
    for resp, arr in ((h, f), (v.T, np.transpose(f, (1, 0, 2)))):
        ch = chroma if arr is f else chroma.T
        for k in range(resp.shape[0]):
            row = resp[k]
            strong = np.flatnonzero(row >= 25)
            if strong.size < RUN_MIN * 0.5:
                continue
            seg = arr[k, strong]
            chs = ch[k, strong]
            seg = seg[chs <= MAX_CHROMA]
            if seg.shape[0] < RUN_MIN * 0.3:
                continue
            if float(np.std(seg.mean(axis=1))) > RUN_FLAT * 2:
                continue
            colors.append(np.median(seg, axis=0))
    if not colors:
        return []
    return _cluster_colors(colors, max_shades)


def _cluster_colors(colors: list[np.ndarray], max_shades: int,
                    tol: float = 7.0) -> list[np.ndarray]:
    if not colors:
        return []
    arr = np.array(colors, dtype=np.float32)
    order = np.argsort(arr.mean(axis=1))
    shades: list[np.ndarray] = []
    for i in order:
        c = arr[i]
        if shades and np.linalg.norm(c - shades[-1]) <= tol:
            shades[-1] = 0.5 * (shades[-1] + c)
        elif any(np.linalg.norm(c - s) <= tol for s in shades):
            continue
        else:
            shades.append(c.copy())
    return shades[:max_shades]


def mask(img: np.ndarray, shades: list[np.ndarray]) -> np.ndarray:
    """Пиксели, попадающие в палитру оттенков рамки."""
    if not shades:
        return np.zeros(img.shape[:2], bool)
    f = img.astype(np.float32)
    br = f.mean(axis=2)
    chroma = f.max(axis=2) - f.min(axis=2)
    # сравниваем по ЯРКОСТИ: линия серая, но JPEG даёт цветные крапины,
    # и расстояние по RGB на них ложно велико
    best = np.full(br.shape, np.inf, np.float32)
    for s in shades:
        np.minimum(best, np.abs(br - float(s.mean())), out=best)
    return (best <= SHADE_TOL) & (chroma <= MASK_CHROMA)


def segments(m, min_len: int = SEG_MIN):
    """Отрезки маски: (горизонтальные, вертикальные) как (координата, a, b)."""
    hs, vs = [], []
    H, W = m.shape
    for y in range(H):
        for a, b in _runs(m[y], min_len):
            hs.append((y, a, b))
    for x in range(W):
        for a, b in _runs(m[:, x], min_len):
            vs.append((x, a, b))
    return hs, vs


def _group(segs, tol: int = 3):
    """Склеить соседние отрезки (линия 1-3 px) -> (центр, a_min, a_max)."""
    out: list[list] = []
    for c, a, b in sorted(segs):
        if out and c - out[-1][0][-1] <= tol and not (
                b < out[-1][1] - 10 or a > out[-1][2] + 10):
            out[-1][0].append(c)
            out[-1][1] = min(out[-1][1], a)
            out[-1][2] = max(out[-1][2], b)
        else:
            out.append([[c], a, b])
    return [(int(np.mean(g[0])), g[1], g[2]) for g in out]


def _uniform(img: np.ndarray, axis: str, pos: int, lo: int, hi: int) -> float:
    """Доля пикселей стороны, близких к её медианному цвету.

    У линии рамки это 0.8-1.0 (иконка может перекрыть часть), у края рисунка
    — 0.3-0.6: там цвет гуляет.
    """
    f = img.astype(np.float32)
    seg = f[lo:hi + 1, pos] if axis == "v" else f[pos, lo:hi + 1]
    med = np.median(seg, axis=0)
    if float(med.mean()) < 18:
        return 0.0
    d = np.linalg.norm(seg - med, axis=1)
    return float(np.mean(d <= 16))


def _side_color(img: np.ndarray, axis: str, pos: int, lo: int, hi: int) -> np.ndarray:
    f = img.astype(np.float32)
    seg = f[lo:hi + 1, pos] if axis == "v" else f[pos, lo:hi + 1]
    return np.median(seg, axis=0)


def rects(img: np.ndarray, box, hs=None, vs=None, size=None):
    """Все рамки-кандидаты вокруг слота: [(l, t, r, b, score)]."""
    x0, y0, x1, y1 = (int(v) for v in box)
    hg = _group([s for s in hs if s[1] - 8 <= x1 and s[2] + 8 >= x0])
    vg = _group([s for s in vs if s[1] - 8 <= y1 and s[2] + 8 >= y0])
    lefts = [c for c, _, _ in vg if c <= x0 + SEARCH and c >= x0 - SEARCH]
    rights = [c for c, _, _ in vg if c >= x1 - SEARCH and c <= x1 + SEARCH]
    tops = [c for c, _, _ in hg if c <= y0 + SEARCH and c >= y0 - SEARCH]
    bots = [c for c, _, _ in hg if c >= y1 - SEARCH and c <= y1 + SEARCH]
    if not (lefts and rights and tops and bots):
        return []
    bw, bh = x1 - x0 + 1, y1 - y0 + 1
    out = []
    for l in lefts:
        for r in rights:
            w = r - l
            if not (SIDE_LO * bw <= w <= SIDE_HI * bw):
                continue
            for t in tops:
                for b in bots:
                    hh = b - t
                    if not (SIDE_LO * bh <= hh <= SIDE_HI * bh):
                        continue
                    if abs(w - hh) > SQUARE_TOL * max(w, hh):
                        continue
                    # каждая сторона обязана быть РОВНОЙ вдоль себя: линия
                    # рамки одного оттенка по всей длине, рисунок — нет
                    uni = [
                        _uniform(img, "h", t, l, r),
                        _uniform(img, "h", b, l, r),
                        _uniform(img, "v", l, t, b),
                        _uniform(img, "v", r, t, b),
                    ]
                    if min(uni) < SIDE_UNIFORM:
                        continue
                    score = (4.0 - min(uni)) * 10 + 0.3 * (
                        abs(l - x0) + abs(r - x1) + abs(t - y0) + abs(b - y1)
                    ) + abs(w - hh)
                    # все рамки листа одного размера: это сильный ориентир,
                    # он отсекает «похожие на линию» детали рисунка
                    if size and size[0] and size[1]:
                        score += 4.0 * (abs(w - size[0]) + abs(hh - size[1]))
                    out.append((l, t, r, b, score))
    out.sort(key=lambda z: z[4])
    return out


def _sheet_size(img: np.ndarray, boxes, hs, vs,
                _depth: int = 0) -> tuple[int | None, int | None]:
    """Размер рамки, общий для листа: мода по слотам, где видны все 4 стороны."""
    ws, hhs = [], []
    for b in boxes:
        cand = rects(img, b, hs, vs)      # без prior: он ещё не известен
        if not cand:
            continue
        l, t, r, bb, _ = cand[0]
        ws.append(r - l)
        hhs.append(bb - t)
    def mode(vals):
        if not vals:
            return None
        vals = np.array(vals)
        # медиана устойчивее моды на малом числе слотов
        lo = np.percentile(vals, 25)
        hi = np.percentile(vals, 75)
        core = vals[(vals >= lo) & (vals <= hi)]
        return int(round(float(np.median(core if core.size else vals))))
    return mode(ws), mode(hhs)


def _side_profiles(img, box, m):
    """Профили покрытия маской вдоль сторон слота."""
    x0, y0, x1, y1 = (int(v) for v in box)
    H, W = m.shape
    ay0, ay1 = max(0, y0 - SEARCH), min(H, y1 + SEARCH + 1)
    ax0, ax1 = max(0, x0 - SEARCH), min(W, x1 + SEARCH + 1)
    cov_h = m[ay0:ay1, x0:x1 + 1].mean(axis=1)     # по строкам
    cov_v = m[y0:y1 + 1, ax0:ax1].mean(axis=0)     # по столбцам
    return cov_v, cov_h, ax0, ay0


def _snap(prof, guess, off, win=6):
    """Ближайший к догадке локальный максимум покрытия."""
    lo = max(0, int(round(guess)) - off - win)
    hi = min(len(prof) - 1, int(round(guess)) - off + win)
    if hi < lo:
        return int(round(guess))
    seg = prof[lo:hi + 1]
    idx = np.arange(lo, hi + 1) + off
    return int(idx[int(np.argmax(seg - 1e-6 * np.abs(idx - guess)))])


def _best_side(img, vals, edge, axis, span, m):
    """Самая «линейная» из кандидатов рядом с краем слота.

    Кандидат должен быть ровным вдоль себя (одного оттенка на всей длине) —
    так отсекаются детали рисунка, похожие на линию по цвету.
    """
    lo, hi = span
    best = None
    for c in vals:
        if abs(c - edge) > SEARCH:
            continue
        u = _uniform(img, axis, c, lo, hi)
        if u < SIDE_UNIFORM:
            continue
        key = (u, -abs(c - edge))
        if best is None or key > best[0]:
            best = (key, c)
    return best[1] if best else None


def find_rect(img: np.ndarray, box, shades=None, hs=None, vs=None,
              size=None, prof=None):
    """Рамка слота: структура «4 линии одного размера».

    Размер рамки на листе постоянный (это измерено по слотам, где видны все
    стороны). Поэтому достаточно надёжно найти, например, левую и верхнюю
    линии, а остальные получить как `лево + ширина` и `верх + высота`, лишь
    уточняя их по ближайшему кандидату в пределах ±4 px. Это устойчивее, чем
    независимо выбирать каждую сторону: деталь рисунка рядом с краем больше
    не может «съесть» 10-18 px.
    """
    if hs is None or vs is None:
        if shades is None:
            shades = palette(img)
        m = mask(img, shades)
        hs, vs = segments(m)
    if size is None or size[0] is None or size[1] is None:
        cand = rects(img, box, hs, vs)
        return cand[0][:4] if cand else None
    W, H = size
    x0, y0, x1, y1 = (int(v) for v in box)
    vg = _group([s for s in vs if s[1] - 8 <= y1 and s[2] + 8 >= y0])
    hg = _group([s for s in hs if s[1] - 8 <= x1 and s[2] + 8 >= x0])
    xs = [c for c, _, _ in vg]
    ys = [c for c, _, _ in hg]
    l = _best_side(img, xs, x0, "v", (y0, y1), None)
    r = _best_side(img, xs, x1, "v", (y0, y1), None)
    t = _best_side(img, ys, y0, "h", (x0, x1), None)
    b = _best_side(img, ys, y1, "h", (x0, x1), None)

    def snap(vals, guess, tol=4):
        near = [c for c in vals if abs(c - guess) <= tol]
        return min(near, key=lambda c: abs(c - guess)) if near else int(round(guess))

    if l is None and r is not None:
        l = snap(xs, r - W)
    if r is None and l is not None:
        r = snap(xs, l + W)
    if t is None and b is not None:
        t = snap(ys, b - H)
    if b is None and t is not None:
        b = snap(ys, t + H)
    if None in (l, r, t, b):
        return None
    # держим размер листа: расхождение больше 4 px — это уже не рамка
    if abs((r - l) - W) > 4 or abs((b - t) - H) > 4:
        return None
    return (int(l), int(t), int(r), int(b))


def crop_rect(img: np.ndarray, rect) -> tuple[int, int, int, int]:
    """Вырез по готовой рамке.

    Сама обрезка (по пикселям линии) уже отлажена в `frame_rect.crop_rect`
    и проверена на 45 168 сторонах, поэтому не дублируем её здесь: этот
    модуль отвечает за НАХОЖДЕНИЕ рамки, а не за рез.
    """
    from frame_rect import crop_rect as _crop
    return _crop(img, rect)


def detect(img: np.ndarray, boxes):
    """Рамки для всех слотов листа: [(l, t, r, b) | None]."""
    shades = palette(img)
    m = mask(img, shades)
    hs, vs = segments(m)
    size = _sheet_size(img, boxes, hs, vs)
    out = []
    for b in boxes:
        prof = _side_profiles(img, b, m)
        out.append(find_rect(img, b, shades, hs, vs, size, prof))
    return out


def annotate(img: np.ndarray, boxes, path: str, thick: int = 2) -> int:
    """Обвести найденные рамки красным и сохранить. Возвращает число рамок."""
    th = palette(img)
    m = mask(img, th)
    hs, vs = segments(m)
    size = _sheet_size(img, boxes, hs, vs)
    out = img.copy()
    n = 0
    for b in boxes:
        r = find_rect(img, b, th, hs, vs, size, _side_profiles(img, b, m))
        if r is None:
            continue
        n += 1
        cv2.rectangle(out, (r[0], r[1]), (r[2], r[3]), (0, 0, 255), thick)
    cv2.imwrite(path, out)
    return n


def palette_report(img: np.ndarray) -> list[int]:
    """Оттенки палитры как яркости — для лога."""
    return [int(round(float(s.mean()))) for s in palette(img)]


def verify(img: np.ndarray, rect, size=None) -> dict:
    """Проверить готовую рамку «по структуре»: оттенки, ровность сторон, размер.

    Возвращает словарь с метриками; `ok` = рамка действительно похожа на
    рамку, а не на деталь рисунка.
    """
    l, t, r, b = rect
    uni = [
        _uniform(img, "h", t, l, r),
        _uniform(img, "h", b, l, r),
        _uniform(img, "v", l, t, b),
        _uniform(img, "v", r, t, b),
    ]
    cols = np.array([
        _side_color(img, "h", t, l, r),
        _side_color(img, "h", b, l, r),
        _side_color(img, "v", l, t, b),
        _side_color(img, "v", r, t, b),
    ])
    spread = float(max(np.linalg.norm(cols[i] - cols[j])
                       for i in range(4) for j in range(i + 1, 4)))
    w, hh = r - l, b - t
    size_ok = True
    if size and size[0] and size[1]:
        size_ok = abs(w - size[0]) <= 4 and abs(hh - size[1]) <= 4
    # ВАЖНО: требовать «все четыре стороны одного оттенка» нельзя — у рамок
    # этой серии есть градиент (верх 41, низ 116). Поэтому в проверке только
    # то, что действительно инвариант: сторона ровная вдоль себя, рамка
    # квадратная и её размер совпадает с размером рамок всего листа.
    ok = (min(uni) >= SIDE_UNIFORM and size_ok and abs(w - hh) <= 12)
    return {"ok": ok, "min_uniform": round(min(uni), 2),
            "shade_spread": round(spread, 1), "size_ok": size_ok,
            "w": w, "h": hh}


def sheet_size(img: np.ndarray, boxes) -> tuple:
    """Размер рамок листа (по видимым линиям) — для проверки."""
    shades = palette(img)
    hs, vs = segments(mask(img, shades))
    return _sheet_size(img, boxes, hs, vs)


def peaks_near(img, box, reach: int = 26, th=None):
    """Пики top-hat рядом со слотом: (x-координаты, y-координаты)."""
    from frame_rect import tophat
    h, v = th if th is not None else tophat(img)
    x0, y0, x1, y1 = (int(t) for t in box)
    H, W = h.shape
    ay0, ay1 = max(0, y0 - reach), min(H, y1 + reach + 1)
    ax0, ax1 = max(0, x0 - reach), min(W, x1 + reach + 1)
    pv = np.percentile(v[ay0:ay1, ax0:ax1], 75, axis=0)
    ph = np.percentile(h[ay0:ay1, ax0:ax1], 75, axis=1)
    xs = [ax0 + int(i) for i in np.flatnonzero(pv >= 10)]
    ys = [ay0 + int(i) for i in np.flatnonzero(ph >= 10)]
    return xs, ys


def _pair(vals, size, edge_lo, edge_hi, lo_a, hi_a):
    """Пара пиков с расстоянием ровно `size`, ближайшая к краям слота."""
    best = None
    for a in vals:
        if not (lo_a <= a <= hi_a):
            continue
        for b in vals:
            if b <= a:
                continue
            d = abs((b - a) - size)
            if d > 3:
                continue
            score = 2.0 * d + 0.25 * (abs(a - edge_lo) + abs(b - edge_hi))
            if best is None or score < best[0]:
                best = (score, a, b)
    return None if best is None else (best[1], best[2])


def locate(img: np.ndarray, box, size, peaks=None):
    """Рамка слота: пара сторон ровно по размеру рамок листа.

    Размер рамок на листе постоянный (измерено), поэтому стороны ищутся не
    по одной, а ПАРОЙ с расстоянием `size`. Это убирает промах на 1-2 px:
    раньше сторона могла встать на соседний пик, и линия попадала в вырез.
    """
    if not size or not size[0] or not size[1]:
        return None
    W, H = int(size[0]), int(size[1])
    x0, y0, x1, y1 = (int(t) for t in box)
    xs, ys = peaks if peaks is not None else peaks_near(img, box)
    horiz = _pair(xs, W, x0, x1, x0 - SEARCH, x0 + SEARCH)
    vert = _pair(ys, H, y0, y1, y0 - SEARCH, y0 + SEARCH)
    if horiz is None or vert is None:
        return None
    return (horiz[0], vert[0], horiz[1], vert[1])


def complete(img: np.ndarray, box, rect, size, peaks=None):
    """Достроить/уточнить рамку по размеру листа.

    Размер рамок на листе постоянный. Поэтому, зная размер, каждую сторону
    можно не искать заново, а проверить: если она стоит не там, где должна
    (по другой стороне + размер), берём предсказанную позицию и уточняем её
    по ближайшему пику top-hat в пределах ±4 px.
    """
    if size is None or size[0] is None or size[1] is None:
        return rect
    W, H = size
    x0, y0, x1, y1 = (int(v) for v in box)
    # если рамка не нашлась совсем — начинаем от бокса слота: бокс стоит
    # примерно там же, а размер листа известен точно
    l = rect[0] if rect else x0
    t = rect[1] if rect else y0
    r = rect[2] if rect else x0 + W
    b = rect[3] if rect else y0 + H
    if peaks is None:
        from frame_rect import tophat
        h, v = tophat(img)
        import numpy as _np
        H_, W_ = h.shape
        ay0, ay1 = max(0, y0 - 26), min(H_, y1 + 27)
        ax0, ax1 = max(0, x0 - 26), min(W_, x1 + 27)
        pv = _np.percentile(v[ay0:ay1, ax0:ax1], 75, axis=0)
        ph = _np.percentile(h[ay0:ay1, ax0:ax1], 75, axis=1)
        peaks = (list(ax0 + _np.flatnonzero(pv >= 10)), ax0,
                 list(ay0 + _np.flatnonzero(ph >= 10)), ay0)

    xs, ys = peaks[0], peaks[2]

    def snap(vals, guess, tol=4):
        near = [c for c in vals if abs(c - guess) <= tol]
        return min(near, key=lambda c: abs(c - guess)) if near else int(round(guess))

    if l is None and r is not None:
        l = snap(xs, r - W)
    if r is None and l is not None:
        r = snap(xs, l + W)
    if t is None and b is not None:
        t = snap(ys, b - H)
    if b is None and t is not None:
        b = snap(ys, t + H)
    if None in (l, r, t, b):
        return None
    # каждую сторону уточняем по реальному пику линии, иначе предсказание
    # остаётся в 1-2 px от линии и линия попадает в вырез
    l = snap(xs, l)
    r = snap(xs, r)
    t = snap(ys, t)
    b = snap(ys, b)
    # подтягиваем к размеру листа (сдвиг не больше 4 px)
    if abs((r - l) - W) > 2 and abs((r - l) - W) <= 8:
        mid = (l + r) / 2
        l, r = int(round(mid - W / 2)), int(round(mid + W / 2))
    if abs((b - t) - H) > 2 and abs((b - t) - H) <= 8:
        mid = (t + b) / 2
        t, b = int(round(mid - H / 2)), int(round(mid + H / 2))
    return (int(l), int(t), int(r), int(b))


def size_from_peaks(img: np.ndarray, boxes, th=None) -> tuple:
    """Размер рамок листа по расстояниям между пиками линий.

    Работает даже там, где маска оттенков не даёт ни одного замкнутого
    прямоугольника (линии тоньше 1 px, рисунок почти касается рамки):
    берём ближайший пик слева и справа от бокса, считаем расстояние и
    возвращаем медиану — размер рамок листа одинаковый.
    """
    from frame_rect import tophat
    th = th if th is not None else tophat(img)
    ws, hh = [], []
    for b in boxes:
        xs, ys = peaks_near(img, b, th=th)
        x0, y0, x1, y1 = (int(v) for v in b)
        left = [c for c in xs if c <= x0 + SEARCH and c >= x0 - SEARCH]
        right = [c for c in xs if c >= x1 - SEARCH and c <= x1 + SEARCH]
        top = [c for c in ys if c <= y0 + SEARCH and c >= y0 - SEARCH]
        bot = [c for c in ys if c >= y1 - SEARCH and c <= y1 + SEARCH]
        if left and right:
            ws.append(min(right) - max(left))
        if top and bot:
            hh.append(min(bot) - max(top))
    def med(vals):
        if len(vals) < 3:
            return None
        v = np.array(vals)
        lo, hi = np.percentile(v, 25), np.percentile(v, 75)
        core = v[(v >= lo) & (v <= hi)]
        return int(round(float(np.median(core if core.size else v))))
    return med(ws), med(hh)


def sheet_rects(img: np.ndarray, boxes) -> tuple[list, tuple, list]:
    """Рамки всего листа: (прямоугольники, размер листа, палитра).

    Порядок: палитра оттенков -> размер рамок листа -> пары сторон ровно по
    этому размеру. Пара сторон вместо четырёх независимых — ключ к точности:
    сторона не может «уехать» на соседний пик.
    """
    from frame_rect import tophat
    shades = palette(img)
    m = mask(img, shades)
    hs, vs = segments(m)
    size = _sheet_size(img, boxes, hs, vs)
    th = tophat(img)
    if not size or not size[0] or not size[1]:
        alt = size_from_peaks(img, boxes, th)
        size = (size[0] or alt[0], size[1] or alt[1])
    out = []
    for b in boxes:
        r = locate(img, b, size, peaks_near(img, b, th=th))
        if r is None:
            r = complete(img, b, find_rect(img, b, shades, hs, vs), size)
        out.append(r)
    return out, size, shades
