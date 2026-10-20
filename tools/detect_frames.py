#!/usr/bin/env python3
"""
detect_frames.py — поиск серо-белых рамок (слотов) на листах иконок.

Что ищем
--------
Каждый слот обведён тонкой (1-2 px) серой линией, а между слотами лежит
почти чёрный "жёлоб". Рамки ровные, оси-параллельные, сетка регулярная.

Алгоритм (4 шага)
-----------------
1. ЖЁЛОБА. Фон листов чёрный (медиана 0..16). Берём 99-й процентиль яркости
   по каждому столбцу/строке: там, где он остаётся почти чёрным на всей длине,
   лежит межслотовая полоса. Склеиваем разрывы <= 6 px и выбрасываем полосы
   уже 5 px (это случайные тёмные столбцы внутри картинки).
2. СЕТКА. Середины жёлобов дают позиции границ; крайние границы — от края
   жёлоба до края листа. Проверяем регулярность шага (медиана вместо среднего).
3. УТОЧНЕНИЕ ЛИНИЙ. Для каждой стороны каждого слота в окне +-10 px ищется
   "плато" тонкой линии: пиксели, которые одновременно
       * ярче соседей на 2 px влево/вправо (или вверх/вниз) на > 6,
       * в сером диапазоне 30..225,
       * почти постоянны вдоль всей стороны (низкий разброс),
   и берётся центр этого плато — так 1-px и 2-px линии дают одинаковый bbox.
4. КОНТРОЛЬ. Размеры слотов сравниваются с номиналом (жёлоб-жёлоб); бокс,
   уехавший дальше 45% от него, откатывается к сеточному. Затем все рамки
   столбца/строки выравниваются по медиане (сетка регулярная, поэтому выброс
   = линия перекрыта иконкой). Если ни одна линия не подтвердилась
   (confidence < 0.15) — лист помечается no_frames и рамки не рисуются.

Использование
-------------
    python3 tools/detect_frames.py                       # все листы -> annotated/preview/
    python3 tools/detect_frames.py --scale 1.0           # разметка в полном размере
    python3 tools/detect_frames.py --limit 6 -v          # первые 6 с логом
    python3 tools/detect_frames.py "RPG Loot Icons 01/RPG Loot Icons 01 Part 1.jpg"
    python3 tools/detect_frames.py --no-draw --csv out.csv   # только координаты

Результат: размеченные листы + CSV/JSON со списком bbox на каждый слот.
"""
from __future__ import annotations

import argparse
import csv
import glob
import json
import os
import sys
from dataclasses import dataclass, field

import cv2
import numpy as np

BG_SLACK = 90          # насколько 99-й процентиль может превышать фон в жёлобе
GUTTER_MIN_W = 5       # мин. ширина жёлоба, px
SEARCH_W = 14          # окно поиска линии вокруг сеточной границы, px
INSET = 0.10           # отступ от углов при оценке линии (доля стороны)


# --------------------------------------------------------------------- helpers
def _runs(mask: np.ndarray, max_gap: int = 6) -> list[tuple[int, int]]:
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return []
    groups = np.split(idx, np.flatnonzero(np.diff(idx) > max_gap) + 1)
    return [(int(g[0]), int(g[-1])) for g in groups]


def _gutter_bands(profile: np.ndarray, limit: int) -> tuple[list, list, list]:
    """Возвращает (inner, edge_left, edge_right) полосы-жёлоба."""
    bands = [b for b in _runs(profile) if b[1] - b[0] + 1 >= GUTTER_MIN_W]
    if not bands:
        return [], [], []
    # полосы ближе 45% типичного шага — один и тот же жёлоб (разорван бликом)
    centers = [(a + b) / 2 for a, b in bands]
    if len(centers) > 1:
        pitch = float(np.median(np.diff(centers)))
        merged = [bands[0]]
        for (a, b), c in zip(bands[1:], centers[1:]):
            pa, pb = merged[-1]
            if c - (pa + pb) / 2 < 0.45 * pitch:
                merged[-1] = (pa, b)
            else:
                merged.append((a, b))
        bands = merged
    inner, left, right = [], [], []
    for a, b in bands:
        if a <= 2:
            left.append((a, b))
        elif b >= limit - 3:
            right.append((a, b))
        else:
            inner.append((a, b))
    return inner, left, right


def _plateau_line(band: np.ndarray) -> tuple[float, float]:
    """
    band: (rows_along_side, cols_candidates) — яркость вокруг границы.
    Возвращает (позиция_центра_плато_в_координатах_band, confidence 0..1).
    """
    rows, w = band.shape
    if w < 3 or rows < 8:
        return (w - 1) / 2.0, 0.0
    P = np.pad(band, ((0, 0), (2, 2)), mode="edge")
    left = (P[:, 0:w] + P[:, 1:w + 1]) / 2.0
    right = (P[:, 3:w + 3] + P[:, 4:w + 4]) / 2.0
    nb = np.maximum(left, right)
    ok = (band - nb > 6) & (band > 30) & (band < 225)
    frac = ok.mean(axis=0)

    uni = np.zeros(w, np.float32)
    for c in range(w):
        v = band[ok[:, c], c]
        if v.size < 5:
            continue
        uni[c] = max(0.0, 1.0 - float(v.std()) / 45.0)
    score = frac * uni

    k = int(np.argmax(score))
    if score[k] <= 0:
        return (w - 1) / 2.0, 0.0
    lo = k
    while lo > 0 and score[lo - 1] >= 0.55 * score[k]:
        lo -= 1
    hi = k
    while hi < w - 1 and score[hi + 1] >= 0.55 * score[k]:
        hi += 1
    return (lo + hi) / 2.0, float(score[k])


def _line_candidates(gray: np.ndarray, lo: int, hi: int,
                     along: tuple[int, int], axis: int,
                     min_score: float = 0.25) -> list[tuple[float, float]]:
    """Все кандидаты-линии в окне [lo, hi): список (позиция, score), по x."""
    a, b = along
    if axis == 0:
        band = gray[a:b, lo:hi].astype(np.float32)
    else:
        band = gray[lo:hi, a:b].T.astype(np.float32)
    if band.shape[1] < 3 or band.shape[0] < 8:
        return []
    w = band.shape[1]
    P = np.pad(band, ((0, 0), (2, 2)), mode="edge")
    nb = np.maximum((P[:, 0:w] + P[:, 1:w + 1]) / 2.0,
                    (P[:, 3:w + 3] + P[:, 4:w + 4]) / 2.0)
    ok = (band - nb > 6) & (band > 30) & (band < 225)
    frac = ok.mean(axis=0)
    score = np.zeros(w, np.float32)
    for c in range(w):
        v = band[ok[:, c], c]
        if v.size >= 5:
            score[c] = frac[c] * max(0.0, 1.0 - float(v.std()) / 45.0)
    # плато: собираем связные участки score >= 0.55*max, берём центр каждого
    if score.max() < min_score:
        return []
    thr = max(min_score, 0.55 * float(score.max()))
    out, c = [], 0
    while c < w:
        if score[c] < thr:
            c += 1
            continue
        e = c
        while e + 1 < w and score[e + 1] >= thr:
            e += 1
        k = c + int(np.argmax(score[c:e + 1]))
        out.append((lo + (c + e) / 2.0, float(score[k])))
        c = e + 1
    return out


def _refine(gray: np.ndarray, lo: int, hi: int, along: tuple[int, int],
            axis: int) -> tuple[float, float]:
    """Уточняет одну границу. axis=0 -> вертикальная линия (ищем по x)."""
    a, b = along
    if axis == 0:
        band = gray[a:b, lo:hi].astype(np.float32)
    else:
        band = gray[lo:hi, a:b].T.astype(np.float32)
    pos, conf = _plateau_line(band)
    return lo + pos, conf


def _refine2(gray: np.ndarray, nominal: int, along: tuple[int, int], axis: int,
             window: int = 8) -> tuple[float, float]:
    """Уточнение с авто-расширением окна, если линия не нашлась."""
    total = gray.shape[1] if axis == 0 else gray.shape[0]
    for win in (window, 2 * window):
        lo, hi = max(1, nominal - win), min(total - 1, nominal + win + 1)
        if hi - lo < 3:
            continue
        pos, conf = _refine(gray, lo, hi, along, axis)
        if conf >= 0.35:
            return pos, conf
    return float(nominal), 0.0


# ----------------------------------------------------------------------- model
@dataclass
class Sheet:
    path: str
    width: int
    height: int
    cols: int = 0
    rows: int = 0
    cell_w: float = 0.0
    cell_h: float = 0.0
    confidence: float = 0.0
    boxes: list = field(default_factory=list)      # (x0, y0, x1, y1)
    grid: list = field(default_factory=list)       # сеточные (без уточнения)
    cells_conf: list = field(default_factory=list)  # по ячейке, 0..1
    corrected: int = 0                              # выбросов, возвращённых к сетке
    warns: list = field(default_factory=list)


def detect_sheet(path: str, verbose: bool = False) -> Sheet:
    gray = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise FileNotFoundError(f"не читается: {path}")
    H, W = gray.shape
    sh = Sheet(path=path, width=W, height=H)
    g16 = gray.astype(np.int16)
    bg = float(np.median(g16))
    thr = bg + BG_SLACK

    # ---------------------------------------------------------------- 1. жёлоба
    prof_x = np.percentile(g16, 99, axis=0) < thr
    prof_y = np.percentile(g16, 99, axis=1) < thr
    inx, lx, rx = _gutter_bands(prof_x, W)
    iny, uy, dy = _gutter_bands(prof_y, H)
    if not inx or not iny:
        sh.warns.append("no_gutters")
        return sh

    cols, rows = len(inx) + 1, len(iny) + 1
    sh.cols, sh.rows = cols, rows

    # ---------------------------------------------------------------- 2. сетка
    # номинальные границы слотов: жёлоб слева/справа от слота
    def nominals(inner, left, right, total):
        xs = [[-1, 0]]                       # номинальные (g0, g1) для каждого слота
        for a, b in inner:
            xs[-1][1] = a
            xs.append([b, total - 1])
        if left:
            xs[0][0] = left[-1][1] + 1
        if right:
            xs[-1][1] = right[0][0] - 1
        return xs

    nx = nominals(inx, lx, rx, W)
    ny = nominals(iny, uy, dy, H)

    # ---------------------------------------------------------------- 3. уточнение
    # Между двумя слотами всегда ДВЕ линии: правая граница левого слота и левая
    # граница правого. Ищем их вместе внутри жёлоба (+-12 px) и берём первую и
    # последнюю найденную. Так линии не теряются, даже если они тусклее жёлоба
    # (обычный случай: линия ярче 30, но темнее порога "не жёлоб").
    MARGIN = 12

    def outer_line(nominal: int, along: tuple[int, int], axis: int, total: int):
        """Внешняя граница листа — линии с одной стороны, окно пошире."""
        for win in (10, 20):
            lo = max(1, nominal - win)
            hi = min(total - 1, nominal + win + 1)
            cands = _line_candidates(gray, lo, hi, along, axis)
            if cands:
                # берём ближайшего к номиналу
                best = min(cands, key=lambda c: abs(c[0] - nominal))
                return best
        return float(nominal), 0.0

    def gutter_lines(band: tuple[int, int], along: tuple[int, int], axis: int,
                     total: int):
        """Две линии жёлоба: правая грань левого слота и левая грань правого.

        Жёлоб делим пополам: слева от середины ищем правую границу (ближайшую
        к середине), справа — левую. Это не даёт линии «уехать» на контур
        иконки, если он попал в окно поиска.
        """
        a, b = band
        lo, hi = max(1, a - MARGIN), min(total - 1, b + MARGIN + 1)
        cands = _line_candidates(gray, lo, hi, along, axis)
        if not cands:
            return (float(a), 0.0), (float(b), 0.0)
        mid = (a + b) / 2.0
        left = [c for c in cands if c[0] <= mid]
        right = [c for c in cands if c[0] >= mid]
        l = max(left, key=lambda c: c[0]) if left else (float(a), 0.0)
        r = min(right, key=lambda c: c[0]) if right else (float(b), 0.0)
        return l, r

    # диапазон вдоль стороны (чтобы не ловить углы и арт по краям)
    yin = (ny[0][0] + int(INSET * (ny[-1][1] - ny[0][0])),
           ny[-1][1] - int(INSET * (ny[-1][1] - ny[0][0])))
    xin = (nx[0][0] + int(INSET * (nx[-1][1] - nx[0][0])),
           nx[-1][1] - int(INSET * (nx[-1][1] - nx[0][0])))

    xs = [[0.0, 0.0] for _ in range(cols)]
    xs_conf = [[0.0, 0.0] for _ in range(cols)]
    # внешние границы
    left_edge = outer_line(max(2, nx[0][0]), (max(0, yin[0]), min(H, yin[1])), 0, W)
    right_edge = outer_line(min(W - 2, nx[-1][1]), (max(0, yin[0]), min(H, yin[1])), 0, W)
    xs[0][0], xs_conf[0][0] = left_edge
    xs[-1][1], xs_conf[-1][1] = right_edge
    for k, band in enumerate(inx):
        l, r = gutter_lines(band, (max(0, yin[0]), min(H, yin[1])), 0, W)
        xs[k][1], xs_conf[k][1] = l        # правая граница левого слота
        xs[k + 1][0], xs_conf[k + 1][0] = r  # левая граница правого слота

    ys = [[0.0, 0.0] for _ in range(rows)]
    ys_conf = [[0.0, 0.0] for _ in range(rows)]
    top_edge = outer_line(max(2, ny[0][0]), (max(0, xin[0]), min(W, xin[1])), 1, H)
    bot_edge = outer_line(min(H - 2, ny[-1][1]), (max(0, xin[0]), min(W, xin[1])), 1, H)
    ys[0][0], ys_conf[0][0] = top_edge
    ys[-1][1], ys_conf[-1][1] = bot_edge
    for k, band in enumerate(iny):
        t, b = gutter_lines(band, (max(0, xin[0]), min(W, xin[1])), 1, H)
        ys[k][1], ys_conf[k][1] = t
        ys[k + 1][0], ys_conf[k + 1][0] = b

    boxes, confs = [], []
    for j in range(rows):
        for i in range(cols):
            x0, x1 = xs[i]
            y0, y1 = ys[j]
            boxes.append([x0, y0, x1, y1])
            c = float(np.mean([xs_conf[i][0], xs_conf[i][1],
                               ys_conf[j][0], ys_conf[j][1]]))
            confs.extend([c] * 4)
            if verbose and c < 0.5:
                print(f"    слабая ячейка {i},{j}: conf {c:.2f} "
                      f"({x0:.0f},{y0:.0f})-({x1:.0f},{y1:.0f})")

    # ---------------------------------------------------------------- 4. контроль
    boxes = np.asarray(boxes, float)
    nx0 = np.array([a for a, _ in nx], float)
    nx1 = np.array([b for _, b in nx], float)
    ny0 = np.array([a for a, _ in ny], float)
    ny1 = np.array([b for _, b in ny], float)

    for j in range(rows):
        for i in range(cols):
            k = j * cols + i
            b = boxes[k]
            nom = [nx0[i], ny0[j], nx1[i], ny1[j]]
            # мусорный бокс (например, линия убежала в соседний слот) -> откат
            if not (0.55 * (nom[2] - nom[0]) < b[2] - b[0] < 1.45 * (nom[2] - nom[0])
                    and 0.55 * (nom[3] - nom[1]) < b[3] - b[1]
                    < 1.45 * (nom[3] - nom[1])):
                boxes[k] = nom
                sh.warns.append(f"fallback_{i}_{j}")
                continue
            # не даём рамкам соседних слотов пересечься
            if i and b[0] <= boxes[k - 1][2]:
                b[0] = boxes[k - 1][2] + 1
            if j and b[1] <= boxes[k - cols][3]:
                b[1] = boxes[k - cols][3] + 1

    # --- выравнивание по сетке
    # Сетка строго регулярная: все слоты листа одного размера. Поэтому после
    # уточнения приводим ширину/высоту всех боксов к медиане листа (проверено:
    # качество попадания в линию от этого только растёт), а затем раздвигаем
    # границы, если соседние слоты начали пересекаться.
    if boxes.size:
        b3 = boxes.reshape(rows, cols, 4)
        w_med = float(np.median(b3[:, :, 2] - b3[:, :, 0]))
        h_med = float(np.median(b3[:, :, 3] - b3[:, :, 1]))
        sh.corrected = int(np.sum(np.abs(b3[:, :, 2] - b3[:, :, 0] - w_med) > 0.5)
                           + np.sum(np.abs(b3[:, :, 3] - b3[:, :, 1] - h_med) > 0.5))
        b3[:, :, 2] = b3[:, :, 0] + w_med
        b3[:, :, 3] = b3[:, :, 1] + h_med
        for j in range(rows):
            for i in range(cols):
                if i and b3[j, i, 0] <= b3[j, i - 1, 2]:
                    b3[j, i, 0] = b3[j, i - 1, 2] + 1
                if j and b3[j, i, 1] <= b3[j - 1, i, 3]:
                    b3[j, i, 1] = b3[j - 1, i, 3] + 1
        boxes = b3.reshape(-1, 4)

    sh.cell_w = float(np.median(boxes[:, 2] - boxes[:, 0]))
    sh.cell_h = float(np.median(boxes[:, 3] - boxes[:, 1]))
    sh.confidence = float(np.mean(confs)) if confs else 0.0
    sh.cells_conf = [float(np.mean(confs[k * 4:k * 4 + 4]))
                     for k in range(len(boxes))]
    sh.boxes = [tuple(int(round(v)) for v in b) for b in boxes]
    sh.grid = [(int(round(nx0[i])), int(round(ny0[j])),
                int(round(nx1[i])), int(round(ny1[j])))
               for j in range(rows) for i in range(cols)]

    # лист без рамок: сетка слотов ещё читается по жёлобам, но серых линий нет
    if sh.confidence < 0.15:
        sh.warns.append("no_frames")
        sh.boxes = []            # не рисуем ложные рамки поверх иконок
    elif sh.confidence < 0.5:
        sh.warns.append("low_confidence")
    return sh


# --------------------------------------------------------------------- drawing
def _scaled(sh: Sheet, scale: float) -> Sheet:
    """Копия листа с координатами, умноженными на scale (для превью)."""
    import copy
    c = copy.copy(sh)
    f = lambda t: tuple(int(round(v * scale)) for v in t)
    c.boxes = [f(b) for b in sh.boxes]
    c.grid = [f(b) for b in sh.grid]
    return c


RED = (40, 40, 255)
ORANGE = (0, 150, 255)
GREY = (150, 150, 150)


def draw(sh: Sheet, src: str, dst: str, quality: int = 88, scale: float = 1.0) -> None:
    """Красным — уверенные рамки, оранжевым — те, где линия не подтвердилась."""
    img = cv2.imread(src)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        sh = _scaled(sh, scale)
    th = max(1, int(round(2 * max(scale, 0.75))))
    if not sh.boxes:                      # рамок на листе нет — показываем сетку слотов
        for (x0, y0, x1, y1) in sh.grid:
            cv2.rectangle(img, (x0, y0), (x1, y1), GREY, 1, cv2.LINE_AA)
        cv2.putText(img, "NO FRAMES", (12, 34), cv2.FONT_HERSHEY_SIMPLEX, 1.0,
                    (0, 0, 255), 2, cv2.LINE_AA)
    for k, (x0, y0, x1, y1) in enumerate(sh.boxes):
        conf = sh.cells_conf[k] if k < len(sh.cells_conf) else 1.0
        cv2.rectangle(img, (x0, y0), (x1, y1),
                      RED if conf >= 0.5 else ORANGE, th, cv2.LINE_AA)
    os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
    cv2.imwrite(dst, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])


# ------------------------------------------------------------------------ main
def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Детектор серых рамок на листах иконок")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--out", default="annotated/full")
    ap.add_argument("--csv", default="annotated/frames.csv")
    ap.add_argument("--json", default="annotated/frames.json")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quality", type=int, default=88)
    ap.add_argument("--scale", type=float, default=0.5,
                    help="масштаб превью с рамками (1.0 = полный размер)")
    ap.add_argument("--no-draw", action="store_true")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args(argv)

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    files = args.images or sorted(glob.glob(os.path.join(root, "*", "*.jpg")))
    if args.limit:
        files = files[: args.limit]

    rows, bad = [], 0
    for n, f in enumerate(files, 1):
        rel = os.path.relpath(os.path.abspath(f), root)
        try:
            sh = detect_sheet(f, verbose=args.verbose)
        except Exception as e:                                  # noqa: BLE001
            print(f"[ERR] {rel}: {e}", file=sys.stderr)
            bad += 1
            continue
        if sh.warns:
            bad += 1
        status = ("no_frames" if "no_frames" in sh.warns
                  else "warn" if sh.warns else "ok")
        out_rel = os.path.join(args.out, os.path.splitext(rel)[0] + ".jpg")
        if not args.no_draw:
            dst = out_rel if os.path.isabs(args.out) else os.path.join(root, out_rel)
            draw(sh, f, dst, quality=args.quality, scale=args.scale)
        print(f"[{n}/{len(files)}] {rel}  {sh.cols}x{sh.rows}  "
              f"cell {sh.cell_w:.1f}x{sh.cell_h:.1f}  conf {sh.confidence:.2f}  "
              f"{status}")
        rows.append({
            "file": rel, "out": out_rel, "status": status,
            "cols": sh.cols, "rows": sh.rows,
            "cell_w": round(sh.cell_w, 1), "cell_h": round(sh.cell_h, 1),
            "confidence": round(sh.confidence, 3),
            "fixed": sh.corrected, "warn": ";".join(sh.warns),
            "boxes": " ".join(f"{a},{b},{c},{d}" for a, b, c, d in sh.boxes),
        })

    if rows:
        os.makedirs(os.path.dirname(os.path.join(root, args.csv)), exist_ok=True)
        with open(os.path.join(root, args.csv), "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        with open(os.path.join(root, args.json), "w", encoding="utf-8") as fh:
            json.dump(rows, fh, ensure_ascii=False, indent=1)
    print(f"\nГотово: {len(rows)} листов, с предупреждениями {bad}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
