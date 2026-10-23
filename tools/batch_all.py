#!/usr/bin/env python3
"""
batch_all.py — прогнать всю коллекцию: нарезка -> чистка -> апскейл 256.

Для каждого листа (`<папка>/<файл>.jpg`):
  detect_frames -> все рамки -> вырез 1:1 с обводкой -> clean_fill('normal')
  -> (по желанию) рамка-бортик поверх готового кадра.

По умолчанию НИЧЕГО не увеличивается: иконка отдаётся в родном размере,
пиксель в пиксель. Апскейл включается только явно: --size 256.
Файлы кладутся в `icons_cut/<папка>/<файл>__slotNN.png`.
Папку можно переименовать ключом `--out`.

В `samples/` пишется то, что нужно глазам:
  batch_summary.csv        — по листам: сколько иконок, шум до/после, время
  batch_contact_sheet.jpg  — сетка: по одной иконке с каждого листа
  batch_montage.jpg        — 12 иконок из разных папок крупно

    python3 tools/batch_all.py                     # вся коллекция, все слоты
    python3 tools/batch_all.py --slots 3           # по 3 иконки с листа
    python3 tools/batch_all.py --limit 5           # только первые 5 листов
    python3 tools/batch_all.py --smooth            # EDSR вместо FSRCNN (медленнее)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from _console import setup as _console_setup  # noqa: E402
import clean_icon as C            # noqa: E402
import detect_frames as D         # noqa: E402
import fetch_models               # noqa: E402
import style_frame as S           # noqa: E402
from progress import Progress      # noqa: E402

MODELS = os.path.join(HERE, "models")
SR = {}


def _sr(algo: str, scale: int = 4):
    key = (algo, scale)
    if key not in SR:
        name = {"edsr": "EDSR", "fsrcnn": "FSRCNN"}[algo]
        m = cv2.dnn_superres.DnnSuperResImpl_create()
        m.readModel(os.path.join(MODELS, f"{name}_x{scale}.pb"))
        m.setModel(algo, scale)
        SR[key] = m
    return SR[key]


def upscale(img: np.ndarray, size: int = 256, unsharp: float = 0.55,
            algo: str = "fsrcnn") -> np.ndarray:
    """Апскейл иконки до size x size.

    algo="anime" - Real-ESRGAN (достраивает детали; ~3 с на иконку);
    algo="edsr"/"fsrcnn" - старый путь через cv2.dnn_superres (мылит).
    """
    if algo == "anime":
        import sr_ncnn
        return sr_ncnn.upscale(img, size=size, device=SR_DEVICE,
                               threads=SR_THREADS)
    sr = _sr(algo, 4)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    big = sr.upsample(cv2.cvtColor(lab[..., 0], cv2.COLOR_GRAY2BGR))
    lo = cv2.cvtColor(cv2.resize(big, (size, size),
                                 interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo = lo + (lo - cv2.GaussianBlur(lo, (0, 0), 0.8)) * unsharp
    ab = cv2.resize(lab[..., 1:], (size, size),
                    interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(np.dstack([np.clip(lo, 0, 255).astype(np.uint8), ab]),
                        cv2.COLOR_LAB2BGR)


# --- рамка на готовом кадре -------------------------------------------------
FRAME_RIM = 10      # толщина борта в пикселях ИТОГОВОГО кадра: одна на всех
FRAME_GAP = 4       # зазор между бортом и рисунком: иконка целиком внутри рамки
INSET_CROP = 0      # оставлено для совместимости: границы берём по белой линии


LINE_MAX = 4        # насколько белая линия сетки может выступать за bbox
LINE_MIN_BRIGHT = 10   # внешний край линии тусклее её ядра (антиалиас): 20-30
LINE_MAX_CHROMA = 25   # линия серая; цветной рисунок у края не трогаем


def _line_like(px: np.ndarray) -> bool:
    """Похоже на белую линию сетки (px - один пиксель BGR).

    Ядро линии яркое (~60), а её внешний сглаженный край - 20-30, поэтому
    порог низкий: иначе срез проходил по ядру и половина линии отрезалась.
    """
    f = px.astype(np.int16)
    return f.mean() >= LINE_MIN_BRIGHT and (f.max() - f.min()) <= LINE_MAX_CHROMA


def crop_by_sheet_line(img: np.ndarray, box) -> np.ndarray:
    """Вырез ровно по внешнему краю белой линии сетки.

    Детектор даёт bbox по ЦЕНТРУ линии, поэтому при обычном вырезе половина
    линии оставалась в кадре, а половина терялась. Здесь мы отступаем наружу
    на всю толщину линии (1-3 px, зависит от листа) — линия входит в иконку
    целиком и служит её родной рамкой, а размер получается максимальным.
    """
    x0, y0, x1, y1 = box
    h, w = img.shape[:2]
    cy, cx = (y0 + y1) // 2, (x0 + x1) // 2

    def run(row, start, step):
        n = 0
        i = start + step
        while n < LINE_MAX and 0 <= i < len(row) and _line_like(row[i]):
            n += 1
            i += step
        return n

    el = run(img[cy], x0, -1)
    er = run(img[cy], x1, +1)
    et = run(img[:, cx], y0, -1)
    eb = run(img[:, cx], y1, +1)
    return img[max(0, y0 - et):min(h, y1 + 1 + eb),
               max(0, x0 - el):min(w, x1 + 1 + er)]


INNER_DARK = 30     # яркость, ниже которой пиксель считается чёрным фоном
LINE_MIN = 15       # ярче этого - уже не чёрный фон, а линия или рисунок
LINE_WIN_IN = 4     # на сколько пикселей внутрь от края bbox ищем линию
LINE_WIN_OUT = 2    # и на сколько наружу (bbox иногда стоит чуть внутри линии)
LINE_DEEP = 12      # запасной проход, если линия оказалась глубже окна
LINE_DEEP_COVER = 0.6   # линия в запасном проходе занимает большую часть стороны
LINE_DEEP_STD = 14      # и ровная вдоль стороны (рисунок гуляет на 20-60)
LINE_COVER = 0.5    # какую долю полосы должны занимать нейтрально-яркие пиксели
LINE_EXTRA = 1      # на сколько пикселей режем глубже самой яркой полосы (кромка)
LINE_COVER_N = 0.25 # линия может быть частично перекрыта цветным рисунком
LINE_STD_N = 12     # но сами серые пиксели линии при этом очень ровные
LINE_BRIGHT = 40    # ярче этого пиксель считается светлым (кандидат в линию)


def inner_rect(img: np.ndarray, box) -> tuple[int, int, int, int]:
    """Границы выреза по внутренней стороне рамки (в координатах листа)."""
    return _inner_scan(img, box)


def crop_by_inner_edge(img: np.ndarray, box) -> np.ndarray:
    """Режем по внутренней стороне белой рамки - скан по КАЖДОЙ строке/столбцу.

    Идём от края слота внутрь, пока пиксель серый/белый (это линия), и
    останавливаемся на первом реально чёрном пикселе фона. Считаем не по
    одной центральной линии, а по всем строкам: рамку можно встретить где
    угодно, а рисунок закрывает её лишь местами.

    Смещения собираем по всем строкам и берём 85-й процентиль: там, где
    рисунок доходит до края, скан упирается в него и не растягивает вырез,
    а медиана не даёт паре «дырок» испортить результат. Если пиксель цветной
    (рисунок) - дальше не идём, картинку не режем.
    """
    r = _inner_scan(img, box)
    return img[r[1]:r[3], r[0]:r[2]]


def _inner_scan(img: np.ndarray, box) -> tuple[int, int, int, int]:
    """Границы выреза по линии рамки.

    Линия рамки ровная: её яркость почти не меняется вдоль всей стороны
    (разброс 0.4-2 единицы), и она занимает сторону целиком. У рисунка
    (серебро, мех, светлое дерево) яркость вдоль полосы гуляет на 18-57,
    даже если цвет такой же нейтральный.

    Поэтому для каждой стороны смотрим 4 px внутрь и 2 px наружу, берём
    полосы с низким разбросом яркости и малой хромой, и режем по самой
    ВНУТРЕННЕЙ такой полосе. Прежний скан «иди внутрь, пока не станет
    чёрным» уходил в светлый рисунок на 10-11 px и срезал лишнее.
    """
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    xi0, xi1 = max(0, x0 + 4), min(W, x1 - 3)
    yi0, yi1 = max(0, y0 + 4), min(H, y1 - 3)

    def strip(y=None, x=None):
        if y is not None:
            f = img[y, xi0:xi1].astype(np.int16)
        else:
            f = img[yi0:yi1, x].astype(np.int16)
        if f.size == 0:
            return None
        bright = f.mean(axis=1)
        chroma = f.max(axis=1) - f.min(axis=1)
        neutral = chroma <= LINE_MAX_CHROMA
        cov = float(((bright > LINE_BRIGHT) & neutral).mean())
        med = float(np.median(bright))
        if neutral.any():
            nb = bright[neutral]
            nmed, nstd = float(np.median(nb)), float(nb.std())
            ncov = float(neutral.mean())
        else:
            nmed, nstd, ncov = 0.0, 99.0, 0.0
        return med, float(np.median(chroma)), cov, nmed, nstd, ncov

    def line(axis, base, limit, inward):
        """Самая яркая нейтральная полоса возле края - это линия рамки.

        Линия проходит по всей стороне слота, поэтому она самая яркая
        полоса в окне у края. Дальше неё не идём: светлый рисунок под
        линией (серебро, мех, дерево) тоже нейтральный, и старый скан
        принимал его за продолжение линии, срезая на 10-11 px лишнего.
        """
        best = None
        for c in range(base - LINE_WIN_IN, base + LINE_WIN_OUT + 1):
            if not (0 <= c < limit):
                continue
            v = strip(y=c) if axis == 'y' else strip(x=c)
            if v is None:
                continue
            bright, chroma, cov, nmed, nstd, ncov = v
            clean = (bright > LINE_BRIGHT and chroma <= LINE_MAX_CHROMA
                     and cov >= LINE_COVER)
            # линия под цветным рисунком: своих серых пикселей немного,
            # но они яркие и очень ровные вдоль всей стороны
            under = (ncov >= LINE_COVER_N and nmed > LINE_BRIGHT
                     and nstd <= LINE_STD_N)
            if clean or under:
                if best is None or bright > best[1]:
                    best = (c, bright)
        if best is None:
            return None
        return best[0] + inward * LINE_EXTRA

    def find(axis, base, limit, inward):
        """Линия у края, с запасным проходом глубже по чёрному фону.

        Детектор рамки иногда ставит box на 5-6 px НАРУЖУ линии (кейс
        `RPG Loot Icons 14 Part 2`, слот 3: линия на x=698, а box - x1=703).
        Тогда в окно поиска линия не попадает, и прежний код откатывался на
        край box - линия оставалась ВНУТРИ выреза.

        Запасной проход идёт от края box внутрь по чёрному фону и
        останавливается на первой НЕ чёрной полосе: если это линия (ровная,
        светлая, нейтральная) - режем сразу за ней; если рисунок - не
        трогаем, чтобы не срезать иконку.
        """
        found = line(axis, base, limit, inward)
        if found is not None:
            return found
        seen = False                 # уже встретили линию
        for step in range(1, LINE_DEEP + 1):
            c = base + inward * step
            if not (0 <= c < limit):
                return None
            v = strip(y=c) if axis == 'y' else strip(x=c)
            if v is None:
                return None
            bright, chroma, cov, nmed, nstd, ncov = v
            if bright < LINE_BRIGHT:     # чёрный фон
                if seen:                 # линия кончилась - режем здесь
                    return c
                continue                 # ещё идём по фону
            is_line = (chroma <= LINE_MAX_CHROMA and cov >= LINE_DEEP_COVER
                       and nstd <= LINE_DEEP_STD)
            if is_line:                  # линия может быть в 2-3 px толщиной
                seen = True
                continue
            return None                  # светлое, но не линия - не режем
        return None

    left = find('x', x0, W, +1)
    right = find('x', x1, W, -1)
    top = find('y', y0, H, +1)
    bottom = find('y', y1, H, -1)
    lx = left + 1 if left is not None else max(0, x0)
    ty = top + 1 if top is not None else max(0, y0)
    rx = right if right is not None else min(W, x1 + 1)
    by = bottom if bottom is not None else min(H, y1 + 1)
    return (lx, ty, max(lx + 1, rx), max(ty + 1, by))


CANVAS = 151        # итоговый кадр: мелкие иконки добираем фоном до этого размера


def pad_to_canvas(img: np.ndarray, rect, target: int = CANVAS) -> np.ndarray:
    """Довести вырез до target x target, дорисовав тёмный фон по сторонам.

    Иконка НЕ масштабируется: остаётся пиксель в пиксель, а вокруг неё
    появляется ровный тёмный фон - цвет берём из настоящего фона листа
    рядом со слотом (тёмный жёлоб, без белой линии). Так мелкая иконка
    встаёт в тот же кадр, что и крупные, но без мыла от растягивания.
    """
    x0, y0, x1, y1 = rect
    crop = img[y0:y1, x0:x1]
    h, w = crop.shape[:2]
    if max(h, w) >= target:      # крупнее кадра - режем по центру, не растягиваем
        oy, ox = (h - target) // 2, (w - target) // 2
        return crop[oy:oy + target, ox:ox + target].copy()

    # фон: тёмные нейтральные пиксели вокруг слота
    H, W = img.shape[:2]
    m = 14
    wnd = img[max(0, y0 - m):min(H, y1 + m), max(0, x0 - m):min(W, x1 + m)]
    f = wnd.astype(np.int16)
    bright = f.mean(axis=2)
    chroma = f.max(axis=2) - f.min(axis=2)
    bg_mask = (bright <= 35) & (chroma <= 25)
    if bg_mask.sum() >= 30:
        bg = np.median(wnd[bg_mask], axis=0)
    else:                                     # совсем нет фона - тёмный нейтральный
        bg = np.array([8.0, 8.0, 8.0])

    canvas = np.empty((target, target, 3), np.uint8)
    canvas[:] = np.clip(bg, 0, 255).astype(np.uint8)
    ly, lx = (target - h) // 2, (target - w) // 2
    canvas[ly:ly + h, lx:lx + w] = crop
    return canvas


CORNER_P = 3        # отступ от линии сетки внутрь слота
CORNER_I = 8        # размер углового плашка, по которому берём эталон фона


def _uniform(px: np.ndarray) -> tuple[np.ndarray, float]:
    """Средний цвет и разброс плашка (по самому «шумному» каналу)."""
    flat = px.reshape(-1, 3).astype(np.float32)
    return flat.mean(axis=0), float(flat.std(axis=0).max())


def corner_box(img: np.ndarray, box, margin: int = 16):
    """Вырез по однотонным чёрным углам (по часовой от левого верхнего).

    Углы внутри слота почти всегда - ровный тёмный фон. Берём угол за эталон,
    растущением находим всю однотонную область фона, по её границам строим
    вырез и отступаем наружу по белой линии. Возвращает (x0, y0, x1, y1)
    в координатах исходного листа или None, если уверенного угла нет.
    """
    x0, y0, x1, y1 = box
    H, W = img.shape[:2]
    wx0, wy0 = max(0, x0 - margin), max(0, y0 - margin)
    wx1, wy1 = min(W, x1 + margin + 1), min(H, y1 + margin + 1)
    win = img[wy0:wy1, wx0:wx1]
    bx0, by0 = x0 - wx0, y0 - wy0
    bx1, by1 = x1 - wx0, y1 - wy0
    bh, bw = win.shape[:2]
    best = None
    for name in ("tl", "tr", "br", "bl"):          # по часовой стрелке
        cx = bx0 + CORNER_P if name in ("tl", "bl") else bx1 - CORNER_P - CORNER_I + 1
        cy = by0 + CORNER_P if name in ("tl", "tr") else by1 - CORNER_P - CORNER_I + 1
        patch = win[max(0, cy):cy + CORNER_I, max(0, cx):cx + CORNER_I]
        if patch.shape[0] < 3 or patch.shape[1] < 3:
            continue
        mean, std = _uniform(patch)
        if std > 12:                                # угол не однотонный
            continue
        tol = max(14.0, 3.0 * std + 16.0)
        diff = np.abs(win.astype(np.int16) - mean.astype(np.int16)).max(axis=2)
        mask = (diff <= tol).astype(np.uint8)
        if not mask[cy, cx]:
            continue
        flood = mask.copy()
        mm = np.zeros((bh + 2, bw + 2), np.uint8)
        cv2.floodFill(flood, mm, (cx, cy), 2)
        ys, xs = np.where(flood == 2)
        if len(xs) < 400:
            continue
        w_, h_ = xs.max() - xs.min() + 1, ys.max() - ys.min() + 1
        if not (130 <= w_ <= 175 and 130 <= h_ <= 175):
            continue
        if best is None or len(xs) > best[0]:
            best = (len(xs), int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))
    if best is None:
        return None
    _, cx0, cy0, cx1, cy1 = best

    def out(run, start, step):
        n = 0
        i = start + step
        while n < LINE_MAX and 0 <= i < len(run) and _line_like(run[i]):
            n += 1
            i += step
        return n
    cy_mid, cx_mid = (cy0 + cy1) // 2, (cx0 + cx1) // 2
    el = out(win[cy_mid], cx0, -1); er = out(win[cy_mid], cx1, +1)
    et = out(win[:, cx_mid], cy0, -1); eb = out(win[:, cx_mid], cy1, +1)
    return (max(0, cx0 - el) + wx0, max(0, cy0 - et) + wy0,
            min(bw - 1, cx1 + er) + wx0, min(bh - 1, cy1 + eb) + wy0)


# auto | gpu | cpu. Ставится в main, а воркеры (на Windows они стартуют
# заново, через spawn) читают значение из переменной окружения.
SR_DEVICE = os.environ.get("SR_DEVICE", "auto")
# сколько потоков дать ncnn на CPU; на GPU всегда 1
SR_THREADS = int(os.environ.get("SR_THREADS", "1"))


def process_sheet(path: str, outroot: str, nslots: int | None,
                  algo: str, size: int, frame: str | None = None,
                  cut: str = "line", canvas: int = CANVAS) -> dict:
    t0 = time.time()
    img = cv2.imread(path)
    if img is None:
        return {"sheet": path, "error": "не читается"}
    H, W = img.shape[:2]
    boxes = D.detect_sheet(path).boxes
    if not boxes:
        return {"sheet": path, "slots": 0, "error": "рамки не найдены"}
    mine = boxes if nslots is None else boxes[:nslots]

    # Порядок важен: сначала иконка (вырез -> чистка -> апскейл), и только
    # потом рамка - на готовом кадре, в пикселях этого кадра.
    # Иначе чистка и супер-разрешение размывали борт, и его ширина гуляла
    # от иконки к иконке.  size <= 0 -> без апскейла вообще (родной размер).
    inner = size if frame is None else inner_size(size)   # сторона самой иконки
    fp = frame_params(frame, size) if frame else None

    folder = os.path.relpath(os.path.dirname(path), ROOT)
    stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    outdir = os.path.join(outroot, folder)
    os.makedirs(outdir, exist_ok=True)

    made, noise_before, noise_after = 0, 0.0, 0.0
    for k, box in enumerate(mine):
        rect = None
        if cut == "inner":
            rect = _inner_scan(img, box)              # по внутренней стороне рамки
            crop = img[rect[1]:rect[3], rect[0]:rect[2]]
        elif cut == "corner":
            cb = corner_box(img, box)
            if cb is not None:
                cx0, cy0, cx1, cy1 = cb
                crop = img[cy0:cy1 + 1, cx0:cx1 + 1]
            else:
                crop = crop_by_sheet_line(img, box)   # не вышло - обычный путь
        else:
            crop = crop_by_sheet_line(img, box)       # режем по внешнему краю линии
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        # добивка кадра родным тёмным фоном (без масштабирования иконки)
        if canvas > 0 and size <= 0 and rect is not None:
            crop = pad_to_canvas(img, rect, canvas)
        clean = C.clean_fill(crop, "normal")
        if size > 0:                       # апскейл только если он запрошен
            big = upscale(clean, size=inner, algo=algo)
        else:
            # пиксель в пиксель, без размытия и без обрезки.
            # Квадрат: не хватает пары пикселей по стороне -> добираем повтором
            # краевой строки/столбца (рисунок от этого не страдает).
            h, w = clean.shape[:2]
            d = max(h, w)
            pad_t, pad_b = 0, d - h
            pad_l, pad_r = 0, d - w
            big = cv2.copyMakeBorder(clean, pad_t, pad_b, pad_l, pad_r,
                                     cv2.BORDER_REPLICATE)
        if fp is not None:
            big = compose_frame(big, size, fp)
        out = os.path.join(outdir, f"{stem}__slot{k:02d}.png")
        cv2.imwrite(out, big)
        noise_before += C.fill_noise(crop)
        noise_after += C.fill_noise(clean)
        made += 1

    dt = time.time() - t0
    return {"sheet": os.path.relpath(path, ROOT), "folder": folder,
            "slots": made,
            "noise_before": round(noise_before / max(made, 1), 2),
            "noise_after": round(noise_after / max(made, 1), 2),
            "sec": round(dt, 1)}


def main() -> int:
    _console_setup()
    ap = argparse.ArgumentParser(description="Пакетный прогон коллекции")
    ap.add_argument("--out", default=os.path.join(ROOT, "icons_cut"),
                    help="куда класть результат (по умолчанию icons_cut/ в репозитории)")
    ap.add_argument("--slots", type=int, default=None,
                    help="сколько иконок с листа (по умолчанию все)")
    ap.add_argument("--limit", type=int, default=None, help="только первые N листов")
    ap.add_argument("--size", type=int, default=0,
                    help="сторона кадра; 0 (по умолчанию) - без апскейла, "
                         "родной размер иконки")
    ap.add_argument("--smooth", action="store_true",
                    help="EDSR (качественнее, ~9 c на иконку) вместо FSRCNN")
    ap.add_argument("--sr", default="anime", choices=["anime", "edsr", "fsrcnn"],
                    help="движок апскейла при --size: anime - Real-ESRGAN "
                         "(по умолчанию, достройка деталей, ~3 c на иконку); "
                         "edsr/fsrcnn - cv2.dnn_superres (мылит)")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"],
                    help="где гонять сеть: auto (GPU если есть Vulkan), gpu, "
                         "cpu. На GPU бери --jobs 1: несколько процессов "
                         "дерутся за видеопамять")
    ap.add_argument("--frame", default="none",
                    choices=["none"] + sorted(S.VARIANTS),
                    help="рамка-бортик вокруг иконки; по умолчанию none - без рамки. "
                         "Рамка требует --size (рисуется в пикселях кадра)")
    ap.add_argument("--canvas", type=int, default=CANVAS,
                    help="довести кадр до N x N, добрав тёмный фон по сторонам "
                         "(0 - выключить); работает без апскейла")
    ap.add_argument("--cut", default="inner", choices=["inner", "line", "corner"],
                    help="как искать границы: inner - по внутренней стороне белой "
                         "рамки, опираясь на тёмные пиксели (по умолчанию); "
                         "line - по внешнему краю линии; corner - по чёрным углам")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    frame = None if args.frame == "none" else args.frame
    if frame and args.size <= 0:          # рамке нужен кадр известного размера
        args.size = 256

    cv2.setNumThreads(1)
    sheets = sorted(p for p in glob.glob(os.path.join(ROOT, "*", "*.jpg"))
                    if os.path.basename(os.path.dirname(p)) not in ("samples",))
    if args.limit:
        sheets = sheets[:args.limit]
    if args.size > 0 and args.sr:
        algo = args.sr
        if algo == "anime":
            import sr_ncnn
            if not sr_ncnn.available():
                print("Real-ESRGAN недоступен: pip install ncnn "
                      "realesrgan-ncnn-py — перехожу на EDSR")
                algo = "edsr"
                args.smooth = True
            else:
                try:
                    dev = sr_ncnn.pick_device(args.device)
                except RuntimeError as e:
                    print(f"{e}\nперехожу на CPU")
                    dev = "cpu"
                globals()["SR_DEVICE"] = dev
                os.environ["SR_DEVICE"] = dev          # для воркеров (spawn)
                if dev == "gpu":
                    card = sr_ncnn.gpu_name()
                    print(f"апскейл: Real-ESRGAN на GPU (Vulkan)"
                          + (f" — {card}" if card else ""))
                else:
                    # на CPU делим ядра между процессами: иначе один поток
                    # на иконку - это треть скорости машины
                    cores = os.cpu_count() or 1
                    per = max(1, cores // max(1, args.jobs))
                    os.environ["SR_THREADS"] = str(per)
                    globals()["SR_THREADS"] = per
                    print(f"апскейл: Real-ESRGAN на CPU, {cores} ядер "
                          f"-> {per} поток(а) на процесс "
                          f"(видеокарта не найдена — tools/gpu_check.py)")
                if dev == "gpu" and args.jobs > 1:
                    print("на GPU ставлю --jobs 1 (иначе процессы дерутся "
                          "за видеопамять)")
                    args.jobs = 1
    else:
        algo = "edsr" if args.smooth else "fsrcnn"
    fetch_models.ensure(all_models=args.smooth)      # модели качаются один раз
    scale_txt = f"{args.size}x{args.size}" if args.size > 0 else "1:1 без апскейла"
    if args.size > 0 and args.canvas > 0:
        print(f"добивка фоном до {args.canvas}px не нужна: при апскейле в сеть "
              f"идёт сам вырез, без чёрных полей")
    print(f"листов: {len(sheets)} | слотов с листа: {args.slots or 'все'} | "
          f"размер: {scale_txt} | рамка: {args.frame} | -> {args.out}")

    rows = []
    t0 = time.time()
    total_icons = 0
    prog = Progress(len(sheets))
    prog.tick(0, 0, note="старт")
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for res in pool.imap_unordered(
                    _worker, [(p, args.out, args.slots, algo, args.size, frame,
                               args.cut, args.canvas) for p in sheets]):
                rows.append(res)
                total_icons += res.get("slots", 0)
                prog.tick(len(rows), total_icons,
                          note=f"{res['sheet']} · {res.get('slots', 0)} шт "
                               f"за {res.get('sec', 0)} c")
    else:
        for p in sheets:
            res = process_sheet(p, args.out, args.slots, algo, args.size, frame,
                                args.cut, args.canvas)
            rows.append(res)
            total_icons += res.get("slots", 0)
            prog.tick(len(rows), total_icons,
                      note=f"{res['sheet']} · {res.get('slots', 0)} шт "
                           f"за {res.get('sec', 0)} c")
    prog.close(total_icons)

    rows.sort(key=lambda r: r["sheet"])
    os.makedirs(os.path.join(ROOT, "samples"), exist_ok=True)
    csv_path = os.path.join(ROOT, "samples", "batch_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "folder", "slots",
                                          "noise_before", "noise_after", "sec"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    total = sum(r.get("slots", 0) for r in rows)
    print(f"готово: {total} иконок с {len(rows)} листов за "
          f"{(time.time() - t0) / 60:.1f} мин -> {args.out}")
    print(f"отчёт: {csv_path}")
    return 0


def _worker(a):
    p, out, n, algo, size, frame, cut, canvas = a
    try:
        return process_sheet(p, out, n, algo, size, frame, cut, canvas)
    except Exception as e:                      # noqa: BLE001
        return {"sheet": os.path.relpath(p, ROOT), "slots": 0,
                "error": str(e)[:120]}


if __name__ == "__main__":
    raise SystemExit(main())
