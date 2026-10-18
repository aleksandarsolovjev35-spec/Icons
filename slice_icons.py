#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
slice_icons.py — вырезание иконок из листов по внутренней грани белой (светлой) обводки,
очистка и апскейл до целевого размера.

Алгоритм (по шагам, всё авто-детектируется на каждом листе отдельно):

 1. Линейный детектор рамок.
    Обводка плитки — это тонкая (1-4 px) светлая линия, снаружи (в «межплиточном»
    канале) окружённая почти чёрным фоном. Это и есть надёжный признак, а не просто
    «ярко». Для каждой из 4 сторон считается асимметричный отклик:
        score_top[y,x] = g[y,x] - max(g[y-2..y-4, x])      (светлая, сверху тёмно)
        score_bot[...] = g[y,x] - max(g[y+2..y+4, x])
        score_left ...  score_right ...
    Линия = длинный горизонтальный/вертикальный «гребень»: ответ сглаживается box-фильтром
    вдоль линии, поэтому край арта (рваный, короткий) отбрасывается, а рамка остаётся.

 2. Полосы сетки (rows x cols).
    Профиль по стороне = доля длины строки/столбца, покрытая линией; из него берутся
    пики (широкие пики режутся по внутреннему минимуму, позиция — квадратичная
    парабола, суб-пиксель). Кандидатов сетки набирается несколько и они СВЕТСЯ:
    линии с покрытием grid_cover, «решётка» (пропуски пиков, если часть рамок убита),
    и полосы содержимого (арт). Выбирается набор с наименьшим числом геометрических
    нарушений (равный шаг, плитки внутри полосы, полные строки/столбцы), поэтому
    «плитки стоят не идеально» (сдвиг 1-3 px, свой размер у листа) разметку не ломает.
    Содержимое полос подтягивается к ближайшему пику линий (не наоборот).

 3. Уточнение рёбра каждой плитки индивидуально (два прохода).
    Для каждой ячейки каждое из 4 рёбер пере-ищется в своём окне (max(search_win,
    0.07*min(плитка)) и не дальше ~45% полосы) по СВОЕЙ карте отклика в реальном
    x/y-диапазоне плитки: левая грань конкретной плитки гуляет независимо от соседей.
    Кандидат должен быть не наружу полосы (на сколько позволяет канал: tol_out =
    1..3 px), снаружи него должно быть темнее, чем внутри, и он должен набирать
    cfg.edge_cov по длине линии. Во втором проходе в тот же критерий добавляется
    prior типового размера плитки листа (median |размер − target|): на листах с
    тонким каналом рамка соседней плитки бывает БЛИЖЕ к границе полосы, чем своя,
    и одна близость её не отвергает.
    Пометки в manifest (`why` по каждой стороне): "weak" — линия не нашлась,
    "relaxed" — найдена слабая (рамку «съел» арт) но внутри своей полосы,
    "past-band" — в окне нет ничего внутри полосы, "no-dark-outside" — у ручной
    сетки, "bright-outside" — снаружи светлое, "stroke-cap" — см. п.4.
    Любая сторона с пометкой делает плитку low_confidence (её стоит глянуть в
    _debug/overlay.png и _debug/edge_zoom.png).

 4. Толщина обводки и «внутренняя грань».
    От ЯКОРЯ (строки, где линия реально нашлась) пробег в обе стороны по рядам,
    которые «как линия»: яркость в коридоре (ниже уровня линии, но не светлее p90
    калибровки листа) и почти бесцветно (sat < stroke_sat). Штрих = максимальный
    непрерывный пробег, проходящий через якорь — тогда грань реза не зависит от
    того, на какую из строк 2px линии сел пик, и не зависит от светлого ореола в
    канале (он двигает только outer). Замер достоверен (t_valid), если пробег сам
    остановился и не упёрся в max_stroke; недостоверные замеры заменяются типовой
    толщиной стороны по листу (percentile stroke_pct).
    Отдельный шаг (tighten): если кадр вышел заметно меньше типового размера листа,
    а с какой-то стороны пробег длиннее типовой толщины — за рамкой лежит не обводка,
    а тёмный арт, обнимающий рамку изнутри; «тин» на этой стороне урезается, наружу
    рез не двигается никогда.
    По умолчанию inset = 0 px: режем строго по внутренней грани. inset > 0 — страховка
    на листы с грязным каналом.

 5. Кадр + геометрия.
    Кадр = внутренний прямоугольник рамки (x0,y0,x1,y1). Рука написанная рамка гуляет
    на 1-2 px, поэтому размер плитки может быть 149/150/151 — это честно, а не «квадрат
    со средним размером» (иначе по краям пойдут дубликаты строк). --fit stretch (по
    умолчанию) тянет кадр в целевой квадрат; --fit square центрирует по большей стороне
    и режет по ней; --fit pad дополняет до квадрата.

 6. Чистка («до практического идеала»):
    - rim-fix: пиксели в 1-2 px от края, которые явно продолжают отрезанный штрих
      (серые, уровень яркости линии листа, а за ними — тёмный фон плитки) подменяются
      медианой соседей внутрь. Одиночный светлый пиксель арта не трогается: решение
      принимается по доле СТОРОНЫ, покрытой такими пикселями (--rim-frac);
    - despeckle (по желанию, --specks): гасит «пылинки» JPEG, только если это
      одиночный пиксель, отличающийся от 3x3 соседей;
    - шум давят двумя проходами NL-means: h~3 в исходнике (иначе интерполяция
      раздует блоки и ringing) и h~6 после ресайза;
    - опционально --matte: мягкая альфа, убирающая чёрный фон плитки (по умолчанию
      фон остаётся частью иконки — тайл и есть иконка).

 7. Апскейл до 256 (--pipeline quality|fast|soft|pixel|raw):
    quality (дефолт): denoise(3) -> Lanczos -> denoise(6) -> unsharp 0.6 с
    масштабированием по локальной контрастности (плоским местам 0, краю всё) ->
    denoise(3). На этих листах даёт x3.3 меньше шума в плоских тёмных областях,
    +18% крутизны края против «просто Lanczos», и сохраняет мелкие искры арта
    (24.7 против 21.8 в исходнике — то есть мелкие блики не съедаются).
    pixel — nearest для пиксель-арта; raw — только Lanczos, без обработки.
    Альфа ресамплижится в premultiplied, поэтому канта по контуру нет.

 8. Контроль качества.
    - manifest.json: bbox рамки, bbox реза, толщина штриха, уверенность, метрики края;
    - debug/overlay.png: сетка резов поверх листа (проверить «не срезал / не лишнего»);
    - debug/edge_zoom.png: зум 4x краёв выбранных плиток (остатки обводки видно сразу);
    - contact_sheet.png: все иконки montage'ом;
    - плитки с подозрением (нет кадра, пустой кадр, край «в обводке») — в warning'и.

Использование:
    python3 slice_icons.py "RPG Loot Icons 01" -o out --size 256
    python3 slice_icons.py sheet.png -o out --inset 1 --sharpen 0.5 --debug
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import sys
from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple

import numpy as np

try:
    import cv2
except ImportError:  # pragma: no cover
    sys.exit("нужен opencv:  pip install opencv-python-headless numpy pillow")

try:
    from PIL import Image
except ImportError:  # pragma: no cover
    sys.exit("нужен pillow:  pip install pillow")


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------

def to_gray(rgb: np.ndarray) -> np.ndarray:
    """float32 luma, 0..255."""
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)


def box_mean_along(arr: np.ndarray, length: int, horizontal: bool) -> np.ndarray:
    """Сглаживание строго вдоль линии (length px) — превращает «гребень» в плато."""
    k = max(5, int(length) | 1)
    if horizontal:
        return cv2.boxFilter(arr.astype(np.float32), -1, (k, 1), normalize=True,
                             borderType=cv2.BORDER_REPLICATE)
    return cv2.boxFilter(arr.astype(np.float32), -1, (1, k), normalize=True,
                         borderType=cv2.BORDER_REPLICATE)


def directional_line_maps(g: np.ndarray, gap: Tuple[int, int] = (2, 4)) -> dict:
    """
    4 карты «тонкая светлая линия, с внешней стороны тёмно».
    top    -> линия, над которой (снаружи, вверх) темно: ВЕРХНЯЯ грань рамки
    bottom -> линия, под которой темно: НИЖНЯЯ грань
    left   -> линия, левее которой темно: ЛЕВАЯ грань
    right  -> линия, правее которой темно: ПРАВАЯ грань
    """
    lo, hi = gap
    out = {}
    # np.roll(+d) => out[y] = g[y-d]
    for name, axis, sign in (("top", 0, +1), ("bottom", 0, -1),
                             ("left", 1, +1), ("right", 1, -1)):
        dark = np.full_like(g, -1e9)
        for d in range(lo, hi + 1):
            dark = np.maximum(dark, np.roll(g, sign * d, axis=axis))
        out[name] = (g - dark).astype(np.float32)
    return out


def peak_positions(profile: np.ndarray, thresh: float, min_len: int,
                   merge_gap: int = 3) -> List[dict]:
    """
    profile: 1D массив отклика (по строкам или столбцам).
    Возвращает пики: {'pos': int, 'sub': float, 'w': int, 'score': float}.
    sub — суб-пиксельный центроид внутри пика (по массиву отклика).
    """
    idx = np.where(profile >= thresh)[0]
    peaks: List[dict] = []
    if idx.size == 0:
        return peaks
    group: List[int] = [int(idx[0])]
    for i in idx[1:]:
        i = int(i)
        if i - group[-1] <= merge_gap:
            group.append(i)
        else:
            peaks.append(group)
            group = [i]
    peaks.append(group)

    res = []
    # широкая серия = часто ДВЕ линии (низ одной плитки и верх следующей при узком
    # канале). Режем такую серию по внутренней впадине, иначе полоса «съедает» соседа.
    wide = []
    for gp in peaks:
        a, b = gp[0], gp[-1] + 1
        seg = profile[a:b]
        if b - a > 5 and float(seg.max()) > 0:
            inside = seg[2:-2]
            cut = int(np.argmin(inside)) + 2
            if float(inside.min()) < 0.55 * float(seg.max()):
                wide.append(list(range(a, a + cut)))
                wide.append(list(range(a + cut, b)))
                continue
        wide.append(gp)
    peaks = wide
    for gp in peaks:
        a, b = gp[0], gp[-1] + 1
        seg = profile[a:b]
        w = int(b - a)
        score = float(seg.max())
        if w > 1 and score > 0:
            # центроид по «треугольнику» пика -> subpixel
            sub = float(np.average(np.arange(a, b), weights=np.clip(seg, 0, None)))
        else:
            sub = float(a)
        if w >= min_len and False:  # min_len относится к длине линии, см. callers
            pass
        res.append({"pos": int(a + int(np.argmax(seg))), "sub": sub, "w": w, "score": score})
    return res


def group_pairs(positions: Sequence[float], min_dist: float, max_dist: float,
                min_pitch: float) -> List[Tuple[float, float]]:
    """
    Из отсортированного списка рёбер собрать пары (начало, конец) плиточной полосы
    в координатах листа (суб-пиксель).
    Жадно: следующее ребро, отстоящее на [min_dist, max_dist] (размер плитки), — это
    противоположная грань; всё, что ближе min_dist, — утолщение/дубль той же линии
    (JPEG ringing), а зазор между полосами (канал) всегда меньше размера плитки.
    """
    pos = [float(v) for v in positions]
    pairs: List[Tuple[float, float]] = []
    i, n = 0, len(pos)
    while i < n:
        j = i + 1
        while j < n and (pos[j] - pos[i]) < min_dist:
            j += 1  # слишком близко -> это утолщение/дубль той же линии
        if j < n and (pos[j] - pos[i]) <= max_dist:
            pairs.append((pos[i], pos[j]))
            i = j + 1
        else:
            i += 1
    return pairs


def modal_thickness(values: Sequence[int], lo: int = 1, hi: int = 5) -> int:
    """Мода (устойчивая к выбросам там, где арт обнимает рамку)."""
    if not len(values):
        return 1
    vals = [int(v) for v in values if lo <= v <= hi]
    if not vals:
        return 1
    cnt = np.bincount(np.array(vals))
    return int(np.argmax(cnt[lo:hi + 1])) + lo


def weighted_mode(vals: Sequence[float], bins: float = 0.5) -> float:
    v = np.asarray(list(vals), dtype=np.float64)
    if v.size == 0:
        return 0.0
    hist, edges = np.histogram(v, bins=max(1, int((v.max() - v.min()) / bins) + 1))
    k = int(np.argmax(hist))
    return float(0.5 * (edges[k] + edges[k + 1]))


# --------------------------------------------------------------------------------------
# config / state
# --------------------------------------------------------------------------------------

@dataclass
class Cfg:
    size: int = 256
    inset: int = 0                    # px запаса внутрь от внутренней грани
    rim: int = 2                      # ширина зоны rim-fix, px
    rim_frac: float = 0.30           # доля стороны, покрытая «линейными» пикселями, чтобы чинить
    line_thresh: float = 0.0          # порог отклика линии; 0 = авто (3 сигма шума тёмных областей)
    line_cover: float = 0.34          # мин. доля линии в окне, чтобы считать это линией
    grid_cover: float = 0.45         # доля сечения, покрытая линиями, чтобы считать ряд/столбец «рамкой»
    grid_cover_lo: float = 0.22    # второй, низкий порог — для листов, где часть линий пропала
    search_win: int = 6               # мин. окно локального поиска ребра плитки, px
    edge_cov_lo: float = 0.30        # ослабленный порог длины линии для «съеденных» рамок
    edge_cov: float = 0.55            # мин. доля ребра, покрытая линией (чтобы считать, что линия тут есть)
    cell_cover: float = 0.45
    edge_gain: float = 0.45        # порог поиска ребра = max(авто-порог, edge_gain * отклик линии)
    max_stroke: int = 8
    stroke_half: float = 0.5         # доля контраста линии, ниже которой штрих считается кончившимся
    stroke_min: float = 0.0          # доп. нижняя граница яркости строки штриха (доля p90 линии)
    stroke_pct: float = 15.0       # перцентиль измеренной толщины штриха по стороне (занижение = не съесть арт)
    stroke_sat: float = 34.0         # насыщенность выше этой = арт, а не серый штрих обводки
    grid_sat: float = 34.0         # тем же гейтом фильтруем пики сетки
    size_pull: float = 1.0         # вес «типового размера плитки» при выборе спорного ребра
    edge_out: float = 0.15         # наружу от границы полосы ребро может уйти на эту долю канала
    content_gain: float = 0.55       # на сколько над каналом считается «контент»
    content_frac: float = 0.035      # доля сечения, ниже которой это канал, а не полоса плиток
    pipeline: str = "quality"         # quality | fast | soft | pixel | raw
    pre_denoise: float = -1.0         # h денойза В ИСХОДНИКЕ (-1 = из пресета)
    denoise: float = -1.0             # h денойза ПОСЛЕ ресайза (-1 = из пресета, 0 = выкл)
    post_denoise: float = -1.0        # h финального денойза (-1 = из пресета)
    specks: float = 0.0             # порог подавления «пылинок» (0 = выкл)
    sharpen: float = -1.0             # unsharp amount (-1 = из пресета)
    sharpen_radius: float = 1.2
    sharpen_thresh: float = 7.0       # ниже этой локальной контрастности не sharpen-им
    engine: str = ""                  # '' = из пресета
    fit: str = "stretch"              # stretch | pad | square
    matte: bool = False
    matte_soft: float = 22.0
    grid: str = ""
    complete: bool = True
    prefix: str = ""
    formats: Tuple[str, ...] = ("png",)
    jpg_q: int = 95
    only: str = ""
    overwrite: bool = False
    scale_hint: float = 1.0
    debug: bool = False
    dry_run: bool = False
    max_side: int = 0
    workers: int = 0                  # 0 = авто
    report: bool = True


@dataclass
class Sheet:
    path: str
    name: str
    rgb: np.ndarray
    gray: np.ndarray
    sat: np.ndarray
    maps: dict = field(default_factory=dict)
    rows: List[Tuple[int, int]] = field(default_factory=list)
    cols: List[Tuple[int, int]] = field(default_factory=list)
    tile_h: int = 0
    tile_w: int = 0
    stroke: int = 2
    stroke_side: dict = field(default_factory=dict)
    line_level: float = 70.0
    gutter: float = 12.0
    noise: float = 3.0
    lthr: float = 14.0
    line_p90: float = 160.0
    line_resp: float = 40.0
    notes: List[str] = field(default_factory=list)
    grid_prof: tuple = ()
    grid_peaks: tuple = ()
    search: int = 6
    trust_y: bool = True           # полосы из пиков линий: ребро ищем у границы полосы
    trust_x: bool = True
    inferred_rows: set = field(default_factory=set)
    inferred_cols: set = field(default_factory=set)

    @property
    def H(self): return self.gray.shape[0]

    @property
    def W(self): return self.gray.shape[1]


# --------------------------------------------------------------------------------------
# detection
# --------------------------------------------------------------------------------------

def build_sheet(path: str, cfg: Cfg) -> Sheet:
    arr = np.asarray(Image.open(path).convert("RGB"))
    g = to_gray(arr)
    mx, my = arr.max(2), arr.min(2)
    sat = (mx - my).astype(np.float32)
    rgb = arr.astype(np.float32)
    maps = directional_line_maps(g)
    sh = Sheet(path=path, name=os.path.splitext(os.path.basename(path))[0],
               rgb=rgb, gray=g, sat=sat, maps=maps)

    # --- авто-порог «это линия»: 3 сигма шума тёмных областей --------------------------
    # абсолютный порог плохо живёт между «шумный JPEG» и «тёмная обводка на чёрном»,
    # поэтому считаем его от шума: MAD по отклику там, где лист практически чёрный.
    resp = np.maximum.reduce([maps[k] for k in ("top", "bottom")])
    sh.gutter = float(np.median(g[g < np.percentile(g, 40)])) if (g > -1).any() else 10.0
    dark = g < (sh.gutter + max(8.0, 0.12 * 255))
    if dark.sum() > 400:
        r = resp[dark]
        noise = float(1.4826 * np.median(np.abs(r - np.median(r))))
    else:
        noise = 3.0
    sh.noise = noise
    sh.lthr = float(cfg.line_thresh) if cfg.line_thresh > 0 else \
        float(np.clip(3.0 * max(noise, 2.0), 8.0, 30.0))
    # грубая прикидка уровня линии до калибровки (после find_grid будет calibrate)
    hi = resp[resp > sh.lthr]
    sh.line_level = float(np.percentile(hi, 90)) if hi.size else 70.0
    return sh


def _line_profile(sh: "Sheet", side: str, klen: int, cfg: Cfg, sat_gate: bool = True) -> np.ndarray:
    """
    Профиль «длинных тонких линий» поперёк листа.
    Для top/bottom -> массив по строкам (доля строки, покрытая горизонтальной линией),
    для left/right -> массив по столбцам.
    Двойной порог: сначала пиксель похож на тонкую светлую линию с тёмной внешней
    стороной (отклик > line_thresh), потом требуется, чтобы такая линия была ДЛИННОЙ
    (доля покрытия в окне вдоль линии). Край арта рваный -> не проходит.
    """
    horiz = side in ("top", "bottom")
    m = sh.maps[side] > sh.lthr
    if sat_gate:
        # линия рамки — ахроматная; насыщенное цветное пятно арта, лежащее впритык
        # к рамке, дало бы лишний пик сетки (и фантомную полосу)
        m = m & (sh.sat < cfg.grid_sat)
    ind = m.astype(np.float32)
    sm = box_mean_along(ind, klen, horizontal=horiz)
    ok = (sm > cfg.line_cover).astype(np.float32)
    return ok.mean(axis=1 if horiz else 0)


def snap_to_peaks(bands: List[Tuple[float, float]], peaks: List[float],
                  win: float) -> List[Tuple[float, float]]:
    """Границы полос подтягиваем к найденным пикам линий (они суб-пиксельные)."""
    out = []
    for a, b in bands:
        na, nb = a, b
        cand_a = [p for p in peaks if abs(p - a) <= win]
        cand_b = [p for p in peaks if abs(p - b) <= win]
        if cand_a:                     # ближайший пик; при равенстве — наружу (наружу меньше риска срезать)
            na = min(cand_a, key=lambda v: (abs(v - a), a - v))
        if cand_b:
            nb = min(cand_b, key=lambda v: (abs(v - b), v - b))
        out.append((float(na), float(nb)) if nb - na >= 0.35 * (b - a) else (float(a), float(b)))
    return out


def find_grid(sh: Sheet, cfg: Cfg) -> None:
    """
    Полосы сетки листа (rows по Y, cols по X) + оценка размера плитки.

    Основной детектор — «тёмный канал»: ряды/столбцы плиток разделены полосами фона,
    в котором нет контента. Это работает, даже когда плитки гуляют независимо,
    разного размера или обводка местами пропала. Пики самих линий (асимметричный
    отклик «светлая линия над тёмным») суб-пиксельные, поэтому границами полос
    правятся грубые края контент-полос. Если контент-полос нет (светлый фон, канала
    нет) — сетка берётся из пиков линий.
    """
    H, W = sh.H, sh.W
    win = max(15, int(min(H, W) * 0.03) | 1)

    def prof(side):
        """Профиль линий; с цветовым гейтом, но если так линий не видно — без него."""
        p = np.maximum(_line_profile(sh, side, win, cfg),
                       _line_profile(sh, {"top": "bottom", "bottom": "top",
                                           "left": "right", "right": "left"}[side], win, cfg))
        if len(peak_positions(p, cfg.grid_cover, 1)) >= 2:
            return p
        return np.maximum(_line_profile(sh, side, win, cfg, sat_gate=False),
                          _line_profile(sh, {"top": "bottom", "bottom": "top",
                                              "left": "right", "right": "left"}[side],
                                          win, cfg, sat_gate=False))

    ph, pv = prof("top"), prof("left")

    stat_y: dict = {}
    stat_x: dict = {}

    def modal_step(vals):
        """
        Типовой размер плитки = медиана КРУПНЫХ шагов между соседними линиями.
        Мелкие шаги — это ширина самого штриха и края арта внутри плитки, поэтому
        брать медиану по всем шагам нельзя: она оказывается «между» модами.
        """
        d = np.asarray(list(vals), float)
        d = d[d > 4]
        if d.size == 0:
            return 0.0
        big = d[d >= 0.5 * float(d.max())]
        return float(np.median(big)) if big.size else float(np.median(d))

    def lattice(peaks, pitch):
        """Пики = ГРАНИЦЫ между рядами (нижняя линия ряда пропала или линии слиплись)."""
        out = [(float(peaks[i]), float(peaks[i + 1])) for i in range(len(peaks) - 1)]
        guard = 0
        while pitch > 1 and len(out) > 1 and guard < 64:
            guard += 1
            sz = [b - a for a, b in out]
            short = [k for k, v in enumerate(sz) if v < 0.55 * pitch]
            if not short:
                break
            i0 = min(short, key=lambda k: sz[k])
            if i0 == 0:
                out[1] = (out[0][0], out[1][1]); del out[0]
            elif i0 == len(out) - 1:
                out[i0 - 1] = (out[i0 - 1][0], out[i0][1]); del out[i0]
            elif abs((out[i0 + 1][1] - out[i0][0]) - pitch) <= abs((out[i0][1] - out[i0 - 1][0]) - pitch):
                out[i0] = (out[i0][0], out[i0 + 1][1]); del out[i0 + 1]
            else:
                out[i0 - 1] = (out[i0 - 1][0], out[i0][1]); del out[i0]
        return out

    def viol(bands, pitch, invented=0):
        """
        Штраф набору полос, посчитанный только по его собственной геометрии (порогов
        яркости здесь нет намеренно: тёмный арт и светлый фон ломают любую такую
        меру). Правила:
          * канал между соседними полосами не может быть шире самой плитки — если он
            в разы шире, туда влез целый ряд плиток, т.е. сетка потеряла ряд;
          * полоса не может быть сильно короче шага — это мусор от слияния линий;
          * каждая «придуманная» граница (разрез длинной контент-полосы по минимуму
            профиля, а не найденная линия) наказывается отдельно: выдуманный рез
            опаснее пропущенного ряда, который достраивается по шагу.
        """
        if not bands:
            return 999, 0
        if pitch <= 1:
            return int(invented), -len(bands)
        sizes = [b - a for a, b in bands]
        v = int(invented)
        v += 2 * sum(1 for q in sizes if q < 0.4 * pitch)
        for (a, b), (c, d) in zip(bands, bands[1:]):
            g = c - b
            if g > 0.6 * pitch:
                v += max(1, int(round(g / pitch)))
        return v, -len(bands)

    def axis_cands(prof1d, cfg):
        """
        Список вариантов сетки по оси: (полосы, имя, приоритет, пики).
        Порог пика сначала строгий; если при нём ряды/столбцы не сходятся (часть
        линий пропала из-за микросмещения плиток внутри ряда — реальность для
        нарисованных от руки листов), пробуем низкий порог.
        """
        out = []
        for pri, cov_thr in enumerate((cfg.grid_cover, cfg.grid_cover_lo)):
            pk = [q["sub"] for q in peak_positions(prof1d, cov_thr, 1)]
            if len(pk) < 2:
                continue
            step = modal_step(np.diff(pk))
            size = modal_step([b - a for a, b in zip(pk, pk[1:])])
            out.append((group_pairs(pk, 0.55 * max(step, 1), 1.65 * max(step, 1), 0),
                        f"lines@{cov_thr:.2f}", pri * 2, pk, step, size, 0))
            out.append((lattice(pk, step), f"lattice@{cov_thr:.2f}", pri * 2 + 1, pk, step, step, 0))
        return out

    cy = axis_cands(ph, cfg)
    cx = axis_cands(pv, cfg)
    size_y = cy[0][5] if cy else 0.0
    band_y = content_bands(sh, cfg, "y", size_y or 0.0, stat_y)
    band_x = content_bands(sh, cfg, "x", 0.0, stat_x)
    if band_x:
        band_x = content_bands(sh, cfg, "x", float(np.median([b - a for a, b in band_x])), stat_x)
    for cands, bnds, stat in ((cy, band_y, stat_y), (cx, band_x, stat_x)):
        if not bnds:
            continue
        cands.append((bnds, "content", 10, list(cands[0][3]) if cands else [],
                      float(np.median([b - a for a, b in bnds])), 0.0,
                      max(0, len(bnds) - int(stat.get("runs", len(bnds))))))



    def pick(cands):
        """1) меньше штрафов; 2) больше найденных полос (не потерять ряд); 3) доверенный источник."""
        if not cands:
            return [], "none", []
        best = min(cands, key=lambda c: (viol(c[0], c[4], c[6])[0], -len(c[0]), c[2]))
        return best[0], best[1], best[3]

    rows, how_y, pk_y = pick(cy)
    cols, how_x, pk_x = pick(cx)
    sh.trust_y, sh.trust_x = (not how_y.startswith("content")), (not how_x.startswith("content"))
    if cfg.debug:
        for nm, cands in (("y", cy), ("x", cx)):
            sh.notes.append("cand " + nm + ": " + " | ".join(
                f"{c[1]}:{viol(c[0], c[4], c[6])[0]}:{len(c[0])}" for c in cands))
    yh, xv = pk_y, pk_x
    sh.grid_prof = (ph, pv)
    sh.grid_peaks = (yh, xv)
    if len(rows) < 2 or len(cols) < 2:
        sh.rows, sh.cols = rows, cols
        sh.notes.append(f"grid: полос {len(rows)}x{len(cols)} (линий {len(yh)}x{len(xv)}) "
                        "— задайте --grid СТРОКИxСТОЛБЦЫ")
        return
    # Подтягивать к суб-пиксельным пикам нужно только ГРУБЫЕ контент-полосы.
    # Линейные полосы уже стоят на пиках, и «подтяжка» могла бы только перескочить
    # на линию соседнего ряда; для контент-полос окно — доля канала (не более 10% плитки).
    def snap_local(bands, peaks, is_content):
        """Контент-полосы грубые — подтягиваем их края к суб-пиксельным пикам линий."""
        if not is_content:
            return list(bands)
        w = 3.0 + 0.5 * max(float(sh.gutter), 0.0)
        out = []
        for a, b in bands:
            for side in (0, 1):
                v = a if side == 0 else b
                pk = [q for q in peaks if abs(q - v) <= w]
                if pk:
                    q = min(pk, key=lambda z: abs(z - v))
                    if side == 0:
                        a = q
                    else:
                        b = q
            out.append((float(a), float(b)) if b - a > 4 else (float(a), float(b)))
        return out

    sh.rows = snap_local(rows, yh, how_y == "content")
    sh.cols = snap_local(cols, xv, how_x == "content")
    sh.tile_h = int(round(np.median([b - a for a, b in sh.rows])))
    sh.tile_w = int(round(np.median([b - a for a, b in sh.cols])))
    sh.search = int(max(cfg.search_win, round(0.07 * min(sh.tile_w, sh.tile_h))))
    sh.notes.append(f"grid {len(sh.rows)}x{len(sh.cols)} ({how_y}/{how_x}) "
                    f"tile~{sh.tile_w}x{sh.tile_h} win={sh.search} "
                    f"peaks={len(yh)}x{len(xv)}")


def _runs(mask: np.ndarray, min_len: int) -> List[Tuple[int, int]]:
    """Серии True в 1D-маске -> [(start, end_exclusive)], короче min_len отбрасываются."""
    idx = np.where(mask)[0]
    out: List[Tuple[int, int]] = []
    if idx.size == 0:
        return out
    a = p0 = int(idx[0])
    for i in idx[1:]:
        i = int(i)
        if i - p0 > 1:
            if p0 - a + 1 >= min_len:
                out.append((a, p0 + 1))
            a = i
        p0 = i
    if p0 - a + 1 >= min_len:
        out.append((a, p0 + 1))
    return out


def content_base(sh: Sheet, cfg: Cfg) -> float:
    """
    Порог «это контент, а не канал листа» по гистограмме яркости самого листа.
    Специально не опирается на calibrate(): find_grid вызывает content_bands до
    калибровки, и тогда абсолютный порог «фон + доля от уровня линии» превращался
    в «всё ярче 6» — т.е. контентом становился и канал, что ломало проверку сетки.
    """
    if getattr(sh, "_cbase", None) is not None:
        return float(sh._cbase)
    lo = float(np.percentile(sh.gray, 20.0))
    hi = float(np.percentile(sh.gray, 97.0))
    v = lo + cfg.content_gain * max(10.0, 0.34 * (hi - lo))
    sh._cbase = v
    return v


def content_bands(sh: Sheet, cfg: Cfg, axis: str, est: float = 0.0,
                  stats: Optional[dict] = None) -> List[Tuple[float, float]]:
    """
    Полосы «контента» по тёмному каналу между рядами плиток — основной детектор
    сетки: он не требует, чтобы ряды/колонки были выровнены, прощает разный размер
    плиток и пропавшую местами обводку.

    1) cont = пиксели ярче «канала + доля от уровня линии» (обводка и арт — контент,
       канал листа — нет);
    2) профиль = доля сечения; сглаживание коротким окном, чтобы редкий тёмный арт
       внутри плитки не рвал полосу;
    3) низкий порог (3.5% от максимума) + склейка разрывов < 0.22 типовой плитки:
       полосы, разорванные тёмными строками внутри плитки, собираются обратно;
    4) слишком длинная полоса режется по самому глубокому локальному минимуму
       профиля (это и есть канал между рядами), рекурсивно.
    """
    horiz = axis == "y"
    L = sh.H if horiz else sh.W
    base = content_base(sh, cfg)
    cont = (sh.gray > base).astype(np.float32)
    cont = cv2.morphologyEx(cont, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    prof = cont.mean(axis=1 if horiz else 0)
    k = int(max(3, round(0.06 * (est if est > 1 else L / 6))) * 2 + 1)
    ps = cv2.boxFilter(prof.reshape(-1, 1).astype(np.float32), -1, (k, 1),
                       borderType=cv2.BORDER_REPLICATE).ravel()
    tau = max(0.008, float(ps.max()) * cfg.content_frac)
    tile = float(est) if est > 1 else 0.0
    runs = _runs(ps > tau, max(5, int(0.25 * (tile if tile > 1 else L / 8))))
    if not runs:
        return []
    if tile <= 1:
        tile = float(np.median([b - a for a, b in runs]))
    # склеить разрывы короче четверти плитки (тёмные строки внутри плитки)
    merged = [runs[0]]
    for a, b in runs[1:]:
        if a - merged[-1][1] < 0.22 * tile:
            merged[-1] = (merged[-1][0], b)
        else:
            merged.append((a, b))
    if stats is not None:
        stats["runs"] = len(merged)      # естественные полосы; всё сверх них — придуманные резы
    out: List[Tuple[float, float]] = []
    stack = list(merged)
    guard = 0
    while stack and guard < 64:
        guard += 1
        x, y = stack.pop(0)
        if (y - x) <= 1.32 * tile:
            out.append((float(x), float(y)))
            continue
        seg = ps[x:y]
        m = int(max(3, 0.42 * tile))
        if seg.size < 2 * m + 4:
            out.append((float(x), float(y)))
            continue
        inner = seg[m:seg.size - m]
        cut = x + m + int(np.argmin(inner))
        if float(inner.min()) > 0.5 * float(ps[x:y].max()):
            out.append((float(x), float(y)))          # впадины нет -> не выдумываем рез
            continue
        stack.insert(0, (cut + 1, y))
        stack.insert(0, (x, cut + 1))
    out.sort()
    return out


def merge_bands(line: List[Tuple[float, float]], extra: List[Tuple[float, float]],
                tile: float, tol: float = 0.0) -> Tuple[List[Tuple[float, float]], set]:
    """
    Вставляет контент-полосу только в настоящий «провал» сетки: место размером
    >= 1.5 плитки между соседними полосами (или перед первой / после последней),
    и только если полоса по размеру похожа на плитку. Так «потерянный ряд»
    восстанавливается, а шумная контент-сегментация не плодит фантомные плитки.
    """
    if tile <= 1:
        return list(line), set()
    out = sorted((float(x), float(y)) for x, y in line)
    added = set()
    for a, b in sorted((float(x), float(y)) for x, y in extra):
        if not (0.7 * tile <= (b - a) <= 1.4 * tile):
            continue
        if any(min(b, bb) - max(a, aa) > 2.0 for aa, bb in out):
            continue                                    # пересечение с существующей
        holes = [(out[i][1], out[i + 1][0]) for i in range(len(out) - 1)]
        ok_hole = any(gh - gl > 1.45 * tile and gl - 2 <= a and b <= gh + 2
                      for gl, gh in holes)
        ok_head = bool(out) and (out[0][0] - a) > 1.45 * tile and b <= out[0][0] - 2
        ok_tail = bool(out) and (a - out[-1][1]) > 1.45 * tile and b >= out[-1][1] + 2
        if not (ok_hole or ok_head or ok_tail):
            continue
        out.append((a, b))
        added.add(len(out) - 1)
        out.sort()
    return out, added


def complete_grid(sh: Sheet, cfg: Cfg) -> None:
    """
    Достройка решётки. Если строка (или столбец) рамок не нашлась — слишком тёмная
    обводка, убитая JPEG, или край листа — между соседними найденными полосами
    возникает двойной провал. Смотрим на шаги НАЧАЛ полос: там, где шаг кратен
    типовому (m x), вставляем m-1 полосу по решётке и помечаем их inferred.
    Рёбра таких полос всё равно уточняются локально (часто находятся), а в manifest
    они уходят с предупреждением, чтобы «один плохой ряд» не стоил молча половины
    набора иконок.
    """
    for key, store in (("rows", "inferred_rows"), ("cols", "inferred_cols")):
        bands = getattr(sh, key)
        if len(bands) < 2:
            continue
        starts = np.array([a for a, _ in bands], float)
        pitch = float(np.median(np.diff(starts)))
        size = float(np.median([b - a for a, b in bands]))
        if pitch <= 1 or size <= 1:
            continue
        inferred = set()
        out = [bands[0]]
        for i in range(1, len(bands)):
            prev = out[-1][0]
            cur = bands[i][0]
            m = int(round((cur - prev) / pitch))
            if 2 <= m <= 6:
                for j in range(1, m):
                    a = prev + j * pitch
                    out.append((a, a + size))
                    inferred.add(len(out) - 1)
            out.append(bands[i])
        if inferred:
            setattr(sh, key, out)
            setattr(sh, store, inferred)
            sh.notes.append(f"grid: достроено {len(inferred)} полос(ы) по оси {key}")


def manual_grid(sh: Sheet, spec: str, cfg: Cfg) -> None:
    """
    Ручная сетка 'ROWSxCOLS' — фолбэк для листов, где обводку не найти вообще
    (например, фон не чёрный, а «белая обводка» это не рамка, а кант вокруг арта).
    Полосы = равномерная разбивка bounding box «весь контент» (ищем границы по
    сумме непрозрачного шума), дальше рёбра уточняются локально как обычно.
    """
    try:
        nr, nc = [int(v) for v in spec.lower().split("x")]
    except Exception:
        sh.notes.append(f"manual-grid: не понял '{spec}'")
        return
    ink = (sh.gray > sh.gutter + max(12.0, 0.35 * sh.line_level)).astype(np.float32)
    ink = cv2.morphologyEx(ink, cv2.MORPH_CLOSE, np.ones((5, 5), np.uint8))
    rs, cs = ink.sum(1), ink.sum(0)
    def bands(prof, n):
        thr = max(1.0, prof.max() * 0.04)
        on = np.where(prof > thr)[0]
        if on.size < 8:
            on = np.arange(len(prof))
        a, b = float(on[0]), float(on[-1]) + 1
        step = (b - a) / n
        return [(a + i * step, a + (i + 1) * step) for i in range(n)]
    sh.rows = bands(rs, nr)
    sh.cols = bands(cs, nc)
    sh.tile_h = int(round(np.median([b - a for a, b in sh.rows])))
    sh.tile_w = int(round(np.median([b - a for a, b in sh.cols])))
    # полосы равномерные -> граница плитки может быть на полполосы рядом: ребро ищем
    # широким окном и по правилу «самая внешняя линия, за которой темно»
    sh.trust_y = sh.trust_x = False
    sh.search = int(max(cfg.search_win, round(0.22 * min(sh.tile_w, sh.tile_h))))
    sh.notes.append(f"manual-grid {nr}x{nc}: полос {len(sh.rows)}x{len(sh.cols)}, "
                    f"плитка ~{sh.tile_w}x{sh.tile_h} win={sh.search}")


def calibrate(sh: Sheet, cfg: Cfg) -> None:
    """
    калибровка по факту: яркость самой обводки и уровень «канала» между плитками.
    Берём медиану luma точно на найденных линиях (все рёбра всех плиток) — это и есть
    уровень штриха листа; и медиану luma в середине каналов — уровень чёрного фона.
    Нужны для rim-fix (не спутать арт с остатками обводки) и для альфы.
    """
    H, W = sh.H, sh.W
    line_vals, gut_vals = [], []
    ys = [v for pair in sh.rows for v in pair]
    xs = [v for pair in sh.cols for v in pair]
    for y in ys:
        yy = int(round(y))
        if 0 <= yy < H:
            seg = []
            for (a, b) in sh.cols:
                x0, x1 = int(round(a)) + 3, int(round(b)) - 3
                if x1 - x0 > 8:
                    seg.append(sh.gray[yy, x0:x1])
            if seg:
                line_vals.append(np.concatenate(seg))
    for x in xs:
        xx = int(round(x))
        if 0 <= xx < W:
            seg = []
            for (a, b) in sh.rows:
                y0, y1 = int(round(a)) + 3, int(round(b)) - 3
                if y1 - y0 > 8:
                    seg.append(sh.gray[y0:y1, xx])
            if seg:
                line_vals.append(np.concatenate(seg))
    # канал между полосами
    for k in range(len(sh.rows) - 1):
        a = int(round(sh.rows[k][1])) + 2
        b = int(round(sh.rows[k + 1][0])) - 2
        if b > a:
            gut_vals.append(sh.gray[a:b, :].ravel())
    for k in range(len(sh.cols) - 1):
        a = int(round(sh.cols[k][1])) + 2
        b = int(round(sh.cols[k + 1][0])) - 2
        if b > a:
            gut_vals.append(sh.gray[:, a:b].ravel())
    if line_vals:
        v = np.concatenate(line_vals)
        sh.line_level = float(np.percentile(v, 60))
        sh.line_p90 = float(np.percentile(v, 92))
    # типовой отклик линии (g - dark_outside) на найденных рёбрах — от него считаем порог
    resp_at = []
    for (a, b) in sh.rows:
        yy = int(round(a))
        if 0 <= yy < H:
            resp_at.append(np.maximum(sh.maps["top"][yy, :], sh.maps["bottom"][yy, :]))
    for (a, b) in sh.cols:
        xx = int(round(a))
        if 0 <= xx < W:
            resp_at.append(np.maximum(sh.maps["left"][:, xx], sh.maps["right"][:, xx]))
    if resp_at:
        sh.line_resp = float(np.percentile(np.concatenate(resp_at), 70))
    else:
        sh.line_resp = 40.0
    if gut_vals:
        sh.gutter = float(np.median(np.concatenate(gut_vals)))
    if not np.isfinite(sh.gutter) or sh.gutter <= 0:
        sh.gutter = float(np.clip(0.08 * max(sh.tile_w, sh.tile_h), 2.0, 40.0))
    sh.notes.append(f"calib: line~{sh.line_level:.0f} (p92 {sh.line_p90:.0f}) "
                    f"resp~{sh.line_resp:.0f} gutter~{sh.gutter:.0f}")


def _edge_cov(sh: Sheet, side: str, lo: int, hi: int, thresh: float,
              coarse: int, win: int) -> np.ndarray:
    """
    Для каждой позиции в окне [coarse-win, coarse+win] — доля отрезка ребра этой
    плитки, где отклик > порога. Т.е. «здесь действительно идёт длинная линия».
    """
    horiz = side in ("top", "bottom")
    m = sh.maps[side]
    L = sh.H if horiz else sh.W
    n0 = int(max(0, min(coarse - win, L - 2)))
    n1 = int(max(n0 + 1, min(coarse + win + 1, L)))
    out = np.full(n1 - n0, -1.0, np.float32)
    if hi - lo < 8:
        return out
    for i, c in enumerate(range(n0, n1)):
        seg = m[c, lo:hi] if horiz else m[lo:hi, c]
        out[i] = float(np.mean(seg > thresh))
    return out


def find_edge(sh: Sheet, side: str, coarse: int, span0: int, span1: int,
              thresh: float, min_cov: float, cfg: Cfg, win: Optional[int] = None,
              lim: Optional[Tuple[int, int]] = None, tight: bool = True,
              far: Optional[int] = None, target: float = 0.0) -> dict:
    """
    Одно ребро одной плитки.

    Ищем НЕ «самый сильный», а «самый внешний» длинный линейный отклик: светлый край
    арта, упирающегося в рамку, всегда сильнее самой обводки, а рамка — первая
    структура снаружи внутрь. Дополнительно требуем, чтобы снаружи от найденной линии
    было темно (это канал между плитками), иначе ореол арта в канале сдвинул бы рез.

    Толщина штриха = пробег ВНУТРЬ, пока покрытие ребра линейным откликом держится
    (>=60% пика). Хитрость в том, что отклик требует темноты СНАРУЖИ: у строки арта,
    лежащей сразу за линией, сверху bright рамка -> отклика нет -> пробег сам
    останавливается на конце штриха, даже когда арт «обнимает» рамку.
    Замер считается валидным, только если за пробегом реально пусто: иначе он
    не идёт в статистику толщины листа, а берётся типовая толщина стороны.
    """
    horiz = side in ("top", "bottom")
    L = sh.H if horiz else sh.W
    inward = +1 if side in ("top", "left") else -1
    win_i = int(win if win else cfg.search_win)
    pad = win_i + cfg.max_stroke + 2
    cov = _edge_cov(sh, side, span0, span1, thresh, coarse, pad)
    if cov.size == 0:
        return {"outer": int(coarse), "t": int(sh.stroke), "cov": 0.0, "ok": False,
                "why": "no-window", "resp": 0.0, "t_valid": False}
    base = int(max(0, min(coarse - pad, L - 2)))
    lo_i, hi_i = max(0, min(span0, span1)), max(0, min(span1, span0) + 1)

    def cov_at(sheet_pos: int) -> float:
        i = sheet_pos - base
        if 0 <= i < cov.size:
            return float(cov[i])
        seg = sh.maps[side][sheet_pos, lo_i:hi_i] if horiz else sh.maps[side][lo_i:hi_i, sheet_pos]
        return float(np.mean(seg > thresh)) if seg.size else 0.0

    # Кандидаты — только в штатном окне поиска (за его краями живёт лишь «пробег»
    # толщины). Иначе в широком канале листа ловится чужая линия и рез распухает.
    c0 = int(np.clip(coarse - base, 0, max(0, cov.size - 1)))
    lo_w, hi_w = max(0, c0 - win_i), min(cov.size - 1, c0 + win_i)
    if lim is not None:                      # не выходить за пределы своей полосы
        l0 = int(np.clip(lim[0] - base, 0, cov.size - 1))
        l1 = int(np.clip(lim[1] - base, 0, cov.size - 1))
        lo_w, hi_w = min(hi_w, max(lo_w, l0)), max(lo_w, min(hi_w, l1))
    sub = cov[lo_w:hi_w + 1]
    cands = (np.where(sub >= min_cov)[0] + lo_w) if sub.size else np.array([], int)
    why, ok = "", True

    def outside_dark(pos: int) -> bool:
        a, b = (pos - 3, pos - 1) if inward > 0 else (pos + 1, pos + 3)
        a = max(0, min(L - 1, a)); b = max(a, min(L - 1, b))
        seg = sh.gray[a:b + 1, span0:span1] if horiz else sh.gray[span0:span1, a:b + 1]
        return bool(seg.size and float(np.mean(seg)) < 0.55 * sh.line_level + 0.45 * sh.gutter)

    if cands.size == 0:
        # линии нужной длины тут нет — но рез всё равно обязан остаться внутри своей
        # полосы: наружу не выходим, иначе в кадр возьмётся чужая рамка. Ищем максимум
        # отклика во ВНУТРЕННЕЙ половине окна (и только если там пусто — во всём).
        lo_i2, hi_i2 = (c0, hi_w) if inward > 0 else (lo_w, c0)
        seg2 = cov[lo_i2:hi_i2 + 1]
        if seg2.size and float(seg2.max()) > 0:
            k = int(np.argmax(seg2)) + lo_i2
        else:
            k = int(np.argmax(sub)) + lo_w if sub.size else c0
        ok, why = False, "weak"
    else:
        # Грубая граница полосы уже стоит на линии (или на середине канала), поэтому
        # берём БЛИЖАЙШИЙ кандидат, а не самый внешний: самый внешний на широком окне
        # прыгает в линию соседней плитки и режет «лишнее». Наружу разрешаем всего
        # ~половину канала — на случай, если полоса чуть уже плитки.
        if tight:
            gut = float(sh.gutter) if np.isfinite(sh.gutter) else 0.08 * max(sh.tile_w, sh.tile_h)
            tol_out = int(max(1, min(3, round(cfg.edge_out * max(gut, 4.0)))))
            pool = [int(c) for c in cands if inward * (base + int(c) - coarse) >= -tol_out]
            if not pool:
                # в штатном окне нет линии: берём самое внутреннее пересечение — рез
                # станет чуть меньше плитки, но в кадр не попадут канал и чужая рамка
                pool = [int(max(cands) if inward > 0 else min(cands))]
                why = "past-band"
            def size_pen(c: int) -> float:
                """Насколько предсказанный по этому кандидату размер ячейки отличается
                от типового размера листа. Спор между «ближе к краю полосы» и
                «плитка обычного размера» решается в пользу размера: на сросшихся
                рамках (тонкий канал) ближайшее к границе — часто чужая плитка."""
                if target <= 0 or far is None:
                    return 0.0
                return abs(abs(far - (base + int(c))) - target)

            if not pool or why == "past-band":
                # Местами рамка «съедена» ярким артом впритык (или просто убита): там
                # длинная линия не набирает нужный порог. Ослабляем требование к
                # ДЛИНЕ линии, но только внутрь своей полосы — наружу по-прежнему
                # нельзя, иначе в кадр возьмётся чужая рамка.
                c_lo = [int(c) for c in np.where(sub >= cfg.edge_cov_lo)[0] + lo_w
                        if inward * (base + int(c) - coarse) >= 0]
                if c_lo:
                    pool = c_lo
                    why = "relaxed"
            k = min(pool, key=lambda c: (0 if outside_dark(base + c) else 1,
                                         abs(base + c - coarse) + cfg.size_pull * size_pen(c),
                                         -int(cov[c])))
            if why == "" and not outside_dark(base + k):
                why = "bright-outside"
                # Снаружи найденной линии — фон/свечение, а не тёмный канал: вероятно,
                # это не наша рамка, а строка рядом со светлым артом. Ищем линию
                # послабее, но строго ВНУТРЬ своей полосы, у которой снаружи темнее.
                c_lo = [int(c) for c in np.where(sub >= cfg.edge_cov_lo)[0] + lo_w
                        if inward * (base + int(c) - coarse) > 0]
                if c_lo:
                    def _key(c):
                        return (0 if outside_dark(base + int(c)) else 1,
                                abs(base + int(c) - coarse) + cfg.size_pull * size_pen(c),
                                -int(cov[int(c)]))
                    alt = min(c_lo, key=_key)
                    if _key(alt) < _key(k):
                        k, why = alt, "bright-outside/relaxed"
        else:
            # Полоса грубая (--grid, контент-полосы): её край может стоять ровно на
            # линии СОСЕДНЕЙ плитки, поэтому идём от центра плитки к краю и берём
            # ПЕРВУЮ линию, за которой темно — это наша рамка, а не чужая.
            k = -1
            for c in (cands[::-1] if inward > 0 else cands):
                if outside_dark(base + int(c)):
                    k = int(c)
                    break
            if k < 0:
                k, why, ok = int(cands[-1] if inward > 0 else cands[0]), "no-dark-outside", False
    outer = base + k
    pk = float(cov[k])

    # Толщина штриха: пробег ВНУТРЬ по рядам, которые всё ещё «как линия» — яркость
    # в узкой полосе вокруг уровня штриха листа И почти бесцветно. Двойной критерий:
    # (а) не провалиться в светлый арт впритык к рамке (он цветной или резче по
    # яркости), (б) не остановиться внутри самого толстого штриха — у его внутреннего
    # ряда отклика «bright над темнотой» уже нет, ряды над ним такие же светлые.
    g, satm = sh.gray, sh.sat
    # Полоса «это ещё штрих»: (1) яркость в узком коридоре вокруг уровня линии листа,
    # (2) нижняя граница коридора НЕ ниже типичной яркости линии (p90): иначе на
    # тёмном листе внутрь штриха доедает тусклый ряд арта, лежащий впритык к рамке
    # (рез становился на 1px шире с каждой стороны = остаток обводки в кадре).
    thr_run = max(sh.gutter + 0.5 * max(4.0, sh.line_level - sh.gutter),
                  cfg.stroke_min * sh.line_p90)   # stroke_min=0 -> только «вдвое темнее линии»
    if thr_run >= 0.9 * sh.line_level:            # иначе сам штрих не пройдёт свой порог
        thr_run = 0.5 * sh.line_level
    hi_run = 1.35 * sh.line_level + 10.0
    if hi_run <= thr_run:
        hi_run = thr_run + 40.0

    def run_ok(sp: int) -> bool:
        if not (0 <= sp < L):
            return False
        lv, sv = (g[sp, span0:span1], satm[sp, span0:span1]) if horiz else \
                 (g[span0:span1, sp], satm[span0:span1, sp])
        if lv.size == 0:
            return False
        return bool(thr_run < float(np.median(lv)) < hi_run
                    and float(np.median(sv)) < cfg.stroke_sat)

    # Штрих = максимальный непрерывный «как линия» пробег, проходящий через выбранный
    # кандидат (в obie стороны). Тогда внутренняя грань реза не зависит от того, на
    # какую именно из строк штриха сел кандидат (у 2px линии это подбрасывало ±1px),
    # и не зависит от светлого ореола в канале СНАРУЖИ (он двигает только outer).
    def grow(p0: int, step: int) -> int:
        q = p0
        while abs(q + step - p0) <= cfg.max_stroke and run_ok(q + step):
            q += step
        return q

    e_out = grow(outer, -inward)
    e_in = grow(outer, inward)
    lo, hi = min(e_out, e_in), max(e_out, e_in)
    t = int(hi - lo + 1)
    hit_lim = bool(t > cfg.max_stroke)
    anchor = int(outer)                       # строка, на которой линия нашлась (пик отклика)
    tin = int((hi - anchor) if inward > 0 else (anchor - lo))   # сколько рядов штриха ВНУТРЬ от якоря
    outer = int(lo if inward > 0 else hi)      # для QC/оверлея — внешняя грань штриха
    t_valid = bool((not hit_lim) and (not run_ok(hi + 1 if inward > 0 else lo - 1)))
    if hit_lim:
        t = cfg.max_stroke

    # суб-пиксельная позиция линии (парабола по покрытию) — для manifest/QC
    pos = float(outer)
    if 0 < k < cov.size - 1:
        y0_, y1_, y2_ = float(cov[k - 1]), float(cov[k]), float(cov[k + 1])
        den = y0_ - 2 * y1_ + y2_
        pos += float(np.clip(0.5 * (y0_ - y2_) / den, -0.75, 0.75)) if abs(den) > 1e-6 else 0.0
    seg = sh.maps[side][outer, span0:span1] if horiz else sh.maps[side][span0:span1, outer]
    return {"outer": int(outer), "pos": round(pos, 2), "t": int(t), "cov": round(pk, 3),
            "anchor": int(anchor), "tin": int(max(0, tin)),
            "ok": bool(ok), "why": why, "t_valid": t_valid,
            "resp": round(float(np.percentile(seg, 85)) if seg.size else 0.0, 1)}


def _note(ed: dict, tag: str) -> None:
    w = str(ed.get("why", "") or "")
    if tag not in w:
        ed["why"] = (w + "/" if w else "") + tag


def tighten(sh: Sheet, cells: List[dict], target_h: float, target_w: float) -> None:
    """Разбор раздутого артом тина по типовому размеру плитки листа.

    Толщина рукописной рамки согласована на всех четырёх сторонах плитки, а размер
    плитки — с типовым размером листа. Если кадр вышел ЗАМЕТНО меньше типового, а с
    какой-то стороны пробег штриха длиннее типовой толщины, значит за рамкой лежит не
    обводка, а тёмный арт, обнимающий рамку изнутри (корпус/тень иконки): такой хвост
    пробега съедает несколько px арта. Уменьшаем «тин» на самой раздутой стороне;
    наружу (увеличивать тин) не идём никогда — там канал и чужая рамка.
    """
    if target_h <= 0 and target_w <= 0:
        return
    for c in cells:
        for _ in range(4):
            x0, y0, x1, y1, _f = inner_box(c, sh.stroke_side, 0, sh.W, sh.H, "none")
            w_, h_ = x1 - x0, y1 - y0
            if target_w > 0 and w_ < target_w - 2:
                pair, size, target = ("right", "left"), w_, target_w
            elif target_h > 0 and h_ < target_h - 2:
                pair, size, target = ("bottom", "top"), h_, target_h
            else:
                break                      # ячейка уже типового размера
            best = None
            for side in pair:
                ed = c["edges"][side]
                cur = int(ed.get("tin", 0))
                if cur <= 0:
                    continue
                cut = int(min(cur, target - size))
                if cut <= 0:
                    continue
                gain = abs(size - target) - abs(size + cut - target)
                if gain > 0 and (best is None or gain > best[0] or
                                 (gain == best[0] and int(c["edges"][best[1]].get("tin", 0)) < cur)):
                    best = (gain, side, cut)
            if best is None:
                break                      # уменьшить нечем (или некуда)
            _g, side, cut = best
            ed = c["edges"][side]
            ed["tin"] = int(max(0, int(ed.get("tin", 0)) - cut))
            w = str(ed.get("why", "") or "")
            if "stroke-cap" not in w:
                ed["why"] = (w + "/" if w else "") + "stroke-cap"


def find_edges(sh: Sheet, cfg: Cfg) -> List[dict]:
    """
    Все рёбра всех ячеек (грубая привязка — из сетки, уточнение — локально).

    Два прохода. Первый — «по близости к границе полосы». По его результатам считается
    типовой размер плитки листа, и второй проход добавляет в критерий выбора ребра
    согласованность с этим размером: на листах с тонким каналом рамка соседней плитки
    бывает БЛИЖЕ к краю полосы, чем своя, и близость сама по себе её не отвергает.
    """
    pad_y = max(4, int(0.06 * sh.tile_h))
    pad_x = max(4, int(0.06 * sh.tile_w))
    thresh = max(sh.lthr, cfg.edge_gain * sh.line_resp)
    win = int(getattr(sh, "search", cfg.search_win) or cfg.search_win)
    trust_y = bool(getattr(sh, "trust_y", True))
    trust_x = bool(getattr(sh, "trust_x", True))

    def one_pass(target_h: float, target_w: float) -> List[dict]:
        cells: List[dict] = []
        for ri, (yt, yb) in enumerate(sh.rows):
            for ci, (xl, xr) in enumerate(sh.cols):
                sx0, sx1 = int(round(xl + pad_x)), int(round(xr - pad_x))
                sy0, sy1 = int(round(yt + pad_y)), int(round(yb - pad_y))
                hband = max(1.0, yb - yt)
                wband = max(1.0, xr - xl)
                if trust_y and trust_x:
                    # окно поиска каждой грани ограничиваем ~45% полосы: соседняя
                    # плитка не должна «перетянуть» ребро на себя при широком окне
                    spec = (("top", yt, (sx0, sx1), (int(yt - win), int(yt + max(win, 0.45 * hband)))),
                            ("bottom", yb, (sx0, sx1), (int(yb - max(win, 0.45 * hband)), int(yb + win))),
                            ("left", xl, (sy0, sy1), (int(xl - win), int(xl + max(win, 0.45 * wband)))),
                            ("right", xr, (sy0, sy1), (int(xr - max(win, 0.45 * wband)), int(xr + win))))
                    far = {"top": yb, "bottom": yt, "left": xr, "right": xl}
                    tgt = {"top": target_h, "bottom": target_h, "left": target_w, "right": target_w}
                else:
                    # полоса НЕ из найденных линий (--grid, контент-полосы): ищем линию
                    # ВНУТРИ своей полосы, наружу нельзя ни на пиксель, иначе рез
                    # дотянется до рамки соседней плитки и возьмёт канал в кадр
                    spec = (("top", yt, (sx0, sx1), (int(yt), max(int(yt) + 2, int(yb - 0.3 * hband)))),
                            ("bottom", yb, (sx0, sx1), (min(int(yb) - 2, int(yt + 0.3 * hband)), int(yb))),
                            ("left", xl, (sy0, sy1), (int(xl), max(int(xl) + 2, int(xr - 0.3 * wband)))),
                            ("right", xr, (sy0, sy1), (min(int(xr) - 2, int(xl + 0.3 * wband)), int(xr))))
                    far = {"top": yb, "bottom": yt, "left": xr, "right": xl}
                    tgt = dict.fromkeys(far, 0.0)
                e = {}
                for side, coarse, (a, b), lim in spec:
                    e[side] = find_edge(sh, side, int(round(coarse)), a, b, thresh,
                                        cfg.edge_cov, cfg, win, lim,
                                        tight=trust_y if side in ("top", "bottom") else trust_x,
                                        far=int(round(far[side])), target=float(tgt[side]))
                cells.append({"row": ri, "col": ci, "edges": e})
        return cells

    def aggregate(cells: List[dict]) -> None:
        """Типовая толщина штриха по сторонам + флаги достоверности ячеек."""
        good = {s: [c["edges"][s]["t"] for c in cells if c["edges"][s]["t_valid"]]
                for s in ("top", "left", "bottom", "right")}
        allgood = [v for lst in good.values() for v in lst]
        base = int(max(1, round(float(np.percentile(allgood, cfg.stroke_pct))))) if allgood else 1
        for s in ("top", "left", "bottom", "right"):
            v = good[s]
            sh.stroke_side[s] = int(max(1, round(float(np.percentile(v, cfg.stroke_pct))))) if v else base
        sh.stroke = base
        for c in cells:
            covs = [c["edges"][s]["cov"] for s in ("top", "left", "bottom", "right")]
            c["cover"] = covs
            if c["row"] in sh.inferred_rows or c["col"] in sh.inferred_cols:
                c["inferred"] = True
            c["low_confidence"] = bool(c.get("inferred") or min(covs) < cfg.cell_cover or
                                       any(c["edges"][s]["why"] for s in ("top", "left", "bottom", "right")))

    cells = one_pass(0.0, 0.0)
    aggregate(cells)
    hs, ws = [], []
    for c in cells:
        x0, y0, x1, y1, _ = inner_box(c, sh.stroke_side, 0, sh.W, sh.H, "none")
        if 0.25 * max(sh.tile_w, 1) < (x1 - x0) < 2.5 * max(sh.tile_w, 1):
            ws.append(x1 - x0)
        if 0.25 * max(sh.tile_h, 1) < (y1 - y0) < 2.5 * max(sh.tile_h, 1):
            hs.append(y1 - y0)
    th_, tw_ = (float(np.median(hs)) if len(hs) >= 3 else 0.0,
                float(np.median(ws)) if len(ws) >= 3 else 0.0)
    if (th_ > 0 or tw_ > 0) and len(cells) > 1:
        c2 = one_pass(th_, tw_)
        a2 = [abs((inner_box(c, sh.stroke_side, 0, sh.W, sh.H, "none")[3] -
                   inner_box(c, sh.stroke_side, 0, sh.W, sh.H, "none")[1]) - th_) for c in c2]
        a1 = [abs((inner_box(c, sh.stroke_side, 0, sh.W, sh.H, "none")[3] -
                   inner_box(c, sh.stroke_side, 0, sh.W, sh.H, "none")[1]) - th_) for c in cells]
        if float(np.median(a2)) <= float(np.median(a1)):
            cells = c2
            sh.notes.append(f"grid: второй проход рёбер по типовому размеру {tw_:.0f}x{th_:.0f}")
    tighten(sh, cells, th_, tw_)
    aggregate(cells)
    return cells



# --------------------------------------------------------------------------------------
# crop / clean / upscale
# --------------------------------------------------------------------------------------

def inner_box(cell: dict, stroke_mode: int, inset: int, W: int, H: int,
              fit: str) -> Tuple[int, int, int, int, tuple]:
    """
    Внутренняя грань обводки -> bbox реза (x0, y0, x1, y1), x1/y1 exclusive,
    плюс outer-бокс рамки для QC.

    edge['outer'] — пиксель САМОЙ ВНЕШНЕЙ строки/столбца линии (считая от канала
    внутрь), edge['t'] — измеренная ширина штриха (сколько строк/столбцов линия
    занимает, двигаясь внутрь). Штрих занимает [outer .. outer + inward*(t-1)],
    поэтому первая «внутренняя» строка = outer + t (для top/left) и последняя
    внутренняя = outer - t (для bottom/right, exclusive +1).
    Если t раздута артом, который обнимает рамку, берём моду листа — иначе мы бы
    срезали лишнего.
    """
    e = cell["edges"]

    # Считаем от ЯКОРЯ (строка, где линия реально нашлась), а не от внешней грани
    # штриха: внешняя грань зависит от ореола в канале и её раздувание сдвинуло бы рез
    # внутрь кадра (в кадр попадала серая строка обводки). Число рядов штриха «внутрь»
    # от якоря ограничиваем типовой толщиной листа — это защита от светлого арта,
    # лежащего впритык к рамке (иначе рез съел бы ряд арта).
    def tin_of(side: str) -> int:
        ed = e[side]
        mode = int(stroke_mode.get(side, stroke_mode.get("all", 1)) if isinstance(stroke_mode, dict)
                   else stroke_mode)
        mode = max(1, mode)
        tin = int(ed.get("tin", max(0, mode - 1)))
        # Замер «пробег сам остановился» (т.е. за штрихом реально видно перепад) —
        # доверяем ему даже если он длиннее типовой толщины листа: на реальных листах
        # рамка местами 2px при типовой 1px, и кламп по моде оставлял в кадре серую
        # строку обводки. Клампим только недостоверные замеры (упёрлись в лимит).
        if not ed.get("t_valid", True):
            tin = mode - 1
        return int(max(0, tin))

    x0 = int(e["left"]["anchor"]) + tin_of("left") + 1 - inset
    y0 = int(e["top"]["anchor"]) + tin_of("top") + 1 - inset
    x1 = int(e["right"]["anchor"]) - tin_of("right") + inset
    y1 = int(e["bottom"]["anchor"]) - tin_of("bottom") + inset
    frame = (int(e["left"]["outer"]), int(e["top"]["outer"]),
             int(e["right"]["outer"]) + 1, int(e["bottom"]["outer"]) + 1)
    x0 = max(0, min(W - 2, x0)); y0 = max(0, min(H - 2, y0))
    x1 = max(x0 + 4, min(W, x1)); y1 = max(y0 + 4, min(H, y1))
    if fit == "square":
        s = max(x1 - x0, y1 - y0)
        cx, cy = 0.5 * (x0 + x1), 0.5 * (y0 + y1)
        nx0 = max(0, min(W - s, int(round(cx - s / 2))))
        ny0 = max(0, min(H - s, int(round(cy - s / 2))))
        x0, y0 = nx0, ny0
        x1, y1 = min(W, nx0 + s), min(H, ny0 + s)
    return x0, y0, x1, y1, frame


def rim_residue_mask(sh: Sheet, x0: int, y0: int, x1: int, y1: int,
                     cfg: Cfg) -> np.ndarray:
    """
    Маска остатков обводки ВНУТРИ реза (1 = это остаток линии, его надо заменить).

    Пиксель считается остатком, только если выполняются все условия сразу:
      * он стоит в первых cfg.rim px от края;
      * его яркость в узкой полосе вокруг уровня линии листа (0.62..1.5 * line_level)
        и он почти бесцветный (sat < 20) — это серый штрих, а не арт;
      * сосед снаружи (последний отрезанный px) такой же «линейный» — т.е. линия
        реально продолжается прямо в рез, а не просто рядом была;
      * соседи внутри (2-3 px) тёмные, на уровне канала листа — т.е. за пикселем
        не стоит арт. Это главная защита: светлый камень, упирающийся в рамку,
        не будет «вычищен».
    Решение принимается по всей стороне сразу (нужна доля cfg.rim_frac), поэтому
    одиночный светлый пиксель арта на краю не трогается, а «полоска поперёк» убирается.
    """
    h, w = y1 - y0, x1 - x0
    mask = np.zeros((h, w), bool)
    if cfg.rim <= 0:
        return mask
    g, sat = sh.gray, sh.sat
    ll, gut = sh.line_level, sh.gutter
    lo, hi = 0.62 * ll, min(200.0, 1.50 * ll + 12)
    rim = int(max(1, min(cfg.rim, min(h, w) // 3)))

    def lineish(vec):
        return (vec > lo) & (vec < hi) & (sat_vec(vec) < 20.0)

    def sat_vec_along(fixed, idxs, horiz):
        return sat[fixed, idxs[0]:idxs[1]] if horiz else sat[idxs[0]:idxs[1], fixed]

    sides = []
    for k in range(rim):
        sides.append(("y", k, +1))     # top rows, inward = +y
    for k in range(rim):
        sides.append(("y", h - 1 - k, -1))
    for k in range(rim):
        sides.append(("x", k, +1))
    for k in range(rim):
        sides.append(("x", w - 1 - k, -1))

    for axis, pos, inward in sides:
        horiz = axis == "y"
        # абсолютные координаты в листе
        ab0, ab1 = (x0, x1) if horiz else (y0, y1)
        fixed = (y0 if pos < h else y0) + pos
        fixed = (y0 + pos) if horiz else (x0 + pos)
        cur_g = g[fixed, ab0:ab1] if horiz else g[ab0:ab1, fixed]
        cur_s = sat[fixed, ab0:ab1] if horiz else sat[ab0:ab1, fixed]
        out_i = fixed - inward
        in_i = fixed + inward * 3
        if not (0 <= out_i < (sh.H if horiz else sh.W)):
            continue
        out_g = g[out_i, ab0:ab1] if horiz else g[ab0:ab1, out_i]
        in_a, in_b = sorted((fixed + inward, in_i + inward))
        in_a = max(0, min((sh.H if horiz else sh.W) - 1, in_a))
        in_b = max(in_a + 1, min((sh.H if horiz else sh.W), in_b))
        inside = g[in_a:in_b, ab0:ab1] if horiz else g[ab0:ab1, in_a:in_b]
        dark_in = inside.mean(axis=0) if horiz else inside.mean(axis=1)
        cand = (cur_g > lo) & (cur_g < hi) & (cur_s < 20.0) & \
               (np.abs(cur_g - out_g) < max(28.0, 0.5 * ll)) & \
               (out_g > lo * 0.8) & (out_g < hi * 1.6) & (dark_in < gut + 26)
        if cand.size and float(cand.mean()) >= cfg.rim_frac:
            if horiz:
                mask[pos, :] = cand
            else:
                mask[:, pos] = cand
    return mask


def apply_rim_fix(tile: np.ndarray, mask: np.ndarray) -> Tuple[np.ndarray, int]:
    """Замена помеченных пикселей на медиану 1..3 px внутрь (цвет фона плитки)."""
    if mask.sum() == 0:
        return tile, 0
    out = tile.copy()
    h, w = tile.shape[:2]
    ys, xs = np.where(mask)
    for y, x in zip(ys, xs):
        if y < h // 2:
            src = tile[min(h - 1, y + 1):min(h, y + 4), x]
        elif y >= h - h // 2:
            src = tile[max(0, y - 3):max(1, y), x]
        elif x < w // 2:
            src = tile[y, min(w - 1, x + 1):min(w, x + 4)]
        else:
            src = tile[y, max(0, x - 3):max(1, x)]
        out[y, x] = np.median(src.reshape(-1, 3), axis=0) if src.size else out[y, x]
    return out, int(mask.sum())


def despeckle(rgb: np.ndarray, thr: float = 34.0) -> np.ndarray:
    """Одиночные выбросы (шум/пыль после JPEG) — мягкая 3x3 медиана только на них."""
    u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    med = cv2.medianBlur(u8, 3)
    d = cv2.absdiff(u8, med).max(2)
    m = (d > thr).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    m = cv2.dilate(m, np.ones((2, 2), np.uint8))
    out = u8.copy()
    out[m.astype(bool)] = med[m.astype(bool)]
    return out.astype(np.float32)


def make_alpha(rgb: np.ndarray, soft: float) -> np.ndarray:
    """
    Мятая альфа: чёрный фон плитки -> прозрачно.
    Фон = уровень канала листа; альфа по яркости + насыщенности (цветной пиксель = арт).
    """
    g = cv2.cvtColor(np.clip(rgb, 0, 255).astype(np.uint8), cv2.COLOR_RGB2GRAY).astype(np.float32)
    sat = rgb.max(2) - rgb.min(2)
    base = max(6.0, float(np.percentile(g, 8)))
    a = (g - base) / max(1e-3, soft)
    a = np.clip(a, 0, 1)
    a = np.maximum(a, np.clip(sat / max(1e-3, soft * 1.6), 0, 1))
    a = cv2.morphologyEx((a > 0.25).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    a = cv2.GaussianBlur(a.astype(np.float32), (0, 0), 0.6)
    return np.clip(a, 0, 1)


UPF = {"nearest": cv2.INTER_NEAREST, "linear": cv2.INTER_LINEAR,
       "cubic": cv2.INTER_CUBIC, "lanczos": cv2.INTER_LANCZOS4}

# Пресеты пост-обработки. Порядок важен: шум давим ДО и ПОСЛЕ ресайза,
# резкость поднимаем только там, где есть настоящий контраст (grad-режим),
# иначе 150->256 вытягивает JPEG-зерно и ringing в «хрусталь».
PIPELINES = {
    # source-res denoise | target-res denoise | unsharp amount | post denoise | fit-note
    "quality": dict(pre=3.0, den=6.0, sharp=0.6, post=3.0, engine="lanczos"),
    "fast":    dict(pre=5.0, den=0.0, sharp=0.5, post=0.0, engine="lanczos"),
    "soft":    dict(pre=4.0, den=5.0, sharp=0.0, post=0.0, engine="lanczos"),
    "pixel":   dict(pre=0.0, den=0.0, sharp=0.0, post=0.0, engine="nearest"),
    "raw":     dict(pre=0.0, den=0.0, sharp=0.0, post=0.0, engine="lanczos"),
}


def nlmeans(rgb: np.ndarray, h: float) -> np.ndarray:
    """Non-local means (щадящий, edge-aware). h<=0 -> бездействие."""
    if h <= 0:
        return rgb
    u8 = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.fastNlMeansDenoisingColored(u8, None, float(h), float(h) * 1.05, 5, 17).astype(np.float32)


def unsharp(rgb: np.ndarray, amount: float, radius: float, thresh: float) -> np.ndarray:
    """Unsharp c масштабированием по локальной контрастности: плоским местам — 0, краю — всё."""
    if amount <= 0:
        return rgb
    det = rgb - cv2.GaussianBlur(rgb, (0, 0), radius)
    if thresh > 0:
        k = np.clip(np.abs(det).max(axis=2) / thresh, 0, 1)[..., None]
    else:
        k = 1.0
    return np.clip(rgb + amount * det * k, 0, 255)


def render(rgb: np.ndarray, alpha: Optional[np.ndarray], size: int, cfg: Cfg) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Апскейл + вся чистка. Всё в float32. Альфа (если есть) ресамплижится в
    premultiplied — по контуру не появляется тёмный/светлый «кант».
    """
    p = PIPELINES.get(cfg.pipeline, PIPELINES["quality"])
    pre, den, sh, post = cfg.pre_denoise if cfg.pre_denoise >= 0 else p["pre"], \
        cfg.denoise if cfg.denoise >= 0 else p["den"], \
        cfg.sharpen if cfg.sharpen >= 0 else p["sharp"], \
        cfg.post_denoise if cfg.post_denoise >= 0 else p["post"]
    eng = cfg.engine or p["engine"]
    f = UPF.get(eng, cv2.INTER_LANCZOS4)
    h0, w0 = rgb.shape[:2]
    out, a2 = rgb, alpha

    # 1) до ресайза: снять JPEG-блоки и ringing, пока пиксель = пиксель
    if pre > 0:
        out = nlmeans(out, pre)
    # 2) ресайз
    if (h0, w0) != (size, size):
        if a2 is None:
            out = cv2.resize(out, (size, size), interpolation=f)
        else:
            pm = cv2.resize(out * a2[..., None], (size, size), interpolation=f)
            a2 = np.clip(cv2.resize(a2, (size, size), interpolation=f), 0, 1)
            nz = a2 > 1e-4
            out = np.zeros_like(pm)
            out[nz] = pm[nz] / a2[nz][..., None]
            out[~nz] = 0
    if eng == "nearest":  # пиксель-арт: только quantize-безопасное сглаживание
        out = np.clip(out, 0, 255)
    # 3) после ресайза: добить «крупички» интерполяции, затем резкость, затем финальный проход
    if den > 0:
        out = nlmeans(out, den)
    out = unsharp(out, sh, cfg.sharpen_radius, cfg.sharpen_thresh)
    if post > 0:
        out = nlmeans(out, post)
    out = np.clip(out, 0, 255)
    if a2 is not None:
        a2 = cv2.GaussianBlur(a2, (0, 0), 0.35) if eng != "nearest" else a2
        a2 = np.clip(a2, 0, 1)
    return out, a2


def pad_to_square(rgb: np.ndarray, alpha: Optional[np.ndarray], size: int, mode: str) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    h, w = rgb.shape[:2]
    if mode != "pad":
        return rgb, alpha
    s = max(h, w)
    oy, ox = (s - h) // 2, (s - w) // 2
    canvas = np.zeros((s, s, 3), rgb.dtype)
    canvas[oy:oy + h, ox:ox + w] = rgb
    a2 = None
    if alpha is not None:
        c = np.zeros((s, s), alpha.dtype)
        c[oy:oy + h, ox:ox + w] = alpha
        a2 = c
    return canvas, a2


# --------------------------------------------------------------------------------------
# per-sheet pipeline
# --------------------------------------------------------------------------------------

def process_sheet(path: str, outdir: str, cfg: Cfg) -> dict:
    """Весь лист: детект -> рез -> чистка -> апскейл -> сохранение + QC-артефакты."""
    sh = build_sheet(path, cfg)
    if cfg.grid:
        manual_grid(sh, cfg.grid, cfg)
        calibrate(sh, cfg)
    else:
        find_grid(sh, cfg)
        calibrate(sh, cfg)
        if cfg.complete:
            complete_grid(sh, cfg)
    cells = find_edges(sh, cfg)
    if not cells:
        return {"sheet": sh.name, "source": path, "error": "no cells",
                "notes": sh.notes, "icons": []}

    os.makedirs(outdir, exist_ok=True)
    dbg = os.path.join(outdir, "_debug")
    if cfg.debug:
        os.makedirs(dbg, exist_ok=True)

    recs: List[dict] = []
    marks: List[tuple] = []
    zoom_items: List[tuple] = []
    for cell in cells:
        x0, y0, x1, y1, fbox = inner_box(cell, dict(sh.stroke_side, all=sh.stroke),
                                         cfg.inset, sh.W, sh.H, cfg.fit)
        w, h = x1 - x0, y1 - y0
        e = cell["edges"]
        base_rec = {"row": cell["row"], "col": cell["col"],
                    "frame": list(fbox),
                    "edge_outer": {s: e[s]["outer"] for s in e},
                    "edge_t": {s: e[s]["t"] for s in e},
                    "edge_resp": {s: e[s]["resp"] for s in e},
                    "cover": cell["cover"],
                    "why": {s: e[s]["why"] for s in e if e[s]["why"]},
                    "low_confidence": cell["low_confidence"],
                    "inferred_band": bool(cell.get("inferred"))}
        if w < 12 or h < 12:
            recs.append({**base_rec, "crop": [x0, y0, x1, y1], "status": "skipped:too_small"})
            continue
        tile = np.clip(sh.rgb[y0:y1, x0:x1], 0, 255)

        # --- QC до обработки: сколько «линейных» пикселей осталось в резе и где ------
        g_t = to_gray(tile.astype(np.uint8))
        band = np.zeros(tile.shape[:2], bool)
        band[:cfg.rim or 1, :] = True; band[-(cfg.rim or 1):, :] = True
        band[:, :cfg.rim or 1] = True; band[:, -(cfg.rim or 1):] = True
        resid = float(np.mean(band & (g_t > 0.62 * sh.line_level) & (g_t < 1.5 * sh.line_level + 12)))
        touch = float(np.mean(g_t[band] > 60))
        # арт, который остался СНАРУЖИ реза (в 2px канте: строка самой обводки):
        # нет нулей -> стоит посмотреть, не режем ли мы дальше внутренней грани
        sat_f = sh.sat
        clipped = 0.0
        for ax in range(4):
            if ax == 0:
                a, sa = sh.gray[max(0, y0 - 2):y0, x0:x1], sat_f[max(0, y0 - 2):y0, x0:x1]
            elif ax == 1:
                a, sa = sh.gray[y1:min(sh.H, y1 + 2), x0:x1], sat_f[y1:min(sh.H, y1 + 2), x0:x1]
            elif ax == 2:
                a, sa = sh.gray[y0:y1, max(0, x0 - 2):x0], sat_f[y0:y1, max(0, x0 - 2):x0]
            else:
                a, sa = sh.gray[y0:y1, x1:min(sh.W, x1 + 2)], sat_f[y0:y1, x1:min(sh.W, x1 + 2)]
            if a.size:
                clipped = max(clipped, float(np.mean((a > 105) & (sa > 40))))

        tile, n_fix = apply_rim_fix(tile, rim_residue_mask(sh, x0, y0, x1, y1, cfg))
        if cfg.specks > 0:
            tile = despeckle(tile, cfg.specks)
        tile = tile.astype(np.float32)

        alpha = make_alpha(tile, cfg.matte_soft) if cfg.matte else None
        if cfg.fit == "pad":
            tile, alpha = pad_to_square(tile, alpha, max(tile.shape[0], tile.shape[1]), "pad")
        out, out_a = render(tile, alpha, cfg.size, cfg)

        base = f"{cfg.prefix}{cell['row'] + 1:02d}_{cell['col'] + 1:02d}"
        rec = {**base_rec, "index": cell["row"] * len(sh.cols) + cell["col"],
               "name": base,
               "crop": [int(x0), int(y0), int(x1), int(y1)],
               "crop_wh": [int(w), int(h)],
               "out_wh": [int(cfg.size), int(cfg.size)],
               "scale": round(cfg.size / max(1e-6, 0.5 * (w + h)), 4),
               "rim_residue_px": round(resid, 4), "rim_fix_px": int(n_fix),
               "art_left_on_frame_px": round(clipped, 4),
               "art_touches_edge": round(touch, 4), "files": []}
        if not cfg.dry_run:
            for fmt in cfg.formats:
                fp = os.path.join(outdir, f"{base}.{fmt}")
                if fmt in ("png", "webp") and out_a is not None:
                    rgba = np.dstack([np.clip(out, 0, 255).astype(np.uint8),
                                      (out_a * 255).astype(np.uint8)])
                    Image.fromarray(rgba, "RGBA").save(fp, optimize=True)
                elif fmt == "png":
                    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").save(fp, optimize=True)
                elif fmt == "webp":
                    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").save(fp, quality=cfg.jpg_q)
                else:
                    Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), "RGB").save(
                        fp, quality=cfg.jpg_q, subsampling=0)
                rec["files"].append(os.path.relpath(fp, outdir))
            zoom_items.append((base, x0, y0, x1, y1, cell))
        recs.append(rec)
        marks.append((fbox, (x0, y0, x1, y1), cell["low_confidence"]))

    info = {"sheet": sh.name, "source": os.path.relpath(path, os.getcwd()),
            "sheet_size": [int(sh.W), int(sh.H)], "tile": [int(sh.tile_w), int(sh.tile_h)],
            "stroke_px": int(sh.stroke), "stroke_px_by_side": {k: int(v) for k, v in sh.stroke_side.items()}, "line_level": round(sh.line_level, 1),
            "gutter_level": round(sh.gutter, 1),
            "grid_rows": [[round(a, 2), round(b, 2)] for a, b in sh.rows],
            "grid_cols": [[round(a, 2), round(b, 2)] for a, b in sh.cols],
            "rows": len(sh.rows), "cols": len(sh.cols),
            "out_size": [cfg.size, cfg.size], "pipeline": cfg.pipeline,
            "engine": cfg.engine or PIPELINES[cfg.pipeline]["engine"], "fit": cfg.fit,
            "inset": cfg.inset, "matte": cfg.matte,
            "sharpen": cfg.sharpen, "n_icons": sum(1 for r in recs if r.get("crop")),
            "residue_max": round(max([r.get("rim_residue_px", 0) for r in recs] or [0]), 4),
            "art_left_on_frame_max": round(max([r.get("art_left_on_frame_px", 0) for r in recs] or [0]), 4),
            "art_left_on_frame_tiles": int(sum(1 for r in recs if r.get("art_left_on_frame_px", 0) > 0.15)),
            "notes": sh.notes, "icons": recs}

    if not cfg.dry_run:
        if cfg.report:
            with open(os.path.join(outdir, "manifest.json"), "w", encoding="utf-8") as f:
                json.dump(info, f, ensure_ascii=False, indent=1)
        contact = make_contact(recs, outdir, cfg)
        info["contact_sheet"] = os.path.relpath(contact, outdir) if contact else None
    if cfg.debug:
        make_overlay(sh, marks, os.path.join(dbg, "overlay.png"))
        make_edge_zoom(sh, zoom_items, os.path.join(dbg, "edge_zoom.png"))
    return info


def make_overlay(sh: Sheet, marks: List[tuple], out_path: str, width: int = 1200) -> None:
    """Сетка резов поверх листа: оранжевый = найденная рамка, голубой = фактический рез."""
    sc = float(width) / sh.W
    img = cv2.resize(np.clip(sh.rgb, 0, 255).astype(np.uint8), None, fx=sc, fy=sc,
                     interpolation=cv2.INTER_AREA).astype(np.float32) / 255.0
    for (fl, ft, fr, fb), (x0, y0, x1, y1), low in marks:
        col = (1.0, 0.25, 0.15) if low else (1.0, 0.62, 0.05)
        cv2.rectangle(img, (int(fl * sc), int(ft * sc)), (int(fr * sc), int(fb * sc)), col, 1)
        cv2.rectangle(img, (int(x0 * sc), int(y0 * sc)), (int(x1 * sc) - 1, int(y1 * sc) - 1),
                      (0.1, 0.75, 1.0), 1)
    Image.fromarray((np.clip(img, 0, 1) * 255).astype(np.uint8)).save(out_path)


def make_contact(recs: List[dict], outdir: str, cfg: Cfg, cell: int = 96) -> Optional[str]:
    items = []
    for r in recs:
        if not r.get("files"):
            continue
        fp = os.path.join(outdir, r["files"][0])
        if not os.path.exists(fp):
            continue
        im = Image.open(fp).convert("RGB").resize((cell, cell), Image.LANCZOS)
        items.append(np.asarray(im))
    if not items:
        return None
    n = len(items)
    cols = int(math.ceil(math.sqrt(n)))
    rows = int(math.ceil(n / cols))
    grid = np.full((rows * (cell + 2) + 2, cols * (cell + 2) + 2, 3), 24, np.uint8)
    for k, a in enumerate(items):
        r, c = divmod(k, cols)
        y = 2 + r * (cell + 2); x = 2 + c * (cell + 2)
        grid[y:y + cell, x:x + cell] = a
    out = os.path.join(outdir, "contact_sheet.png")
    Image.fromarray(grid).save(out)
    return out


def make_edge_zoom(sh: Sheet, items, out_path: str, zoom: int = 5, box: int = 28,
                   max_tiles: int = 8) -> None:
    """Зум УГЛОВ реза: сразу видно и остаток обводки, и срезанный обводкой арт.

    Плитка = мозаика 2x2 (верхний-левый, верхний-правый, нижний-левый, нижний-правый
    углы), каждый угол — квадрат 2*box исходных пикселей вокруг угла реза, красным
    показана линия фактического реза. Первыми идут плитки с пометками (low_confidence,
    непустой why) и с худшим покрытием линии — именно их надо проверять глазами.
    """
    def sev(it) -> tuple:
        cell = it[5]
        cov = cell.get("cover") or [1.0]
        return (not cell.get("low_confidence"), float(np.min(cov)))

    tiles = []
    for base, x0, y0, x1, y1, cell in sorted(items, key=sev)[:max_tiles]:
        cs = []
        for cx, cy in ((x0, y0), (x1, y0), (x0, y1), (x1, y1)):
            a, b = max(0, int(cx) - box), max(0, int(cy) - box)
            e, f = min(sh.W, int(cx) + box), min(sh.H, int(cy) + box)
            c = sh.rgb[b:f, a:e]
            if c.size == 0:
                cs = []
                break
            c = cv2.resize(np.clip(c, 0, 255).astype(np.uint8), None, fx=zoom, fy=zoom,
                           interpolation=cv2.INTER_NEAREST).copy()
            cv2.line(c, (0, (int(cy) - b) * zoom), (c.shape[1], (int(cy) - b) * zoom), (255, 40, 40), 1)
            cv2.line(c, ((int(cx) - a) * zoom, 0), ((int(cx) - a) * zoom, c.shape[0]), (255, 40, 40), 1)
            cs.append(c)
        if len(cs) != 4:
            continue
        hh, ww = max(c.shape[0] for c in cs), max(c.shape[1] for c in cs)
        cs = [np.pad(c, ((0, hh - c.shape[0]), (0, ww - c.shape[1]), (0, 0)),
                     constant_values=40) for c in cs]
        gap_v = np.full((hh, 3, 3), 40, np.uint8)
        top = np.concatenate([cs[0], gap_v, cs[1]], 1)
        bot = np.concatenate([cs[2], gap_v, cs[3]], 1)
        tiles.append(np.concatenate([top, np.full((3, top.shape[1], 3), 40, np.uint8), bot], 0))
    if not tiles:
        return
    hm = max(t.shape[0] for t in tiles)
    tiles = [np.pad(t, ((0, hm - t.shape[0]), (0, 5), (0, 0)), constant_values=40) for t in tiles]
    Image.fromarray(np.concatenate(tiles, 1).astype(np.uint8)).save(out_path)


# --------------------------------------------------------------------------------------
# io / cli
# --------------------------------------------------------------------------------------

IMG_EXT = (".png", ".jpg", ".jpeg", ".webp", ".tif", ".tiff", ".bmp", ".tga")


def collect_inputs(inputs: Sequence[str], recursive: bool = True) -> List[str]:
    files: List[str] = []
    for p in inputs:
        if os.path.isdir(p):
            pat = os.path.join(p, "**", "*") if recursive else os.path.join(p, "*")
            for f in sorted(glob.glob(pat, recursive=recursive)):
                if f.lower().endswith(IMG_EXT):
                    files.append(f)
        elif p.lower().endswith(IMG_EXT):
            files.append(p)
        else:
            for f in sorted(glob.glob(p)):
                if f.lower().endswith(IMG_EXT):
                    files.append(f)
    seen, out = set(), []
    for f in files:
        if f not in seen:
            seen.add(f); out.append(f)
    return out


def parse_cli(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("inputs", nargs="+", help="листы или папки с листами")
    ap.add_argument("-o", "--out", default="icons_out", help="папка результата (по листам: out/<sheet>/)")
    ap.add_argument("--size", type=int, default=256, help="целевой размер, px (256)")
    ap.add_argument("--inset", type=int, default=0,
                    help="запас внутрь от внутренней грани обводки, px (0 = строго по грани)")
    ap.add_argument("--rim", type=int, default=2, help="ширина зоны точечной чистки канта, px (0 = выкл)")
    ap.add_argument("--rim-frac", type=float, default=0.30, help="доля стороны с остатками обводки, чтобы её подменить")
    ap.add_argument("--pipeline", default="quality", choices=list(PIPELINES.keys()),
                    help="пресет чистки+апскейла: quality|fast|soft|pixel|raw")
    ap.add_argument("--pre-denoise", type=float, default=-1.0, help="h денойза в исходнике (-1 = из пресета)")
    ap.add_argument("--denoise", type=float, default=-1.0, help="h денойза после ресайза (-1 = из пресета, 0 = выкл)")
    ap.add_argument("--post-denoise", type=float, default=-1.0, help="h финального денойза (-1 = из пресета)")
    ap.add_argument("--sharpen-thresh", type=float, default=7.0, help="порог контраста для unsharp")
    ap.add_argument("--specks", type=float, default=0.0, help="подавление пылинок, порог отклонения от 3x3 медианы (0 = выкл)")
    ap.add_argument("--sharpen", type=float, default=-1.0, help="unsharp amount (-1 = из пресета, 0 = выкл)")
    ap.add_argument("--engine", default="", choices=[""] + list(UPF.keys()),
                    help="ядро апскейла (пусто = взять из --pipeline)")
    ap.add_argument("--fit", default="stretch", choices=["stretch", "square", "pad"],
                    help="как собирать квадрат: stretch — тянем как есть (нет пустот), "
                         "square — берём больший квадрат, pad — вписываем в квадрат с полями")
    ap.add_argument("--matte", action="store_true", help="вырезать чёрный фон плитки в альфу (PNG RGBA)")
    ap.add_argument("--matte-soft", type=float, default=22.0, help="мягкость альфы")
    ap.add_argument("--line-thresh", type=float, default=0.0, help="порог отклика обводки (0 = авто по шуму листа)")
    ap.add_argument("--line-cover", type=float, default=0.34, help="мин. покрытие линии при поиске сетки")
    ap.add_argument("--search-win", type=int, default=6, help="мин. окно локального поиска ребра (авто = 7%% плитки)")
    ap.add_argument("--grid-cover", type=float, default=0.45, help="доля сечения, покрытая линиями, для поиска сетки")
    ap.add_argument("--formats", default="png", help="png, jpg, webp (через запятую)")
    ap.add_argument("--jpg-quality", type=int, default=95)
    ap.add_argument("--grid", default="", help="принудительная сетка 'СТРОКИxСТОЛБЦЫ' (фолбэк, если обводку не найти)")
    ap.add_argument("--no-complete", action="store_true", help="не достраивать пропущенные полосы решётки")
    ap.add_argument("--prefix", default="", help="префикс имён файлов")
    ap.add_argument("--no-recursive", action="store_true")
    ap.add_argument("--debug", action="store_true", help="overlay + edge_zoom в _debug/")
    ap.add_argument("--no-report", action="store_true")
    ap.add_argument("--dry-run", action="store_true", help="только геометрия и QC, без записи иконок")
    return ap.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_cli(argv)
    cfg = Cfg(size=args.size, inset=args.inset, rim=args.rim, pipeline=args.pipeline,
              pre_denoise=args.pre_denoise, denoise=args.denoise, post_denoise=args.post_denoise,
              sharpen=args.sharpen, sharpen_thresh=args.sharpen_thresh, engine=args.engine, fit=args.fit, matte=args.matte,
              specks=args.specks, rim_frac=args.rim_frac, grid=args.grid, complete=not args.no_complete,
              matte_soft=args.matte_soft, line_thresh=args.line_thresh,
              line_cover=args.line_cover, search_win=args.search_win, grid_cover=args.grid_cover, prefix=args.prefix,
              formats=tuple(f.strip().lower() for f in args.formats.split(",") if f.strip()),
              jpg_q=args.jpg_quality, debug=args.debug, dry_run=args.dry_run,
              report=not args.no_report)
    files = collect_inputs(args.inputs, recursive=not args.no_recursive)
    if not files:
        print("нет входных изображений"); return 2
    multi = len(files) > 1 or os.path.isdir(args.inputs[0])
    total = 0
    for f in files:
        sub = os.path.splitext(os.path.basename(f))[0] if multi else ""
        outdir = os.path.join(args.out, sub) if sub else args.out
        try:
            info = process_sheet(f, outdir, cfg)
        except Exception as e:  # noqa: BLE001
            print(f"[!] {f}: {type(e).__name__}: {e}")
            if os.environ.get("SLICE_DEBUG_TB"):
                import traceback; traceback.print_exc()
            continue
        if info.get("error"):
            print(f"[!] {f}: {info['error']}. {info.get('notes')}")
            continue
        icons = [r for r in info.get("icons", []) if r.get("crop")]
        low = [r for r in icons if r.get("low_confidence")]
        resid = [r for r in icons if r.get("rim_residue_px", 0) > 0.02]
        total += len(icons)
        tile = info.get("tile") or [0, 0]
        print(f"[{info['sheet']}] сетка {info.get('rows')}x{info.get('cols')}, "
              f"плитка ~{tile[0]}x{tile[1]}, штрих {info.get('stroke_px')}px "
              f"{info.get('stroke_px_by_side', '')} -> "
              f"{len(icons)} иконок {cfg.size}px"
              + (f", low-conf {len(low)}" if low else "")
              + (f", residue {len(resid)}" if resid else ""))
        for r in low:
            print(f"   ? r{r['row']+1}c{r['col']+1}: cover={r['cover']} — проверь overlay.png")
    print(f"итого {total} иконок -> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
