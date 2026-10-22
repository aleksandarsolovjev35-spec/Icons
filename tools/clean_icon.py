#!/usr/bin/env python3
"""
clean_icon.py — убрать JPEG-шум с иконки, не размывая рисунок.

Что чистим (по результатам tools/analyze_noise.py на этой коллекции):
  1. ХРОМА-ШУМ — цветные крапинки (зеленоватые/сиреневые) на ровных участках.
     В JPEG хрома и так лежит в половинном разрешении (4:2:0) и квантуется
     грубее яркости, поэтому её можно сглаживать смело, без потери деталей:
     рисунок почти целиком держится на яркости.
  2. ШУМ НА ЗАЛИВКАХ и БЛОЧНОСТЬ 8×8 — «сыпь» там, где по рисунку ровный цвет.
     Чистится NLM (non-local means) по каналам LAB: пиксель заменяется средним
     по похожим фрагментам кадра, поэтому шум уходит, а кромки, скругления и
     плавные тени остаются на месте — в отличие от mean shift/медианы, которые
     схлопывают заливку в плато и дают ступеньки и «надрезы» на гранях.
     Альтернативные движки: `clean_fill_freq` (частотный, пресет `freq`) и
     `clean_fill_ms` (mean shift, ручной режим --protect/--iters).
  3. ЗВОН у контуров (mosquito noise) — лёгкое подавление в узкой полосе
     вокруг сильных кромок, край остаётся резким.

    python3 tools/clean_icon.py in.png -o out.png
    python3 tools/clean_icon.py in.png -o out.png --chroma 1.0 --luma 0.8
    python3 tools/clean_icon.py in.png -o out.png --no-luma      # только цвет
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _console import setup as _console_setup  # noqa: E402


# --------------------------------------------------------------------- masks
def _edges(L: np.ndarray) -> np.ndarray:
    g = L.astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    return np.maximum(lap, np.hypot(gx, gy))


def flat_mask(L: np.ndarray, thr: float = 10.0, grow: int = 1) -> np.ndarray:
    """1 = ровная зона (шум можно давить), 0 = контур/деталь (не трогаем)."""
    m = (_edges(L) < thr).astype(np.float32)
    if grow:
        m = cv2.erode(m, np.ones((2 * grow + 1, 2 * grow + 1), np.uint8))
    return cv2.GaussianBlur(m, (0, 0), 1.2)


def edge_band(L: np.ndarray, lo: float = 11.0, hi: float = 30.0) -> np.ndarray:
    """Мягкая полоса вокруг заметных, но не «жёстких» переходов (звон)."""
    e = _edges(L)
    m = np.clip((e - lo) / max(hi - lo, 1e-3), 0.0, 1.0)
    return cv2.GaussianBlur(m, (0, 0), 0.8) * 0.5


# ---------------------------------------------------------------------- presets
PRESETS = {
    # detail — сколько исходной фактуры возвращается поверх чистки (0..1):
    # в ровных заливках возвращается мало, у деталей — больше.
    "soft":     dict(hL=8.0, hC=4.0, t=7, s=15, soft=0.0, detail=0.95, flat_T=10.0),
    "normal":   dict(hL=14.0, hC=5.0, t=7, s=15, soft=0.0, detail=0.90, flat_T=12.0),
    "strong":   dict(hL=20.0, hC=6.0, t=7, s=15, soft=0.0, detail=0.75, flat_T=14.0),
    "max":      dict(hL=28.0, hC=8.0, t=9, s=21, soft=0.0, detail=0.55, flat_T=16.0),
    # частотный способ (альтернатива)
    "freq":     dict(T=(6.0, 4.0, 4.0), soft=0.0, detail=0.0),
}


# ------------------------------------------------------------------- cleaning
def clean(img: np.ndarray, chroma: float = 1.0, luma: float = 0.8,
          edge_thr: float = 10.0, protect: int = 1, ringing: bool = True,
          preset: str | None = None) -> np.ndarray:
    if preset:
        return clean_preset(img, preset)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = [c.astype(np.float32) for c in cv2.split(lab)]
    m = flat_mask(L, edge_thr, protect)

    # --- 1. хрома: медиана 5×5 (давит крапинки, не размазывая цветовые зоны),
    #     плюс лёгкое сглаживание; сила зависит от плоскости участка
    A8 = np.clip(A, 0, 255).astype(np.uint8)
    B8 = np.clip(B, 0, 255).astype(np.uint8)
    A_med = cv2.medianBlur(A8, 5).astype(np.float32)
    B_med = cv2.medianBlur(B8, 5).astype(np.float32)
    A_mix = 0.5 * A_med + 0.5 * cv2.GaussianBlur(A, (0, 0), 1.4)
    B_mix = 0.5 * B_med + 0.5 * cv2.GaussianBlur(B, (0, 0), 1.4)
    k = float(np.clip(chroma, 0.0, 2.0))
    w_ch = np.clip(0.35 + 0.65 * m, 0.0, 1.0) * k     # у контуров чистим слабее
    A = A + (A_mix - A) * w_ch
    B = B + (B_mix - B) * w_ch

    # --- 2. яркость: только в ровных зонах
    if luma > 0:
        L8 = np.clip(L, 0, 255).astype(np.uint8)
        L_med = cv2.medianBlur(L8, 3).astype(np.float32)
        if hasattr(cv2, "ximgproc"):
            L_gui = cv2.ximgproc.guidedFilter(L8, L8, 3, 30).astype(np.float32)
        else:
            L_gui = cv2.bilateralFilter(L, 5, 18, 3.0)
        L_s = 0.6 * L_med + 0.4 * L_gui
        w = m * float(np.clip(luma, 0.0, 2.0))
        if ringing:                       # в полосе у кромок — только медиана, мягко
            w = np.maximum(w, edge_band(L) * 0.8 * float(np.clip(luma, 0.0, 2.0)))
        L = L * (1 - w) + L_s * w

    out = cv2.cvtColor(np.clip(np.stack([L, A, B], -1), 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR)
    return out


def fill_mask(L: np.ndarray, lo: float = 10.0, hi: float = 22.0,
              soft: float = 0.6) -> np.ndarray:
    """1 = ровная заливка (шум давим), 0 = контур/мелкая деталь (не трогаем).

    Важно: маска считается по АМПЛИТУДЕ мелкой структуры (|L - медиана 3×3|),
    а не по градиенту. Градиент от самого шума и «съедал» заливку: шум делал
    участок «контурным», и чистка его пропускала. Амплитуда шума ±3..8, у
    настоящей линии/кромки — 20 и больше, поэтому порог их разделяет.
    """
    L8 = np.clip(L, 0, 255).astype(np.uint8)
    hp = np.abs(L - cv2.medianBlur(L8, 3).astype(np.float32))
    m = 1.0 - np.clip((hp - lo) / max(hi - lo, 1e-6), 0.0, 1.0)
    return cv2.GaussianBlur(m, (0, 0), soft)


def guard_mask(L: np.ndarray, amp: float = 25.0, ramp: float = 6.0,
               dil: int = 1, feather: float = 0.5,
               smooth: float = 1.5) -> np.ndarray:
    """1 = защищённая зона (кромки, тонкие линии, полутоновые переходы).

    Детектор двойной, иначе шум и рисунок не разделить:
      • `amp`  — амплитуда мелкой структуры по СЫРОЙ яркости: тонкая линия и
                 жёсткая кромка дают 25+, шум 3..10;
      • `ramp` — то же, но по сглаженной яркости: шум там гаснет до ±3, а
                 мягкий полутоновой переход (градиент 6..15 на пиксель) виден.
    Пиксель считается деталью, если сработал любой из критериев.
    """
    L8 = np.clip(L, 0, 255).astype(np.uint8)
    h1 = np.abs(L - cv2.medianBlur(L8, 3).astype(np.float32))
    G = cv2.GaussianBlur(L, (0, 0), smooth)
    h2 = np.abs(G - cv2.medianBlur(np.clip(G, 0, 255).astype(np.uint8), 3).astype(np.float32))
    p = ((h1 > amp) | (h2 > ramp)).astype(np.uint8)
    if dil:
        p = cv2.dilate(p, np.ones((2 * dil + 1, 2 * dil + 1), np.uint8))
    p = p.astype(np.float32)
    if feather:                       # мягкий край зоны — без видимого стыка
        p = cv2.GaussianBlur(p, (0, 0), feather)
    return p


def soften_edges(img: np.ndarray, k: float = 0.5, band: int = 1,
                 sigma: float = 0.7) -> np.ndarray:
    """Лёгкое сглаживание ТОЛЬКО в полосе кромок: убирает «острую» иглу
    контура после чистки, заливки и детали внутри не затрагивает."""
    if k <= 0:
        return img
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    m = (_edges(g) > 14).astype(np.uint8)
    if band:
        m = cv2.dilate(m, np.ones((2 * band + 1, 2 * band + 1), np.uint8))
    m = cv2.GaussianBlur(m.astype(np.float32), (0, 0), 0.8) * float(k)
    f = img.astype(np.float32)
    blur = cv2.GaussianBlur(f, (0, 0), sigma)
    return np.clip(f + (blur - f) * m[..., None], 0, 255).astype(np.uint8)


def _shrink(d: np.ndarray, T: float, soft: float = 2.0) -> np.ndarray:
    """Мягкий порог: амплитуду меньше T считаем шумом и гасим, больше T (кромки,
    детали) оставляем как есть."""
    a = np.abs(d)
    k = np.clip((a - T) / max(soft, 1e-6), 0.0, 1.0)
    return np.sign(d) * np.maximum(a - T, 0.0) * k


def restore_detail(src: np.ndarray, cleaned: np.ndarray,
                   amount: float = 0.0, mid_thr: float = 1.0,
                   mid_span: float = 4.0, flat_keep: float = 0.30,
                   chroma: float = 0.45) -> np.ndarray:
    """Возврат исходной фактуры поверх чистки, раздельно по масштабам.

    Картинка раскладывается на две полосы:
      * средняя  `med` — то, что переживает медиану 3×3: пятна, мраморность,
        мягкая штриховка. Именно она и есть «узор», который смывает NLM.
      * тонкая   `src - med` — однопиксельная крупа и кромки.

    Средняя полоса возвращается почти всегда (в ровной заливке `med` и так
    совпадает с чисткой, так что шум не вернётся), тонкая — только там, где
    есть средняя структура: на крапинке в пустой заливке `med` ровная, вес
    нулевой, крупа остаётся вычищенной, а тонкая линия/контур возвращаются
    целиком. Цвет возвращается слабее яркости — цветной шум заметнее всего.
    """
    if amount <= 0:
        return cleaned
    lab = cv2.cvtColor(src, cv2.COLOR_BGR2LAB).astype(np.float32)
    clab = cv2.cvtColor(cleaned, cv2.COLOR_BGR2LAB).astype(np.float32)

    def _med(ch: np.ndarray) -> np.ndarray:
        return cv2.medianBlur(np.clip(ch, 0, 255).astype(np.uint8), 3).astype(np.float32)

    med_l = _med(lab[..., 0])
    mid = np.abs(med_l - cv2.GaussianBlur(med_l, (0, 0), 2.2))
    w = np.clip((mid - mid_thr) / max(mid_span, 1e-3), 0.0, 1.0)
    w = cv2.GaussianBlur(w, (0, 0), 1.5)
    k = flat_keep + (1.0 - flat_keep) * w

    out = clab.copy()
    for i, f in ((0, 1.0), (1, chroma), (2, chroma)):
        m = _med(lab[..., i])
        out[..., i] = clab[..., i] + float(amount) * (
            k * (m - clab[..., i]) + w * (lab[..., i] - m) * f)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


def _std9(a: np.ndarray, k: int = 9) -> np.ndarray:
    """Локальный разброс яркости в окне k×k — «сколько тут вообще есть структуры»."""
    m = cv2.blur(a, (k, k))
    m2 = cv2.blur(a * a, (k, k))
    return np.sqrt(np.maximum(m2 - m * m, 0.0))


def clean_flats(img: np.ndarray, cleaned: np.ndarray, T: float = 12.0,
                k: int = 9, msize: int = 5, iters: int = 2,
                feather: float = 2.0) -> np.ndarray:
    """Добить остаточный шум ТАМ, ГДЕ НЕТ УЗОРА.

    Заливки, фон и мягкие тени от узора отличаются локальным разбросом яркости
    (std в окне 9×9): у мраморности, штриховки и бликов он большой, у ровного
    цвета — маленький. Где разброс ниже T, дополнительно прогоняем медиану:
    она снимает остаточную крупу и блочность 8×8, но не трогает градиент —
    низкие частоты (сама тень) остаются на месте. Маска мягкая (`feather`),
    поэтому стыка между зонами не видно.
    """
    if T <= 0:
        return cleaned
    s = _std9(cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32), k)
    m = np.clip((T - s) / max(T * 0.45, 1e-3), 0.0, 1.0)
    m = cv2.GaussianBlur(m, (0, 0), feather)[..., None]
    flat = cleaned
    for _ in range(max(1, iters)):
        flat = cv2.medianBlur(flat, msize)
    return np.clip(flat.astype(np.float32) * m + cleaned.astype(np.float32) * (1 - m),
                   0, 255).astype(np.uint8)


def clean_fill(img: np.ndarray, preset: str = "normal") -> np.ndarray:
    """NLM (non-local means): пиксель заменяется средним по похожим на его
    окрестность фрагментам всего кадра, с весом по схожести. Шум на заливке
    уходит (похожих фрагментов много, они усредняются), а кромка и мелкая
    деталь сохраняются: похожие фрагменты ищутся по обе стороны от неё, но
    веса у непохожих близки к нулю. Плавные градиенты не «платофицируются»,
    поэтому нет ступенек и надрезов на гранях."""
    p = PRESETS[preset]
    if "hL" not in p:
        return clean_fill_freq(img, preset)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    ch = []
    for i in range(3):
        h = p["hL"] if i == 0 else p["hC"]
        ch.append(cv2.fastNlMeansDenoising(lab[..., i], None, float(h),
                                           int(p["t"]), int(p["s"])))
    res = cv2.cvtColor(np.stack(ch, -1), cv2.COLOR_LAB2BGR)
    res = restore_detail(img, res, float(p.get("detail", 0.0)))
    res = clean_flats(img, res, T=float(p.get("flat_T", 12.0)),
                      msize=int(p.get("flat_med", 5)))
    return soften_edges(res, float(p.get("soft", 0.0)))


def clean_fill_freq(img: np.ndarray, preset: str = "normal") -> np.ndarray:
    """Частотная чистка: изображение раскладывается на полосы (σ = 1, 3, 7 px),
    в каждой гасится только мелкая по амплитуде часть — «сыпь» и блочность
    JPEG, — а крупные перепады (кромки, градиенты, полутона) остаются ровно
    такими, как были. Поэтому нет ни ступенек на скруглениях, ни «надрезов»
    на гранях, которые дают методы, схлопывающие заливку в плато."""
    p = PRESETS[preset]
    T = tuple(p.get("T", (6.0, 4.0, 4.0)))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L0 = lab[..., 0].astype(np.float32)
    A0 = lab[..., 1].astype(np.float32)
    B0 = lab[..., 2].astype(np.float32)

    # полосы: (L − σ1), (σ1 − σ3), (σ3 − σ7); основа — σ7
    G = [L0] + [cv2.GaussianBlur(L0, (0, 0), s) for s in (1.0, 3.0, 7.0)]
    L = G[-1].copy()
    for i in range(3):
        L = L + _shrink(G[i] - G[i + 1], T[i])

    # хрома: медиана + сглаживание (зоны цвета не размазываются)
    A = 0.35 * cv2.medianBlur(np.clip(A0, 0, 255).astype(np.uint8), 5).astype(np.float32) \
        + 0.65 * cv2.GaussianBlur(A0, (0, 0), 1.5)
    B = 0.35 * cv2.medianBlur(np.clip(B0, 0, 255).astype(np.uint8), 5).astype(np.float32) \
        + 0.65 * cv2.GaussianBlur(B0, (0, 0), 1.5)

    res = cv2.cvtColor(np.clip(np.stack([L, A, B], -1), 0, 255).astype(np.uint8),
                       cv2.COLOR_LAB2BGR)
    return soften_edges(res, float(p.get("soft", 0.0)))


def clean_fill_ms(img: np.ndarray, preset: str = "normal") -> np.ndarray:
    """Mean shift по LAB: шум на заливках «схлопывается» в один тон, кромки
    остаются резкими (режим берётся по моде, а не среднее). Затем результат
    подмешивается ТОЛЬКО вне защищённой зоны — пиксели кромок, тонких линий и
    полутоновых переходов остаются байт-в-байт исходными."""
    p = PRESETS[preset]
    lab8 = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    cur = lab8
    for _ in range(int(p["it"])):
        cur = cv2.pyrMeanShiftFiltering(cur, int(p["sp"]), int(p["sr"]))
    L0 = lab8[..., 0].astype(np.float32)
    A0 = lab8[..., 1].astype(np.float32)
    B0 = lab8[..., 2].astype(np.float32)
    w = 1.0 - guard_mask(L0, p["amp"], p["ramp"], int(p["dil"]), float(p["feather"]))
    Lm, Am, Bm = (cur[..., i].astype(np.float32) for i in range(3))
    out = np.stack([L0 + (Lm - L0) * w,
                    A0 + (Am - A0) * w,
                    B0 + (Bm - B0) * w], -1)
    res = cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)
    return soften_edges(res, float(p.get("soft", 0.0)))


def clean_preset(img: np.ndarray, preset: str = "normal") -> np.ndarray:
    """Чистка по пресету: хрома — медиана + сглаживание, яркость — guided
    по маске плоских зон в несколько проходов, затем лёгкий возврат резкости
    только на контурах (без гало: маска краёв узкая)."""
    p = dict(PRESETS[preset])
    amt = float(np.clip(p["amount"], 0.0, 1.0))     # сила чистки яркости
    k = float(np.clip(p.get("ck", amt), 0.0, 1.0))  # сила чистки цвета
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = [c.astype(np.float32) for c in cv2.split(lab)]

    # --- 1. хрома: медиана + сглаживание (цветовые зоны не размазываются)
    A_med = cv2.medianBlur(np.clip(A, 0, 255).astype(np.uint8), p["cmed"]).astype(np.float32)
    B_med = cv2.medianBlur(np.clip(B, 0, 255).astype(np.uint8), p["cmed"]).astype(np.float32)
    A = A + (0.45 * A_med + 0.55 * cv2.GaussianBlur(A, (0, 0), p["csig"]) - A) * k
    B = B + (0.45 * B_med + 0.55 * cv2.GaussianBlur(B, (0, 0), p["csig"]) - B) * k

    # --- 2. яркость: раскраска только в заливках, края держим
    #     bilateral убирает крапинки (малый пространственный радиус — тонкие
    #     линии не разъезжаются), guided добивает остаток мягко.
    for _ in range(int(p["passes"])):
        w = fill_mask(L, p["lo"], p["hi"]) * amt
        Lb = cv2.bilateralFilter(L, int(p["d"]), float(p["sc"]), float(p["ss"]))
        L = L + (Lb - L) * w
        Lg = cv2.ximgproc.guidedFilter(L, L, int(p["grad"]), float(p["geps"])) \
            if hasattr(cv2, "ximgproc") else cv2.bilateralFilter(L, 5, 12, 1.0)
        L = L + (Lg - L) * w

    return cv2.cvtColor(np.clip(np.stack([L, A, B], -1), 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


# -------------------------------------------------------------------- metrics
def fill_noise(img: np.ndarray) -> float:
    """Робастная оценка шума ИМЕННО на заливках: сигма мелкой структуры
    (|L − медиана 3×3|) по пикселям, где эта структура слабая."""
    L = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    hp = np.abs(L - cv2.medianBlur(np.clip(L, 0, 255).astype(np.uint8), 3).astype(np.float32))
    sel = hp < 12.0
    return round(float(hp[sel].mean()), 2)


def _ring(L: np.ndarray, edge: np.ndarray) -> float:
    near = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~edge
    return float(np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.0))[near].mean())


def metrics(img: np.ndarray) -> dict:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    flat = flat_mask(L) > 0.6
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(y, cv2.CV_32F))
    edge = lap > 25

    def md(c, k=5):
        u8 = np.clip(c, 0, 255).astype(np.uint8)
        return np.abs(c - cv2.medianBlur(u8, k).astype(np.float32))

    return {
        "цветной шум (a/b)": round(float((md(A)[flat].mean() + md(B)[flat].mean()) / 2), 2),
        "шум яркости": round(float(md(L)[flat].mean()), 2),
        "резкость контуров": round(float(lap[edge].mean()), 1),
        "шум на заливках": fill_noise(img),
        "звон у контуров": round(_ring(L, edge), 2),
        "средняя яркость": round(float(L.mean()), 1),
    }


def main() -> int:
    _console_setup()
    ap = argparse.ArgumentParser(description="Чистка иконок от JPEG-шума")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--preset", default="normal", choices=sorted(PRESETS),
                    help="готовый режим чистки (по умолчанию normal)")
    ap.add_argument("--protect", type=int, default=None,
                    help="ширина защитной зоны вокруг деталей в px (0..3)")
    ap.add_argument("--iters", type=int, default=None,
                    help="сколько проходов mean shift (1..4)")
    ap.add_argument("--soft-edges", type=float, default=None,
                    help="смягчение кромок 0..1 (0 — жёсткие)")
    ap.add_argument("--detail", type=float, default=None,
                    help="возврат исходной фактуры 0..1 (0 — только чистка)")
    ap.add_argument("--flat-T", type=float, default=None,
                    help="порог 'ровная зона' 0..40: больше — чистим шире (по умолчанию из пресета)")
    ap.add_argument("--hL", type=float, default=None,
                    help="сила чистки яркости 1..40 (по умолчанию из пресета)")
    ap.add_argument("--chroma", type=float, default=1.0, help="сила чистки цвета 0..2")
    ap.add_argument("--luma", type=float, default=0.8, help="сила чистки яркости 0..2")
    ap.add_argument("--edge-thr", type=float, default=10.0)
    ap.add_argument("--no-ringing", action="store_true")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="во сколько раз увеличить результат (nearest, по умолчанию 1:1)")
    args = ap.parse_args()

    for src in args.images:
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"не читается: {src}")
        if args.protect is not None or args.iters is not None \
                or args.soft_edges is not None or args.detail is not None \
                or args.hL is not None or args.flat_T is not None:
            p = dict(PRESETS[args.preset])
            if args.protect is not None:
                p["dil"] = args.protect
                p["feather"] = 0.6 if args.protect >= 2 else 0.4
            if args.iters is not None:
                p["it"] = args.iters
            if args.soft_edges is not None:
                p["soft"] = args.soft_edges
            if args.detail is not None:
                p["detail"] = args.detail
            if args.hL is not None:
                p["hL"] = args.hL
            if args.flat_T is not None:
                p["flat_T"] = args.flat_T
            PRESETS["_custom"] = p
            if "hL" in p or "T" in p:
                res = clean_fill(img, "_custom")
            else:
                res = clean_fill_ms(img, "_custom")
        else:
            res = clean_fill(img, args.preset) if args.preset else \
                clean(img, args.chroma, args.luma, args.edge_thr,
                      ringing=not args.no_ringing)
        if args.out and len(args.images) == 1:
            dst = args.out
        else:
            base, ext = os.path.splitext(src)
            dst = f"{base}_clean{ext or '.png'}"
        if args.scale != 1.0:
            res = cv2.resize(res, None, fx=args.scale, fy=args.scale,
                             interpolation=cv2.INTER_NEAREST)
        os.makedirs(os.path.dirname(dst) or ".", exist_ok=True)
        cv2.imwrite(dst, res)
        print(f"{src} -> {dst}")
        before, after = metrics(img), metrics(res)
        for k in before:
            print(f"   {k:22s} {before[k]:>7}  ->  {after[k]:>7}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
