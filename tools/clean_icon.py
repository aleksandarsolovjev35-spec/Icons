#!/usr/bin/env python3
"""
clean_icon.py — убрать JPEG-шум с иконки, не размывая рисунок.

Что чистим (по результатам tools/analyze_noise.py на этой коллекции):
  1. ХРОМА-ШУМ — цветные крапинки (зеленоватые/сиреневые) на ровных участках.
     В JPEG хрома и так лежит в половинном разрешении (4:2:0) и квантуется
     грубее яркости, поэтому её можно сглаживать смело, без потери деталей:
     рисунок почти целиком держится на яркости.
  2. ЯРКОСТНЫЙ ШУМ и БЛОЧНОСТЬ 8×8 — дрожание яркости на плато: аккуратный
     медианный фильтр + guided-фильтр малого радиуса, ТОЛЬКО в плоских зонах
     (маска по градиенту яркости), поэтому контуры и мелкие детали не мылятся.
  3. ЗВОН у контуров (mosquito noise) — лёгкое подавление в узкой полосе
     вокруг сильных кромок, край остаётся резким.

    python3 tools/clean_icon.py in.png -o out.png
    python3 tools/clean_icon.py in.png -o out.png --chroma 1.0 --luma 0.8
    python3 tools/clean_icon.py in.png -o out.png --no-luma      # только цвет
"""
from __future__ import annotations

import argparse
import os

import cv2
import numpy as np


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
    # мягко: только явный мусор, рисунок почти не меняется
    "gentle": dict(chroma=1.0, cmed=5, csig=1.4, passes=1, mthr=10, mrad=3,
                   meps=30, unsharp=0.0),
    # по умолчанию: цвет чистится сильно, яркость — 2 прохода по плоским зонам,
    # контуры возвращаются лёгким unsharp только по маске краёв
    "strong": dict(chroma=1.0, cmed=7, csig=2.5, passes=2, mthr=14, mrad=6,
                   meps=120, unsharp=0.35),
    # максимум: для сильно пережатых листов (следить за мелкими деталями)
    "max":    dict(chroma=1.0, cmed=9, csig=3.5, passes=3, mthr=16, mrad=10,
                   meps=300, unsharp=0.6),
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


def clean_preset(img: np.ndarray, preset: str = "strong") -> np.ndarray:
    """Чистка по пресету: хрома — медиана + сглаживание, яркость — guided
    по маске плоских зон в несколько проходов, затем лёгкий возврат резкости
    только на контурах (без гало: маска краёв узкая)."""
    p = dict(PRESETS[preset])
    k = float(np.clip(p["chroma"], 0.0, 2.0))
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    L, A, B = [c.astype(np.float32) for c in cv2.split(lab)]

    # --- 1. хрома
    A_med = cv2.medianBlur(np.clip(A, 0, 255).astype(np.uint8), p["cmed"]).astype(np.float32)
    B_med = cv2.medianBlur(np.clip(B, 0, 255).astype(np.uint8), p["cmed"]).astype(np.float32)
    A_t = 0.45 * A_med + 0.55 * cv2.GaussianBlur(A, (0, 0), p["csig"])
    B_t = 0.45 * B_med + 0.55 * cv2.GaussianBlur(B, (0, 0), p["csig"])
    A = A + (A_t - A) * k
    B = B + (B_t - B) * k

    # --- 2. яркость: guided-фильтр только в плоских зонах, несколько проходов
    for _ in range(int(p["passes"])):
        m = flat_mask(L, p["mthr"], 1)
        L8 = np.clip(L, 0, 255).astype(np.uint8)
        if hasattr(cv2, "ximgproc"):
            L_g = cv2.ximgproc.guidedFilter(L8, L8, int(p["mrad"]),
                                            float(p["meps"])).astype(np.float32)
        else:
            L_g = cv2.bilateralFilter(L, int(p["mrad"]), 22, 3.0)
        L_m = cv2.medianBlur(L8, 3).astype(np.float32)
        L_s = 0.35 * L_m + 0.65 * L_g
        L = L + (L_s - L) * np.clip(m, 0.0, 1.0)

    # --- 3. возврат резкости по контурам (узкая маска — без ореолов)
    if p["unsharp"] > 0:
        hp = L - cv2.GaussianBlur(L, (0, 0), 1.5)
        em = np.clip((_edges(L) - 6.0) / 24.0, 0.0, 1.0)
        L = L + hp * float(p["unsharp"]) * em

    return cv2.cvtColor(np.clip(np.stack([L, A, B], -1), 0, 255).astype(np.uint8),
                        cv2.COLOR_LAB2BGR)


# -------------------------------------------------------------------- metrics
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
        "звон у контуров": round(_ring(L, edge), 2),
        "средняя яркость": round(float(L.mean()), 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Чистка иконок от JPEG-шума")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--preset", default="strong", choices=sorted(PRESETS),
                    help="готовый режим чистки (по умолчанию strong)")
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
        res = clean_preset(img, args.preset) if args.preset else \
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
