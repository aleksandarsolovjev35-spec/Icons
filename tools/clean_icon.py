#!/usr/bin/env python3
"""
clean_icon.py — убрать JPEG-шум с иконки, не размывая рисунок.

Что чистим (по результатам tools/analyze_noise.py на этой коллекции):
  1. ХРОМА-ШУМ — цветные крапинки (зеленоватые/сиреневые) на ровных участках.
     В JPEG хрома и так лежит в половинном разрешении (4:2:0) и квантуется
     грубее яркости, поэтому её можно сглаживать смело, без потери деталей:
     рисунок почти целиком держится на яркости.
  2. ШУМ НА ЗАЛИВКАХ и БЛОЧНОСТЬ 8×8 — «сыпь» там, где по рисунку ровный цвет.
     Лечится mean shift по LAB: шум схлопывается в один тон, кромки остаются
     резкими. Результат подмешивается только ВНЕ защищённой зоны (детектор
     деталей — по амплитуде мелкой структуры), поэтому линии и полутона
     остаются исходными (Δ ≤ 2 из 255).
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
    # сильная защита: чистим только глубокие заливки
    "gentle": dict(it=1, sp=2, sr=10, amp=28, ramp=7, dil=2),
    # по умолчанию: заливки выравниваются, детали не сдвигаются (Δ ≤ 2)
    "normal": dict(it=2, sp=2, sr=10, amp=25, ramp=6, dil=1),
    # чуть сильнее по заливкам
    "strong": dict(it=3, sp=2, sr=10, amp=25, ramp=6, dil=1),
    # максимум: защита минимальна
    "max":    dict(it=3, sp=2, sr=10, amp=20, ramp=5, dil=0),
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
               dil: int = 1, smooth: float = 1.5) -> np.ndarray:
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
    return p.astype(np.float32)


def clean_fill(img: np.ndarray, preset: str = "normal") -> np.ndarray:
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
    w = 1.0 - guard_mask(L0, p["amp"], p["ramp"], int(p["dil"]))
    Lm, Am, Bm = (cur[..., i].astype(np.float32) for i in range(3))
    out = np.stack([L0 + (Lm - L0) * w,
                    A0 + (Am - A0) * w,
                    B0 + (Bm - B0) * w], -1)
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)


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
    ap = argparse.ArgumentParser(description="Чистка иконок от JPEG-шума")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--preset", default="normal", choices=sorted(PRESETS),
                    help="готовый режим чистки (по умолчанию normal)")
    ap.add_argument("--protect", type=int, default=None,
                    help="ширина защитной зоны вокруг деталей в px (0..3)")
    ap.add_argument("--iters", type=int, default=None,
                    help="сколько проходов mean shift (1..4)")
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
        if args.protect is not None or args.iters is not None:
            p = dict(PRESETS[args.preset])
            if args.protect is not None:
                p["dil"] = args.protect
            if args.iters is not None:
                p["it"] = args.iters
            PRESETS["_custom"] = p
            res = clean_fill(img, "_custom")
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
