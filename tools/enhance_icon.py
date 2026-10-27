#!/usr/bin/env python3
"""
enhance_icon.py — довести одиночную иконку до товарного вида.

На входе маленький JPEG из игры или со стока (56×56: шум, артефакты JPEG,
по углам — обрезки рамки слота), на выходе чистая иконка 512×512, которую не
стыдно поставить в магазин, в вики или в превью набора.

Почему не «resize ×8»: интерполяция мылит кромки и растягивает JPEG-крупу в
крупные пятна, а тёмный фон после усиления контраста идёт полосами. Здесь
четыре шага, каждый лечит свою проблему.

  1. РЕМОНТ КРОМКИ (`clean_border`). У вырезанной иконки по периметру часто
     остаются пиксели рамки листа (светло-серые, обесцвеченные) — апскейл
     раздувает их в яркую кайму. Ищем такие пиксели только у самой кромки,
     проверяем, что они «серые и яркие» (S < 60, V > 90), и заливаем цветом
     соседнего фона. Углы 4×4 чистим всегда: там рамка сходится и цвет
     смазан больше всего.

  2. АПСКЕЙЛ (`upscale`) — Real-ESRGAN ×4 (`realesrgan-x4plus-anime`, он
     обучен на иллюстрациях и ДОСТРАИВАЕТ контуры, а не размазывает их).
     Для иконки 56 px делается два прохода (56 -> 224 -> 896): первый держит
     форму, второй уточняет градиенты и блики. Второй проход включается
     только если после первого кадр меньше половины нужного размера — иначе
     он тратил бы минуту ради пикселей, которые уйдут в даунскейл.
     Всё, что больше `safe` (256 px) по стороне, режется на куски с
     перехлёстом (`sr_tiled`): ncnn держит блобы пропорционально площади
     входа, и на кадре 208x208 модель съедает ~2.7 ГБ — машину с 4 ГБ просто
     убивает OOM. Шов между кусками гасят плавные веса.

  3. ЧИСТКА ФОНА (`clean_background`). Сеть местами поднимает старую
     JPEG-крупу в тёмном фоне, а усиление контраста превращает её в
     полосы. В тёмных участках без деталей (низкая яркость + низкая
     локальная амплитуда) яркость пересчитывается NLM — фон становится
     ровным, золото и камень не затрагиваются.

  4. ПОЛИРОВКА (`polish`). После сети картинка «плоская»: локальный
     контраст в средних тонах, два unsharp разного радиуса, мягкое
     свечение по ярким пикселям (золото и самоцвет начинают гореть),
     вибранс, тепло и виньетка. В тенях контраст ослаблен, иначе фон
     идёт кольцами; насыщенность поднимается вибрансом, иначе уже
     насыщенные камни выкручиваются в кислотный цвет.

Плюс `sheet()` — превью «как было / как стало» в одном файле.

    python3 tools/enhance_icon.py icon.jpg
    python3 tools/enhance_icon.py icon.jpg --sizes 256 512 1024
    python3 tools/enhance_icon.py icon.jpg --variant rich       # смелее
    python3 tools/enhance_icon.py icon.jpg --variant faithful   # ближе к исходнику
    python3 tools/enhance_icon.py icon.jpg --no-sr              # без сети, только чистка

Результат: `enhanced/<имя>_<размер>.png` + `enhanced/<имя>_sravnenie.png`.
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _console import setup as _console_setup  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


# ------------------------------------------------------------------ 1. border
def clean_border(img: np.ndarray, ring: int = 3, corner: int = 4) -> np.ndarray:
    """Убрать остатки рамки листа по кромке.

    Светло-серая линия рамки в HSV почти обесцвечена (S < 60) и заметно
    ярче фона иконки (V > 90), поэтому её видно даже на тёмно-красном фоне.
    Такие пиксели есть только в полосе `ring` у края — там их и меняем на
    медиану соседнего фона (медиана, а не среднее: рядом может лежать
    золото, и среднее подмешает его в фон).
    """
    out = img.copy()
    h, w = img.shape[:2]
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    s, v = hsv[..., 1], hsv[..., 2]

    near = np.zeros((h, w), np.uint8)
    near[:ring, :] = 1
    near[h - ring:, :] = 1
    near[:, :ring] = 1
    near[:, w - ring:] = 1
    hit = ((near > 0) & (s < 60) & (v > 90)).astype(np.uint8)
    if hit.any():
        bg = np.median(img[(near == 0)].reshape(-1, 3), axis=0)
        out[hit > 0] = np.clip(bg, 0, 255).astype(np.uint8)

    # углы рамки смазаны сильнее всего — их меняем всегда
    if corner:
        bg = np.array([2, 2, 2], np.uint8)
        for ys in (slice(0, corner), slice(h - corner, h)):
            for xs in (slice(0, corner), slice(w - corner, w)):
                out[ys, xs] = bg
    return out


# ---------------------------------------------------------------------- 2. SR
def _sr_once(img: np.ndarray, model: str, threads: int) -> np.ndarray:
    import sr_ncnn
    big, _ = sr_ncnn._sr(img, model, "cpu", threads, False)
    return big


def sr_tiled(img: np.ndarray, model: str = "realesrgan-x4plus-anime",
             scale: int = 4, tile: int = 96, pad: int = 16,
             threads: int = 2, safe: int = 256) -> np.ndarray:
    """Апскейл кадра кусками с перехлёстом.

    Зачем куски: ncnn держит промежуточные блобы пропорционально площади
    входа, и уже на кадре 208x208 Real-ESRGAN x4 съедает ~2.7 ГБ — машину с
    4 ГБ просто убивает OOM (проверено). Поэтому всё, что больше `safe`
    пикселей по стороне, режется на куски, а в кадр идёт только их середина:
    сеть на краю куска не имеет контекста и врёт, поэтому куски берутся с
    полями `pad`, и соседние перекрываются плавными весами — шва не видно.
    """
    h, w = img.shape[:2]
    if max(h, w) <= safe:
        return _sr_once(img, model, threads)

    canvas = np.zeros((h * scale, w * scale, 3), np.float32)
    wsum = np.zeros((h * scale, w * scale, 1), np.float32)

    for y in range(0, h, tile):
        for x in range(0, w, tile):
            y1, x1 = min(y + tile, h), min(x + tile, w)
            py0, px0 = max(0, y - pad), max(0, x - pad)
            py1, px1 = min(h, y1 + pad), min(w, x1 + pad)
            up = _sr_once(img[py0:py1, px0:px1], model, threads).astype(np.float32)

            hh, ww = up.shape[:2]
            wy = np.ones(hh, np.float32)
            wx = np.ones(ww, np.float32)
            tp, bp = (y - py0) * scale, (py1 - y1) * scale
            lp, rp = (x - px0) * scale, (px1 - x1) * scale
            if tp:
                tp = min(tp, hh)
                wy[:tp] = np.linspace(0, 1, tp, endpoint=False)
            if bp:
                bp = min(bp, hh)
                wy[hh - bp:] = np.linspace(1, 0, bp, endpoint=False)
            if lp:
                lp = min(lp, ww)
                wx[:lp] = np.linspace(0, 1, lp, endpoint=False)
            if rp:
                rp = min(rp, ww)
                wx[ww - rp:] = np.linspace(1, 0, rp, endpoint=False)
            wgt = (wy[:, None] * wx[None, :])[..., None]

            dy, dx = y * scale, x * scale
            # у правого/нижнего края кадра кусок может вылезти за холст —
            # обрезаем его и веса одинаково, иначе shapes не сойдутся
            vh = min(hh, canvas.shape[0] - dy)
            vw = min(ww, canvas.shape[1] - dx)
            canvas[dy:dy + vh, dx:dx + vw] += up[:vh, :vw] * wgt[:vh, :vw]
            wsum[dy:dy + vh, dx:dx + vw] += wgt[:vh, :vw]

    return np.clip(canvas / np.maximum(wsum, 1e-6), 0, 255).astype(np.uint8)


def upscale(img: np.ndarray, passes: int = 2, threads: int = 2,
            target: int = 1024) -> np.ndarray:
    """Первый проход x4 — форма и контуры, второй — градиенты и блики.

    Второй проход нужен, только если после первого кадр меньше половины
    нужного размера: для иконки 56 px это 224 -> 896 (совсем другое
    качество), а готовый вырез 150 px уже на первом проходе даёт 600, и
    второй проход тратил бы минуту и гигабайты ради пикселей, которые всё
    равно уйдут в даунскейл.
    """
    cur = sr_tiled(img, threads=threads)
    for _ in range(max(1, passes) - 1):
        if max(cur.shape[:2]) >= target * 0.5:
            break
        cur = sr_tiled(cur, tile=96, pad=16, threads=threads, safe=256)
    return cur


# ------------------------------------------------------------------ 3. bg
def clean_background(img: np.ndarray, vmax: float = 70.0,
                     amp: float = 6.0, h: float = 6.0,
                     max_side: int = 1024) -> np.ndarray:
    """Ровный тёмный фон: NLM по яркости только там, где нет деталей.

    Маска = «темно» (яркость ниже `vmax`) x «гладко» (амплитуда мелкой
    структуры ниже `amp`). По ней подмешивается NLM (`h` — его сила): он
    убирает крупу и полосы, но сохраняет низкие частоты — саму тень и
    свечение вокруг иконки.

    На кадре больше `max_side` NLM считается по уменьшенной копии, а обратно
    возвращается только ПОПРАВКА (разница «гладко минус как было»): она
    низкочастотная, поэтому переносится в полный размер без потери деталей,
    зато не требует 5 минут и гигабайт памяти.
    """
    hh, ww = img.shape[:2]
    scale = min(1.0, max_side / float(max(hh, ww)))
    small = img if scale >= 1.0 else cv2.resize(
        img, (max(1, int(round(ww * scale))), max(1, int(round(hh * scale)))),
        interpolation=cv2.INTER_AREA)

    lab = cv2.cvtColor(small, cv2.COLOR_BGR2LAB)
    l8 = lab[..., 0]
    l = l8.astype(np.float32)
    hp = np.abs(l - cv2.medianBlur(l8, 3).astype(np.float32))
    dark = np.clip(1.0 - (l - vmax * 0.35) / max(vmax * 0.65, 1.0), 0, 1)
    flat = np.clip(1.0 - hp / amp, 0, 1)
    m = cv2.GaussianBlur(dark * flat, (0, 0), 1.5)
    smooth = cv2.fastNlMeansDenoising(l8, None, float(h), 7, 21).astype(np.float32)
    fix = (smooth - l) * m

    if scale < 1.0:
        fix = cv2.resize(fix, (ww, hh), interpolation=cv2.INTER_LINEAR)
    out = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    out[..., 0] = np.clip(out[..., 0].astype(np.float32) + fix, 0, 255).astype(np.uint8)
    return cv2.cvtColor(out, cv2.COLOR_LAB2BGR)


# ------------------------------------------------------------------- 4. polish
def _unsharp_l(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0]
    lab[..., 0] = np.clip(l + (l - cv2.GaussianBlur(l, (0, 0), sigma)) * amount, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def local_contrast(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    """Усилить только средние частоты: контуры и большие заливки не трогаем.

    В глубоких тенях вес падает до нуля: там нет деталей, а усиленный шум
    выглядит кольцами и полосами.
    """
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0]
    detail = l - cv2.GaussianBlur(l, (0, 0), sigma)
    k = np.clip((l - 24.0) / 48.0, 0.0, 1.0) * np.clip(1.0 - np.abs(l - 128) / 128, 0.25, 1.0)
    lab[..., 0] = np.clip(l + detail * amount * k, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def bloom(img: np.ndarray, thr: float = 190, sigma: float = 9.0,
          amount: float = 0.3) -> np.ndarray:
    """Мягкое свечение от ярких пикселей — золото и камень начинают гореть.

    Маска по яркости, поэтому светятся только металл и самоцвет, а не фон.
    Оттенок свечения берётся с самих пикселей (тёплый у золота, красный у
    камня) — иначе золото подсвечивалось бы красным.
    """
    l = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)[..., 0].astype(np.float32)
    m = np.clip((l - thr) / max(255 - thr, 1), 0, 1) ** 1.4
    src = img.astype(np.float32) * m[..., None]
    glow = cv2.GaussianBlur(src, (0, 0), sigma)
    glow = glow / (glow.max() + 1e-6) * 255.0 * amount
    return np.clip(img.astype(np.float32) + glow, 0, 255).astype(np.uint8)


def saturate(img: np.ndarray, k: float = 1.1, protect: float = 0.55) -> np.ndarray:
    """Вибранс, а не «умножить насыщенность на k».

    Умножение вслепую выкручивает уже насыщенные места: розовый самоцвет
    становится кислотно-красным, бирюза — неоновой, а белый блик уходит в
    цвет. Поэтому прибавка тем меньше, чем пиксель уже насыщеннее:
    при `k` = 1.10 прибавка полная на сером и падает до нуля там, где
    насыщенность выше `protect`.
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    sat = hsv[..., 1].astype(np.float32)
    gain = 1.0 + (k - 1.0) * np.clip(1.0 - sat / (255.0 * protect), 0.0, 1.0)
    hsv[..., 1] = np.clip(sat * gain, 0, 255).astype(np.uint8)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)


def warm(img: np.ndarray, k: float = 0.8) -> np.ndarray:
    f = img.astype(np.float32)
    f[..., 2] *= 1.0 + 0.03 * k     # R
    f[..., 0] *= 1.0 - 0.04 * k     # B
    return np.clip(f, 0, 255).astype(np.uint8)


def vignette(img: np.ndarray, strength: float = 0.3) -> np.ndarray:
    h, w = img.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    r = np.hypot(xx / w - 0.5, yy / h - 0.5) / 0.707
    m = 1.0 - strength * np.clip(r, 0, 1) ** 2.2
    return np.clip(img.astype(np.float32) * m[..., None], 0, 255).astype(np.uint8)


PRESETS = {
    # локальный контраст | unsharp 1 | unsharp 2 | свечение | насыщенность | виньетка | тепло
    "faithful": dict(lc=0.18, us1=0.18, us2=0.20, bloom=0.15, sat=1.04,
                     vig=0.15, warm=0.4),
    "balanced": dict(lc=0.35, us1=0.30, us2=0.35, bloom=0.30, sat=1.10,
                     vig=0.30, warm=0.8),
    "rich":     dict(lc=0.55, us1=0.45, us2=0.55, bloom=0.50, sat=1.18,
                     vig=0.45, warm=1.2),
}


def polish(big: np.ndarray, size: int, variant: str = "balanced") -> np.ndarray:
    """Большой кадр после сети -> аккуратная иконка size×size.

    Все радиусы заданы в долях размера, поэтому 256 и 1024 выглядят
    одинаково, а не «мелко» на маленьком размере.
    """
    p = PRESETS[variant]
    s = float(size)
    interp = cv2.INTER_AREA if big.shape[0] >= size else cv2.INTER_LANCZOS4
    out = cv2.resize(big, (size, size), interpolation=interp)
    out = local_contrast(out, p["lc"], s / 64.0)
    out = _unsharp_l(out, p["us1"], max(0.6, s / 420.0))
    out = bloom(out, thr=190, sigma=s / 55.0, amount=p["bloom"])
    out = saturate(out, p["sat"])
    out = warm(out, p["warm"])
    out = _unsharp_l(out, p["us2"], max(0.5, s / 512.0))
    return vignette(out, p["vig"])


# --------------------------------------------------------------------- preview
def sheet(before: np.ndarray, after: np.ndarray, side: int = 360,
          pad: int = 10, bg: int = 14) -> np.ndarray:
    """«Как было / как стало» в одном кадре: слева исходник (по пикселям,
    NEAREST), справа результат."""
    a = cv2.resize(before, (side, side), interpolation=cv2.INTER_NEAREST)
    b = cv2.resize(after, (side, side), interpolation=cv2.INTER_AREA)
    gap = np.full((side, pad, 3), bg, np.uint8)
    row = np.hstack([a, gap, b])
    edge = np.full((pad, row.shape[1], 3), bg, np.uint8)
    return np.vstack([edge, row, edge])


# ------------------------------------------------------------------------ main
def enhance(path: str, sizes=(256, 512, 1024), variant: str = "balanced",
            use_sr: bool = True, passes: int = 2, threads: int = 2) -> dict:
    src = cv2.imread(path)
    if src is None:
        raise SystemExit(f"не читается: {path}")
    work = clean_border(src)
    if use_sr:
        import sr_ncnn
        if not sr_ncnn.available():
            raise SystemExit("нет Real-ESRGAN: pip install ncnn realesrgan-ncnn-py")
        big = upscale(work, passes, threads)
    else:
        big = cv2.resize(work, (work.shape[1] * 8, work.shape[0] * 8),
                         interpolation=cv2.INTER_LANCZOS4)
    big = clean_background(big)
    return {"src": src, "big": big,
            "sizes": {s: polish(big, s, variant) for s in sizes}}


def main() -> int:
    _console_setup()
    ap = argparse.ArgumentParser(description="Улучшить одиночную иконку")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=os.path.join(ROOT, "enhanced"))
    ap.add_argument("--sizes", type=int, nargs="+", default=[256, 512, 1024],
                    help="какие размеры сохранить (по умолчанию 256 512 1024)")
    ap.add_argument("--variant", default="balanced", choices=sorted(PRESETS),
                    help="сила полировки: faithful - ближе к исходнику, "
                         "balanced - как в примерах, rich - смелее")
    ap.add_argument("--passes", type=int, default=2,
                    help="сколько проходов сети ×4 (1 - быстрее и мягче)")
    ap.add_argument("--no-sr", action="store_true",
                    help="без нейросети: только чистка и Lanczos (быстро, мыльнее)")
    ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--no-sheet", action="store_true", help="без превью")
    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    for src in args.images:
        r = enhance(src, args.sizes, args.variant, not args.no_sr,
                    args.passes, args.threads)
        base = os.path.splitext(os.path.basename(src))[0]
        for s, im in r["sizes"].items():
            p = os.path.join(args.out, f"{base}_{s}.png")
            cv2.imwrite(p, im)
            print(f"{p}  {im.shape[1]}x{im.shape[0]}")
        if not args.no_sheet:
            biggest = max(args.sizes)
            p = os.path.join(args.out, f"{base}_sravnenie.png")
            cv2.imwrite(p, sheet(r["src"], r["sizes"][biggest]))
            print(f"{p}  (слева исходник, справа результат {biggest}px)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
