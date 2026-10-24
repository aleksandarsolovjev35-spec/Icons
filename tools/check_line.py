#!/usr/bin/env python3
"""check_line.py — не осталось ли линии рамки на кромке выреза.

Считает по каждому слоту каждого листа, похожа ли крайняя полоса выреза на
линию рамки: ровная (std <= 12), светлая (медиана >= 40), нейтральная
(chroma <= 25) и занимает большую часть стороны. Рисунок у края так не
выглядит — у него яркость вдоль стороны сильно гуляет.

    python tools/check_line.py            # вся коллекция
    python tools/check_line.py --limit 20 # первые 20 листов
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

import batch_all as B            # noqa: E402
import detect_frames as D        # noqa: E402


def line_like(strip: np.ndarray) -> bool:
    """Похожа ли полоса на линию рамки (а не на рисунок)."""
    f = strip.astype(np.int16)
    bright = f.mean(axis=1)
    chroma = f.max(axis=1) - f.min(axis=1)
    neutral = chroma <= B.LINE_MAX_CHROMA
    cov = float(((bright > B.LINE_BRIGHT) & neutral).mean())
    med = float(np.median(bright))
    if not neutral.any():
        return False
    nb = bright[neutral]
    # линия очень ровная (std ~1) и занимает почти всю сторону (cov ~1).
    # Мягкий градиент фона даёт std 10 и cov 0.7 - это не линия.
    return (med > B.LINE_BRIGHT and float(nb.std()) <= 2.5
            and cov >= 0.85)


def leftover(crop: np.ndarray, side: str, depth: int = 6) -> bool:
    """Линия внутри кромки: ровная светлая полоса, а до неё - чёрный фон.

    Смотрим не только сам крайний пиксель: линия может остаться в 2-6 px от
    края (детектор рамки иногда ставит box на несколько пикселей наружу).
    Тогда на кромке чёрный фон, а линия - чуть глубже, и глаз её видит.
    """
    for d in range(depth):
        strip = {"верх": lambda: crop[d],
                 "низ": lambda: crop[-1 - d],
                 "лево": lambda: crop[:, d],
                 "право": lambda: crop[:, -1 - d]}[side]()
        if not line_like(strip):
            continue
        # между краем и линией должен быть тёмный фон, иначе это рисунок
        gap = {"верх": lambda: crop[:d], "низ": lambda: crop[-d:],
               "лево": lambda: crop[:, :d], "право": lambda: crop[:, -d:]}[side]()
        if d == 0 or gap.size == 0 or float(np.median(gap)) < B.INNER_DARK:
            return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--sheets", nargs="*", default=None)
    args = ap.parse_args()

    if args.sheets:
        sheets = args.sheets
    else:
        sheets = sorted(p for p in glob.glob(os.path.join(ROOT, "*", "*.jpg"))
                        if os.path.basename(os.path.dirname(p)) != "samples")
        if args.limit:
            sheets = sheets[:args.limit]

    bad_sides = 0
    total_sides = 0
    bad_slots = []
    for i, path in enumerate(sheets, 1):
        img = cv2.imread(path)
        if img is None:
            continue
        try:
            boxes = D.detect_sheet(path).boxes
        except Exception:                       # noqa: BLE001
            continue
        for k, box in enumerate(boxes):
            x0, y0, x1, y1 = B._inner_scan(img, box)
            if x1 - x0 < 8 or y1 - y0 < 8:
                continue
            crop = img[y0:y1, x0:x1]
            hit = [n for n in ("верх", "низ", "лево", "право")
                   if leftover(crop, n)]
            total_sides += 4
            if hit:
                bad_sides += len(hit)
                bad_slots.append((os.path.relpath(path, ROOT), k, hit))
        if i % 20 == 0:
            print(f"  ...{i}/{len(sheets)} листов, проблемных сторон: {bad_sides}")

    print(f"\nсторон проверено: {total_sides}")
    print(f"с линией на кромке: {bad_sides}")
    print(f"слотов с проблемой: {len(bad_slots)}")
    for s, k, hit in bad_slots[:40]:
        print(f"  {s} слот {k}: {', '.join(hit)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
