#!/usr/bin/env python3
"""
bezel_gallery.py — сравнение вариантов отделки рамки на живых иконках.

Собирает сетку: строки — пресеты отделки (steel, bronze, cold, pin, brackets,
matte, glass, slim), столбцы — разные иконки. Каждая плитка подписана.
Результат: samples/bezel_options.jpg (+ отдельные PNG по желанию).

    python3 tools/bezel_gallery.py
    python3 tools/bezel_gallery.py --scale 2 --icons 3
"""
from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_frames as D  # noqa: E402
import style_frame as S    # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# (лист, номер слота) — разные по цвету и форме иконки
ICONS = [
    ("RPG Loot Icons 01/RPG Loot Icons 01 Part 1.jpg", 0),    # камень, светлый
    ("RPG Loot Icons 01/RPG Loot Icons 01 Part 1.jpg", 1),    # красный кристалл
    ("Alchemical Reagents Icons/file (84).jpg", 0),           # миска, тёплая
]
ORDER = ["steel", "bronze", "cold", "pin", "brackets", "matte", "glass", "slim"]


def build(scale: float, n_icons: int, out_jpg: str) -> None:
    sheets: dict[str, tuple[np.ndarray, list]] = {}
    tiles_by_row = {k: [] for k in ORDER}

    for name in ORDER:
        p = S.decor_params(name)
        row = []
        for sheet, idx in ICONS[:n_icons]:
            if sheet not in sheets:
                sh = D.detect_sheet(os.path.join(ROOT, sheet))
                sheets[sheet] = (cv2.imread(os.path.join(ROOT, sheet)), sh.boxes)
            base, boxes = sheets[sheet]
            img = base.copy()
            for box in boxes:
                S.style_box(img, box, p)
            x0, y0, x1, y1 = boxes[idx]
            pad = int(p["rim"] + 12)
            crop = img[max(0, y0 - pad):min(img.shape[0], y1 + 1 + pad),
                       max(0, x0 - pad):min(img.shape[1], x1 + 1 + pad)]
            crop = cv2.resize(crop, None, fx=scale, fy=scale,
                              interpolation=cv2.INTER_NEAREST)
            crop = cv2.copyMakeBorder(crop, 26, 6, 6, 6, cv2.BORDER_CONSTANT,
                                      value=(18, 18, 18))
            cv2.putText(crop, name, (7, 19), cv2.FONT_HERSHEY_SIMPLEX, 0.55,
                        (150, 220, 150), 1, cv2.LINE_AA)
            row.append(crop)
        tiles_by_row[name] = row

    # выравниваем плитки по размеру
    h = max(t.shape[0] for row in tiles_by_row.values() for t in row)
    w = max(t.shape[1] for row in tiles_by_row.values() for t in row)
    grid = []
    for name in ORDER:
        row = [cv2.copyMakeBorder(t, 0, h - t.shape[0], 0, w - t.shape[1],
                                  cv2.BORDER_CONSTANT, value=(18, 18, 18))
               for t in tiles_by_row[name]]
        grid.append(np.hstack(row))
    img = np.vstack(grid)
    os.makedirs(os.path.dirname(out_jpg) or ".", exist_ok=True)
    cv2.imwrite(out_jpg, img, [int(cv2.IMWRITE_JPEG_QUALITY), 93])
    print(f"{out_jpg}  {img.shape[1]}x{img.shape[0]}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.6)
    ap.add_argument("--icons", type=int, default=3)
    ap.add_argument("--out", default="samples/bezel_options.jpg")
    args = ap.parse_args()
    build(args.scale, args.icons, os.path.join(ROOT, args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
