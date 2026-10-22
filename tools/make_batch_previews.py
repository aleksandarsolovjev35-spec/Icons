#!/usr/bin/env python3
"""
make_batch_previews.py — превью для пакетного прогона.

Читает `icons_256/**/*.png` (результат tools/batch_all.py) и собирает в
`samples/`:

  batch_contact_sheet.jpg  — сетка по одной иконке с каждого листа
  batch_montage.jpg        — 12 иконок из разных папок крупно (×2)
  batch_grid_<папка>.jpg   — полная сетка иконок одной папки (--per-folder N)

    python3 tools/make_batch_previews.py
    python3 tools/make_batch_previews.py --per-folder 5
"""
from __future__ import annotations

import argparse
import glob
import os

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
SRC = os.path.join(ROOT, "icons_256")
OUT = os.path.join(ROOT, "samples")


def _label(img: np.ndarray, text: str, h: int = 22) -> np.ndarray:
    cv2.rectangle(img, (0, 0), (img.shape[1], h), (0, 0, 0), -1)
    cv2.putText(img, text, (4, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.36,
                (0, 255, 0), 1, cv2.LINE_AA)
    return img


def _grid(tiles: list[np.ndarray], cols: int, gap: int = 4,
          label: str | None = None) -> np.ndarray:
    if not tiles:
        return np.zeros((8, 8, 3), np.uint8)
    h, w = tiles[0].shape[:2]
    rows = (len(tiles) + cols - 1) // cols
    sheet = np.full((rows * (h + gap) + gap, cols * (w + gap) + gap, 3),
                    18, np.uint8)
    for i, t in enumerate(tiles):
        r, c = divmod(i, cols)
        y, x = gap + r * (h + gap), gap + c * (w + gap)
        sheet[y:y + h, x:x + w] = t
    if label:
        sheet = _label(sheet, label, 24)
    return sheet


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tile", type=int, default=96, help="размер миниатюры")
    ap.add_argument("--src", default=SRC, help="папка с результатами")
    ap.add_argument("--per-folder", type=int, default=0,
                    help="ширины сеток по папкам (0 — не делать)")
    args = ap.parse_args()

    src = args.src
    files = sorted(glob.glob(os.path.join(src, "*", "*.png")))
    if not files:
        print("нет файлов в", src)
        return 1
    print("иконок:", len(files))

    # 1) контактный лист: по одной иконке с каждого листа (slot00)
    per_sheet = [f for f in files if "__slot00" in os.path.basename(f)]
    tiles = []
    for f in per_sheet:
        im = cv2.imread(f)
        if im is None:
            continue
        tiles.append(cv2.resize(im, (args.tile, args.tile),
                                interpolation=cv2.INTER_AREA))
    cols = int(np.ceil(np.sqrt(len(tiles) * 1.6)))
    sheet = _grid(tiles, cols, 3)
    cv2.imwrite(os.path.join(OUT, "batch_contact_sheet.jpg"), sheet,
                [int(cv2.IMWRITE_JPEG_QUALITY), 90])
    print("contact sheet:", sheet.shape[1], "×", sheet.shape[0],
          f"({len(tiles)} листов)")

    # 2) монтаж: по одной иконке из разных папок, крупно
    folders = sorted({os.path.dirname(f) for f in files})
    step = max(1, len(folders) // 12)
    picks = [cv2.imread(p) for p in
             [sorted(glob.glob(os.path.join(dd, "*.png")))[
                 len(glob.glob(os.path.join(dd, "*.png"))) // 2]
              for dd in folders[::step][:12]]]
    picks = [p for p in picks if p is not None]
    m = _grid([cv2.resize(p, (256, 256), interpolation=cv2.INTER_AREA)
               for p in picks], 4, 4)
    cv2.imwrite(os.path.join(OUT, "batch_montage.jpg"), m,
                [int(cv2.IMWRITE_JPEG_QUALITY), 92])
    print("монтаж:", m.shape[1], "×", m.shape[0], f"({len(picks)} иконок)")

    # 3) сетки по папкам
    if args.per_folder:
        for dd in folders:
            sub = sorted(glob.glob(os.path.join(dd, "*.png")))
            if len(sub) < 8:
                continue
            name = os.path.basename(dd).replace(" ", "_")
            tt = [cv2.resize(cv2.imread(s), (args.per_folder, args.per_folder),
                             interpolation=cv2.INTER_AREA)
                  for s in sub[: args.per_folder ** 2]]
            tt = [t for t in tt if t is not None]
            g = _grid(tt, args.per_folder, 2, label=name)
            cv2.imwrite(os.path.join(OUT, f"batch_grid_{name}.jpg"), g,
                        [int(cv2.IMWRITE_JPEG_QUALITY), 88])
        print("сетки по папкам готовы")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
