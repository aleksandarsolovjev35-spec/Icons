#!/usr/bin/env python3
"""
cut_icons.py — вырезает отдельные иконки с рамкой (и с красной обводкой поверх).

Режет ровно по внешнему краю красной обводки: bbox + толщина линии + 1 px,
поэтому в вырезе остаётся вся красная рамка, без обрезки.

    python3 tools/cut_icons.py                      # по 2 иконки с 2 листов, 1:1
    python3 tools/cut_icons.py --scale 3            # то же с апскейлом x3
    python3 tools/cut_icons.py --n 6                # по 6 с каждого указанного листа
    python3 tools/cut_icons.py "RPG Loot Icons 01/RPG Loot Icons 01 Part 1.jpg" --slots 0 11
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_frames as D  # noqa: E402
import style_frame as S  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PAD = 2          # запас вокруг красной линии (сама линия 2 px)


def cut(path: str, slots: list[int], outdir: str, scale: float = 1.0,
        style: str | None = None, radius: float | None = None) -> list[str]:
    sh = D.detect_sheet(path)
    img = cv2.imread(path)
    H, W = img.shape[:2]
    p = None
    if style:                       # объёмный бортик вместо красной линии
        p = S.params(style, radius=radius)
        for box in sh.boxes:
            S.style_box(img, box, p)
    else:
        for (x0, y0, x1, y1) in sh.boxes:
            cv2.rectangle(img, (x0, y0), (x1, y1), D.RED, 2, cv2.LINE_AA)

    made = []
    stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    for k in slots:
        if k >= len(sh.boxes):
            continue
        x0, y0, x1, y1 = sh.boxes[k]
        pad = int(PAD + (p["rim"] if style else 0))
        cx0 = max(0, x0 - pad)
        cy0 = max(0, y0 - pad)
        cx1 = min(W, x1 + 1 + pad)
        cy1 = min(H, y1 + 1 + pad)
        crop = img[cy0:cy1, cx0:cx1]
        big = crop if scale == 1.0 else cv2.resize(
            crop, None, fx=scale, fy=scale, interpolation=cv2.INTER_LANCZOS4)
        out = os.path.join(outdir, f"{stem}__slot{k:02d}.png")
        cv2.imwrite(out, big)
        made.append(out)
        print(f"  {out}  {crop.shape[1]}x{crop.shape[0]} -> x{scale}")
    return made


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="*")
    ap.add_argument("--slots", type=int, nargs="*", default=None)
    ap.add_argument("--n", type=int, default=2, help="сколько иконок с листа")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="1.0 = пиксель в пиксель (по умолчанию)")
    ap.add_argument("--out", default="cutouts")
    ap.add_argument("--style", default=None, choices=sorted(S.VARIANTS),
                    help="рисовать объёмный бортик вместо красной линии")
    ap.add_argument("--radius", type=float, default=None,
                    help="радиус скругления бортика, 0 = прямые углы")
    args = ap.parse_args()

    files = args.images or [
        "RPG Loot Icons 01/RPG Loot Icons 01 Part 1.jpg",
        "Alchemical Reagents Icons/file (84).jpg",
    ]
    outdir = os.path.join(ROOT, args.out)
    os.makedirs(outdir, exist_ok=True)
    for f in files:
        f = f if os.path.isabs(f) else os.path.join(ROOT, f)
        print(os.path.relpath(f, ROOT))
        sh = D.detect_sheet(f)
        slots = args.slots if args.slots is not None else list(range(args.n))
        cut(f, slots, outdir, scale=args.scale, style=args.style,
            radius=args.radius)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
