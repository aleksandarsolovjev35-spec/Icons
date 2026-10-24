#!/usr/bin/env python3
"""one_icon.py — разбор одного слота: как было / как отработал / что вышло.

Запуск:
    python one_icon.py "RPG Potion Icons 01/file (83).jpg" 27 имя

Кладёт в samples/one_icon/ три файла (x4, nearest — чтобы видеть пиксели):
    01_<имя>_kak_bylo.png        — кусок листа без разметки
    02_<имя>_kak_otrabotal.png   — жёлтый bbox детектора, красная найденная
                                   рамка, зелёные границы выреза
    03_<имя>_rezultat.png        — сам вырез
"""
from __future__ import annotations

import os
import sys

import cv2

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import detect_frames as D     # noqa: E402
import frame_struct as S      # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                   "samples", "one_icon")
SCALE = 4


def build(path: str, k: int, name: str, pad: int = 45) -> dict:
    img = cv2.imread(path)
    boxes = D.detect_sheet(path).boxes
    box = boxes[k]
    rects, size, shades = S.sheet_rects(img, boxes)
    rect = rects[k]
    crop = S.crop_rect(img, rect)

    x0 = max(0, min(box[0], rect[0]) - 30)
    y0 = max(0, min(box[1], rect[1]) - pad)
    x1 = min(img.shape[1], max(box[2], rect[2]) + 30)
    y1 = min(img.shape[0], max(box[3], rect[3]) + pad)
    frag = img[y0:y1, x0:x1]

    os.makedirs(OUT, exist_ok=True)

    def save(sub: str, im):
        p = os.path.join(OUT, f"{sub}.png")
        cv2.imwrite(p, cv2.resize(im, None, fx=SCALE, fy=SCALE,
                                  interpolation=cv2.INTER_NEAREST))
        return p

    save(f"01_{name}_kak_bylo", frag)

    v = frag.copy()
    cv2.rectangle(v, (box[0] - x0, box[1] - y0), (box[2] - x0, box[3] - y0),
                  (0, 215, 255), 1)                      # bbox детектора
    cv2.rectangle(v, (rect[0] - x0, rect[1] - y0), (rect[2] - x0, rect[3] - y0),
                  (0, 0, 255), 1)                        # найденная рамка
    cv2.rectangle(v, (crop[0] - x0, crop[1] - y0), (crop[2] - x0, crop[3] - y0),
                  (0, 255, 0), 1)                        # границы выреза
    save(f"02_{name}_kak_otrabotal", v)

    save(f"03_{name}_rezultat", img[crop[1]:crop[3] + 1, crop[0]:crop[2] + 1])

    return {"path": path, "slot": k, "box": box, "rect": rect, "crop": crop,
            "size": size, "shades": [int(round(float(s.mean()))) for s in shades]}


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print(__doc__)
        raise SystemExit(1)
    info = build(sys.argv[1], int(sys.argv[2]), sys.argv[3])
    print(info)
