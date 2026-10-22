#!/usr/bin/env python3
"""
check_sizes.py — контроль физического размера рамок.

Иконки со обводкой не могут быть больше MAX_CELL px, поэтому каждая найденная
рамка проверяется по ширине и высоте (обе считаются включительно: x1-x0+1).
Скрипт:
  * печатает статистику размеров и список листов, где хоть одна рамка больше
    лимита (и, отдельно, заметно меньше — признак недодетекта);
  * собирает картинку annotated/size_check.jpg: самые крупные рамки с
    подписанным размером в пикселях;
  * пишет annotated/size_check.csv.

Запуск:
    python3 tools/check_sizes.py              # лимит 160 px
    python3 tools/check_sizes.py --limit 160
"""
from __future__ import annotations

import argparse
import csv
import os

import cv2
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ANN = os.path.join(ROOT, "annotated")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=160)
    ap.add_argument("--top", type=int, default=9)
    ap.add_argument("--csv", default=os.path.join(ANN, "frames.csv"))
    args = ap.parse_args()

    with open(args.csv, encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))

    recs = []
    for r in rows:
        for k, b in enumerate(r["boxes"].split()):
            x0, y0, x1, y1 = (int(v) for v in b.split(","))
            recs.append({
                "file": r["file"], "slot": k,
                "x0": x0, "y0": y0, "x1": x1, "y1": y1,
                "w": x1 - x0 + 1, "h": y1 - y0 + 1,
            })
    if not recs:
        print("нет данных — сначала запустите tools/detect_frames.py")
        return 1

    w = np.array([r["w"] for r in recs])
    h = np.array([r["h"] for r in recs])
    over = [r for r in recs if r["w"] > args.limit or r["h"] > args.limit]
    under = [r for r in recs if min(r["w"], r["h"]) < args.limit * 0.85]

    print(f"рамок всего: {len(recs)}  лимит: {args.limit} px (включительно)")
    print(f"ширина : min {w.min()}  p1 {np.percentile(w, 1):.0f}  "
          f"медиана {np.median(w):.0f}  p99 {np.percentile(w, 99):.0f}  max {w.max()}")
    print(f"высота : min {h.min()}  p1 {np.percentile(h, 1):.0f}  "
          f"медиана {np.median(h):.0f}  p99 {np.percentile(h, 99):.0f}  max {h.max()}")
    print(f"\nбольше {args.limit} px: {len(over)}")
    for r in over[:20]:
        print(f"  {r['file']}  слот {r['slot']}: {r['w']}x{r['h']} "
              f"({r['x0']},{r['y0']})-({r['x1']},{r['y1']})")
    print(f"заметно меньше ({(1 - 0.85) * 100:.0f}% от лимита): {len(under)}")
    for r in under[:10]:
        print(f"  {r['file']}  слот {r['slot']}: {r['w']}x{r['h']}")

    # CSV по всем рамкам
    with open(os.path.join(ANN, "size_check.csv"), "w", newline="",
              encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(recs[0]))
        wr.writeheader()
        wr.writerows(recs)

    # картинка: самые крупные рамки с подписью размера
    biggest = sorted(recs, key=lambda r: max(r["w"], r["h"]), reverse=True)[: args.top]
    tiles = []
    for r in biggest:
        img = cv2.imread(os.path.join(ROOT, r["file"]))
        pad = 16
        crop = img[max(0, r["y0"] - pad):r["y1"] + pad,
                   max(0, r["x0"] - pad):r["x1"] + pad].copy()
        cv2.rectangle(crop, (r["x0"] - max(0, r["x0"] - pad), r["y0"] - max(0, r["y0"] - pad)),
                      (r["x1"] - max(0, r["x0"] - pad), r["y1"] - max(0, r["y0"] - pad)),
                      (0, 0, 255), 1, cv2.LINE_AA)
        sc = 240 / max(crop.shape[:2])
        crop = cv2.resize(crop, None, fx=sc, fy=sc, interpolation=cv2.INTER_NEAREST)
        crop = cv2.copyMakeBorder(crop, 26, 6, 6, 6, cv2.BORDER_CONSTANT,
                                  value=(25, 25, 25))
        label = f"{r['w']}x{r['h']} px"
        col = (0, 0, 255) if max(r["w"], r["h"]) > args.limit else (120, 220, 120)
        cv2.putText(crop, label, (6, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 1,
                    cv2.LINE_AA)
        tiles.append(crop)
    if tiles:
        hmax = max(t.shape[0] for t in tiles)
        wmax = max(t.shape[1] for t in tiles)
        tiles = [cv2.copyMakeBorder(t, 0, hmax - t.shape[0], 0, wmax - t.shape[1],
                                    cv2.BORDER_CONSTANT, value=(25, 25, 25))
                 for t in tiles]
        cols = 3
        while len(tiles) % cols:
            tiles.append(np.full_like(tiles[0], 25))
        grid = np.vstack([np.hstack(tiles[i:i + cols])
                          for i in range(0, len(tiles), cols)])
        out = os.path.join(ANN, "size_check.jpg")
        cv2.imwrite(out, grid, [int(cv2.IMWRITE_JPEG_QUALITY), 90])
        print(f"\nкартинка с самыми крупными рамками: {out}")
    return 0 if not over else 2


if __name__ == "__main__":
    raise SystemExit(main())
