#!/usr/bin/env python3
"""
analyze_noise.py — разбор шума на иконке: что именно «шумит» и где.

Считает раздельно:
  * хрома-шум (Lab a/b) на плоских участках — цветные крапинки;
  * яркостный шум/блочность (Lab L) — дрожание яркости и сетка 8×8 от JPEG;
  * звон (ringing) у сильных контуров — ореолы;
  * запас по резкости (насколько контуры сильнее плоских зон).

    python3 tools/analyze_noise.py samples/icon_sharp_bezel.png
    python3 tools/analyze_noise.py img.png --region 38 50 78 92
"""
from __future__ import annotations

import argparse

import cv2
import numpy as np


def flat_mask(L: np.ndarray, thr: float = 10.0) -> np.ndarray:
    g = L.astype(np.float32)
    lap = np.abs(cv2.Laplacian(g, cv2.CV_32F))
    gx = cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3) / 4.0
    gy = cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3) / 4.0
    return np.maximum(lap, np.hypot(gx, gy)) < thr


def med_dev(ch: np.ndarray, k: int = 5) -> np.ndarray:
    u8 = np.clip(ch, 0, 255).astype(np.uint8)
    return np.abs(ch - cv2.medianBlur(u8, k).astype(np.float32))


def report(img: np.ndarray, title: str = "") -> None:
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    L, A, B = lab[..., 0], lab[..., 1], lab[..., 2]
    flat = flat_mask(L)
    y = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    lap = np.abs(cv2.Laplacian(y, cv2.CV_32F))
    edge = lap > 25

    print(f"\n=== {title}  ({img.shape[1]}x{img.shape[0]}) ===")
    print(f"плоских пикселей: {100 * flat.mean():.0f}%")
    print("хрома (цветные крапинки):")
    print(f"   std  a {A[flat].std():5.2f}   b {B[flat].std():5.2f}")
    print(f"   |x-med5| a {med_dev(A)[flat].mean():5.2f}   b {med_dev(B)[flat].mean():5.2f}"
          f"   p99 {np.percentile(med_dev(B)[flat], 99):.1f}")
    print("яркость:")
    print(f"   |x-med5| {med_dev(L)[flat].mean():5.2f}  p99 "
          f"{np.percentile(med_dev(L)[flat], 99):.1f}   std {L[flat].std():6.2f}")
    # блочность 8x8: насколько границы блоков «выше» среднего
    d = np.abs(np.diff(y, axis=1)).mean(axis=0)
    phase = np.array([d[p::8].mean() for p in range(8)])
    print(f"   блочность 8×8 (по фазам): разброс {100 * (phase.max() / phase.mean() - 1):.1f}%"
          f"  [{', '.join(f'{v:.1f}' for v in phase)}]")
    # звон у контуров
    ring = np.abs(L - cv2.GaussianBlur(L, (0, 0), 1.0))
    near = cv2.dilate(edge.astype(np.uint8), np.ones((7, 7), np.uint8)).astype(bool) & ~edge
    print(f"звон у контуров (σ=1): {ring[near].mean():.2f}"
          f"   против плоских зон {ring[flat].mean():.2f}")
    print(f"яркость кромок (резкость): {lap[edge].mean():.1f}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("images", nargs="+")
    ap.add_argument("--region", type=int, nargs=4, default=None,
                    metavar=("Y0", "Y1", "X0", "X1"))
    args = ap.parse_args()
    for p in args.images:
        img = cv2.imread(p)
        if img is None:
            raise SystemExit(f"не читается: {p}")
        if args.region:
            y0, y1, x0, x1 = args.region
            img = img[y0:y1, x0:x1]
        report(img, p)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
