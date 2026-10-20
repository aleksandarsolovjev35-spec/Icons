#!/usr/bin/env python3
"""High-quality upscale of RPG icon PNGs to 256x256.

Pipeline per image:
  1. Decode sRGB -> linear light using the exact IEC 61966-2-1 transfer curve.
     (Resampling in linear light avoids the dark halos / muddy mid-tones that a
     plain gamma-space resize produces on dark backgrounds.)
  2. Lanczos-4 resample to 256x256 (antialiased, can up- or down-scale).
  3. Edge-aware unsharp mask on the LAB L channel only:
       - threshold kills low-amplitude noise (no grain amplification)
       - the detail signal is clamped before being added back, so steep edges
         get crisp without white halos or clipped highlights
       - chroma (a/b) is left untouched, so no colour fringing appears.
  4. Save as lossless PNG (compress_level 9, no metadata).

Usage:
  python3 tools/upscale_256.py --src . --dst icons_256 --jobs 2
  python3 tools/upscale_256.py --src "RPG Gems Icons 01" --dst icons_256  # subset
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

import cv2
import numpy as np
from PIL import Image

SIZE = 256
# exact sRGB transfer function, implemented as 8-bit <-> float LUTs (fast, exact)
def _build_luts() -> tuple[np.ndarray, np.ndarray]:
    x = np.arange(256, dtype=np.float32) / 255.0
    to_lin = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    y = np.linspace(0.0, 1.0, 4096, dtype=np.float32)
    to_srgb = np.where(y <= 0.0031308, y * 12.92, 1.055 * y ** (1 / 2.4) - 0.055)
    return to_lin.astype(np.float32), to_srgb.astype(np.float32)


SRGB_TO_LIN, LIN_TO_SRGB = _build_luts()


_BAYER8 = np.array([
    [0, 32, 8, 40, 2, 34, 10, 42],
    [48, 16, 56, 24, 50, 18, 58, 26],
    [12, 44, 4, 36, 14, 46, 6, 38],
    [60, 28, 52, 20, 62, 30, 54, 22],
    [3, 35, 11, 43, 1, 33, 9, 41],
    [51, 19, 59, 27, 49, 17, 57, 25],
    [15, 47, 7, 39, 13, 45, 5, 37],
    [63, 31, 55, 23, 61, 29, 53, 21],
], dtype=np.float32)


def _encode(linear: np.ndarray, dither: bool = True) -> np.ndarray:
    """linear float [0..1] -> 8-bit sRGB with ordered dithering (no banding)."""
    idx = np.clip(linear, 0.0, 1.0) * (len(LIN_TO_SRGB) - 1)
    out = np.interp(idx, np.arange(len(LIN_TO_SRGB)), LIN_TO_SRGB) * 255.0
    if dither:
        h, w = out.shape[:2]
        pat = np.tile(_BAYER8, (h // 8 + 1, w // 8 + 1))[:h, :w, None]
        out = out + (pat - 31.5) / 64.0  # +/- 0.49 LSB, deterministic
    return np.clip(out + 0.5, 0, 255).astype(np.uint8)


def upscale(img_bgr: np.ndarray, size: int = SIZE,
            amount: float = 0.55, radius: float = 1.0,
            threshold: float = 2.0, detail_clip: float = 0.10) -> np.ndarray:
    lin = SRGB_TO_LIN[img_bgr]
    h, w = img_bgr.shape[:2]
    # resample in linear light; Lanczos-4 keeps the most detail without ringing
    big = cv2.resize(lin, (size, size), interpolation=cv2.INTER_LANCZOS4)
    out = _encode(big)

    # --- edge-aware unsharp on L only -------------------------------------
    lab = cv2.cvtColor(out, cv2.COLOR_BGR2LAB)
    l = lab[:, :, 0].astype(np.float32) / 255.0
    blur = cv2.GaussianBlur(l, (0, 0), radius)
    detail = l - blur
    if threshold > 0:
        detail = np.where(np.abs(detail) < threshold / 255.0, 0.0, detail)
    detail = np.clip(detail * amount, -detail_clip, detail_clip)
    lab[:, :, 0] = np.clip((l + detail) * 255.0 + 0.5, 0, 255).astype(np.uint8)
    return cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)


def process_one(args: tuple[str, str, int, dict]) -> tuple[str, bool, str]:
    src, dst, size, kw = args
    try:
        with Image.open(src) as im:
            arr = np.array(im.convert("RGB"))
        bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
        res = upscale(bgr, size, **kw)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        Image.fromarray(cv2.cvtColor(res, cv2.COLOR_BGR2RGB)).save(
            dst, "PNG", optimize=True, compress_level=9)
        return src, True, ""
    except Exception as exc:  # noqa: BLE001
        return src, False, repr(exc)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=".", help="source dir or file (searched recursively for *.png)")
    ap.add_argument("--dst", default="icons_256", help="output dir (tree is mirrored)")
    ap.add_argument("--size", type=int, default=SIZE)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 1)
    ap.add_argument("--amount", type=float, default=0.55, help="sharpening strength")
    ap.add_argument("--radius", type=float, default=1.0, help="sharpening radius")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()

    src_root = os.path.abspath(a.src)
    dst_root = os.path.abspath(a.dst)
    jobs: list[tuple[str, str, int, dict]] = []
    if os.path.isfile(src_root):
        files = [src_root]
        base = os.path.dirname(src_root)
    else:
        files, base = [], src_root
        for root, dirs, names in os.walk(src_root):
            dirs[:] = [d for d in dirs if not d.startswith(".") and d not in ("icons_256", "tools")]
            for n in sorted(names):
                if n.lower().endswith(".png"):
                    files.append(os.path.join(root, n))
    kw = dict(amount=a.amount, radius=a.radius)
    for f in files:
        rel = os.path.relpath(f, base)
        out = os.path.join(dst_root, rel)
        if a.skip_existing and os.path.exists(out):
            continue
        jobs.append((f, out, a.size, kw))

    total = len(jobs)
    print(f"{total} images -> {dst_root} ({a.size}x{a.size})", flush=True)
    ok = bad = 0
    with Pool(max(1, a.jobs)) as pool:
        for i, (src, success, err) in enumerate(pool.imap_unordered(process_one, jobs, chunksize=8), 1):
            if success:
                ok += 1
            else:
                bad += 1
                print("FAIL", src, err, flush=True)
            if i % 250 == 0 or i == total:
                print(f"  {i}/{total}", flush=True)
    print(f"done: {ok} ok, {bad} failed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
