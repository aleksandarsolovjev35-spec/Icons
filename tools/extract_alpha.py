#!/usr/bin/env python3
"""
Cut a generated icon out of its flat studio background and normalise it.

The generator is instructed to render every icon on a seamless pure-white
background, which makes the cut deterministic and lossless:

    C = a * F + (1 - a) * 255          (per channel, W = pure white)

so the alpha is recovered from the darkest channel, the background is
confirmed with a border-connected flood fill, and the remaining fringe is
despilled by un-premultiplying against white.

The subject is then re-centred and scaled to a fixed fraction of a square
canvas, which is what keeps a template library visually consistent: every
icon occupies the same amount of visual weight regardless of how the
generator framed it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

# --- tunables -------------------------------------------------------------

LIGHT_TOL = 185      # min channel above this counts as "bright"
CHROMA_TOL = 60       # max channel-min channel above this counts as "saturated"
GAP_AREA = 0.008      # enclosed area (fraction of the frame) that counts as a hole
FLOOD_TOL = 2         # extra slack used by the border flood fill
MIN_ALPHA = 0.02      # below this a pixel is discarded entirely
SUBJECT_FILL = 0.84   # fraction of the canvas the subject should occupy
MASTER = 512          # master PNG size
EXPORT = 200          # shipped webp size


def _alpha_from_white(rgb: np.ndarray) -> np.ndarray:
    """Recover alpha assuming a pure white background (white-screen formula)."""
    return 1.0 - rgb.min(axis=2) / 255.0


def _background_mask(rgb: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Split bright & unsaturated pixels into background and enclosed regions.

    Returns
    -------
    bg      : pixels connected to the image border - the studio backdrop and
              any cast shadow lying on it.
    enclosed: bright regions the subject surrounds, e.g. the white paper of a
              scroll or the see-through middle of a coiled rope.
    """
    lightest = rgb.min(axis=2)
    chroma = rgb.max(axis=2) - lightest
    candidate = (lightest >= LIGHT_TOL) & (chroma <= CHROMA_TOL)

    lab, n = ndimage.label(candidate)
    if n == 0:
        zeros = np.zeros(rgb.shape[:2], dtype=bool)
        return zeros, zeros

    border = set(np.unique(np.concatenate([lab[0, :], lab[-1, :], lab[:, 0], lab[:, -1]])))
    border.discard(0)
    bg = np.isin(lab, list(border)) if border else np.zeros(rgb.shape[:2], dtype=bool)
    bg = ndimage.binary_dilation(bg, iterations=FLOOD_TOL)
    return bg, candidate & (~bg)


def _is_flat_white(mask: np.ndarray, rgb: np.ndarray) -> bool:
    """True for a region that is just the flat studio background.

    Enclosed regions are usually painted highlights (white paper, a specular
    streak on glass) and must stay opaque. The exception is a genuine
    see-through gap - the middle of a coiled rope - which is flat, colourless
    and as bright as the backdrop, and therefore has to stay transparent.
    """
    values = rgb[mask]
    if values.size == 0:
        return False
    chroma = values.max(axis=1) - values.min(axis=1)
    return bool(
        values.min(axis=1).mean() > 240
        and chroma.mean() < 12
        and values.std() < 8
    )


def cut(source: Path) -> Image.Image:
    img = Image.open(source)
    if img.mode != "RGBA":
        img = img.convert("RGBA")

    arr = np.asarray(img).astype(np.float32)
    rgb = arr[..., :3]
    src_alpha = arr[..., 3] / 255.0

    alpha = _alpha_from_white(rgb)
    alpha = np.where(src_alpha < 1.0, src_alpha, alpha)  # respect existing alpha

    # 1. background + cast shadow (border connected) and enclosed bright regions
    bg, enclosed = _background_mask(rgb)

    # 2. an enclosed region is a see-through gap when it is a large, flat,
    #    colourless patch of backdrop; anything else is painted surface
    min_gap = int(GAP_AREA * rgb.shape[0] * rgb.shape[1])
    labels, count = ndimage.label(enclosed)
    gaps = np.zeros_like(bg)
    for index in range(1, count + 1):
        region = labels == index
        if region.sum() >= min_gap and _is_flat_white(region, rgb):
            gaps |= region

    # 3. silhouette: the object plus the painted surface it encloses
    silhouette = (~bg) & (~gaps)
    silhouette = ndimage.binary_closing(silhouette, iterations=2)
    silhouette = ndimage.binary_fill_holes(silhouette) & (~gaps)

    # 4. inside the silhouette the icon is solid; only the outer fringe is soft
    alpha = np.where(silhouette, 1.0, alpha)
    alpha = np.where(bg, 0.0, alpha)
    alpha = np.where(gaps, 0.0, alpha)
    alpha = np.clip(alpha, 0.0, 1.0)
    alpha[alpha < MIN_ALPHA] = 0.0

    # 4. de-fringe the soft edge: un-premultiply against white so the rim
    #    keeps the subject's own colour instead of a white halo
    out = rgb.copy()
    soft = (alpha > 0.0) & (alpha < 0.995)
    if soft.any():
        a = alpha[soft][:, None]
        un = (rgb[soft] - (1.0 - a) * 255.0) / np.maximum(a, 1e-4)
        out[soft] = np.clip(un, 0.0, 255.0)

    rgba = np.dstack([out, alpha * 255.0]).astype(np.uint8)
    return Image.fromarray(rgba, "RGBA")


def normalise(img: Image.Image, fill: float = SUBJECT_FILL, size: int = MASTER) -> Image.Image:
    """Crop to the subject, then centre and scale it to a fixed canvas fill."""
    bbox = img.getbbox()
    if bbox is None:
        raise SystemExit("empty image after cut")
    subject = img.crop(bbox)

    side = max(subject.size)
    side = max(side, 8)
    target = int(round(size * fill))
    scale = target / side
    new = (max(1, int(round(subject.width * scale))), max(1, int(round(subject.height * scale))))
    subject = subject.resize(new, Image.LANCZOS)

    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    canvas.paste(subject, ((size - subject.width) // 2, (size - subject.height) // 2), subject)
    return canvas


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("source", type=Path)
    ap.add_argument("--out", type=Path, required=True, help="output PNG (master)")
    ap.add_argument("--webp", type=Path, help="also write a shipped webp here")
    ap.add_argument("--size", type=int, default=MASTER)
    ap.add_argument("--export", type=int, default=EXPORT)
    ap.add_argument("--fill", type=float, default=SUBJECT_FILL)
    args = ap.parse_args()

    img = normalise(cut(args.source), fill=args.fill, size=args.size)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, optimize=True)

    if args.webp:
        args.webp.parent.mkdir(parents=True, exist_ok=True)
        img.resize((args.export, args.export), Image.LANCZOS).save(
            args.webp, "WEBP", quality=92, method=6, exact=True
        )

    bbox = img.getbbox()
    print(f"{args.source.name} -> {args.out.name} content={bbox} alpha>0.5px={(np.asarray(img)[...,3] > 128).sum()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
