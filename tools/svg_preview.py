#!/usr/bin/env python3
"""
Render a traced vector to PNG without any SVG library.

  python tools/svg_preview.py icons/treasure/gem.svg [out.png]

The tracer emits a small, well defined subset of SVG - absolute paths made of
M / L / C / Z with a flat fill - so a 100 line rasteriser is enough to check
the result. Useful because the sandbox (and plenty of build servers) have no
SVG renderer at all.
"""

from __future__ import annotations

import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

from PIL import Image, ImageChops, ImageDraw

BEZIER_STEPS = 12


def parse_path(d: str):
    """Return a list of subpaths, each a list of (x, y) points."""
    tokens = re.findall(r"[MLCZmlcz]|-?\d*\.?\d+(?:e[-+]?\d+)?", d)
    subpaths: list[list[tuple[float, float]]] = []
    current: list[tuple[float, float]] = []
    x = y = 0.0
    start = (0.0, 0.0)
    i = 0

    def bezier(p0, p1, p2, p3):
        for step in range(1, BEZIER_STEPS + 1):
            t = step / BEZIER_STEPS
            u = 1 - t
            current.append((
                u**3 * p0[0] + 3 * u**2 * t * p1[0] + 3 * u * t**2 * p2[0] + t**3 * p3[0],
                u**3 * p0[1] + 3 * u**2 * t * p1[1] + 3 * u * t**2 * p2[1] + t**3 * p3[1],
            ))

    while i < len(tokens):
        token = tokens[i]
        if token == "M":
            x, y = float(tokens[i + 1]), float(tokens[i + 2])
            i += 3
            if current:
                subpaths.append(current)
            current = [(x, y)]
            start = (x, y)
        elif token == "L":
            x, y = float(tokens[i + 1]), float(tokens[i + 2])
            current.append((x, y))
            i += 3
        elif token == "C":
            c1 = (float(tokens[i + 1]), float(tokens[i + 2]))
            c2 = (float(tokens[i + 3]), float(tokens[i + 4]))
            end = (float(tokens[i + 5]), float(tokens[i + 6]))
            bezier((x, y), c1, c2, end)
            x, y = end
            i += 7
        elif token == "Z":
            if current:
                current.append(start)
                subpaths.append(current)
            current = []
            x, y = start
            i += 1
        else:
            i += 1

    if current:
        subpaths.append(current)
    return subpaths


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def parse_transform(value: str | None) -> tuple[float, float, float]:
    """Return (scale_x, scale_y, translate_x, translate_y) for an SVG transform."""
    sx = sy = 1.0
    tx = ty = 0.0
    if not value:
        return sx, sy, tx, ty
    for name, args in re.findall(r"(translate|scale|matrix)\s*\(([^)]*)\)", value):
        numbers = [float(n) for n in re.findall(r"-?\d*\.?\d+(?:e[-+]?\d+)?", args)]
        if name == "translate":
            tx += numbers[0]
            ty += numbers[1] if len(numbers) > 1 else 0.0
        elif name == "scale":
            sx *= numbers[0]
            sy *= numbers[1] if len(numbers) > 1 else numbers[0]
        elif name == "matrix" and len(numbers) == 6:
            sx *= numbers[0]
            sy *= numbers[3]
            tx += numbers[4]
            ty += numbers[5]
    return sx, sy, tx, ty


def render(svg_path: Path, out_path: Path, size: int = 512, background=(0, 0, 0, 0)) -> int:
    tree = ET.parse(svg_path)
    root = tree.getroot()
    width = float(root.get("width", size))
    height = float(root.get("height", size))

    canvas = Image.new("RGBA", (size, size), background)
    for element in root.iter():
        if not element.tag.endswith("path"):
            continue
        d = element.get("d")
        if not d:
            continue
        fill = element.get("fill", "#000000")
        colour = hex_to_rgb(fill) if fill.startswith("#") else (0, 0, 0)
        opacity = float(element.get("fill-opacity", 1) or 1)
        tsx, tsy, ttx, tty = parse_transform(element.get("transform"))

        layer = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        scale_x = size / width
        scale_y = size / height

        # a path may carry several subpaths that wind around each other; the
        # holes only survive if the subpaths are combined instead of filled
        # one by one, so they are combined with the even-odd rule
        mask = Image.new("L", (size, size), 0)
        for points in parse_path(d):
            scaled = [
                (px * tsx * scale_x + ttx * scale_x, py * tsy * scale_y + tty * scale_y)
                for px, py in points
            ]
            if len(scaled) < 3:
                continue
            piece = Image.new("L", (size, size), 0)
            ImageDraw.Draw(piece).polygon(scaled, fill=255)
            mask = ImageChops.difference(mask, piece)

        layer.paste(colour + (int(255 * opacity),), (0, 0), mask)
        canvas.alpha_composite(layer)

    canvas.save(out_path)
    return sum(1 for e in root.iter() if e.tag.endswith("path"))


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: svg_preview.py <file.svg> [out.png] [size]", file=sys.stderr)
        return 1
    source = Path(sys.argv[1])
    target = Path(sys.argv[2]) if len(sys.argv) > 2 else source.with_suffix(".png")
    size = int(sys.argv[3]) if len(sys.argv) > 3 else 512
    paths = render(source, target, size)
    print(f"wrote {target} ({paths} paths)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
