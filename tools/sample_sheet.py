#!/usr/bin/env python3
"""
Quality sheet for a finished sample: PNG vs vector, big vs 40px.

  python tools/sample_sheet.py "S*-poster-tilt.png"

Four columns per icon:
    master PNG | traced SVG | PNG at 40px | SVG at 40px

The last two columns are the ones that matter: that is how an item icon
actually appears in a Foundry inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from build_icons import PALETTE, posterise  # noqa: E402
from extract_alpha import cut, normalise  # noqa: E402
from svg_preview import render  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
STYLES = ROOT / "src" / "styles"
WORK = ROOT / "preview" / "_sample"

BG = (24, 26, 33, 255)
PANEL = (36, 39, 49, 255)
TEXT = (214, 219, 232, 255)

BIG = 190
SMALL = 40
PAD = 12
LABEL = 22


def main() -> int:
    pattern = sys.argv[1] if len(sys.argv) > 1 else "*.png"
    files = sorted(p for p in STYLES.glob(pattern))
    if not files:
        print(f"no candidates matching {pattern}", file=sys.stderr)
        return 1

    WORK.mkdir(parents=True, exist_ok=True)
    import vtracer

    columns = 4
    rows = len(files)
    width = columns * (BIG + PAD) + PAD
    height = rows * (BIG + LABEL + SMALL + PAD * 2) + PAD

    canvas = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for index, path in enumerate(files):
        y = PAD + index * (BIG + LABEL + SMALL + PAD * 2)

        with Image.open(path) as img:
            master = normalise(cut(img), size=BIG)

        flat = WORK / f"{path.stem}.flat.png"
        posterise(master, PALETTE).save(flat)
        svg = WORK / f"{path.stem}.svg"
        vtracer.convert_image_to_svg_py(
            str(flat), str(svg), colormode="color", filter_speckle=4,
            corner_threshold=60, length_threshold=4.0, max_iterations=10,
            splice_threshold=45, layer_difference=8,
        )
        rendered = WORK / f"{path.stem}.render.png"
        render(svg, rendered, BIG, background=PANEL)

        cells = [
            (master, "PNG"),
            (Image.open(rendered).convert("RGBA"), "SVG"),
            (master.resize((SMALL, SMALL), Image.LANCZOS), "40px PNG"),
            (Image.open(rendered).convert("RGBA").resize((SMALL, SMALL), Image.LANCZOS), "40px SVG"),
        ]

        for column, (art, label) in enumerate(cells):
            x = PAD + column * (BIG + PAD)
            if column < 2:
                draw.rounded_rectangle([x, y, x + BIG, y + BIG], radius=8, fill=PANEL)
                canvas.alpha_composite(art, (x, y))
                draw.text((x + 3, y + BIG + 5), f"{label}  {path.stem}", fill=TEXT)
            else:
                canvas.alpha_composite(art, (x + 3, y + BIG + LABEL))
                draw.text((x + 3, y + BIG + 5), label, fill=TEXT)

        svg_kb = svg.stat().st_size / 1024
        paths = svg.read_text(encoding="utf-8").count("<path")
        draw.text((PAD + BIG + PAD + 4, y + BIG + 5), f"{svg_kb:.1f} KB, {paths} paths", fill=TEXT)

    out = ROOT / "preview" / "sample-sheet.png"
    canvas.convert("RGB").save(out)
    print(f"wrote {out} ({len(files)} icons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
