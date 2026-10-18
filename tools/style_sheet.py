#!/usr/bin/env python3
"""
Build a comparison sheet for style exploration.

  python tools/style_sheet.py [dir]

Lays out every candidate in the folder twice: large, to judge the drawing, and
at 40px, which is how an item icon actually looks inside a Foundry inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_alpha import cut, normalise  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

BG = (24, 26, 33, 255)
PANEL = (36, 39, 49, 255)
TEXT = (216, 221, 234, 255)

BIG = 220
SMALL = 40
PAD = 14
LABEL = 22


def main() -> int:
    folder = Path(sys.argv[1]) if len(sys.argv) > 1 else ROOT / "src" / "styles"
    files = sorted(p for p in folder.glob("*.png"))
    if not files:
        print(f"no images in {folder}", file=sys.stderr)
        return 1

    columns = max(1, len(files) // 2)
    rows = (len(files) + columns - 1) // columns
    cell_h = BIG + LABEL + SMALL + PAD * 2
    width = columns * (BIG + PAD) + PAD
    height = rows * cell_h + PAD

    canvas = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for index, path in enumerate(files):
        row, col = divmod(index, columns)
        x = PAD + col * (BIG + PAD)
        y = PAD + row * cell_h

        with Image.open(path) as img:
            # line art is a stroke, not a filled shape: keep the inside empty
            art = normalise(cut(img, fill_interior="ink-line" not in path.stem), size=BIG)

        draw.rounded_rectangle([x, y, x + BIG, y + BIG], radius=8, fill=PANEL)
        canvas.alpha_composite(art, (x, y))
        draw.text((x + 2, y + BIG + 4), path.stem, fill=TEXT)

        # 40px, the size that actually matters
        canvas.alpha_composite(art.resize((SMALL, SMALL), Image.LANCZOS), (x + 2, y + BIG + LABEL))

    out = ROOT / "preview" / "style-sheet.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)
    print(f"wrote {out} ({len(files)} candidates)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
