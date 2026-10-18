#!/usr/bin/env python3
"""
Review sheet of the shipped library.

  python tools/library_sheet.py

For every icon in icons/: the master PNG, the traced SVG, and both of them at
40px - the size an item icon actually has inside a Foundry inventory.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_preview import render  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
ICONS = ROOT / "icons"
WORK = ROOT / "preview" / "_lib"

BG = (24, 26, 33, 255)
PANEL = (36, 39, 49, 255)
TEXT = (214, 219, 232, 255)
DIM = (140, 148, 170, 255)

BIG = 180
SMALL = 40
PAD = 10
LABEL = 20
COLUMNS = 2  # icon blocks per row, each block is PNG + SVG


def main() -> int:
    masters = sorted(ICONS.rglob("*.png"))
    if not masters:
        print("library is empty", file=sys.stderr)
        return 1

    WORK.mkdir(parents=True, exist_ok=True)

    block_w = BIG * 2 + PAD
    block_h = BIG + LABEL + SMALL + PAD * 2
    rows = (len(masters) + COLUMNS - 1) // COLUMNS
    width = COLUMNS * (block_w + PAD) + PAD
    height = rows * (block_h + PAD) + PAD

    canvas = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for index, master_path in enumerate(masters):
        row, col = divmod(index, COLUMNS)
        x = PAD + col * (block_w + PAD)
        y = PAD + row * (block_h + PAD)

        with Image.open(master_path) as img:
            master = img.convert("RGBA").resize((BIG, BIG), Image.LANCZOS)

        # PNG column
        draw.rounded_rectangle([x, y, x + BIG, y + BIG], radius=8, fill=PANEL)
        canvas.alpha_composite(master, (x, y))

        # SVG column
        svg_path = master_path.with_suffix(".svg")
        svg_x = x + BIG + PAD
        if svg_path.exists():
            rendered = WORK / f"{master_path.stem}.render.png"
            render(svg_path, rendered, BIG, background=PANEL)
            draw.rounded_rectangle([svg_x, y, svg_x + BIG, y + BIG], radius=8, fill=PANEL)
            canvas.alpha_composite(Image.open(rendered).convert("RGBA"), (svg_x, y))
            kb = svg_path.stat().st_size / 1024
            paths = svg_path.read_text(encoding="utf-8").count("<path")
            draw.text((svg_x + 3, y + BIG + 4), f"svg {kb:.0f} KB / {paths} paths", fill=DIM)
            canvas.alpha_composite(
                Image.open(rendered).convert("RGBA").resize((SMALL, SMALL), Image.LANCZOS),
                (svg_x + 3, y + BIG + LABEL)
            )

        name = master_path.relative_to(ICONS).with_suffix("").as_posix()
        draw.text((x + 3, y + BIG + 4), name, fill=TEXT)
        canvas.alpha_composite(
            master.resize((SMALL, SMALL), Image.LANCZOS), (x + 3, y + BIG + LABEL)
        )
        draw.text((x + 3, y + BIG + LABEL + SMALL + 2), "40px png", fill=DIM)

    out = ROOT / "preview" / "library-sheet.png"
    canvas.convert("RGB").save(out)
    print(f"wrote {out} ({len(masters)} icons)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
