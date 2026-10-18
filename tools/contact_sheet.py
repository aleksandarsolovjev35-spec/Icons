#!/usr/bin/env python3
"""
Render a contact sheet of the shipped library, for reviewing consistency.

  python tools/contact_sheet.py

Writes
    preview/library-sheet.png       icons at 200px on the Foundry dark background
    preview/library-sheet-small.png the same icons at 40px, i.e. how they
                                   actually look in an inventory list
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "icons"
PREVIEW = ROOT / "preview"

BG = (24, 26, 33, 255)
PANEL = (34, 37, 46, 255)
TEXT = (214, 219, 232, 255)
DIM = (138, 146, 168, 255)

CELL = 200
PAD = 16
LABEL = 22
COLUMNS = 6


def sheet(icons: dict[str, str], cell: int, with_labels: bool) -> Image.Image:
    items = sorted(icons)
    rows = (len(items) + COLUMNS - 1) // COLUMNS
    row_h = cell + (LABEL if with_labels else 0)
    width = COLUMNS * (cell + PAD) + PAD
    height = rows * (row_h + PAD) + PAD

    canvas = Image.new("RGBA", (width, height), BG)
    draw = ImageDraw.Draw(canvas)

    for index, icon_id in enumerate(items):
        row, col = divmod(index, COLUMNS)
        x = PAD + col * (cell + PAD)
        y = PAD + row * (row_h + PAD)

        draw.rounded_rectangle(
            [x, y, x + cell, y + cell], radius=max(4, cell // 16), fill=PANEL
        )

        path = ICON_DIR / f"{icon_id}.webp"
        with Image.open(path) as img:
            art = img.convert("RGBA").resize((cell, cell), Image.LANCZOS)
        canvas.alpha_composite(art, (x, y))

        if with_labels:
            name = icon_id.split("/")[-1]
            draw.text((x + 2, y + cell + 5), name, fill=TEXT)
            draw.text((x + 2, y + cell + 5), "", fill=DIM)

    return canvas


def main() -> int:
    PREVIEW.mkdir(parents=True, exist_ok=True)
    data = json.loads((ROOT / "data" / "icons.json").read_text(encoding="utf-8"))
    icons = data["icons"]
    if not icons:
        print("library is empty", file=sys.stderr)
        return 1

    sheet(icons, CELL, True).convert("RGB").save(PREVIEW / "library-sheet.png")
    sheet(icons, 40, False).convert("RGB").save(PREVIEW / "library-sheet-small.png")
    print(f"wrote {PREVIEW / 'library-sheet.png'} ({len(icons)} icons)")
    print(f"wrote {PREVIEW / 'library-sheet-small.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
