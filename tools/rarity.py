#!/usr/bin/env python3
"""
Rarity colouring: the icon carries the shape, the colour carries the rarity.

  python tools/rarity.py icons/treasure/gem.svg --preview preview/rarity.png
  python tools/rarity.py icons/treasure/gem.svg --export legendary

A neutral icon is a greyscale image, so a rarity colour can be applied by
keeping the lightness of every tone and replacing the hue. The result is a
duotone icon that still reads as a shaded object.

Rarity colours follow the dnd5e convention:

    common     grey, no tint
    uncommon   green
    rare       blue
    very rare  violet
    legendary  gold
    artifact   red
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_preview import hex_to_rgb, render  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

RARITIES = {
    "common": "#9aa0ad",
    "uncommon": "#6fcf5b",
    "rare": "#4aa3ff",
    "very-rare": "#b47cff",
    "legendary": "#ffb03a",
    "artifact": "#ff6b4a",
}


# --------------------------------------------------------------- png tinting

def tint_png(img: Image.Image, colour: str) -> Image.Image:
    """Greyscale the image, then dye it with the rarity colour."""
    red, green, blue = hex_to_rgb(colour)
    alpha = img.getchannel("A") if img.mode == "RGBA" else None
    grey = img.convert("L")
    tinted = Image.merge(
        "RGB",
        (
            grey.point(lambda v: v * red // 255),
            grey.point(lambda v: v * green // 255),
            grey.point(lambda v: v * blue // 255),
        ),
    )
    if alpha is not None:
        tinted = tinted.convert("RGBA")
        tinted.putalpha(alpha)
    return tinted


# --------------------------------------------------------------- svg tinting

def tint_svg(svg: str, colour: str) -> str:
    """Repaint a vector, keeping the lightness of every tone."""
    red, green, blue = hex_to_rgb(colour)

    def replace(match: re.Match) -> str:
        value = match.group(2)
        raw = value.lstrip("#")
        if len(raw) == 3:
            raw = "".join(c * 2 for c in raw)
        r, g, b = (int(raw[i:i + 2], 16) for i in (0, 2, 4))
        luminance = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255
        new = (round(red * luminance), round(green * luminance), round(blue * luminance))
        return f'{match.group(1)}="#{new[0]:02X}{new[1]:02X}{new[2]:02X}"'

    return re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', replace, svg)


# ------------------------------------------------------------------- preview

def preview(sources: list[Path], out: Path, cell: int = 150) -> None:
    """One row per icon, one column per rarity."""
    pad = 12
    label = 20
    width = len(RARITIES) * (cell + pad) + pad
    height = len(sources) * (cell + label + pad) + pad + 24

    canvas = Image.new("RGBA", (width, height), (36, 39, 49, 255))
    draw = ImageDraw.Draw(canvas)

    for column, (name, colour) in enumerate(RARITIES.items()):
        x = pad + column * (cell + pad)
        draw.text((x + 2, 6), f"{name}  {colour}", fill=(214, 219, 232, 255))

    for row, source in enumerate(sources):
        y = pad + 24 + row * (cell + label + pad)
        for column, (name, colour) in enumerate(RARITIES.items()):
            x = pad + column * (cell + pad)

            if source.suffix == ".svg":
                svg = tint_svg(source.read_text(encoding="utf-8"), colour)
                tmp = out.parent / f".{source.stem}-{name}.svg"
                tmp.write_text(svg, encoding="utf-8")
                png = tmp.with_suffix(".png")
                render(tmp, png, cell, background=(36, 39, 49, 255))
                canvas.alpha_composite(Image.open(png).convert("RGBA"), (x, y))
                tmp.unlink()
                png.unlink()
            else:
                with Image.open(source) as img:
                    canvas.alpha_composite(
                        tint_png(img.convert("RGBA"), colour).resize((cell, cell), Image.LANCZOS),
                        (x, y)
                    )
        draw.text((pad, y + cell + 4), source.stem, fill=(160, 168, 188, 255))

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)
    print(f"wrote {out}")


def export(sources: list[Path], rarity: str, out_dir: Path) -> None:
    colour = RARITIES[rarity]
    for source in sources:
        target = out_dir / f"{source.stem}.{rarity}{source.suffix}"
        target.parent.mkdir(parents=True, exist_ok=True)
        if source.suffix == ".svg":
            target.write_text(tint_svg(source.read_text(encoding="utf-8"), colour), encoding="utf-8")
        else:
            with Image.open(source) as img:
                tint_png(img.convert("RGBA"), colour).save(target)
        print(f"wrote {target}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("icons", nargs="+", type=Path)
    ap.add_argument("--preview", type=Path, help="render a rarity matrix")
    ap.add_argument("--export", choices=sorted(RARITIES), help="write tinted copies")
    ap.add_argument("--out", type=Path, default=ROOT / "icons" / "rarity")
    args = ap.parse_args()

    if args.preview:
        preview(args.icons, args.preview)
    if args.export:
        export(args.icons, args.export, args.out)
    if not args.preview and not args.export:
        ap.print_help()
    return 0


if __name__ == "__main__":
    sys.exit(main())
