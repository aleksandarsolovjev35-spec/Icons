#!/usr/bin/env python3
"""
Recolour traced icons - a palette study that costs no generation at all.

  python tools/recolor.py icons/treasure/gem.svg --matrix preview/palette.png

Because the traced artwork is made of flat fills, a palette swap is a matter
of remapping the fill colours. The role of each colour is guessed from its
luminance and saturation:

    light  (very bright, unsaturated)  -> a tint of the accent
    accent (the most saturated colour)  -> the accent
    dark   (everything else)            -> the body colour
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from svg_preview import parse_path, render  # noqa: E402

# name, body colour, accent colour
PALETTES = [
    ("black + red", "#141414", "#C8202E"),
    ("navy + gold", "#1B2437", "#E0A32E"),
    ("light grey + teal", "#C9CDD6", "#1F8A8C"),
    ("charcoal + violet", "#23252B", "#8B5CF6"),
    ("black + emerald", "#141414", "#2E9E5B"),
]


def hex_to_rgb(value: str) -> tuple[int, int, int]:
    value = value.strip().lstrip("#")
    if len(value) == 3:
        value = "".join(c * 2 for c in value)
    return tuple(int(value[i:i + 2], 16) for i in (0, 2, 4))


def rgb_to_hex(rgb) -> str:
    return "#{:02X}{:02X}{:02X}".format(*rgb)


def mix(a, b, t: float):
    return tuple(round(x + (y - x) * t) for x, y in zip(a, b))


def colour_families(svg: str) -> dict[str, str]:
    """Group every colour used in the SVG into 'accent', 'light' or 'dark'.

    The tracer emits a slightly different hex for every facet, so colours are
    grouped by family rather than matched exactly.
    """
    fills = re.findall(r'(?:fill|stroke)="(#[0-9a-fA-F]{3,6})"', svg)
    if not fills:
        return {}

    stats = {}
    for fill in set(fills):
        rgb = hex_to_rgb(fill)
        luminance = 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2]
        chroma = max(rgb) - min(rgb)
        stats[fill] = (luminance, chroma)

    peak_chroma = max(c for _, c in stats.values())
    families = {}
    for fill, (luminance, chroma) in stats.items():
        if chroma >= 40 and chroma >= peak_chroma * 0.45:
            families[fill] = "accent"
        elif luminance > 200 and chroma < 40:
            families[fill] = "light"
        else:
            families[fill] = "dark"
    return families


def recolour(svg: str, body: str, accent: str) -> str:
    """Remap the palette while keeping the relative lightness of every tone.

    Faceted artwork relies on several shades of one hue, so each family is
    rescaled instead of being flattened into a single flat colour.
    """
    import colorsys

    families = colour_families(svg)
    if not families:
        return svg

    body_rgb = hex_to_rgb(body)
    accent_rgb = hex_to_rgb(accent)
    body_h, body_l, body_s = colorsys.rgb_to_hls(*[c / 255 for c in body_rgb])
    accent_h, accent_l, accent_s = colorsys.rgb_to_hls(*[c / 255 for c in accent_rgb])

    # the lightest tone of a family anchors the family
    anchors = {}
    for family in set(families.values()):
        members = [f for f, r in families.items() if r == family]
        anchors[family] = max(
            members,
            key=lambda f: 0.2126 * hex_to_rgb(f)[0] + 0.7152 * hex_to_rgb(f)[1] + 0.0722 * hex_to_rgb(f)[2],
        )

    mapping = {}
    for fill, family in families.items():
        rgb = hex_to_rgb(fill)
        if family == "light":
            mapping[fill] = rgb_to_hex(mix(accent_rgb, (255, 255, 255), 0.72))
            continue

        target_h, target_l, target_s = (accent_h, accent_l, accent_s) if family == "accent" else (body_h, body_l, body_s)
        anchor_l = colorsys.rgb_to_hls(*[c / 255 for c in hex_to_rgb(anchors[family])])[1]
        src_l = colorsys.rgb_to_hls(*[c / 255 for c in rgb])[1]
        ratio = (src_l / anchor_l) if anchor_l else 1.0
        new = colorsys.hls_to_rgb(target_h, min(0.99, max(0.02, target_l * ratio)), target_s)
        mapping[fill] = rgb_to_hex(tuple(round(c * 255) for c in new))

    def replace(match: re.Match) -> str:
        return f'{match.group(1)}="{mapping.get(match.group(2), match.group(2))}"'

    return re.sub(r'(fill|stroke)="(#[0-9a-fA-F]{3,6})"', replace, svg)


def matrix(svgs: list[Path], out: Path, cell: int = 200) -> None:
    from PIL import Image, ImageDraw

    pad = 12
    label_h = 22
    width = len(svgs) * (cell + pad) + pad
    height = len(PALETTES) * (cell + label_h + pad) + pad

    canvas = Image.new("RGBA", (width, height), (36, 39, 49, 255))
    draw = ImageDraw.Draw(canvas)

    for row, (name, body, accent) in enumerate(PALETTES):
        y = pad + row * (cell + label_h + pad)
        draw.text((pad, y + 4), f"{name}   ({body} / {accent})", fill=(216, 221, 234, 255))
        for col, svg_path in enumerate(svgs):
            x = pad + col * (cell + pad)
            svg = recolour(svg_path.read_text(encoding="utf-8"), body, accent)
            tmp = out.parent / f".{svg_path.stem}-{row}.svg"
            tmp.write_text(svg, encoding="utf-8")
            png = tmp.with_suffix(".png")
            render(tmp, png, cell, background=(36, 39, 49, 255))
            canvas.alpha_composite(Image.open(png).convert("RGBA"), (x, y + label_h))
            tmp.unlink()
            png.unlink()

    out.parent.mkdir(parents=True, exist_ok=True)
    canvas.convert("RGB").save(out)
    print(f"wrote {out}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("svg", nargs="+", type=Path)
    ap.add_argument("--matrix", type=Path, help="render a palette matrix instead of single files")
    ap.add_argument("--body", default="#141414")
    ap.add_argument("--accent", default="#C8202E")
    args = ap.parse_args()

    if args.matrix:
        matrix(args.svg, args.matrix)
        return 0

    for source in args.svg:
        target = source.with_name(source.stem + "-recolored.svg")
        target.write_text(
            recolour(source.read_text(encoding="utf-8"), args.body, args.accent),
            encoding="utf-8"
        )
        print(f"wrote {target}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
