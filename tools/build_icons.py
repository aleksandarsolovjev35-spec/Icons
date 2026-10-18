#!/usr/bin/env python3
"""
Build the icon set from the generated originals.

  python tools/build_icons.py            build everything listed in library.json
  python tools/build_icons.py --only gem build a single icon

For every entry it writes
    icons/<category>/<name>.png    master, 512x512, transparent
    icons/<category>/<name>.svg    traced vector
    icons/<category>/<name>.webp   200x200, ready to drop into Foundry

The vector is traced from the flat artwork, which is why a flat silhouette
style produces a handful of clean paths instead of hundreds of noisy ones.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_alpha import cut, normalise  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "src" / "raw"
ICONS = ROOT / "icons"

MASTER = 512
EXPORT = 200

# Tracing parameters. The artwork is posterised to a handful of flat colours
# first, and the tracer is then run aggressively: smooth gradients in the
# original otherwise turn into hundreds of tiny paths, and the posterised
# hard edges need a generous speckle filter to stay clean.
TRACE = dict(
    colormode="color",
    filter_speckle=48,
    corner_threshold=180,
    length_threshold=10.0,
    splice_threshold=90,
    layer_difference=32,
    max_iterations=10,
    path_precision=1,
)

# how many colours the artwork is reduced to before it is traced
PALETTE = 6


def posterise(img: Image.Image, colours: int = PALETTE) -> Image.Image:
    """Reduce the artwork to a handful of flat colours.

    The tracer gives every facet its own slightly different hex, which both
    bloats the vector and makes the colours impossible to recolour. Collapsing
    them first keeps the paths and the palette under control.
    """
    rgb = img.convert("RGB")
    alpha = img.getchannel("A")

    # paint the transparent area with the average colour of the subject, so it
    # does not drag the palette towards black
    visible = alpha.point(lambda a: 255 if a > 8 else 0)
    if visible.getbbox() is None:
        return img
    flat = Image.new("RGB", img.size, rgb.getpixel(visible.getbbox()[:2]))
    flat.paste(rgb, mask=visible)

    quantised = flat.quantize(colors=colours, method=Image.Quantize.FASTOCTREE).convert("RGB")
    result = quantised.convert("RGBA")
    result.putalpha(alpha)
    return result


def trace_svg(png: Path, svg: Path) -> int:
    """Raster -> vector. Returns the number of paths in the result."""
    try:
        import vtracer
    except ImportError:
        print("  (vtracer not installed, skipping the vector)", file=sys.stderr)
        return 0

    vtracer.convert_image_to_svg_py(str(png), str(svg), **TRACE)
    return svg.read_text(encoding="utf-8").count("<path")


def build(icon_id: str, entry: dict, make_svg: bool = True, flat: bool = False) -> bool:
    source = RAW / f"{entry['raw']}.png"
    if not source.exists():
        print(f"  skip {icon_id}: no {source.relative_to(ROOT)}")
        return False

    with Image.open(source) as img:
        master = normalise(cut(img), size=MASTER)

    if flat:
        master = posterise(master, PALETTE)

    target = ICONS / f"{icon_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    master.save(target, optimize=True)

    master.resize((EXPORT, EXPORT), Image.LANCZOS).save(
        ICONS / f"{icon_id}.webp", "WEBP", quality=92, method=6, exact=True
    )

    paths = 0
    if make_svg:
        flat = ICONS / f"{icon_id}.flat.png"
        posterise(master, PALETTE).save(flat)
        paths = trace_svg(flat, ICONS / f"{icon_id}.svg")
        flat.unlink()

    size = target.stat().st_size / 1024
    svg_size = (ICONS / f"{icon_id}.svg").stat().st_size / 1024 if paths else 0
    print(f"  {icon_id:<28} png {size:6.1f} KB   svg {svg_size:6.1f} KB ({paths} paths)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="build a single icon id")
    ap.add_argument("--no-svg", action="store_true")
    ap.add_argument("--flat", action="store_true",
                    help="also flatten the master PNG to the trace palette")
    args = ap.parse_args()

    library = json.loads((ROOT / "library.json").read_text(encoding="utf-8"))["icons"]

    built = 0
    for icon_id, entry in sorted(library.items()):
        if args.only and icon_id != args.only:
            continue
        if build(icon_id, entry, make_svg=not args.no_svg, flat=args.flat):
            built += 1

    print(f"\nbuilt {built} icon(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
