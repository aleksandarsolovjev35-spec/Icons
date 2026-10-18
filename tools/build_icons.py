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

# tracing parameters, tuned for flat artwork
TRACE = dict(
    colormode="color",
    filter_speckle=4,
    corner_threshold=60,
    length_threshold=4.0,
    max_iterations=10,
    splice_threshold=45,
    layer_difference=8,
)


def trace_svg(png: Path, svg: Path) -> int:
    """Raster -> vector. Returns the number of paths in the result."""
    try:
        import vtracer
    except ImportError:
        print("  (vtracer not installed, skipping the vector)", file=sys.stderr)
        return 0

    vtracer.convert_image_to_svg_py(str(png), str(svg), **TRACE)
    return svg.read_text(encoding="utf-8").count("<path")


def build(icon_id: str, entry: dict, make_svg: bool = True) -> bool:
    source = RAW / f"{entry['raw']}.png"
    if not source.exists():
        print(f"  skip {icon_id}: no {source.relative_to(ROOT)}")
        return False

    with Image.open(source) as img:
        master = normalise(cut(img), size=MASTER)

    target = ICONS / f"{icon_id}.png"
    target.parent.mkdir(parents=True, exist_ok=True)
    master.save(target, optimize=True)

    master.resize((EXPORT, EXPORT), Image.LANCZOS).save(
        ICONS / f"{icon_id}.webp", "WEBP", quality=92, method=6, exact=True
    )

    paths = 0
    if make_svg:
        paths = trace_svg(target, ICONS / f"{icon_id}.svg")

    size = target.stat().st_size / 1024
    svg_size = (ICONS / f"{icon_id}.svg").stat().st_size / 1024 if paths else 0
    print(f"  {icon_id:<28} png {size:6.1f} KB   svg {svg_size:6.1f} KB ({paths} paths)")
    return True


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", help="build a single icon id")
    ap.add_argument("--no-svg", action="store_true")
    args = ap.parse_args()

    library = json.loads((ROOT / "library.json").read_text(encoding="utf-8"))["icons"]

    built = 0
    for icon_id, entry in sorted(library.items()):
        if args.only and icon_id != args.only:
            continue
        if build(icon_id, entry, make_svg=not args.no_svg):
            built += 1

    print(f"\nbuilt {built} icon(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
