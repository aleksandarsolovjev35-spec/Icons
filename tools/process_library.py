#!/usr/bin/env python3
"""
Batch-process every generated raw image into the shipped library.

  python tools/process_library.py

For each src/raw/<id>.png it writes
    src/processed/<id>.png     master, 512x512 RGBA
    icons/<category>/<id>.webp  shipped asset, 200x200 RGBA webp
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from extract_alpha import EXPORT, MASTER, SUBJECT_FILL, cut, normalise  # noqa: E402
from PIL import Image  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RAW = ROOT / "src" / "raw"
PROCESSED = ROOT / "src" / "processed"
ICONS = ROOT / "icons"

# raw file name -> library id (category/name)
CATEGORIES = {
    "flask-potion": "consumables/flask-potion",
    "vial": "consumables/vial",
    "bottle": "consumables/bottle",
    "scroll": "consumables/scroll",
    "book": "adventuring/book",
    "torch": "adventuring/torch",
    "lantern": "adventuring/lantern",
    "rope": "adventuring/rope",
    "backpack": "containers/backpack",
    "chest": "containers/chest",
    "sack": "containers/sack",
    "ingot": "treasure/ingot",
    "coin-stack": "treasure/coin-stack",
    "gem": "treasure/gem",
    "ring": "treasure/ring",
    "ore": "materials/ore",
    "mushroom": "nature/mushroom",
    "bread": "food/bread",
}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--fill", type=float, default=SUBJECT_FILL)
    args = ap.parse_args()

    PROCESSED.mkdir(parents=True, exist_ok=True)

    done = 0
    for stem, icon_id in sorted(CATEGORIES.items()):
        source = RAW / f"{stem}.png"
        if not source.exists():
            print(f"  skip {stem} (no raw file)")
            continue

        master = normalise(cut(source), fill=args.fill, size=MASTER)
        master_path = PROCESSED / f"{stem}.png"
        master.save(master_path, optimize=True)

        webp_path = ICONS / f"{icon_id}.webp"
        webp_path.parent.mkdir(parents=True, exist_ok=True)
        master.resize((EXPORT, EXPORT), Image.LANCZOS).save(
            webp_path, "WEBP", quality=92, method=6, exact=True
        )

        weight = round(webp_path.stat().st_size / 1024, 1)
        print(f"  {icon_id:<32} {master_path.name}  ->  {webp_path.name} ({weight} KB)")
        done += 1

    print(f"\nprocessed {done} icon(s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
