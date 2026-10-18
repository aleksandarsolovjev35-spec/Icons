#!/usr/bin/env python3
"""
Build the module's data file from the shipped icons and validate the rules.

  python tools/build_library.py [--check]

Writes data/icons.json:
    { "version": 1, "icons": { "<category>/<name>": "icons/<category>/<name>.webp" }, ... }

and reports every rule that points at an icon which does not exist.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
ICON_DIR = ROOT / "icons"
DATA_DIR = ROOT / "data"
EXPORT_SIZE = 200


def scan_icons() -> dict[str, str]:
    icons: dict[str, str] = {}
    for path in sorted(ICON_DIR.rglob("*.webp")):
        rel = path.relative_to(ICON_DIR).with_suffix("").as_posix()
        icons[rel] = f"icons/{rel}.webp"
    return icons


def check_sizes(icons: dict[str, str]) -> list[str]:
    from PIL import Image

    problems = []
    for rel in icons:
        path = ICON_DIR / f"{rel}.webp"
        with Image.open(path) as im:
            if im.size != (EXPORT_SIZE, EXPORT_SIZE):
                problems.append(f"{rel}: {im.size[0]}x{im.size[1]} (expected {EXPORT_SIZE})")
            if im.mode not in ("RGBA", "LA"):
                problems.append(f"{rel}: mode {im.mode} has no alpha channel")
    return problems


def check_rules(rules: dict, icons: dict[str, str]) -> list[str]:
    problems = []
    for name, icon in (rules.get("exact") or {}).items():
        if icon not in icons:
            problems.append(f"exact[{name!r}] -> {icon} (missing)")
    for rule in rules.get("keywords") or []:
        if rule["icon"] not in icons:
            problems.append(f"keywords -> {rule['icon']} (missing)")
    for kind, icon in (rules.get("type") or {}).items():
        if icon not in icons:
            problems.append(f"type[{kind}] -> {icon} (missing)")
    return problems


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--check", action="store_true", help="only validate, do not write")
    ap.add_argument("--strict", action="store_true", help="also fail when a rule points at a missing icon")
    args = ap.parse_args()

    icons = scan_icons()
    if not icons:
        print("no icons found in icons/", file=sys.stderr)
        return 1

    rules_path = DATA_DIR / "rules.json"
    rules = json.loads(rules_path.read_text(encoding="utf-8")) if rules_path.exists() else {}

    size_problems = check_sizes(icons)
    rule_problems = check_rules(rules, icons)

    categories: dict[str, list[str]] = {}
    for rel in icons:
        category = rel.split("/")[0] if "/" in rel else "misc"
        categories.setdefault(category, []).append(rel)

    payload = {
        "version": 1,
        "count": len(icons),
        "icons": icons,
        "categories": {k: sorted(v) for k, v in sorted(categories.items())},
    }

    out = DATA_DIR / "icons.json"
    if not args.check:
        out.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(f"icons: {len(icons)} in {len(categories)} categories")
    for category, names in sorted(categories.items()):
        print(f"  {category:<14} {len(names)}")
    if size_problems:
        print("\nsize problems:", file=sys.stderr)
        for p in size_problems:
            print(f"  {p}", file=sys.stderr)
    if rule_problems:
        print("\nrule warnings (icon not built yet):", file=sys.stderr)
        for p in rule_problems:
            print(f"  {p}", file=sys.stderr)
    if not args.check:
        print(f"\nwrote {out.relative_to(ROOT)}")

    if size_problems:
        return 1
    if rule_problems and args.strict:
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
