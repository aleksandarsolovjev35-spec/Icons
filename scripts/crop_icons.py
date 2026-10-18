#!/usr/bin/env python3
"""Crop the framed icon tiles from the source contact sheets.

Requires ImageMagick's ``convert`` and ``identify`` commands. Original sheets
are never modified. Extracted tiles are stored in one ZIP to keep the repository
compact and the number of tracked files small.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import tempfile
import zipfile
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}
ARCHIVE_README = """Cropped icon tiles

Each source image with a visible rectangular frame was split into individual
icons. Files are arranged as:
  <source folder>/<source image name without extension>/icon_001.jpg ... icon_050.jpg

Tile order is left-to-right, then top-to-bottom. The frame was included in the
crop, and a 4-pixel band around each tile's edge was painted black. Unframed
sheets are not included and were left unchanged in the repository. Original
sheets are also left unchanged.
"""


class DetectionError(RuntimeError):
    """Raised when an image looks framed but its grid cannot be located safely."""


def source_images() -> list[Path]:
    images = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        relative = path.relative_to(ROOT)
        if any(part.startswith(".") for part in relative.parts):
            continue
        if any(part in {"cropped_icons", "cropped-icons"} for part in relative.parts):
            continue
        images.append(path)
    return sorted(images)


def load_pixels(path: Path) -> tuple[int, int, bytes]:
    dimensions = subprocess.run(
        ["identify", "-format", "%w %h", str(path)],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    ).stdout
    width, height = (int(value) for value in dimensions.split())
    raw = subprocess.run(
        ["convert", str(path), "-depth", "8", "rgb:-"],
        check=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    ).stdout
    if len(raw) != width * height * 3:
        raise DetectionError(f"Unexpected decoded pixel count in {path}")
    return width, height, raw


def border_scores(width: int, height: int, pixels: bytes) -> tuple[list[int], list[int]]:
    """Count neutral mid/dark pixels by column and row; frame strokes are gray."""
    columns = [0] * width
    rows = [0] * height
    for offset in range(0, len(pixels), 3):
        red, green, blue = pixels[offset], pixels[offset + 1], pixels[offset + 2]
        if max(red, green, blue) - min(red, green, blue) > 18:
            continue
        luminance = (red + green + blue) // 3
        if not 24 <= luminance <= 180:
            continue
        pixel_index = offset // 3
        y, x = divmod(pixel_index, width)
        columns[x] += 1
        rows[y] += 1
    return columns, rows


def line_candidates(scores: list[int], minimum: float) -> list[tuple[int, int]]:
    """Find local score peaks and merge adjacent JPEG/compression peaks."""
    peaks: list[tuple[int, int]] = []
    for index in range(1, len(scores) - 1):
        value = scores[index]
        if value >= minimum and value >= scores[index - 1] and value >= scores[index + 1]:
            peaks.append((index, value))
    if scores[0] >= minimum and scores[0] >= scores[1]:
        peaks.insert(0, (0, scores[0]))
    if scores[-1] >= minimum and scores[-1] >= scores[-2]:
        peaks.append((len(scores) - 1, scores[-1]))

    groups: list[list[tuple[int, int]]] = []
    for peak in peaks:
        if not groups or peak[0] - groups[-1][-1][0] > 5:
            groups.append([peak])
        else:
            groups[-1].append(peak)
    return [
        (max(group, key=lambda item: item[1])[0], max(value for _, value in group))
        for group in groups
    ]


def choose_grid_lines(scores: list[int], count: int, axis: str) -> list[int] | None:
    """Choose a regular alternating sequence of tile edges and gutters."""
    maximum = max(scores, default=0)
    # A genuine frame crosses most of each side; unframed images score far lower.
    minimum_frame_score = 350 if axis == "x" else 650
    if maximum < minimum_frame_score:
        return None

    floor = max(250 if axis == "x" else 400, maximum * 0.25)
    candidates = line_candidates(scores, floor)
    if len(candidates) < count:
        return None

    # Dynamic programming over ordered peaks. Every other distance is a tile
    # edge (~135-165 px); the alternating distance is the inter-tile gutter.
    best: dict[int, tuple[float, list[int]]] = {
        index: (float(value), [index])
        for index, (_, value) in enumerate(candidates)
    }
    for selected_count in range(2, count + 1):
        next_best: dict[int, tuple[float, list[int]]] = {}
        is_tile_width = (selected_count - 2) % 2 == 0
        min_distance, max_distance, nominal = (
            (130, 165, 150) if is_tile_width else (15, 60, 28)
        )
        for current, (position, value) in enumerate(candidates):
            for previous, (previous_position, _) in enumerate(candidates):
                if previous not in best:
                    continue
                distance = position - previous_position
                if not min_distance <= distance <= max_distance:
                    continue
                old_score, old_path = best[previous]
                score = old_score + value - 2 * abs(distance - nominal)
                if current not in next_best or score > next_best[current][0]:
                    next_best[current] = (score, old_path + [current])
        best = next_best
        if not best:
            return None

    final = max(best.values(), key=lambda entry: entry[0])
    chosen = [candidates[index][0] for index in final[1]]
    if len(chosen) != count:
        return None

    widths = [chosen[index + 1] - chosen[index] for index in range(0, count, 2)]
    gutters = [chosen[index + 1] - chosen[index] for index in range(1, count - 2, 2)]
    if not widths or not gutters:
        return None
    if min(widths) < 130 or max(widths) > 165 or max(widths) - min(widths) > 10:
        return None
    if min(gutters) < 15 or max(gutters) > 60 or max(gutters) - min(gutters) > 12:
        return None
    return chosen


def detect_grid(width: int, height: int, pixels: bytes) -> tuple[list[int], list[int]] | None:
    columns, rows = border_scores(width, height, pixels)
    x_lines = choose_grid_lines(columns, 20, "x")
    y_lines = choose_grid_lines(rows, 10, "y")
    if x_lines is None and y_lines is None:
        return None
    if x_lines is None or y_lines is None:
        raise DetectionError("Only part of a 10-by-5 border grid was detected")
    if x_lines[0] < 0 or x_lines[-1] >= width or y_lines[0] < 0 or y_lines[-1] >= height:
        raise DetectionError("Detected frame lies outside the image")
    return x_lines, y_lines


def ppm_tiles(width: int, source_pixels: bytes, x_lines: list[int], y_lines: list[int]) -> tuple[bytes, int]:
    """Build a sequence of PPM tiles, blackening a four-pixel outer band."""
    payload = bytearray()
    number_of_tiles = 0
    for row in range(5):
        top, bottom = y_lines[row * 2], y_lines[row * 2 + 1]
        tile_height = bottom - top + 1
        for column in range(10):
            left, right = x_lines[column * 2], x_lines[column * 2 + 1]
            tile_width = right - left + 1
            if not (130 <= tile_width <= 166 and 130 <= tile_height <= 166):
                raise DetectionError(f"Unexpected crop dimensions: {tile_width}x{tile_height}")
            payload.extend(f"P6\n{tile_width} {tile_height}\n255\n".encode("ascii"))
            for y in range(top, bottom + 1):
                start = (y * width + left) * 3
                line = bytearray(source_pixels[start : start + tile_width * 3])
                if y - top < 4 or bottom - y < 4:
                    line[:] = b"\x00" * len(line)
                else:
                    line[:12] = b"\x00" * 12
                    line[-12:] = b"\x00" * 12
                payload.extend(line)
            number_of_tiles += 1
    return bytes(payload), number_of_tiles


def write_status(
    path: Path,
    processed: dict[str, int],
    skipped: dict[str, list[str]],
    total_images: int,
    total_icons: int,
    archive_name: str,
) -> None:
    lines = [
        "# Журнал вырезания иконок",
        "",
        f"Результат: `{archive_name}`. Исходные листы сохранены без изменений.",
        "Рамки включены в обрезку и закрашены чёрной полосой шириной 4 пикселя по краю.",
        "Порядок `icon_001`–`icon_050` — слева направо, сверху вниз.",
        "",
        f"**Вырезано:** {len(processed)} папок, {total_images} листов, {total_icons:,} иконок.",
        "",
        "## Папки, где вырезал",
        "",
        "| Папка | Листов с рамкой | Иконок |",
        "|---|---:|---:|",
    ]
    for folder in sorted(processed, key=str.casefold):
        sheet_count = processed[folder]
        lines.append(f"| `{folder}` | {sheet_count} | {sheet_count * 50} |")

    lines += [
        "",
        f"## Папки, где не вырезал — рамок нет ({len(skipped)})",
        "",
        "В этих папках изображения оставлены как есть.",
        "",
    ]
    for folder in sorted(skipped, key=str.casefold):
        names = ", ".join(f"`{name}`" for name in sorted(skipped[folder], key=str.casefold))
        lines.append(f"- `{folder}` — {names}")
    lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archive", default="cropped_icons.zip", help="ZIP output path")
    parser.add_argument("--status", default="CROP_STATUS.md", help="folder status report path")
    parser.add_argument("--quality", type=int, default=88, choices=range(70, 101), metavar="70..100")
    parser.add_argument("--dry-run", action="store_true", help="scan and report only")
    args = parser.parse_args()

    try:
        subprocess.run(["convert", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["identify", "-version"], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (FileNotFoundError, subprocess.CalledProcessError) as error:
        print("ImageMagick (convert and identify) is required.", file=sys.stderr)
        return 2

    images = source_images()
    if not images:
        print("No source images found.", file=sys.stderr)
        return 1

    archive_path = (ROOT / args.archive).resolve()
    status_path = (ROOT / args.status).resolve()
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    status_path.parent.mkdir(parents=True, exist_ok=True)
    temp_archive: Path | None = None
    if not args.dry_run:
        fd, temp_archive_name = tempfile.mkstemp(
            prefix="cropped_icons_", suffix=".zip", dir=archive_path.parent
        )
        os.close(fd)
        temp_archive = Path(temp_archive_name)

    skipped: dict[str, list[str]] = defaultdict(list)
    processed: dict[str, int] = defaultdict(int)
    framed_sheet_count = 0

    def process_page(archive: zipfile.ZipFile, temp_dir: Path, relative: Path,
                     width: int, pixels: bytes, x_lines: list[int], y_lines: list[int]) -> None:
        for existing in temp_dir.iterdir():
            existing.unlink()
        payload, count = ppm_tiles(width, pixels, x_lines, y_lines)
        if count != 50:
            raise DetectionError(f"Expected 50 tiles in {relative}, got {count}")
        output_pattern = str(temp_dir / "icon_%03d.jpg")
        subprocess.run(
            [
                "convert",
                "ppm:-",
                "-strip",
                "-quality",
                str(args.quality),
                "-define",
                "jpeg:optimize-coding=true",
                output_pattern,
            ],
            input=payload,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        crops = sorted(temp_dir.glob("icon_*.jpg"))
        if len(crops) != 50:
            raise DetectionError(f"ImageMagick wrote {len(crops)} crops for {relative}")
        base = relative.parent / relative.stem
        for crop_index, crop in enumerate(crops, start=1):
            archive_name = (base / f"icon_{crop_index:03d}.jpg").as_posix()
            archive.write(crop, archive_name)

    try:
        if args.dry_run:
            for index, path in enumerate(images, start=1):
                width, height, pixels = load_pixels(path)
                grid = detect_grid(width, height, pixels)
                relative = path.relative_to(ROOT)
                if grid is None:
                    skipped[relative.parent.as_posix()].append(relative.name)
                else:
                    processed[relative.parent.as_posix()] += 1
                    framed_sheet_count += 1
                if index % 25 == 0 or index == len(images):
                    print(f"Scanned {index}/{len(images)} sheets", flush=True)
        else:
            assert temp_archive is not None
            with zipfile.ZipFile(temp_archive, "w", compression=zipfile.ZIP_STORED) as archive:
                archive.writestr("README.txt", ARCHIVE_README)
                with tempfile.TemporaryDirectory(prefix="icon-crops-") as temp_dir_name:
                    temp_dir = Path(temp_dir_name)
                    for index, path in enumerate(images, start=1):
                        width, height, pixels = load_pixels(path)
                        grid = detect_grid(width, height, pixels)
                        relative = path.relative_to(ROOT)
                        if grid is None:
                            skipped[relative.parent.as_posix()].append(relative.name)
                        else:
                            x_lines, y_lines = grid
                            process_page(archive, temp_dir, relative, width, pixels, x_lines, y_lines)
                            processed[relative.parent.as_posix()] += 1
                            framed_sheet_count += 1
                        if index % 25 == 0 or index == len(images):
                            print(f"Processed {index}/{len(images)} sheets", flush=True)

        total_icons = framed_sheet_count * 50
        print(
            f"\nDetected frames: {framed_sheet_count} sheets / {total_icons:,} icons "
            f"in {len(processed)} folders.",
            flush=True,
        )
        if skipped:
            print("No frames detected (left untouched):", flush=True)
            for folder in sorted(skipped, key=str.casefold):
                print(f"  {folder}", flush=True)
        if args.dry_run:
            return 0

        assert temp_archive is not None
        os.replace(temp_archive, archive_path)
        write_status(status_path, processed, skipped, framed_sheet_count, total_icons, archive_path.name)
    except Exception:
        if temp_archive is not None:
            temp_archive.unlink(missing_ok=True)
        raise

    print(f"\nSaved: {archive_path.relative_to(ROOT)}", flush=True)
    print(f"Status: {status_path.relative_to(ROOT)}", flush=True)
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except DetectionError as error:
        print(f"Image processing stopped safely: {error}", file=sys.stderr)
        raise SystemExit(2)
