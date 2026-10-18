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
crop, its edge was painted black, and dark neutral compression fringes were
removed. Unframed sheets are not included and were left unchanged. Original
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


def border_scores(width: int, height: int, pixels: bytes) -> tuple[list[int], list[int], bytearray]:
    """Count neutral mid/dark pixels by row and column, retaining their mask."""
    columns = [0] * width
    rows = [0] * height
    mask = bytearray(width * height)
    for offset in range(0, len(pixels), 3):
        red, green, blue = pixels[offset], pixels[offset + 1], pixels[offset + 2]
        if max(red, green, blue) - min(red, green, blue) > 18:
            continue
        luminance = (red + green + blue) // 3
        if not 24 <= luminance <= 180:
            continue
        pixel_index = offset // 3
        y, x = divmod(pixel_index, width)
        mask[pixel_index] = 1
        columns[x] += 1
        rows[y] += 1
    return columns, rows, mask


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


def detect_grid(
    width: int, height: int, pixels: bytes
) -> tuple[list[int], list[int], bytearray] | None:
    columns, rows, mask = border_scores(width, height, pixels)
    x_lines = choose_grid_lines(columns, 20, "x")
    y_lines = choose_grid_lines(rows, 10, "y")
    if x_lines is None and y_lines is None:
        return None
    if x_lines is None or y_lines is None:
        raise DetectionError("Only part of a 10-by-5 border grid was detected")
    if x_lines[0] < 0 or x_lines[-1] >= width or y_lines[0] < 0 or y_lines[-1] >= height:
        raise DetectionError("Detected frame lies outside the image")
    return x_lines, y_lines, mask


def find_local_edge(
    mask: bytearray,
    width: int,
    height: int,
    axis: str,
    expected: int,
    segment_start: int,
    segment_end: int,
    radius: int = 18,
) -> tuple[int, bool]:
    """Refine one side of a tile by following its local gray frame line."""
    if segment_end < segment_start:
        return expected, False
    axis_length = width if axis == "x" else height
    low = max(0, expected - radius)
    high = min(axis_length - 1, expected + radius)
    segment_length = segment_end - segment_start + 1
    best_score = -1
    best_position = expected
    best_distance = radius + 1

    for position in range(low, high + 1):
        if axis == "x":
            score = sum(mask[y * width + position] for y in range(segment_start, segment_end + 1))
        else:
            score = sum(mask[position * width + x] for x in range(segment_start, segment_end + 1))
        distance = abs(position - expected)
        if score > best_score or (score == best_score and distance < best_distance):
            best_score = score
            best_position = position
            best_distance = distance

    # Require a visible line across a substantial part of this side. A nearby
    # art detail can also be gray; only move away from the grid if the local
    # frame peak is meaningfully stronger than the pixels at the grid estimate.
    minimum_score = max(12, int(segment_length * 0.32))
    if axis == "x":
        expected_score = sum(mask[y * width + expected] for y in range(segment_start, segment_end + 1))
    else:
        expected_score = sum(mask[expected * width + x] for x in range(segment_start, segment_end + 1))
    if best_score < minimum_score:
        return expected, False
    if best_position != expected:
        required_improvement = max(18, int(segment_length * 0.18))
        if best_score - expected_score < required_improvement:
            return expected, expected_score >= minimum_score
    return best_position, True


def refine_tile_box(
    mask: bytearray,
    width: int,
    height: int,
    box: tuple[int, int, int, int],
) -> tuple[tuple[int, int, int, int], tuple[int, int, int, int]]:
    """Independently follow each icon's frame instead of assuming a perfect grid."""
    grid_left, grid_right, grid_top, grid_bottom = box
    left, right, top, bottom = grid_left, grid_right, grid_top, grid_bottom
    found = [False, False, False, False]

    # Iterate once so horizontal edges use the locally adjusted left/right span,
    # and vertical edges use the locally adjusted top/bottom span.
    for _ in range(2):
        y_start = max(0, top + 6)
        y_end = min(height - 1, bottom - 6)
        x_start = max(0, left + 6)
        x_end = min(width - 1, right - 6)
        left, found[0] = find_local_edge(mask, width, height, "x", grid_left, y_start, y_end)
        right, found[1] = find_local_edge(mask, width, height, "x", grid_right, y_start, y_end)
        x_start = max(0, left + 6)
        x_end = min(width - 1, right - 6)
        top, found[2] = find_local_edge(mask, width, height, "y", grid_top, x_start, x_end)
        bottom, found[3] = find_local_edge(mask, width, height, "y", grid_bottom, x_start, x_end)

    if not (0 <= left < right < width and 0 <= top < bottom < height):
        raise DetectionError(f"Refined frame is outside the image: {(left, right, top, bottom)}")
    tile_width, tile_height = right - left + 1, bottom - top + 1
    if not (130 <= tile_width <= 166 and 130 <= tile_height <= 166):
        # Reject an implausible local line match and keep the valid grid estimate.
        left, right, top, bottom = grid_left, grid_right, grid_top, grid_bottom
        found = [False, False, False, False]
    return (left, right, top, bottom), tuple(found)


def ppm_tiles(
    width: int,
    height: int,
    source_pixels: bytes,
    mask: bytearray,
    x_lines: list[int],
    y_lines: list[int],
) -> tuple[bytes, int, int, int, int]:
    """Build individually aligned PPM tiles and blacken each frame edge."""
    payload = bytearray()
    number_of_tiles = 0
    corrected_tiles = 0
    maximum_shift = 0
    fallback_edges = 0
    for row in range(5):
        grid_top, grid_bottom = y_lines[row * 2], y_lines[row * 2 + 1]
        for column in range(10):
            grid_left, grid_right = x_lines[column * 2], x_lines[column * 2 + 1]
            bounds, found = refine_tile_box(
                mask, width, height, (grid_left, grid_right, grid_top, grid_bottom)
            )
            left, right, top, bottom = bounds
            offsets = (left - grid_left, right - grid_right, top - grid_top, bottom - grid_bottom)
            maximum_shift = max(maximum_shift, *(abs(offset) for offset in offsets))
            if any(found_edge and abs(offset) >= 3 for found_edge, offset in zip(found, offsets)):
                corrected_tiles += 1
            fallback_edges += sum(not found_edge for found_edge in found)

            tile_width, tile_height = right - left + 1, bottom - top + 1
            if not (130 <= tile_width <= 166 and 130 <= tile_height <= 166):
                raise DetectionError(f"Unexpected crop dimensions: {tile_width}x{tile_height}")
            payload.extend(f"P6\n{tile_width} {tile_height}\n255\n".encode("ascii"))
            for y in range(top, bottom + 1):
                start = (y * width + left) * 3
                line = bytearray(source_pixels[start : start + tile_width * 3])
                local_y = y - top
                if local_y < 2 or bottom - y < 2:
                    line[:] = b"\x00" * len(line)
                else:
                    line[:6] = b"\x00" * 6
                    line[-6:] = b"\x00" * 6
                    edge_pixels = {2, tile_width - 3}
                    if local_y == 2 or bottom - y == 2:
                        edge_pixels.update(range(tile_width))
                    for local_x in edge_pixels:
                        pixel_offset = (y * width + left + local_x) * 3
                        red, green, blue = source_pixels[pixel_offset : pixel_offset + 3]
                        if max(red, green, blue) - min(red, green, blue) <= 18:
                            luminance = (red + green + blue) // 3
                            if 5 <= luminance <= 130:
                                start = local_x * 3
                                line[start : start + 3] = b"\x00\x00\x00"
                payload.extend(line)
            number_of_tiles += 1
    return bytes(payload), number_of_tiles, corrected_tiles, maximum_shift, fallback_edges


def write_status(
    path: Path,
    processed: dict[str, int],
    skipped: dict[str, list[str]],
    total_images: int,
    total_icons: int,
    archive_name: str,
    corrected_tiles: int,
    maximum_shift: int,
) -> None:
    lines = [
        "# Журнал вырезания иконок",
        "",
        f"Результат: `{archive_name}`. Исходные листы сохранены без изменений.",
        "Рамки включены в обрезку и закрашены чёрным; тёмные следы сжатия по краям убраны.",
        "Положение каждой рамки уточнялось отдельно, чтобы учесть небольшие сдвиги относительно сетки.",
        "Порядок `icon_001`–`icon_050` — слева направо, сверху вниз.",
        "",
        f"**Вырезано:** {len(processed)} папок, {total_images} листов, {total_icons:,} иконок.",
        f"При перепроверке рамки уточнены для {corrected_tiles} иконок (сдвиг от сетки ≥3 px; максимальный — {maximum_shift} px).",
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
    corrected_tile_count = 0
    maximum_boundary_shift = 0
    fallback_edge_count = 0

    def process_page(
        archive: zipfile.ZipFile,
        temp_dir: Path,
        relative: Path,
        width: int,
        height: int,
        pixels: bytes,
        mask: bytearray,
        x_lines: list[int],
        y_lines: list[int],
    ) -> tuple[int, int, int]:
        for existing in temp_dir.iterdir():
            existing.unlink()
        payload, count, corrected, maximum_shift, fallback_edges = ppm_tiles(
            width, height, pixels, mask, x_lines, y_lines
        )
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
        return corrected, maximum_shift, fallback_edges

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
                            x_lines, y_lines, mask = grid
                            corrected, shift, fallback = process_page(
                                archive, temp_dir, relative, width, height, pixels, mask, x_lines, y_lines
                            )
                            corrected_tile_count += corrected
                            maximum_boundary_shift = max(maximum_boundary_shift, shift)
                            fallback_edge_count += fallback
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
        if not args.dry_run:
            print(
                f"Locally refined {corrected_tile_count} tile boxes by at least 3 px; "
                f"maximum boundary shift {maximum_boundary_shift} px "
                f"({fallback_edge_count} low-contrast sides kept the grid estimate).",
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
        write_status(
            status_path,
            processed,
            skipped,
            framed_sheet_count,
            total_icons,
            archive_path.name,
            corrected_tile_count,
            maximum_boundary_shift,
        )
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
