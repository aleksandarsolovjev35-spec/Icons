#!/usr/bin/env python3
"""
batch_all.py — прогнать всю коллекцию: нарезка -> чистка -> апскейл 256.

Для каждого листа (`<папка>/<файл>.jpg`):
  detect_frames -> все рамки -> чёрная рамка-бортик (style_frame, --frame) ->
  вырез 1:1 с обводкой -> clean_fill('normal') ->
  супер-разрешение до 256x256 (FSRCNN x4 + unsharp).

Файлы кладутся в `icons_256/<папка>/<файл>__slotNN.png` (11 300 PNG).
Папку можно переименовать ключом `--out`.

В `samples/` пишется то, что нужно глазам:
  batch_summary.csv        — по листам: сколько иконок, шум до/после, время
  batch_contact_sheet.jpg  — сетка: по одной иконке с каждого листа
  batch_montage.jpg        — 12 иконок из разных папок крупно

    python3 tools/batch_all.py                     # вся коллекция, все слоты
    python3 tools/batch_all.py --slots 3           # по 3 иконки с листа
    python3 tools/batch_all.py --limit 5           # только первые 5 листов
    python3 tools/batch_all.py --smooth            # EDSR вместо FSRCNN (медленнее)
"""
from __future__ import annotations

import argparse
import csv
import glob
import os
import sys
import time

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)

from _console import setup as _console_setup  # noqa: E402
import clean_icon as C            # noqa: E402
import detect_frames as D         # noqa: E402
import fetch_models               # noqa: E402
import style_frame as S           # noqa: E402

MODELS = os.path.join(HERE, "models")
SR = {}


def _sr(algo: str, scale: int = 4):
    key = (algo, scale)
    if key not in SR:
        name = {"edsr": "EDSR", "fsrcnn": "FSRCNN"}[algo]
        m = cv2.dnn_superres.DnnSuperResImpl_create()
        m.readModel(os.path.join(MODELS, f"{name}_x{scale}.pb"))
        m.setModel(algo, scale)
        SR[key] = m
    return SR[key]


def upscale(img: np.ndarray, size: int = 256, unsharp: float = 0.55,
            algo: str = "fsrcnn") -> np.ndarray:
    """SR по яркости (цвет — Lanczos), затем доводка микроконтраста."""
    sr = _sr(algo, 4)
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    big = sr.upsample(cv2.cvtColor(lab[..., 0], cv2.COLOR_GRAY2BGR))
    lo = cv2.cvtColor(cv2.resize(big, (size, size),
                                 interpolation=cv2.INTER_AREA),
                      cv2.COLOR_BGR2GRAY).astype(np.float32)
    lo = lo + (lo - cv2.GaussianBlur(lo, (0, 0), 0.8)) * unsharp
    ab = cv2.resize(lab[..., 1:], (size, size),
                    interpolation=cv2.INTER_LANCZOS4)
    return cv2.cvtColor(np.dstack([np.clip(lo, 0, 255).astype(np.uint8), ab]),
                        cv2.COLOR_LAB2BGR)


# --- рамка на готовом кадре -------------------------------------------------
FRAME_RIM = 10      # толщина борта в пикселях ИТОГОВОГО кадра: одна на всех
INSET_CROP = 2      # срезаем с края слота родную серо-белую линию листа


def frame_params(variant: str, size: int) -> dict:
    """Параметры бортика, пересчитанные в пиксели итогового кадра.

    style_frame.py мериет расстояния в пикселях того изображения, на котором
    рисует. Эталон у него - борт 5.5 px, поэтому длины масштабируем так,
    чтобы борт стал ровно FRAME_RIM px, независимо от размера кадра.
    """
    fp = S.params(variant, radius=0.0)
    k = FRAME_RIM / fp["rim"]
    for key in ("rim", "inset", "deep", "radius"):
        fp[key] = max(0.0, float(fp[key]) * k)
    fp["rim"] = float(FRAME_RIM)
    # глубина внутренней тени при 256 px не должна залезать на рисунок:
    # в эталоне она мерилась для слота ~145 px
    fp["deep"] = min(fp["deep"], 9.0)
    fp["radius"] = 0.0
    return fp


def compose_frame(body: np.ndarray, size: int, fp: dict) -> np.ndarray:
    """Вписать готовую иконку в кадр и нарисовать борт поверх краёв."""
    canvas = np.zeros((size, size, 3), np.uint8)
    inner = size - 2 * FRAME_RIM
    if body.shape[0] != inner or body.shape[1] != inner:
        body = cv2.resize(body, (inner, inner), interpolation=cv2.INTER_AREA)
    canvas[FRAME_RIM:FRAME_RIM + inner, FRAME_RIM:FRAME_RIM + inner] = body
    S.style_box(canvas, (FRAME_RIM, FRAME_RIM,
                         FRAME_RIM + inner - 1, FRAME_RIM + inner - 1), fp)
    return canvas


def process_sheet(path: str, outroot: str, nslots: int | None,
                  algo: str, size: int, frame: str | None = "sharp") -> dict:
    t0 = time.time()
    img = cv2.imread(path)
    if img is None:
        return {"sheet": path, "error": "не читается"}
    H, W = img.shape[:2]
    boxes = D.detect_sheet(path).boxes
    if not boxes:
        return {"sheet": path, "slots": 0, "error": "рамки не найдены"}
    mine = boxes if nslots is None else boxes[:nslots]

    # Порядок важен: сначала иконка (вырез -> чистка -> апскейл), и только
    # потом рамка - на готовом кадре 256x256, в пикселях этого кадра.
    # Иначе чистка и супер-разрешение размывали борт, и его ширина гуляла
    # от иконки к иконке.
    inner = size            # сторона иконки внутри рамки
    pad = INSET_CROP        # сколько срезать с края слота (родная линия рамки)
    fp = None
    if frame:
        fp = frame_params(frame, size)
        inner = size - 2 * FRAME_RIM

    folder = os.path.relpath(os.path.dirname(path), ROOT)
    stem = os.path.splitext(os.path.basename(path))[0].replace(" ", "_")
    outdir = os.path.join(outroot, folder)
    os.makedirs(outdir, exist_ok=True)

    made, noise_before, noise_after = 0, 0.0, 0.0
    for k, (x0, y0, x1, y1) in enumerate(mine):
        crop = img[max(0, y0 + pad):min(H, y1 + 1 - pad),
                   max(0, x0 + pad):min(W, x1 + 1 - pad)]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            continue
        clean = C.clean_fill(crop, "normal")
        big = upscale(clean, size=inner, algo=algo)
        if fp is not None:
            big = compose_frame(big, size, fp)
        out = os.path.join(outdir, f"{stem}__slot{k:02d}.png")
        cv2.imwrite(out, big)
        noise_before += C.fill_noise(crop)
        noise_after += C.fill_noise(clean)
        made += 1

    dt = time.time() - t0
    return {"sheet": os.path.relpath(path, ROOT), "folder": folder,
            "slots": made,
            "noise_before": round(noise_before / max(made, 1), 2),
            "noise_after": round(noise_after / max(made, 1), 2),
            "sec": round(dt, 1)}


def main() -> int:
    _console_setup()
    ap = argparse.ArgumentParser(description="Пакетный прогон коллекции")
    ap.add_argument("--out", default=os.path.join(ROOT, "icons_256"),
                    help="куда класть результат (по умолчанию icons_256/ в репозитории)")
    ap.add_argument("--slots", type=int, default=None,
                    help="сколько иконок с листа (по умолчанию все)")
    ap.add_argument("--limit", type=int, default=None, help="только первые N листов")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--smooth", action="store_true",
                    help="EDSR (качественнее, ~9 c на иконку) вместо FSRCNN")
    ap.add_argument("--frame", default="sharp",
                    choices=["none"] + sorted(S.VARIANTS),
                    help="рамка-бортик вокруг иконки (по умолчанию sharp: "
                         "чёрная, прямые углы, свет сверху-слева); none - без рамки")
    ap.add_argument("--jobs", type=int, default=2)
    args = ap.parse_args()
    frame = None if args.frame == "none" else args.frame

    cv2.setNumThreads(1)
    sheets = sorted(p for p in glob.glob(os.path.join(ROOT, "*", "*.jpg"))
                    if os.path.basename(os.path.dirname(p)) not in ("samples",))
    if args.limit:
        sheets = sheets[:args.limit]
    algo = "edsr" if args.smooth else "fsrcnn"
    fetch_models.ensure(all_models=args.smooth)      # модели качаются один раз
    print(f"листов: {len(sheets)} | слотов с листа: {args.slots or 'все'} | "
          f"модель: {algo.upper()} | рамка: {args.frame} | -> {args.out}")

    rows = []
    t0 = time.time()
    if args.jobs > 1:
        from multiprocessing import Pool
        with Pool(args.jobs) as pool:
            for res in pool.imap_unordered(
                    _worker, [(p, args.out, args.slots, algo, args.size, frame)
                              for p in sheets]):
                rows.append(res)
                print(f"  [{len(rows)}/{len(sheets)}] {res['sheet']}  "
                      f"{res.get('slots', 0)} шт  {res.get('sec', 0)} c")
    else:
        for p in sheets:
            res = process_sheet(p, args.out, args.slots, algo, args.size, frame)
            rows.append(res)
            print(f"  [{len(rows)}/{len(sheets)}] {res['sheet']}  "
                  f"{res.get('slots', 0)} шт  {res.get('sec', 0)} c")

    rows.sort(key=lambda r: r["sheet"])
    os.makedirs(os.path.join(ROOT, "samples"), exist_ok=True)
    csv_path = os.path.join(ROOT, "samples", "batch_summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sheet", "folder", "slots",
                                          "noise_before", "noise_after", "sec"])
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in w.fieldnames})

    total = sum(r.get("slots", 0) for r in rows)
    print(f"\nготово: {total} иконок с {len(rows)} листов за "
          f"{(time.time() - t0) / 60:.1f} мин -> {args.out}")
    print(f"отчёт: {csv_path}")
    return 0


def _worker(a):
    p, out, n, algo, size, frame = a
    try:
        return process_sheet(p, out, n, algo, size, frame)
    except Exception as e:                      # noqa: BLE001
        return {"sheet": os.path.relpath(p, ROOT), "slots": 0,
                "error": str(e)[:120]}


if __name__ == "__main__":
    raise SystemExit(main())
