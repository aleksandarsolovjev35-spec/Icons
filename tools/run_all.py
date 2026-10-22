#!/usr/bin/env python3
"""
run_all.py — «нажал и всё сделалось»: вся коллекция → 11 300 иконок 256×256.

Обычно запускается двойным кликом по run_all.bat (Windows) или
`./run_all.sh` (macOS/Linux). Но и напрямую работает:

    python3 tools/run_all.py              # все листы, все иконки
    python3 tools/run_all.py --slots 3    # по 3 иконки с листа
    python3 tools/run_all.py --limit 5    # первые 5 листов (проверка)

Шаги:
  1. проверка окружения (Python 3.9+, numpy, opencv с dnn_superres);
  2. скачивание моделей в tools/models/ (в git их нет);
  3. нарезка → чистка → апскейл 256 для всех листов (tools/batch_all.py);
  4. превью и отчёт в samples/ (tools/make_batch_previews.py);
  5. итоговая сводка в консоли + открыть папку с результатом.
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def head(t: str) -> None:
    print("\n" + "=" * 66)
    print(t)
    print("=" * 66)


def check_env() -> None:
    head("1/5  Проверка окружения")
    print(f"  Python {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        sys.exit("  ОШИБКА: нужен Python 3.9 или новее")
    try:
        import numpy as np
        import cv2
    except ImportError as e:
        print(f"  ОШИБКА: {e}")
        print("  Установи зависимости: pip install -r requirements.txt")
        sys.exit(1)
    print(f"  numpy {np.__version__}, opencv {cv2.__version__}")
    if not hasattr(cv2, "dnn_superres"):
        print("  ОШИБКА: в opencv нет dnn_superres.")
        print("  Нужен именно contrib-пакет:")
        print("    pip install --force-reinstall opencv-contrib-python-headless")
        sys.exit(1)
    print("  dnn_superres: есть")


def run(cmd: list[str], title: str) -> None:
    head(title)
    t = time.time()
    r = subprocess.run(cmd, cwd=ROOT)
    if r.returncode != 0:
        sys.exit(f"  ОШИБКА: команда завершилась с кодом {r.returncode}")
    print(f"  готово за {time.time() - t:.1f} c")


def main() -> int:
    ap = argparse.ArgumentParser(description="Полный прогон коллекции иконок")
    ap.add_argument("--slots", type=int, default=None, help="иконок с листа")
    ap.add_argument("--limit", type=int, default=None, help="первые N листов")
    ap.add_argument("--out", default=os.path.join(ROOT, "icons_256"))
    ap.add_argument("--smooth", action="store_true", help="EDSR вместо FSRCNN")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--open", action="store_true", help="открыть папку в конце")
    args = ap.parse_args()

    check_env()

    base = [sys.executable, os.path.join(HERE, "batch_all.py"),
            "--out", args.out, "--jobs", str(args.jobs)]
    if args.slots:
        base += ["--slots", str(args.slots)]
    if args.limit:
        base += ["--limit", str(args.limit)]
    if args.smooth:
        base += ["--smooth"]

    run([sys.executable, os.path.join(HERE, "fetch_models.py")]
        + (["--all"] if args.smooth else []),
        "2/5  Модели апскейла")
    run(base, "3/5  Нарезка → чистка → 256")
    run([sys.executable, os.path.join(HERE, "make_batch_previews.py"),
         "--src", args.out], "4/5  Превью в samples/")

    head("5/5  Готово")
    files = glob.glob(os.path.join(args.out, "*", "*.png"))
    size = sum(os.path.getsize(f) for f in files) / 1e6
    print(f"  иконок: {len(files)}")
    print(f"  папка:  {args.out}")
    print(f"  размер: {size:.0f} МБ")
    print("\n  превью: samples/batch_contact_sheet.jpg")
    print("          samples/batch_montage.jpg")
    print("  отчёт:  samples/batch_summary.csv")

    if args.open:
        try:
            if sys.platform.startswith("win"):
                os.startfile(args.out)          # type: ignore[attr-defined]
            elif sys.platform == "darwin":
                subprocess.run(["open", args.out])
            else:
                subprocess.run(["xdg-open", args.out])
        except Exception:                       # noqa: BLE001
            pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
