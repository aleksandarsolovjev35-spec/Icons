#!/usr/bin/env python3
"""
run_all.py — "нажал и всё сделалось": вся коллекция -> 11 300 иконок 256x256.

Обычно запускается двойным кликом по run_all.bat (Windows) или
`./run_all.sh` (macOS/Linux). Но и напрямую работает:

    python3 tools/run_all.py              # все листы, все иконки
    python3 tools/run_all.py --slots 3    # по 3 иконки с листа
    python3 tools/run_all.py --limit 5    # первые 5 листов (проверка)

Шаги:
  1. проверка окружения (Python 3.9+, numpy, opencv с dnn_superres);
  2. скачивание моделей в tools/models/ (в git их нет);
  3. нарезка -> чистка -> апскейл 256 для всех листов (tools/batch_all.py);
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

from _console import setup as _console_setup, child_env  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
LOG_PATH = os.path.join(ROOT, "run_log.txt")
_LOG = None


def _open_log() -> None:
    """Дублируем вывод в run_log.txt: если окно всё же закроется, будет что читать."""
    global _LOG
    try:
        _LOG = open(LOG_PATH, "w", encoding="utf-8", buffering=1)
        _LOG.write(f"run_all.py  Python {sys.version.split()[0]}\n")
    except OSError:
        _LOG = None


def log(line: str = "") -> None:
    print(line, flush=True)
    if _LOG:
        _LOG.write(line + "\n")


def head(t: str) -> None:
    log()
    log("=" * 66)
    log(t)
    log("=" * 66)


def check_env() -> None:
    head("1/5  Проверка окружения")
    log(f"  Python {sys.version.split()[0]}")
    if sys.version_info < (3, 9):
        sys.exit("  ОШИБКА: нужен Python 3.9 или новее")
    try:
        import numpy as np
        import cv2
    except ImportError as e:
        log(f"  ОШИБКА: {e}")
        log("  Установи зависимости: pip install -r requirements.txt")
        sys.exit(1)
    log(f"  numpy {np.__version__}, opencv {cv2.__version__}")
    if not hasattr(cv2, "dnn_superres"):
        log("  ОШИБКА: в opencv нет dnn_superres.")
        log("  Нужен именно contrib-пакет:")
        log("    pip install --force-reinstall opencv-contrib-python-headless")
        sys.exit(1)
    log("  dnn_superres: есть")


def run(cmd: list[str], title: str) -> None:
    """Запускает шаг, показывая его вывод живьём и дублируя в лог."""
    head(title)
    t = time.time()
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True,
                            encoding="utf-8", errors="replace", bufsize=1,
                            env=child_env())
    assert proc.stdout is not None
    for line in proc.stdout:
        log(line.rstrip("\n"))
    rc = proc.wait()
    if rc != 0:
        log(f"  ОШИБКА: шаг завершился с кодом {rc}")
        raise SystemExit(1)
    log(f"  готово за {time.time() - t:.1f} c")


def main() -> int:
    _console_setup()
    ap = argparse.ArgumentParser(description="Полный прогон коллекции иконок")
    ap.add_argument("--slots", type=int, default=None, help="иконок с листа")
    ap.add_argument("--limit", type=int, default=None, help="первые N листов")
    ap.add_argument("--out", default=os.path.join(ROOT, "icons_cut"))
    ap.add_argument("--size", type=int, default=0,
                    help="сторона кадра; 0 (по умолчанию) - без апскейла, "
                         "родной размер иконки")
    ap.add_argument("--up", type=int, default=0,
                    help="апскейл: 128, 256 или 512 (то же, что --size N)")
    ap.add_argument("--smooth", action="store_true", help="EDSR вместо FSRCNN")
    ap.add_argument("--sr", default="anime", choices=["anime", "edsr", "fsrcnn"],
                    help="движок апскейла при --up/--size: anime - Real-ESRGAN "
                         "(по умолчанию), edsr/fsrcnn - cv2.dnn_superres")
    ap.add_argument("--frame", default="none",
                    help="рамка-бортик: none (по умолчанию), sharp, deep, inset, soft")
    ap.add_argument("--no-frame", action="store_true",
                    help="без рамки (то же, что --frame none)")
    ap.add_argument("--canvas", type=int, default=151,
                    help="довести кадр до N x N, добрав тёмный фон (0 - выключить)")
    ap.add_argument("--cut", default="inner", choices=["inner", "line", "corner"],
                    help="границы: inner - по внутренней стороне белой рамки, "
                         "по тёмным пикселям (по умолчанию); line - по внешнему "
                         "краю линии; corner - по однотонным чёрным углам")
    ap.add_argument("--jobs", type=int, default=2)
    ap.add_argument("--open", action="store_true", help="открыть папку в конце")
    args = ap.parse_args()

    _open_log()
    log(f"папка: {ROOT}")
    check_env()

    frame = "none" if args.no_frame else args.frame
    size = args.up or args.size
    base = [sys.executable, os.path.join(HERE, "batch_all.py"),
            "--out", args.out, "--jobs", str(args.jobs),
            "--frame", frame, "--size", str(size), "--cut", args.cut,
            "--sr", args.sr,
            "--canvas", str(args.canvas)]
    if args.slots:
        base += ["--slots", str(args.slots)]
    if args.limit:
        base += ["--limit", str(args.limit)]
    if args.smooth:
        base += ["--smooth"]

    run([sys.executable, os.path.join(HERE, "fetch_models.py")]
        + (["--all"] if args.smooth else []),
        "2/5  Модели апскейла")
    run(base, "3/5  Нарезка -> чистка -> 256")
    run([sys.executable, os.path.join(HERE, "make_batch_previews.py"),
         "--src", args.out], "4/5  Превью в samples/")

    head("5/5  Готово")
    files = glob.glob(os.path.join(args.out, "*", "*.png"))
    size = sum(os.path.getsize(f) for f in files) / 1e6
    log(f"  иконок: {len(files)}")
    log(f"  папка:  {args.out}")
    log(f"  размер: {size:.0f} МБ")
    log("  превью: samples/batch_contact_sheet.jpg")
    log("          samples/batch_montage.jpg")
    log("  отчёт:  samples/batch_summary.csv")
    log(f"  лог:    {LOG_PATH}")

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
