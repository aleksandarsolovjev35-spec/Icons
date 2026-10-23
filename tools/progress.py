#!/usr/bin/env python3
"""progress.py — строка прогресса для длинных прогонов.

Показывает процент, полосу, счётчики, сколько прошло и сколько осталось.
Два режима:

  * в настоящей консоли строка перерисовывается на месте (\\r), поэтому
    видно «живой» прогресс, а не простыню из 226 строк;
  * когда вывод идёт в пайп (run_all.py читает batch_all.py и печатает
    его лог), строка печатается обычными строками, но не чаще, чем раз
    в `min_interval` секунд — лог остаётся читаемым.

Полоса нарисована ASCII-символами: их печатает и cp866, и utf-8, поэтому
ничего не ломается на русской Windows.
"""
from __future__ import annotations

import sys
import time

BAR = 24


def hhmmss(sec: float) -> str:
    sec = int(max(0, sec))
    h, m, s = sec // 3600, sec % 3600 // 60, sec % 60
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m}:{s:02d}"


def _tty(stream) -> bool:
    try:
        return bool(stream.isatty())
    except Exception:                      # noqa: BLE001
        return False


class Progress:
    """Счётчик выполненной работы с оценкой остатка."""

    def __init__(self, total: int, stream=None, min_interval: float = 3.0,
                 width: int = 52):
        self.total = max(1, int(total))
        self.stream = stream if stream is not None else sys.stdout
        self.tty = _tty(self.stream)
        self.interval = 0.0 if self.tty else min_interval
        self.width = width
        self.t0 = time.time()
        self.last = 0.0
        self.done = 0
        self.icons = 0
        self.note = ""
        self.shown = False
        self.last_len = 0

    # ---- отрисовка ---------------------------------------------------
    def _bar(self) -> str:
        frac = min(1.0, self.done / self.total)
        fill = int(round(frac * BAR))
        return "[" + "#" * fill + "-" * (BAR - fill) + "]"

    def render(self) -> str:
        frac = self.done / self.total
        pct = int(round(frac * 100))
        el = time.time() - self.t0
        parts = [f"{pct:3d}%", self._bar(), f"{self.done}/{self.total}"]
        if self.icons:
            parts.append(f"{self.icons} иконок")
        parts.append(f"прошло {hhmmss(el)}")
        if 0 < self.done < self.total:
            eta = el / frac - el
            parts.append(f"осталось ~{hhmmss(eta)}")
        line = "  " + " · ".join(parts)
        if self.note:
            note = self.note
            if not self.tty and len(note) > self.width:
                note = "…" + note[-(self.width - 1):]
            line += " · " + note
        return line

    def _write(self, text: str) -> None:
        if self.tty:
            # без ANSI-последовательностей: старую строку забиваем пробелами,
            # иначе в обычном cmd.exe они печатаются как «←[K»
            pad = " " * max(0, self.last_len - len(text))
            self.stream.write("\r" + text + pad)
            self.last_len = len(text)
        else:
            self.stream.write(text + "\n")
        self.stream.flush()
        self.shown = True

    # ---- API ---------------------------------------------------------
    def tick(self, done: int | None = None, icons: int | None = None,
             note: str | None = None, force: bool = False) -> None:
        if done is not None:
            self.done = done
        if icons is not None:
            self.icons = icons
        if note is not None:
            self.note = note
        now = time.time()
        last_item = self.done >= self.total
        if not (force or self.interval == 0.0 or last_item
                or now - self.last >= self.interval):
            return
        self.last = now
        self._write(self.render())

    def close(self, icons: int | None = None) -> None:
        if icons is not None:
            self.icons = icons
        if self.done < self.total or not self.shown:
            self.done = self.total
            self._write(self.render())
        if self.tty:
            self.stream.write("\n")
            self.stream.flush()
            self.last_len = 0
