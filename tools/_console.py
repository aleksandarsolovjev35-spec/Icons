#!/usr/bin/env python3
"""
_console.py - нормальный вывод в консоль Windows.

Почему это нужно. На русской Windows кодировка зависит от того, куда пишет
Python:

  * stdout - настоящая консоль (cmd.exe) -> Python берёт GetConsoleOutputCP,
    обычно cp866: русский виден, а вот `→`, `«»`, `×` в cp866 отсутствуют;
  * stdout - ПАЙП (например, tools/run_all.py запускает batch_all.py и читает
    его вывод) -> Python берёт locale.getpreferredencoding() = cp1251, и уже
    `→` (U+2192) в cp1251 нет:

        UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'

Именно поэтому падал шаг 3/5: batch_all.py - дочерний процесс, его вывод
идёт в пайп.

Что делаем:

  1. В НАСТОЯЩЕЙ консоли Windows переключаем кодовую страницу на UTF-8
     (SetConsoleOutputCP(65001)) и переключаем stdout/stderr на utf-8 -
     тогда весь русский текст и любая типографика видны как есть;
     если переключить не удалось - падаем на реальную кодовую страницу
     (cp866/cp1251) с errors="replace".
  2. В ПАЙПЕ кодировку не трогаем (родитель ждёт utf-8 через
     PYTHONIOENCODING и читает с encoding="utf-8"), но ставим
     errors="replace", чтобы никакой символ не мог уронить прогон.
  3. На Linux/macOS всё остаётся как было (utf-8).

Использование:

    from _console import setup, child_env
    def main():
        setup()
        ...

Дочерние процессы запускать с env=child_env() - тогда они гарантированно
пишут utf-8 в пайп.
"""
from __future__ import annotations

import os
import sys


def _kernel32():
    if os.name != "nt":
        return None
    try:
        import ctypes
        return ctypes.windll.kernel32
    except Exception:                      # noqa: BLE001
        return None


def console_encoding() -> str | None:
    """Кодовая страница консоли Windows (например 'cp866'), иначе None."""
    k = _kernel32()
    if k is None:
        return None
    try:
        cp = k.GetConsoleOutputCP()
    except Exception:                      # noqa: BLE001
        return None
    if not cp:
        return None
    if cp == 65001:
        return "utf-8"
    return f"cp{cp}"


def force_utf8_console() -> bool:
    """Переключить консоль на UTF-8. True - если получилось и проверилось."""
    k = _kernel32()
    if k is None:
        return False
    try:
        if not k.SetConsoleOutputCP(65001):
            return False
        return k.GetConsoleOutputCP() == 65001
    except Exception:                      # noqa: BLE001
        return False


def _isatty(stream) -> bool:
    try:
        return bool(stream is not None and stream.isatty())
    except Exception:                      # noqa: BLE001
        return False


def _setup_stream(stream, enc: str | None) -> None:
    """Пайп: только errors="replace". Консоль: ещё и подходящая кодировка."""
    if stream is None or not hasattr(stream, "reconfigure"):
        return
    tty = _isatty(stream)
    try:
        if tty and enc:
            stream.reconfigure(encoding=enc, errors="replace")
        else:
            stream.reconfigure(errors="replace")
    except Exception:                      # noqa: BLE001
        pass


def setup() -> None:
    """Перенастроить потоки так, чтобы вывод не падал и читался."""
    enc = None
    if any(_isatty(s) for s in (sys.stdout, sys.stderr)):
        enc = "utf-8" if force_utf8_console() else console_encoding()
    for stream in (sys.stdout, sys.stderr):
        _setup_stream(stream, enc)


def child_env() -> dict:
    """Окружение для дочерних процессов: пусть пишут в UTF-8, родитель
    прочитает их с encoding="utf-8" и покажет сам."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env
