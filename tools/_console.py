#!/usr/bin/env python3
"""
_console.py — нормальный вывод в консоль Windows.

Проблема: cmd.exe по умолчанию работает в cp866 (русская локаль), а Python
берёт кодировку консоли как cp1251. Из-за рассинхрона русский текст выглядит
"кракозябрами", а любой символ вне cp1251 (`->`, `x`, `"`) роняет скрипт с
`UnicodeEncodeError: 'charmap' codec can't encode character '\\u2192'`.

Здесь мы спрашиваем у системы реальную кодовую страницу консоли
(GetConsoleOutputCP) и переключаем stdout/stderr на неё, а непереводимые
символы заменяем на `?` вместо падения. На Linux/macOS ничего не меняется.

    from _console import setup
    setup()
"""
from __future__ import annotations

import os
import sys


def console_encoding() -> str | None:
    """Кодовая страница консоли Windows (например 'cp866'), иначе None."""
    if os.name != "nt":
        return None
    try:
        import ctypes
        cp = ctypes.windll.kernel32.GetConsoleOutputCP()
    except Exception:                      # noqa: BLE001
        return None
    if not cp:
        return None
    if cp == 65001:
        return "utf-8"
    if cp == 1200:
        return "utf-16-le"
    return f"cp{cp}"


def setup() -> None:
    """Перенастроить потоки так, чтобы вывод не падал и читался."""
    enc = console_encoding()
    for stream in (sys.stdout, sys.stderr):
        if stream is None or not hasattr(stream, "reconfigure"):
            continue
        try:
            if enc:
                stream.reconfigure(encoding=enc, errors="replace")
            else:
                stream.reconfigure(errors="replace")
        except Exception:                  # noqa: BLE001
            pass


def child_env() -> dict:
    """Окружение для дочерних процессов: пусть пишут в UTF-8, родитель
    прочитает и перекодирует под консоль сам."""
    env = dict(os.environ)
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONUTF8"] = "1"
    return env
