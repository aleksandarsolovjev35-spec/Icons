#!/usr/bin/env python3
"""gpu_check.py — видит ли ncnn видеокарту и почему нет.

Печатает короткий отчёт: версия Python и ncnn, найден ли системный
загрузчик Vulkan (`vulkan-1.dll` на Windows, `libvulkan.so.1` на Linux),
сколько устройств видит Vulkan и как они называются.

Запуск:
    python tools/gpu_check.py            # только отчёт
    python tools/gpu_check.py --bench    # плюс замер GPU против CPU
"""
from __future__ import annotations

import ctypes
import os
import platform
import sys
import time

LOADERS = {
    "Windows": ["vulkan-1.dll"],
    "Linux": ["libvulkan.so.1", "libvulkan.so"],
    "Darwin": ["libvulkan.dylib", "libMoltenVK.dylib"],
}


def loader_name() -> str | None:
    """Путь к системному Vulkan-загрузчику, если он вообще есть."""
    for name in LOADERS.get(platform.system(), ["libvulkan.so.1"]):
        try:
            lib = ctypes.WinDLL(name) if os.name == "nt" else ctypes.CDLL(name)
            return f"{name} -> {getattr(lib, '_name', name)}"
        except OSError:
            continue
    return None


def ncnn_vulkan_built() -> bool | None:
    """Собран ли сам ncnn с Vulkan (есть ли функции GPU в биндинге)."""
    try:
        import ncnn
    except Exception:
        return None
    return hasattr(ncnn, "get_gpu_count")


def gpu_names() -> list[str]:
    """Имена видеокарт, которые видит Vulkan (пусто — не видит ни одной)."""
    try:
        import ncnn
        ncnn.create_gpu_instance()
        count = int(ncnn.get_gpu_count())
    except Exception:
        return []
    names = []
    for i in range(count):
        try:
            info = ncnn.get_gpu_info(i)
            name = str(info.device_name())
            api = str(info.api_version())
            drv = str(info.driver_version())
            names.append(f"{name} (Vulkan {api}, драйвер {drv})")
        except Exception:
            names.append(f"устройство {i} (имя не прочитать)")
    return names


def report() -> list[str]:
    """Строки отчёта — их же печатает проверка окружения в run_all."""
    lines = [f"Python {platform.python_version()} / {platform.machine()} / "
             f"{platform.system()}"]
    try:
        import ncnn
        lines.append(f"ncnn: {getattr(ncnn, '__version__', 'версия неизвестна')}")
    except Exception as e:                              # noqa: BLE001
        lines.append(f"ncnn: НЕ УСТАНОВЛЕН ({e.__class__.__name__}) — "
                     "pip install ncnn")
        return lines

    built = ncnn_vulkan_built()
    if built is False:
        lines.append("ncnn собран БЕЗ Vulkan — GPU невозможен, только CPU")
        return lines

    loader = loader_name()
    if loader:
        lines.append(f"загрузчик Vulkan: {loader}")
    else:
        lines.append("загрузчик Vulkan: НЕ НАЙДЕН (vulkan-1.dll / libvulkan.so.1)")

    names = gpu_names()
    if names:
        lines.append(f"видеокарт найдено: {len(names)}")
        lines += [f"  {i}: {n}" for i, n in enumerate(names)]
    else:
        lines.append("видеокарт найдено: 0 — Vulkan не видит ни одного устройства")
        if not loader:
            if platform.system() == "Windows":
                lines.append("  причина: нет системного Vulkan. На Windows он "
                             "идёт вместе с драйвером видеокарты — обнови "
                             "драйвер (NVIDIA/AMD/Intel) и перезагрузись.")
            else:
                lines.append("  причина: нет системного Vulkan — поставь "
                             "libvulkan1 и ICD-драйвер (mesa-vulkan-drivers "
                             "или драйвер NVIDIA).")
        else:
            lines.append("  причина: загрузчик есть, но драйвер не отдал "
                         "устройство. Обычно это «Базовый видеоадаптер "
                         "Майкрософт» (нет родного драйвера) или виртуалка "
                         "без проброса GPU.")
    return lines


def bench(size: int = 256) -> list[str]:
    """Замер одной иконки на GPU и CPU — видно, есть ли смысл в карте."""
    import glob
    import cv2
    import sr_ncnn

    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    pats = [p for p in glob.glob(os.path.join(root, "*", "*.jpg"))
            if os.path.basename(os.path.dirname(p)) != "samples"]
    if not pats:
        return ["замер пропущен: нет исходных листов"]
    sheet = cv2.imdecode(np_fromfile(pats[0]), cv2.IMREAD_COLOR)
    if sheet is None:
        return ["замер пропущен: лист не читается"]
    crop = sheet[40:40 + 130, 40:40 + 130]
    out = []
    for dev in ("cpu", "gpu"):
        if dev == "gpu" and not gpu_names():
            out.append("GPU: пропуск, видеокарта не найдена")
            continue
        try:
            sr_ncnn.upscale(crop, size=size, device=dev)     # прогрев
            t0 = time.time()
            sr_ncnn.upscale(crop, size=size, device=dev)
            out.append(f"{dev.upper()}: {time.time() - t0:.2f} с на иконку "
                       f"{size}x{size}")
        except Exception as e:                               # noqa: BLE001
            out.append(f"{dev.upper()}: ошибка — {e}")
    return out


def np_fromfile(path: str):
    import numpy as np
    return np.frombuffer(open(path, "rb").read(), np.uint8)


def main() -> int:
    print("=" * 66)
    print("  Проверка видеокарты для апскейла")
    print("=" * 66)
    lines = report()
    for line in lines:
        print(line)
    if "--bench" in sys.argv:
        print("-" * 66)
        for line in bench():
            print(line)
    print("=" * 66)
    if gpu_names():
        print("Итог: GPU работает. Запускай прогон как обычно — устройство "
              "выберется само,\nили укажи явно --device gpu.")
    else:
        print("Итог: GPU недоступен, прогон пойдёт на CPU (~3 с на иконку).\n"
              "Посмотри строку «причина» выше — там что именно не так.")
    try:
        with open(os.path.join(os.path.dirname(os.path.dirname(
                os.path.abspath(__file__))), "gpu_check.txt"), "w",
                encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")
    except OSError:
        pass
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
