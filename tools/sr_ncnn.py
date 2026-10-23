#!/usr/bin/env python3
"""sr_ncnn.py — Real-ESRGAN для иконок, на CPU, без torch и без Vulkan.

Почему так, а не cv2.dnn_superres
---------------------------------
EDSR/FSRCNN из OpenCV обучены на фотографиях: на иконках они размазывают
заливки и съедают контуры (резкость по лапласиану падает с 920 до 125 при
увеличении 149 -> 256). Real-ESRGAN обучен на иллюстрациях: он не
интерполирует, а ДОСТРАИВАЕТ детали — штрихи, чешуйки, нити.

Обычно Real-ESRGAN требует torch и веса с github. Здесь взят пакет
`realesrgan-ncnn-py` с PyPI: веса (`realesrgan-x4plus-anime` и другие)
лежат ВНУТРИ пакета, а движок `ncnn` из PyPI работает на CPU. Родная
обёртка пакета требует libvulkan, поэтому модель грузим напрямую через
ncnn — Vulkan не нужен.

Схема:
  1. модель ×4 (по умолчанию realesrgan-x4plus-anime);
  2. INTER_AREA до нужного размера;
  3. сглаживание ровных зон (bilateral по маске локальной активности),
     чтобы сеть не оставляла «зерно» в тенях;
  4. лёгкий unsharp по L — вернуть микроконтраст после ресемплинга.

Модели: realesrgan-x4plus-anime, realesrgan-x4plus,
        realesr-animevideov3-x2 / -x3 / -x4
Скорость: ~3 с на иконку (anime ×4), ~10 с (x4plus), ~0.3 с (videov3).
"""
from __future__ import annotations

import os
import sys

import cv2
import numpy as np

DEFAULT_MODEL = "realesrgan-x4plus-anime"
SCALES = {"realesrgan-x4plus-anime": 4, "realesrgan-x4plus": 4,
          "realesr-animevideov3-x2": 2, "realesr-animevideov3-x3": 3,
          "realesr-animevideov3-x4": 4}


_GPU: bool | None = None


def gpu_count() -> int:
    """Сколько GPU видит Vulkan (0 - видеокарты нет или нет драйвера)."""
    try:
        import ncnn
        ncnn.create_gpu_instance()
        return int(ncnn.get_gpu_count())
    except Exception:
        return 0


def pick_device(device: str = "auto") -> str:
    """auto -> gpu, если Vulkan видит видеокарту, иначе cpu."""
    if device == "cpu":
        return "cpu"
    if device == "gpu":
        if gpu_count() == 0:
            raise RuntimeError(
                "GPU не найден: Vulkan не видит видеокарту. "
                "На Windows нужен свежий драйвер, на Linux - libvulkan1 + ICD. "
                "Запусти с --device cpu.")
        return "gpu"
    return "gpu" if gpu_count() > 0 else "cpu"


def models_dir() -> str | None:
    """Каталог с .param/.bin из пакета realesrgan-ncnn-py."""
    override = os.environ.get("REALESRGAN_MODELS")
    if override and os.path.isdir(override):
        return override
    # пакет нельзя импортировать (его обёртка требует libvulkan), но путь
    # к его каталогу находится и без импорта
    try:
        import importlib.util
        spec = importlib.util.find_spec("realesrgan_ncnn_py")
        if spec is None or not spec.origin:
            return None
        return os.path.join(os.path.dirname(spec.origin), "models")
    except Exception:
        return None


def available() -> bool:
    """Есть ли ncnn и веса. Без них пакетный прогон должен честно сказать."""
    try:
        import ncnn  # noqa: F401
    except Exception:
        return False
    d = models_dir()
    if not d:
        return False
    return all(os.path.exists(os.path.join(d, f"{DEFAULT_MODEL}{ext}"))
               for ext in (".param", ".bin"))


def _net(model: str, device: str = "auto"):
    import ncnn
    d = models_dir()
    if not d:
        raise RuntimeError(
            "нет весов: pip install realesrgan-ncnn-py ncnn")
    base = os.path.join(d, model)
    if not (os.path.exists(base + ".param") and os.path.exists(base + ".bin")):
        raise RuntimeError(f"нет модели {model} в {d}")
    net = ncnn.Net()
    if device == "gpu":
        ncnn.create_gpu_instance()
        net.opt.use_vulkan_compute = True
        net.opt.num_threads = 1
        try:
            net.set_vulkan_device(ncnn.get_default_gpu_index())
        except Exception:
            pass
    else:
        net.opt.use_vulkan_compute = False  # CPU: libvulkan не нужен
        net.opt.num_threads = 1
    net.load_param(base + ".param")
    net.load_model(base + ".bin")
    return net


def _sr(img: np.ndarray, model: str, device: str = "cpu") -> np.ndarray:
    """Прогон модели; ncnn ждёт CHW float 0..1."""
    import ncnn
    h, w = img.shape[:2]
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    net = _net(model, device)
    ex = net.create_extractor()
    ex.input("data", ncnn.Mat(np.ascontiguousarray(rgb.transpose(2, 0, 1))))
    _, out = ex.extract("output")
    a = np.array(out).transpose(1, 2, 0)
    return cv2.cvtColor(np.clip(a * 255.0, 0, 255).astype(np.uint8),
                        cv2.COLOR_RGB2BGR)


def flat_denoise(img: np.ndarray, limit: float = 12.0,
                 strength: float = 30.0) -> np.ndarray:
    """Приглушить «зерно» сети в ровных зонах, не задев контуры."""
    g = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY).astype(np.float32)
    act = cv2.GaussianBlur(np.abs(cv2.Laplacian(g, cv2.CV_32F, ksize=3)),
                           (0, 0), 2.0)
    mask = np.clip(1.0 - act / limit, 0, 1)[..., None]
    den = cv2.bilateralFilter(img, 9, strength, 9)
    return (img * (1 - mask) + den * mask).astype(np.uint8)


def unsharp(img: np.ndarray, amount: float = 0.25,
            sigma: float = 0.8) -> np.ndarray:
    if amount <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0]
    l = l + (l - cv2.GaussianBlur(l, (0, 0), sigma)) * float(amount)
    lab[..., 0] = np.clip(l, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def upscale(img: np.ndarray, size: int = 256, model: str = DEFAULT_MODEL,
            denoise: bool = True, sharp: float = 0.25,
            device: str = "auto") -> np.ndarray:
    """Иконка (вырез, без полей) -> size×size с достройкой деталей.

    device: "auto" (GPU если есть Vulkan, иначе CPU), "gpu", "cpu".
    """
    big = _sr(img, model, pick_device(device))
    out = cv2.resize(big, (size, size), interpolation=cv2.INTER_AREA)
    if denoise:
        out = flat_denoise(out)
    return unsharp(out, sharp)


def main() -> int:
    import argparse
    ap = argparse.ArgumentParser(description="Real-ESRGAN для иконок (CPU)")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(SCALES))
    ap.add_argument("--sharp", type=float, default=0.25)
    ap.add_argument("--no-denoise", action="store_true")
    ap.add_argument("--device", default="auto", choices=["auto", "gpu", "cpu"],
                    help="auto - GPU через Vulkan, если он есть; иначе CPU")
    args = ap.parse_args()

    if not available():
        print("нет ncnn/весов:\n  pip install ncnn realesrgan-ncnn-py",
              file=sys.stderr)
        return 2
    for src in args.images:
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"не читается: {src}")
        res = upscale(img, args.size, args.model, not args.no_denoise,
                      args.sharp, args.device)
        if args.out and len(args.images) == 1:
            dst = args.out
        else:
            base = os.path.splitext(os.path.basename(src))[0]
            dst = os.path.join(args.out or ".", f"{base}_{args.size}.png")
        os.makedirs(os.path.dirname(os.path.abspath(dst)), exist_ok=True)
        cv2.imwrite(dst, res)
        print(f"{src} -> {dst} ({res.shape[1]}x{res.shape[0]})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
