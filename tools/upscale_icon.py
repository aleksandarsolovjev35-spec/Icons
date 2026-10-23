#!/usr/bin/env python3
"""
upscale_icon.py — увеличить иконку до 256×256 с сохранением деталей.

Почему не обычный resize: билинейка/ланцош на пиксель-арте и на иконках с
обводкой дают мыло и «звон» вдоль контуров. Здесь работает нейросетевой
супер-разрешение (EDSR из OpenCV dnn_superres): он достраивает кромки и
мелкую фактуру, а не размазывает их.

Схема:
  1. EDSR ×4 — поднимаем картинку вчетверо (модель x4, обучалась на
     естественных изображениях, кромки держит лучше FSRCNN);
  2. INTER_AREA до 256×256 — точечная выборка без повторного «звона»;
  3. лёгкий unsharp по яркости (0.4, σ0.8) — возвращает микроконтраст,
     который съедает ресемплинг. Сильнее не надо: у контуров появляется гало.

Сеть гоняем ТОЛЬКО по каналу яркости, цвет (a/b) поднимаем Lanczos отдельно.
На трёх каналах BGR сеть подтягивает насыщенность (у зелья +3 по S) и цвет
уезжает; через L цвет остаётся ровно тем же (a/b mean и std совпадают), а
резкость та же — рисунок и так держится на яркости.

Модель качается один раз в tools/models/ (в git не кладём — 38 МБ).

    python3 tools/upscale_icon.py in.png -o out.png
    python3 tools/upscale_icon.py in.png -o out.png --sharp 0.6 --size 256
    python3 tools/upscale_icon.py in.png -o out.png --sharp 0     # без доводки
"""
from __future__ import annotations

import argparse
import os
import sys
import urllib.request

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _console import setup as _console_setup  # noqa: E402

MODEL_URL = ("https://raw.githubusercontent.com/Saafke/EDSR_Tensorflow/"
             "master/models/EDSR_x4.pb")
MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                          "models", "EDSR_x4.pb")


def ensure_model(path: str = MODEL_PATH) -> str:
    if os.path.exists(path) and os.path.getsize(path) > 1_000_000:
        return path
    os.makedirs(os.path.dirname(path), exist_ok=True)
    print(f"качаю модель EDSR x4 → {path}")
    urllib.request.urlretrieve(MODEL_URL, path)
    return path


def _unsharp(img: np.ndarray, amount: float, sigma: float) -> np.ndarray:
    if amount <= 0:
        return img
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB).astype(np.float32)
    l = lab[..., 0]
    l = l + (l - cv2.GaussianBlur(l, (0, 0), sigma)) * float(amount)
    lab[..., 0] = np.clip(l, 0, 255)
    return cv2.cvtColor(lab.astype(np.uint8), cv2.COLOR_LAB2BGR)


def upscale_anime(img: np.ndarray, size: int = 256) -> np.ndarray:
    """Real-ESRGAN (anime) - достраивает детали, а не размазывает их."""
    import sr_ncnn
    return sr_ncnn.upscale(img, size=size)


def upscale(img: np.ndarray, size: int = 256, sharp: float = 0.4,
            sigma: float = 0.8, model: str | None = None) -> np.ndarray:
    """EDSR ×4 по яркости → точный размер `size` → лёгкий unsharp по L.

    Цвет (a/b) поднимается Lanczos: сеть на хроме только портит оттенок,
    а весь видимый рисунок и так в яркости.
    """
    model = ensure_model(model or MODEL_PATH)
    sr = cv2.dnn_superres.DnnSuperResImpl_create()
    sr.readModel(model)
    sr.setModel("edsr", 4)

    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l3 = cv2.cvtColor(lab[..., 0], cv2.COLOR_GRAY2BGR)   # модель ждёт 3 канала
    big = sr.upsample(l3)
    l_out = cv2.cvtColor(cv2.resize(big, (size, size),
                                    interpolation=cv2.INTER_AREA),
                         cv2.COLOR_BGR2GRAY)
    ab_out = cv2.resize(lab[..., 1:], (size, size),
                        interpolation=cv2.INTER_LANCZOS4)
    out = cv2.cvtColor(np.dstack([l_out, ab_out]), cv2.COLOR_LAB2BGR)
    return _unsharp(out, sharp, sigma)


def main() -> int:
    _console_setup()
    ap = argparse.ArgumentParser(description="Апскейл иконки до 256×256")
    ap.add_argument("images", nargs="+")
    ap.add_argument("-o", "--out", default=None, help="выход (для одного файла)")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--sharp", type=float, default=0.4,
                    help="доводка микроконтраста 0..1 (0 — выключить)")
    ap.add_argument("--sigma", type=float, default=0.8)
    ap.add_argument("--model", default=None, help="путь к .pb (по умолчанию EDSR x4)")
    ap.add_argument("--engine", default="anime", choices=["anime", "edsr"],
                    help="anime - Real-ESRGAN (по умолчанию, достройка деталей), "
                         "edsr - старый путь через cv2.dnn_superres")
    args = ap.parse_args()

    if not hasattr(cv2, "dnn_superres"):
        print("нужен opencv-contrib-python-headless:\n"
              "  pip install opencv-contrib-python-headless", file=sys.stderr)
        return 2

    for src in args.images:
        img = cv2.imread(src)
        if img is None:
            raise SystemExit(f"не читается: {src}")
        if args.engine == "anime":
            import sr_ncnn
            if sr_ncnn.available():
                res = upscale_anime(img, args.size)
            else:
                print("Real-ESRGAN недоступен (pip install ncnn "
                      "realesrgan-ncnn-py), беру EDSR", file=sys.stderr)
                res = upscale(img, args.size, args.sharp, args.sigma, args.model)
        else:
            res = upscale(img, args.size, args.sharp, args.sigma, args.model)
        if args.out and len(args.images) == 1:
            dst = args.out
        else:
            base, ext = os.path.splitext(src)
            dst = f"{base}_x{args.size}{ext or '.png'}"
        ok = cv2.imwrite(dst, res)
        print(f"{src} {img.shape[1]}×{img.shape[0]} -> {dst} "
              f"{res.shape[1]}×{res.shape[0]} {'ok' if ok else 'ОШИБКА'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
