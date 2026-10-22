#!/usr/bin/env python3
"""
fetch_models.py — скачать нейросетевые модели для апскейла.

Модели лежат в tools/models/ и в git не хранятся (EDSR ~38 МБ). Скрипт
идемпотентный: если файл уже есть и размер правдоподобный — не качает.

    python3 tools/fetch_models.py            # FSRCNN x4 (для пакета)
    python3 tools/fetch_models.py --all      # + EDSR x4 (для --smooth)
"""
from __future__ import annotations

import argparse
import os
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = os.path.join(HERE, "models")

# две ссылки на файл: api.github.com работает там, где raw.githubusercontent
# зарезан корпоративным прокси. Пробуем по очереди.
API = "https://api.github.com/repos/Saafke/{repo}/contents/models/{name}"
RAW = "https://raw.githubusercontent.com/Saafke/{repo}/master/models/{name}"

FILES = {
    # FSRCNN лёгкий (41 КБ) и лежит в репозитории — качать не нужно
    "FSRCNN_x4.pb": (API.format(repo="FSRCNN_Tensorflow", name="FSRCNN_x4.pb"),
                     RAW.format(repo="FSRCNN_Tensorflow", name="FSRCNN_x4.pb"), 20_000),
    "EDSR_x4.pb": (API.format(repo="EDSR_Tensorflow", name="EDSR_x4.pb"),
                   RAW.format(repo="EDSR_Tensorflow", name="EDSR_x4.pb"), 5_000_000),
}


def _download(url: str, dst: str) -> None:
    req = urllib.request.Request(url, headers={
        "Accept": "application/vnd.github.raw",
        "User-Agent": "icons-tools",
    })
    with urllib.request.urlopen(req, timeout=120) as r, open(dst, "wb") as f:
        while True:
            chunk = r.read(1 << 20)
            if not chunk:
                break
            f.write(chunk)


def _get(name: str, urls: tuple[str, ...], min_size: int) -> str:
    dst = os.path.join(MODELS, name)
    if os.path.exists(dst) and os.path.getsize(dst) >= min_size:
        print(f"  {name}: уже есть")
        return dst
    os.makedirs(MODELS, exist_ok=True)
    tmp = dst + ".part"
    last = None
    for i, url in enumerate(urls, 1):
        print(f"  качаю {name} (источник {i}/{len(urls)}) ...")
        try:
            _download(url, tmp)
            if os.path.getsize(tmp) < min_size:
                raise RuntimeError("файл слишком маленький — похоже, это ошибка 404")
            os.replace(tmp, dst)
            print(f"  {name}: {os.path.getsize(dst) / 1e6:.1f} МБ — ok")
            return dst
        except Exception as e:                  # noqa: BLE001
            last = e
            print(f"    не вышло: {e}")
            if os.path.exists(tmp):
                os.remove(tmp)
    raise RuntimeError(
        f"не удалось скачать {name}: {last}\n"
        f"  Скачай вручную и положи в {MODELS}/\n"
        f"  ссылка: {urls[-1]}")


def ensure(all_models: bool = False) -> None:
    names = list(FILES) if all_models else ["FSRCNN_x4.pb"]
    for n in names:
        api, raw, min_size = FILES[n]
        _get(n, (api, raw), min_size)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true", help="скачать и EDSR (тяжёлая)")
    args = ap.parse_args()
    ensure(args.all)
    print("модели на месте:", MODELS)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
