#!/usr/bin/env python3
"""
style_frame.py — объёмный чёрный бортик («бортик из теней») вокруг слотов.

Рисуем не линию, а рельефную рамку:
  * борт — тёмная полоса (#141518) по периметру слота со скруглёнными углами;
  * фаска: верхняя/левая кромка борта светлеет (свет падает сверху-слева),
    нижняя/правая темнеет — бортик читается как физический выступ;
  * внутренняя тень по верхней/левой кромке слота (мягкая, как от борта);
  * тонкий блик по нижней/правой внутренней кромке;
  * лёгкая внешняя падающая тень под бортом.

Всё считается через signed distance до скруглённого прямоугольника и
сглаженные маски, поэтому линии получаются антиалиасенными на любом масштабе.

    python3 tools/style_frame.py                       # все листы -> styled/
    python3 tools/style_frame.py --variant inset --limit 3
    python3 tools/style_frame.py "RPG Loot Icons 01/RPG Loot Icons 01 Part 1.jpg"
"""
from __future__ import annotations

import argparse
import glob
import os
import sys

import cv2
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import detect_frames as D  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

#            rim  inset  radius  bevel  shadow  deep  gloss  drop  lip   tone
VARIANTS = {
    # тонкий сдержанный бортик
    "soft":  dict(rim=3.5, inset=1.0, radius=5.0, bevel=0.50, shadow=0.6,
                  deep=6.0, gloss=0.12, drop=0.30, lip=0.18, tone=0.78),
    # выраженный бортик-выступ (основной вариант)
    "deep":  dict(rim=5.5, inset=1.5, radius=9.0, bevel=0.85, shadow=1.0,
                  deep=14.0, gloss=0.22, drop=0.55, lip=0.38, tone=1.05),
    # «утопленный» слот: бортик узкий, градиент мягкий
    "inset": dict(rim=4.0, inset=1.0, radius=8.0, bevel=0.55, shadow=1.3,
                  deep=20.0, gloss=0.14, drop=0.25, lip=0.24, tone=0.85),
    # то же, что deep, но с прямыми углами (свет сверху-слева, как в deep)
    "sharp": dict(rim=5.5, inset=1.5, radius=0.0, bevel=0.85, shadow=1.0,
                  deep=14.0, gloss=0.22, drop=0.55, lip=0.38, tone=1.05,
                  light="top"),
    # прямые углы + свет снизу (тёмный верх, глубина читается снизу)
    "sharp_bottom": dict(rim=5.5, inset=1.5, radius=0.0, bevel=0.85, shadow=1.0,
                         deep=14.0, gloss=0.22, drop=0.55, lip=0.30, tone=1.05,
                         light="bottom"),
}

# готовые пресеты отделки (база — sharp: прямые углы, свет сверху-слева)
DECOR = {
    "steel":    dict(),                                        # нейтральный
    "bronze":   dict(tint=(1.45, 1.05, 0.55)),                  # тёплый металл
    "cold":     dict(tint=(0.70, 0.95, 1.45)),                  # холодный металл
    "pin":      dict(pin=0.85),                                 # светлая нить внутри
    "brackets": dict(bracket=0.45, bracket_len=11.0),           # акценты в углах
    "matte":    dict(bevel=0.0, lip=0.08, gloss=0.04, drop=0.70),
    "glass":    dict(bevel=0.45, lip=0.85, gloss=0.55, tint=(0.9, 1.0, 1.1)),
    "slim":     dict(rim=3.0, inset=1.0, deep=9.0, lip=0.30),
    "glow":     dict(glow=0.55, glow_tint=(1.0, 0.55, 0.25)),   # подсветка снаружи
    "double":   dict(outer_line=0.75),                          # внешний волосяной контур
}


def decor_params(name: str, base: str = "sharp") -> dict:
    """Параметры рамки с пресетом отделки."""
    p = dict(VARIANTS[base])
    p.update(DECOR[name])
    return p


# параметры, которые можно переопределить из командной строки
OVERRIDES = ("rim", "inset", "radius", "bevel", "shadow", "deep", "gloss",
             "drop", "lip", "tone", "light")


def params(variant: str = "deep", **kw) -> dict:
    """Параметры варианта + переопределения (например radius=0)."""
    p = dict(VARIANTS[variant])
    for k, v in kw.items():
        if v is None or k not in OVERRIDES:
            continue
        p[k] = v if k == "light" else float(v)
    if p["radius"] <= 0.05:       # прямые углы: без скругления
        p["radius"] = 0.0
    return p
RIM_DARK = np.array([0x1b, 0x1c, 0x20], np.float32)   # внутренняя кромка борта
RIM_LIGHT = np.array([0x55, 0x59, 0x61], np.float32)  # внешняя кромка на свету
LIP = np.array([0x8e, 0x93, 0x9c], np.float32)        # тонкая световая линия
GLOSS = np.array([0xa8, 0xad, 0xb6], np.float32)


def _sdf(X, Y, cx, cy, hw, hh, r, sharp_outside: bool = False):
    """Signed distance до скруглённого прямоугольника (<0 внутри).

    sharp_outside=True считает внешнюю часть по L∞ (шахматной) метрике — тогда
    смещение борта наружу не скругляет внешние углы, углы остаются прямыми.
    """
    qx = np.abs(X - cx) - (hw - r)
    qy = np.abs(Y - cy) - (hh - r)
    ax, ay = np.maximum(qx, 0.0), np.maximum(qy, 0.0)
    outside = np.maximum(ax, ay) if sharp_outside else np.sqrt(ax * ax + ay * ay)
    return outside + np.minimum(np.maximum(qx, qy), 0.0) - r


def _cov(sd, px=1.0):
    """Покрытие: 1 внутри (sd<0), плавный переход шириной px."""
    return np.clip(0.5 - sd / max(px, 1e-3), 0.0, 1.0)


def style_box(img: np.ndarray, box, p: dict, rim_rgb=RIM_DARK) -> None:
    x0, y0, x1, y1 = box
    m = int(np.ceil(p["rim"] + p["inset"] + 24))
    H, W = img.shape[:2]
    ax0, ay0 = max(0, x0 - m), max(0, y0 - m)
    ax1, ay1 = min(W, x1 + m + 1), min(H, y1 + m + 1)
    patch = img[ay0:ay1, ax0:ax1].astype(np.float32)

    Y, X = np.mgrid[ay0:ay1, ax0:ax1].astype(np.float32)
    cx, cy = (x0 + x1) / 2.0, (y0 + y1) / 2.0
    hw, hh = (x1 - x0 + 1) / 2.0, (y1 - y0 + 1) / 2.0
    r = min(p["radius"], hw * 0.6, hh * 0.6)
    sd = _sdf(X, Y, cx, cy, hw, hh, r,
              sharp_outside=p.get("sharp_outside", r <= 0.05))

    nx = (X - cx) / hw
    ny = (Y - cy) / hh
    if p.get("light", "top") == "bottom":
        # свет снизу-справа: блик по нижней кромке, верх остаётся тёмным
        lit = np.clip(nx * 0.5 + ny * 0.9, -1.0, 1.0)
        shade_dir = np.clip(-ny, 0.0, 1.0)          # внутренняя тень сверху
    else:
        lit = np.clip(-(nx * 0.6 + ny * 0.8), -1.0, 1.0)   # 1 = верх-лево
        shade_dir = np.clip(-lit, 0.0, 1.0)

    # --- 1. внешняя падающая тень (под бортом, снизу-справа)
    drop_band = np.clip(_cov(sd - p["rim"] - 1.0, 6.0) -
                        _cov(sd - p["rim"] - 7.0, 6.0), 0.0, 1.0)
    drop = drop_band * np.clip(0.5 - lit, 0.0, 1.0) * p["drop"]

    # --- 2. тело борта: градиент поперёк борта (светлее у внешней кромки)
    edge = 0.8
    rim = np.clip(_cov(sd - p["rim"], edge) - _cov(sd + p["inset"], edge), 0, 1)
    # t = 0 у внешней кромки, 1 у внутренней
    t = np.clip((sd + p["inset"]) / max(p["rim"] + p["inset"], 1e-3), 0.0, 1.0)
    t = (t ** 1.4)[..., None]
    tint = np.asarray(p.get("tint", (1.0, 1.0, 1.0)), np.float32)
    tint = tint[::-1]                     # пресеты заданы в RGB, картинка в BGR
    tone = rim_rgb * p["tone"] * tint
    light_base = RIM_LIGHT * tint
    lip_base = LIP * tint
    light_col = tone * 0.35 + light_base * 0.65
    # на свету (верх-лево) внешняя кромка светлее, на теневой стороне темнее
    side = np.clip((lit + 1.0) / 2.0, 0.0, 1.0)
    if p.get("light", "top") == "bottom":
        # свет снизу: верхняя кромка почти не подсвечивается (углубляем верх)
        side = side ** 2.5
    mix = (1.0 - t) * (p["bevel"] * (0.10 + 0.90 * side))[..., None]
    rim_col = tone * (1.0 - mix) + light_col * mix

    # тонкая световая линия по внешней кромке (только на свету) — «отражение»
    lip = np.clip(_cov(sd - p["rim"] + 0.3, 0.8) -
                  _cov(sd - p["rim"] + 1.2, 0.8), 0.0, 1.0)
    lip = (lip * np.clip(lit, 0.0, 1.0) * p["lip"])[..., None]

    # --- опция: внутренняя пин-линия (тонкая светлая нить внутри борта)
    pin = p.get("pin", 0.0)
    pin_m = 0.0
    if pin:
        pin_m = np.clip(_cov(sd + p["inset"] + 1.6, 0.8) -
                        _cov(sd + p["inset"] + 2.6, 0.8), 0.0, 1.0)
        pin_m = (pin_m * pin)[..., None]

    # --- опция: угловые скобки (акценты в углах): только зона самого борта
    brk = p.get("bracket", 0.0)
    brk_m = 0.0
    if brk:
        L = float(p.get("bracket_len", 16.0))
        band = np.clip(_cov(sd - p["rim"], 1.0) -
                       _cov(sd + p["inset"] + 1.0, 1.0), 0.0, 1.0)
        zone = ((np.abs(X - cx) > hw - L) &
                (np.abs(Y - cy) > hh - L)).astype(np.float32)
        brk_m = (band * zone * min(brk, 1.0))[..., None]

    # --- 3. внутренняя тень слота (сверху-слева) и блик (снизу-справа)
    d0, d1 = p["inset"], p["inset"] + p["deep"]
    ish = np.clip(_cov(sd + d0, 1.0) - _cov(sd + d1, d1 - d0), 0, 1)
    ish = cv2.GaussianBlur(ish, (0, 0), max(2.0, p["deep"] * 0.35))
    ish = ish * (0.35 + 0.65 * shade_dir) * p["shadow"]
    gl = np.clip(_cov(sd + p["inset"] + 0.4, 1.0) -
                 _cov(sd + p["inset"] + 1.8, 1.0), 0.0, 1.0)
    gl = gl * np.clip(lit, 0.0, 1.0) * p["gloss"]

    # --- опция: внешнее свечение (например, для hover/выделения)
    glow = p.get("glow", 0.0)
    if glow:
        gb = np.clip(_cov(sd - p["rim"] - 1.0, 9.0) * 0.5, 0.0, 1.0)
        gt = np.asarray(p.get("glow_tint", (1.0, 0.6, 0.3)), np.float32)[::-1]
        patch = patch + gb[..., None] * gt * 90.0 * glow

    # --- опция: внешний волосяной контур (двойная рамка)
    oline = p.get("outer_line", 0.0)
    if oline:
        ol = np.clip(_cov(sd - p["rim"] - 2.6, 0.9) -
                     _cov(sd - p["rim"] - 3.4, 0.9), 0.0, 1.0)
        om = (ol * oline)[..., None]
        patch = patch * (1.0 - om) + lip_base * 1.1 * om

    # --- композит
    patch *= (1.0 - drop)[..., None]                       # падающая тень
    patch *= (1.0 - ish)[..., None]                        # внутренняя тень
    a = rim[..., None]
    patch = patch * (1.0 - a) + rim_col * a                # борт
    g = gl[..., None]
    patch = patch * (1.0 - g) + GLOSS * g                  # блик
    patch = patch * (1.0 - lip) + lip_base * lip           # световая линия
    if pin:                                                 # внутренняя нить
        patch = patch * (1.0 - pin_m) + lip_base * 0.95 * pin_m
    if brk:                                                 # угловые скобки
        patch = patch * (1.0 - brk_m) + lip_base * 1.25 * brk_m

    img[ay0:ay1, ax0:ax1] = np.clip(patch, 0, 255).astype(np.uint8)


def style_sheet(path: str, out: str, variant: str = "deep",
                quality: int = 92, scale: float = 1.0, **kw) -> int:
    p = params(variant, **kw)
    sh = D.detect_sheet(path)
    img = cv2.imread(path)
    for box in sh.boxes:
        style_box(img, box, p)
    if scale != 1.0:
        img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)
    cv2.imwrite(out, img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    return len(sh.boxes)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Объёмный чёрный бортик вокруг слотов")
    ap.add_argument("images", nargs="*")
    ap.add_argument("--out", default="styled")
    ap.add_argument("--variant", default="deep", choices=sorted(VARIANTS))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--quality", type=int, default=92)
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--radius", type=float, default=None,
                    help="радиус скругления, 0 = прямые углы")
    ap.add_argument("--light", default=None, choices=["top", "bottom"],
                    help="источник света: top (по умолчанию) или bottom")
    args = ap.parse_args(argv)

    files = args.images or D.all_sheets()
    if args.limit:
        files = files[: args.limit]
    for n, f in enumerate(files, 1):
        f = f if os.path.isabs(f) else os.path.join(ROOT, f)
        rel = os.path.relpath(f, ROOT)
        out_rel = os.path.join(args.out, os.path.splitext(rel)[0] + ".jpg")
        out = out_rel if os.path.isabs(args.out) else os.path.join(ROOT, out_rel)
        k = style_sheet(f, out, variant=args.variant, quality=args.quality,
                        scale=args.scale, radius=args.radius, light=args.light)
        print(f"[{n}/{len(files)}] {rel}  рамок {k}  -> {out_rel}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
