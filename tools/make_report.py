#!/usr/bin/env python3
"""
make_report.py — собирает отчёт по результатам detect_frames.py:
  * annotated/index.html   — галерея всех размеченных листов (фильтр по статусу)
  * annotated/sheets/*.jpg — контактные листы по 12 листов (для быстрого просмотра)
  * annotated/REPORT.md    — сводка + примеры

Запуск:  python3 tools/make_report.py
"""
from __future__ import annotations

import csv
import html
import json
import os

import cv2
import numpy as np
from urllib.parse import quote

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "annotated")
THUMB_W = 480
PER_CONTACT = 8           # 2 в ряд x 4 ряда
CONTACT_COLS = 2


def load_rows() -> list[dict]:
    with open(os.path.join(OUT, "frames.csv"), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def make_thumbs(rows: list[dict]) -> None:
    d = os.path.join(OUT, "thumbs")
    os.makedirs(d, exist_ok=True)
    for r in rows:
        src = os.path.join(ROOT, r["out"])  # annotated/preview/<...>.jpg
        img = cv2.imread(src)
        h = int(round(img.shape[0] * THUMB_W / img.shape[1]))
        cv2.imwrite(os.path.join(d, r["thumb"]),
                    cv2.resize(img, (THUMB_W, h), interpolation=cv2.INTER_AREA),
                    [int(cv2.IMWRITE_JPEG_QUALITY), 82])


def make_contacts(rows: list[dict]) -> list[str]:
    d = os.path.join(OUT, "sheets")
    os.makedirs(d, exist_ok=True)
    made = []
    for page in range(0, len(rows), PER_CONTACT):
        chunk = rows[page:page + PER_CONTACT]
        tiles = []
        for r in chunk:
            img = cv2.imread(os.path.join(ROOT, r["out"]))
            w = 700
            img = cv2.resize(img, (w, int(round(img.shape[0] * w / img.shape[1]))),
                             interpolation=cv2.INTER_AREA)
            cv2.putText(img, r["file"][:60], (8, 22), cv2.FONT_HERSHEY_SIMPLEX,
                        0.55, (0, 255, 0), 1, cv2.LINE_AA)
            tag = r["status"]
            col = (0, 0, 255) if tag != "ok" else (0, 200, 0)
            cv2.putText(img, f"{tag}  {r['cols']}x{r['rows']} cell {r['cell_w']}x"
                             f"{r['cell_h']} conf {r['confidence']}",
                        (8, img.shape[0] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                        col, 1, cv2.LINE_AA)
            tiles.append(img)
        while len(tiles) % CONTACT_COLS:
            tiles.append(np.zeros_like(tiles[0]))
        hh = min(t.shape[0] for t in tiles)
        rows_img = [np.hstack([t[:hh] for t in tiles[i:i + CONTACT_COLS]])
                    for i in range(0, len(tiles), CONTACT_COLS)]
        page_img = np.vstack(rows_img)
        name = f"sheet_{page // PER_CONTACT + 1:02d}.jpg"
        cv2.imwrite(os.path.join(d, name), page_img,
                    [int(cv2.IMWRITE_JPEG_QUALITY), 72])
        made.append(f"annotated/sheets/{name}")
    return made


def make_gallery(rows: list[dict], contacts: list[str]) -> None:
    css = """
    body{background:#111;color:#ddd;font:14px/1.5 -apple-system,Segoe UI,Roboto,sans-serif;margin:0}
    header{padding:18px 24px;background:#191919;border-bottom:1px solid #2c2c2c;position:sticky;top:0;z-index:5}
    h1{font-size:18px;margin:0 0 6px}
    .stats span{display:inline-block;margin-right:16px;color:#9be89b}
    .filters{margin-top:10px}
    .filters a{color:#8fc9ff;margin-right:14px;text-decoration:none;font-size:13px}
    .filters a.active{color:#fff;border-bottom:2px solid #e04}
    .grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(420px,1fr));gap:14px;padding:18px 24px}
    .card{background:#191919;border:1px solid #2a2a2a;border-radius:6px;overflow:hidden}
    .card img{width:100%;display:block}
    .meta{padding:8px 10px;font-size:12px;color:#aaa;display:flex;justify-content:space-between;gap:8px}
    .meta b{color:#eee;font-weight:600}
    .ok{color:#6ad46a}.warn{color:#ff9b3d}.nofr{color:#ff5b5b}
    .contacts{padding:8px 24px 30px;display:flex;flex-wrap:wrap;gap:10px}
    .contacts a{color:#8fc9ff;font-size:13px}
    """
    cards = []
    for r in rows:
        st = r["status"]
        cls = "nofr" if st == "no_frames" else ("warn" if st != "ok" else "ok")
        boxes = 0 if st == "no_frames" else int(r["cols"]) * int(r["rows"])
        cards.append(f"""<div class="card" data-status="{st}">
  <a href="{quote(r['thumb'])}"><img loading="lazy" src="{quote(r['thumb'])}"></a>
  <div class="meta"><b>{html.escape(r['file'])}</b>
  <span class="{cls}">{st} · {r['cols']}×{r['rows']} · рамок {boxes} · conf {r['confidence']}</span></div>
</div>""")

    ok = sum(1 for r in rows if r["status"] == "ok")
    nofr = sum(1 for r in rows if r["status"] == "no_frames")
    wrn = len(rows) - ok - nofr
    total_boxes = sum(0 if r["status"] == "no_frames"
                      else int(r["cols"]) * int(r["rows"]) for r in rows)
    links = "".join(f'<a href="{c.split("/", 1)[1]}" target="_blank">{os.path.basename(c)}</a>'
                    for c in contacts)
    doc = f"""<!doctype html><html lang="ru"><head><meta charset="utf-8">
<title>Рамки иконок — отработка алгоритма</title><style>{css}</style></head><body>
<header><h1>Поиск серо-белых рамок: отработка на {len(rows)} листах</h1>
<div class="stats"><span>рамок найдено: {total_boxes}</span>
<span>листов ок: {ok}</span><span>с замечаниями: {wrn}</span>
<span>без рамок: {nofr}</span></div>
<div class="filters">
<a href="#" data-f="all" class="active">все</a>
<a href="#" data-f="ok">только ок</a>
<a href="#" data-f="warn">с замечаниями</a>
<a href="#" data-f="no_frames">без рамок</a>
</div></header>
<div class="contacts">Контактные листы: {links}</div>
<div class="grid">{''.join(cards)}</div>
<script>
const links=[...document.querySelectorAll('.filters a')];
links.forEach(a=>a.onclick=e=>{{e.preventDefault();
 links.forEach(x=>x.classList.remove('active'));a.classList.add('active');
 const f=a.dataset.f;
 document.querySelectorAll('.card').forEach(c=>{{
   c.style.display=(f==='all'||c.dataset.status===f)?'':'none';}});}});
</script></body></html>"""
    with open(os.path.join(OUT, "index.html"), "w", encoding="utf-8") as fh:
        fh.write(doc)


def make_markdown(rows: list[dict], contacts: list[str]) -> None:
    ok = [r for r in rows if r["status"] == "ok"]
    nofr = [r for r in rows if r["status"] == "no_frames"]
    wrn = [r for r in rows if r["status"] not in ("ok", "no_frames")]
    confs = [float(r["confidence"]) for r in rows if r["status"] != "no_frames"]
    total_boxes = sum(int(r["cols"]) * int(r["rows"]) for r in ok + wrn)
    lines = [
        "# Отработка детектора рамок",
        "",
        f"- листов обработано: **{len(rows)}**",
        f"- рамок найдено: **{total_boxes}** (найденных по всем листам, где рамки есть)",
        f"- листов без замечаний: **{len(ok)}**",
        f"- листов с замечаниями: **{len(wrn)}**",
        f"- листов, где рамок нет вообще: **{len(nofr)}**",
        f"- confidence: min {min(confs):.2f}, median {np.median(confs):.2f}, "
        f"max {max(confs):.2f}",
        "",
        "Красным на разметке обведены уверенно найденные рамки, оранжевым — те, "
        "где линия не подтвердилась (откат к сетке), серым на листах без рамок — "
        "сетка слотов.",
        "",
        "## Листы без рамок (алгоритм честно говорит «рамок нет»)",
        "",
    ]
    lines += [f"- `{r['file']}`" for r in nofr]
    lines += ["", "## Листы с замечаниями", ""]
    lines += [f"- `{r['file']}` — {r['warn']} (conf {r['confidence']})" for r in wrn] or ["—"]
    lines += ["", "## Контактные листы", ""]
    for c in contacts:
        lines.append(f"![{os.path.basename(c)}]({c})")
    with open(os.path.join(OUT, "REPORT.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")


def main() -> int:
    rows = load_rows()
    for r in rows:
        r["thumb"] = r["out"].split("/", 1)[1]
        if "status" not in r:
            w = r["warn"]
            r["status"] = "ok" if not w else ("no_frames" if "no_frames" in w else "warn")
    contacts = make_contacts(rows)
    make_gallery(rows, contacts)
    make_markdown(rows, contacts)

    # сводка по CSV с человекочитаемым статусом
    with open(os.path.join(OUT, "frames.csv"), "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(OUT, "frames.json"), "w", encoding="utf-8") as fh:
        json.dump(rows, fh, ensure_ascii=False, indent=1)
    print("готово:", ", ".join([f"index.html, REPORT.md, frames.csv, {len(contacts)} контактных листов"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
