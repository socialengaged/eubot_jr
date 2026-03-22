#!/usr/bin/env python3
"""Scarica classici italiani da Project Gutenberg (testo UTF-8)."""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data" / "raw" / "gutenberg"

# (gutenberg_id, filename) — edizioni italiane / traduzioni (Project Gutenberg)
BOOKS: list[tuple[int, str]] = [
    (1012, "dante_divina_commedia.txt"),
    (1232, "machiavelli_il_principe.txt"),
    (1610, "seneca_lettere_lucilio.txt"),
    (12924, "manzoni_promessi_sposi.txt"),
    (28206, "leopardi_canti.txt"),
]


def _url_candidates(gid: int) -> list[str]:
    return [
        f"https://www.gutenberg.org/files/{gid}/{gid}-0.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}-8.txt",
        f"https://www.gutenberg.org/files/{gid}/{gid}.txt",
    ]


def download_one(gid: int, dest: Path) -> bool:
    for url in _url_candidates(gid):
        try:
            req = urllib.request.Request(
                url,
                headers={"User-Agent": "EubotJunior/1.0 (educational; +https://github.com/socialengaged/eubot_jr)"},
            )
            with urllib.request.urlopen(req, timeout=60) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"OK {gid} -> {dest.name} ({len(data)} bytes) via {url}")
            return True
        except Exception as e:
            print(f"  try failed {url}: {e}")
            time.sleep(0.5)
    return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for gid, name in BOOKS:
        dest = OUT / name
        if dest.is_file() and dest.stat().st_size > 1000:
            print(f"skip (exists): {dest.name}")
            ok += 1
            continue
        if download_one(gid, dest):
            ok += 1
    print(f"Done: {ok}/{len(BOOKS)} files in {OUT}")
    if ok < len(BOOKS):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
