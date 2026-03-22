#!/usr/bin/env python3
"""Scarica testi filosofici / orientali / esoterici da Project Gutenberg (pubblico dominio)."""
from __future__ import annotations

import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

OUT = ROOT / "data" / "raw" / "sacred"

# ID Gutenberg: verificare licenza PD nel proprio paese
SACRED: list[tuple[int, str]] = [
    (4397, "lao_tzu_tao_te_ching.txt"),
    (2388, "bhagavad_gita.txt"),
    (2145, "kybalion_three_initiates.txt"),
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
            with urllib.request.urlopen(req, timeout=90) as r:
                data = r.read()
            dest.write_bytes(data)
            print(f"OK {gid} -> {dest.name} ({len(data)} bytes)")
            return True
        except Exception as e:
            print(f"  try failed {url}: {e}")
            time.sleep(0.5)
    return False


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    ok = 0
    for gid, name in SACRED:
        dest = OUT / name
        if dest.is_file() and dest.stat().st_size > 500:
            print(f"skip (exists): {dest.name}")
            ok += 1
            continue
        if download_one(gid, dest):
            ok += 1
    print(f"Done: {ok}/{len(SACRED)} files in {OUT}")
    if ok < len(SACRED):
        print("Warning: alcuni download sono falliti (ID o rete). Riprova piu tardi.")


if __name__ == "__main__":
    main()
