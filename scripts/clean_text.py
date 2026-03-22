#!/usr/bin/env python3
"""Pulisce testi Gutenberg: encoding, header/footer, righe vuote."""
from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"


def strip_gutenberg_boilerplate(text: str) -> str:
    """Rimuove blocchi tipici *** START / END OF THIS PROJECT GUTENBERG EBOOK."""
    t = text.replace("\r\n", "\n")
    # taglia dopo START ... se presente
    m = re.search(
        r"\*\*\*\s*START OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*?\*\*\*",
        t,
        re.IGNORECASE | re.DOTALL,
    )
    if m:
        t = t[m.end() :]
    m2 = re.search(
        r"\*\*\*\s*END OF (?:THIS|THE) PROJECT GUTENBERG EBOOK.*",
        t,
        re.IGNORECASE | re.DOTALL,
    )
    if m2:
        t = t[: m2.start()]
    # normalizza spazi
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()


def clean_file(src: Path, dst: Path) -> None:
    raw = src.read_bytes()
    for enc in ("utf-8", "utf-8-sig", "latin-1", "cp1252"):
        try:
            text = raw.decode(enc)
            break
        except UnicodeDecodeError:
            text = None
    else:
        text = raw.decode("utf-8", errors="replace")
    text = strip_gutenberg_boilerplate(text)
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(text, encoding="utf-8")
    print(f"cleaned -> {dst.relative_to(PROC)} ({len(text)} chars)")


def main() -> None:
    if not RAW.is_dir():
        print(f"Missing {RAW} — run download_gutenberg.py / download_sacred.py first.")
        raise SystemExit(1)
    n = 0
    for src in RAW.rglob("*.txt"):
        rel = src.relative_to(RAW)
        dst = PROC / rel
        clean_file(src, dst)
        n += 1
    print(f"Done: {n} files -> {PROC}")


if __name__ == "__main__":
    main()
