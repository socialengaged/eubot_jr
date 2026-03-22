#!/usr/bin/env python3
"""
Costruisce train.jsonl / val.jsonl in formato chat:
- Dataset istruzioni italiano (HF: stambecco + fallback)
- Coppie consecutive da testi puliti (continuazione / stile)
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

from datasets import load_dataset

from jr_utils import load_yaml, repo_root


def load_system_prompt() -> str:
    p = ROOT / "configs" / "finetune.yaml"
    cfg = load_yaml(p)
    return (cfg.get("system_prompt") or "").strip()


def alpaca_like_to_messages(row: dict, system: str) -> dict | None:
    inst = (row.get("instruction") or row.get("prompt") or row.get("domanda") or "").strip()
    inp = (row.get("input") or row.get("context") or row.get("Input") or "").strip()
    out = (row.get("output") or row.get("response") or row.get("risposta") or row.get("answer") or "").strip()
    if not inst or not out:
        return None
    user = inst if not inp else f"{inst}\n\n{inp}"
    return {
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
            {"role": "assistant", "content": out},
        ]
    }


def load_stambecco_rows(system: str, max_rows: int) -> list[dict]:
    rows: list[dict] = []
    candidates = [
        "mchl-labs/stambecco_data_it",
        "marmolito/dolly-it",
    ]
    last_err: Exception | None = None
    for name in candidates:
        try:
            ds = load_dataset(name, split="train", trust_remote_code=True)
            for i, row in enumerate(ds):
                if i >= max_rows * 2:
                    break
                m = alpaca_like_to_messages(dict(row), system)
                if m:
                    rows.append(m)
                if len(rows) >= max_rows:
                    break
            print(f"Loaded {len(rows)} rows from {name}")
            if len(rows) > 0:
                return rows
            print(f"  (no usable rows from {name}, trying next)")
        except Exception as e:
            last_err = e
            print(f"Skip {name}: {e}")
    if last_err:
        raise RuntimeError(f"Nessun dataset istruzioni caricato: {last_err}") from last_err
    return rows


def chunk_text(text: str, min_len: int = 400, max_len: int = 900) -> list[str]:
    """Spezza in blocchi per paragrafi / lunghezza."""
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) < min_len:
        return [text] if text else []
    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_len, len(text))
        if end < len(text):
            sp = text.rfind(" ", start + min_len, end)
            if sp > start:
                end = sp
        piece = text[start:end].strip()
        if len(piece) >= min_len // 2:
            chunks.append(piece)
        start = end if end > start else end + 1
    return chunks


def literature_pairs_from_file(path: Path, system: str, max_pairs: int) -> list[dict]:
    out: list[dict] = []
    try:
        raw = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return out
    chunks = chunk_text(raw)
    if len(chunks) < 2:
        return out
    for i in range(len(chunks) - 1):
        if len(out) >= max_pairs:
            break
        a, b = chunks[i], chunks[i + 1]
        user = (
            "Ecco un estratto di un'opera classica o filosofica:\n\n"
            f"{a}\n\n"
            "Come potrebbe proseguire il discorso, in linea con lo stile e i temi dell'autore? "
            "Rispondi con un breve brano coerente."
        )
        out.append(
            {
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": b[: min(1200, len(b))]},
                ]
            }
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max_instruction", type=int, default=25000, help="Max esempi da dataset HF")
    ap.add_argument("--max_literature_pairs", type=int, default=5000, help="Max coppie da testi locali")
    ap.add_argument("--val_ratio", type=float, default=0.02)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    random.seed(args.seed)
    system = load_system_prompt()
    if not system:
        raise SystemExit("system_prompt mancante in configs/finetune.yaml")

    all_rows: list[dict] = []

    print("Loading Italian instruction dataset …")
    all_rows.extend(load_stambecco_rows(system, args.max_instruction))

    proc = ROOT / "data" / "processed"
    lit: list[dict] = []
    if proc.is_dir():
        txt_files = sorted(proc.rglob("*.txt"))
        per_file = max(50, args.max_literature_pairs // max(1, len(txt_files)))
        for p in txt_files:
            if len(lit) >= args.max_literature_pairs:
                break
            lit.extend(literature_pairs_from_file(p, system, per_file))
        lit = lit[: args.max_literature_pairs]
        all_rows.extend(lit)
        print(f"Added {len(lit)} literature continuation pairs")
    else:
        print(f"No {proc} — run download_* and clean_text.py first (optional).")

    if not all_rows:
        raise SystemExit("Nessun esempio: verifica connessione HF e/o data/processed/*.txt")

    random.shuffle(all_rows)
    n_val = max(1, int(len(all_rows) * args.val_ratio))
    val_rows = all_rows[:n_val]
    train_rows = all_rows[n_val:]

    out_train = repo_root() / "data" / "training" / "train.jsonl"
    out_val = repo_root() / "data" / "training" / "val.jsonl"
    out_train.parent.mkdir(parents=True, exist_ok=True)

    with open(out_train, "w", encoding="utf-8") as f:
        for r in train_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(out_val, "w", encoding="utf-8") as f:
        for r in val_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"Wrote {len(train_rows)} train, {len(val_rows)} val -> {out_train.parent}")


if __name__ == "__main__":
    main()
