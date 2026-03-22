# RunPod — Eubot Junior (Hermes)

## Condivisione GPU con Eubot Coder

- **Un solo job pesante alla volta** sulla stessa GPU: non avviare `eubot-coder/scripts/finetune.py` e `eubot_jr/scripts/finetune.py` insieme.
- Se **Eubot Coder** è ancora in training: prepara solo questo repo (`git clone`, `pip install`, `build_dataset`) e **aspetta** la fine prima di lanciare il fine-tune Junior.
- Se il Pod è spento: clona e installa quando lo riaccendi; nessun carico finché non lanci training.

## Comandi sul Pod (path tipico)

```bash
cd /workspace
git clone https://github.com/socialengaged/eubot_jr.git
cd eubot_jr
pip install -r requirements.txt

python scripts/download_gutenberg.py
python scripts/download_sacred.py
python scripts/clean_text.py
python scripts/build_dataset.py

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  nohup python scripts/finetune.py > training.log 2>&1 &
tail -f training.log
```

Primo run: `max_steps: 2000` in `configs/finetune.yaml` (~1h). Poi test (`chat.py` / `serve.py`), aumentare `max_steps` o `num_train_epochs` e ripartire con `--resume`.

## Verifica stato Eubot Coder (stesso server)

```bash
tail -30 /workspace/eubot/eubot-coder/training.log
pgrep -af finetune
```

Se compare ancora loss / progress, **non** avviare Junior.
