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

## Pod "images" (GPU condivisa con ComfyUI / video)

Su un Pod dove girano **generazione immagini o video** (es. ComfyUI), non lanciare il training “nudo” in concorrenza: usa **`scripts/gpu_guard.sh`** così il fine-tune si **mette in pausa** quando altri processi occupano troppa VRAM e **riprende con `--resume`** quando la GPU torna libera.

### SSH (TCP diretto, esempio)

Aggiungi in `~/.ssh/config` (Windows: `C:\Users\info\.ssh\config`):

```
Host images
    HostName 213.173.103.227
    Port 40181
    User root
    IdentityFile ~/.ssh/eubot_ed25519
    StrictHostKeyChecking accept-new
```

Poi: `ssh images`

### Setup e avvio con gpu_guard

```bash
cd /workspace
git clone https://github.com/socialengaged/eubot_jr.git
cd eubot_jr
pip install -r requirements.txt
chmod +x scripts/gpu_guard.sh

python scripts/download_gutenberg.py
python scripts/download_sacred.py
python scripts/clean_text.py
python scripts/build_dataset.py

WORKDIR=/workspace/eubot_jr nohup bash scripts/gpu_guard.sh >> gpu_guard.log 2>&1 &
tail -f gpu_guard.log
# training stdout/stderr: training.log
```

### Soglie VRAM (opzionale)

Variabili d’ambiente (default ragionevoli per dare priorità a img/video):

- `PAUSE_OTHER_MB` — se la VRAM usata da processi **diversi** dal training è >= questo valore (MiB), il training viene fermato con SIGTERM (checkpoint recenti ogni `save_steps`, default 250). Default script: **14000** (soglia “carico pesante” oltre baseline SD/ComfyUI).
- `RESUME_OTHER_MB` — per **avviare** o **riprendere**, la VRAM “altra” deve restare **sotto** questo valore per `STABLE_SEC` secondi. Default script: **11000** (deve essere **sopra** la VRAM “idle” dei servizi img/video sul Pod, altrimenti il guard non parte mai).
- `POLL_SEC` — intervallo di polling (default 30).
- `STABLE_SEC` — secondi di GPU “calma” prima di ripartire (default 60).

Esempio (Pod quasi vuoto, poca VRAM occupata da altri processi):

```bash
export PAUSE_OTHER_MB=4000 RESUME_OTHER_MB=3000 POLL_SEC=20 STABLE_SEC=90
WORKDIR=/workspace/eubot_jr nohup bash scripts/gpu_guard.sh >> gpu_guard.log 2>&1 &
```

### Log utili

```bash
tail -f /workspace/eubot_jr/gpu_guard.log
tail -f /workspace/eubot_jr/training.log
ls -la /workspace/eubot_jr/models/lora_adapter/
```
