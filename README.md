# Eubot Junior (Hermes)

Bot conversazionale in italiano, specializzato su letteratura classica, filosofia ed esoterismo.  
Fine-tuning **QLoRA** su **Qwen2.5-3B-Instruct**.

## Repository

- GitHub: <https://github.com/socialengaged/eubot_jr>

## Struttura

```
configs/          # finetune.yaml, personality.yaml
scripts/          # download, dataset, training, serve, chat
webapp/           # UI chat statica
data/             # raw/processed/training (non in git)
models/           # adapter e modello merged (non in git)
```

## Setup (locale o RunPod)

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Pipeline

1. **Scarica testi** (Gutenberg + testi filosofici/esoterici pubblici dominio):

   ```bash
   python scripts/download_gutenberg.py
   python scripts/download_sacred.py
   ```

2. **Pulizia** (opzionale, già chiamata dagli script di download):

   ```bash
   python scripts/clean_text.py
   ```

3. **Dataset chat JSONL** (HF `mchl-labs/stambecco_data_it` + QA da testi locali):

   ```bash
   python scripts/build_dataset.py
   ```

4. **Training** (~1h primo run con `max_steps: 2000` in `configs/finetune.yaml`):

   ```bash
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/finetune.py
   # oppure resume:
   PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python scripts/finetune.py --resume
   ```

5. **Merge adapter** (opzionale, per inference senza PEFT):

   ```bash
   python scripts/merge_adapter.py
   ```

6. **Inference**

   ```bash
   python scripts/serve.py --host 0.0.0.0 --port 8080
   python scripts/chat.py
   ```

## Coesistenza con Eubot Coder

Usa la **stessa GPU** solo in sequenza: quando il training di `eubot-coder` è finito, clona questo repo in `/workspace` e lancia `build_dataset` + `finetune`. Non avviare due training contemporaneamente sulla stessa GPU.

## Licenze dati

- Project Gutenberg: pubblico dominio (USA).
- Dataset Hugging Face: verificare la scheda del dataset prima dell’uso commerciale.

## Personalità

Definita in `configs/personality.yaml` e nel campo `system_prompt` di `configs/finetune.yaml`.
