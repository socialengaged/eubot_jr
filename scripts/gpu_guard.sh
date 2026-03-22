#!/usr/bin/env bash
# GPU-aware training scheduler for Eubot Junior (Hermes).
# Pauses QLoRA training when other processes use GPU memory above a threshold
# (e.g. ComfyUI / image-video generation), then resumes with --resume when VRAM is free.
#
# Usage (from repo root):
#   WORKDIR=/workspace/eubot_jr nohup bash scripts/gpu_guard.sh >> gpu_guard.log 2>&1 &
#
# Env (optional):
#   WORKDIR          repo root (default: parent of scripts/)
#   POLL_SEC         poll interval seconds (default: 30)
#   PAUSE_OTHER_MB   pause training if non-training GPU usage >= this MiB (default: 4000)
#   RESUME_OTHER_MB  allow training if non-training GPU usage <= this MiB for STABLE_SEC (default: 2800)
#   STABLE_SEC       seconds "quiet" GPU before starting/resuming (default: 60)
#   TRAINING_LOG     append training stdout/stderr (default: WORKDIR/training.log)

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKDIR="${WORKDIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
cd "$WORKDIR" || exit 1

POLL_SEC="${POLL_SEC:-30}"
PAUSE_OTHER_MB="${PAUSE_OTHER_MB:-4000}"
RESUME_OTHER_MB="${RESUME_OTHER_MB:-2800}"
STABLE_SEC="${STABLE_SEC:-60}"
TRAINING_LOG="${TRAINING_LOG:-$WORKDIR/training.log}"

STABLE_NEEDS=$(( (STABLE_SEC + POLL_SEC - 1) / POLL_SEC ))
STABLE_COUNT=0

log() {
  echo "[$(date -Iseconds)] $*"
}

training_pids() {
  # Main training process(es)
  pgrep -f "python.*scripts/finetune\.py" 2>/dev/null || true
}

is_training_running() {
  [[ -n "$(training_pids)" ]]
}

training_completed() {
  if [[ -f "$TRAINING_LOG" ]] && grep -q "Adapter saved to" "$TRAINING_LOG" 2>/dev/null; then
    return 0
  fi
  return 1
}

# Sum GPU memory (MiB) used by processes NOT in the training PID set.
other_gpu_memory_mib() {
  local train_set line pid mib sum
  train_set="$(training_pids | tr '\n' ' ')"
  sum=0
  while IFS= read -r line; do
    [[ -z "$line" ]] && continue
    # csv: pid, used_gpu_memory (MiB) — tolerate with/without "nounits"
    pid="$(echo "$line" | awk -F',' '{gsub(/ /,"",$1); print $1}')"
    mib="$(echo "$line" | awk -F',' '{print $2}' | grep -oE '[0-9]+' | head -1)"
    [[ -z "$pid" || -z "$mib" ]] && continue
    local skip=0
    for t in $train_set; do
      if [[ "$pid" == "$t" ]]; then
        skip=1
        break
      fi
    done
    [[ "$skip" -eq 1 ]] && continue
    sum=$((sum + mib))
  done < <(nvidia-smi --query-compute-apps=pid,used_gpu_memory --format=csv,noheader 2>/dev/null || true)
  echo "$sum"
}

resume_args() {
  if compgen -G "$WORKDIR/models/lora_adapter/checkpoint-*" > /dev/null 2>&1; then
    echo "--resume"
  else
    echo ""
  fi
}

start_training() {
  local extra
  extra="$(resume_args)"
  log "Starting finetune.py ${extra:-<fresh>} (nice/ionice low priority) ..."
  export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
  # shellcheck disable=SC2086
  nohup nice -n 19 ionice -c 3 -n 7 \
    python scripts/finetune.py $extra >>"$TRAINING_LOG" 2>&1 &
  log "Training started PID=$!"
}

stop_training_graceful() {
  local p
  log "Pausing training: non-training GPU memory above threshold (yielding to image/video workloads)."
  for p in $(training_pids); do
    kill -TERM "$p" 2>/dev/null || true
  done
  # Wait up to 120s for clean exit (checkpoint save may run)
  local waited=0
  while is_training_running && [[ $waited -lt 120 ]]; do
    sleep 2
    waited=$((waited + 2))
  done
  if is_training_running; then
    log "Training still running after SIGTERM; sending SIGKILL."
    for p in $(training_pids); do
      kill -KILL "$p" 2>/dev/null || true
    done
  fi
  STABLE_COUNT=0
}

log "gpu_guard: WORKDIR=$WORKDIR POLL_SEC=$POLL_SEC PAUSE_OTHER_MB=$PAUSE_OTHER_MB RESUME_OTHER_MB=$RESUME_OTHER_MB STABLE_SEC=$STABLE_SEC"

while true; do
  if training_completed && ! is_training_running; then
    log "Training completed (see Adapter saved in $TRAINING_LOG). gpu_guard exiting."
    exit 0
  fi

  other_mb="$(other_gpu_memory_mib || echo 0)"

  if is_training_running; then
    if [[ "${other_mb:-0}" -ge "$PAUSE_OTHER_MB" ]]; then
      log "Non-training GPU ~${other_mb} MiB >= pause ${PAUSE_OTHER_MB} MiB — stopping training."
      stop_training_graceful
    else
      log "Training running; other GPU ~${other_mb} MiB (ok)."
    fi
  else
    if [[ "${other_mb:-0}" -le "$RESUME_OTHER_MB" ]]; then
      STABLE_COUNT=$((STABLE_COUNT + 1))
      log "Idle enough for training: other GPU ~${other_mb} MiB (stable ${STABLE_COUNT}/${STABLE_NEEDS})."
      if [[ "$STABLE_COUNT" -ge "$STABLE_NEEDS" ]]; then
        start_training
        STABLE_COUNT=0
      fi
    else
      if [[ "$STABLE_COUNT" -ne 0 ]]; then
        log "GPU busy for other workloads: other ~${other_mb} MiB > resume ${RESUME_OTHER_MB} MiB — reset stable counter."
      else
        log "Waiting: other GPU ~${other_mb} MiB > resume ${RESUME_OTHER_MB} MiB."
      fi
      STABLE_COUNT=0
    fi
  fi

  sleep "$POLL_SEC"
done
