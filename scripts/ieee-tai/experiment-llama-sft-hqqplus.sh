#!/bin/bash
#
# HQQ+ SFT baseline with uniform LoRA allocation (the control for KurtBoost).
#
# The 1-bit and 2-bit backbones are quantized then recovered with the fixed
# hqq_plus allocation (attn r=32, mlp r=8) for the three KurtBoost llama models.
# Trained LoRA checkpoints are saved under SNAPSHOT_DIR.
#
# Runs the `sft` sub-command of src/cli.py.

set -u

# --- configuration ----------------------------------------------------------
RESULT_DIR="${RESULT_DIR:-results}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-snapshots}"
LOG_DIR="${LOG_DIR:-logs}"
MODELS="${MODELS:-0 1 2}"      # indices into the 3 KurtBoost llama models
NBITS="${NBITS:-1 2}"          # backbone bit-widths (include 1-bit)
GROUP_SIZE="${GROUP_SIZE:-8}"

# SFT hyper-parameters (defaults mirror hqq_plus.py)
LR="${LR:-1e-5}"
N_EPOCHS="${N_EPOCHS:-2}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# Use cached dataset to speed up wikitext loading.
export HF_DATASETS_OFFLINE=1

# Resolve repo paths so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$SNAPSHOT_DIR"

log_file="$LOG_DIR/bench-sft-hqqplus-$(date +%Y%m%d%H%M%S).log"
echo "========= HQQ+ SFT baseline (uniform LoRA) ========="
python "$REPO_DIR/src/cli.py" sft \
    --experiment-name "hqq-plus-sft-hqqplus" \
    --algorithm "HQQ+" \
    --model ${MODELS} \
    --nbits ${NBITS} \
    --group-size "$GROUP_SIZE" \
    --lr "$LR" \
    --n-epochs "$N_EPOCHS" \
    --max-tokens "$MAX_TOKENS" \
    --result-dir "$RESULT_DIR" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    2>&1 | tee -a "$log_file"
exit_code=${PIPESTATUS[0]}
if [ "$exit_code" -ne 0 ]; then
    echo "HQQ+ baseline failed (exit ${exit_code})!"
    exit "$exit_code"
fi

echo "========= HQQ+ baseline complete. Results under ${RESULT_DIR} ========="
