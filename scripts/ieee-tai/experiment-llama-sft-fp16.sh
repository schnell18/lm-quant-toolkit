#!/bin/bash
#
# FP16 upper-bound baseline for the HQQ+ SFT benchmark.
#
# The un-quantized model is evaluated as-is: quantization and SFT are skipped
# altogether, so this is independent of the bit-width / boost knobs and is run
# once per model.
#
# Runs the `sft` sub-command of src/cli.py.

set -u

# --- configuration ----------------------------------------------------------
RESULT_DIR="${RESULT_DIR:-results}"
SNAPSHOT_DIR="${SNAPSHOT_DIR:-snapshots}"
LOG_DIR="${LOG_DIR:-logs}"
MODELS="${MODELS:-0 1 2}"   # indices into the 3 KurtBoost llama models

# Use cached dataset to speed up wikitext loading.
export HF_DATASETS_OFFLINE=1

# Resolve repo paths so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

mkdir -p "$LOG_DIR" "$SNAPSHOT_DIR"

log_file="$LOG_DIR/bench-sft-fp16-$(date +%Y%m%d%H%M%S).log"
echo "========= FP16 baseline ========="
python "$REPO_DIR/src/cli.py" sft \
    --experiment-name "hqq-plus-sft-fp16" \
    --algorithm fp16 \
    --model ${MODELS} \
    --result-dir "$RESULT_DIR" \
    --snapshot-dir "$SNAPSHOT_DIR" \
    2>&1 | tee -a "$log_file"
exit_code=${PIPESTATUS[0]}
if [ "$exit_code" -ne 0 ]; then
    echo "FP16 baseline failed (exit ${exit_code})!"
    exit "$exit_code"
fi

echo "========= FP16 baseline complete. Results under ${RESULT_DIR} ========="
