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
EXPERIMENT_NAME="baseline-fp16"
BASE_DIR="/fdata/llm/ieee-tai/hqqplus/$EXPERIMENT_NAME"
RESULT_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"
SNAPSHOT_DIR="$BASE_DIR/snapshot"

# MODELS="${MODELS:-0 1 2}"   # indices into the 3 KurtBoost llama models
MODELS="${MODELS:-0 2}"

# Use cached dataset to speed up wikitext loading.
export HF_DATASETS_OFFLINE=1

# Resolve repo paths so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

mkdir -p $LOG_DIR
mkdir -p $RESULT_DIR
mkdir -p $SNAPSHOT_DIR

log_file="$LOG_DIR/$EXPERIMENT_NAME-$(date +%Y%m%d%H%M%S).log"
echo "========= FP16 baseline ========="
python "$REPO_DIR/src/cli.py" sft \
    --experiment-name $EXPERIMENT_NAME \
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
