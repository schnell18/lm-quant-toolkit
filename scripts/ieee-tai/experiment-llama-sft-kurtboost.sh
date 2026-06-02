#!/bin/bash
#
# HQQ+ SFT with KurtBoost-guided LoRA allocation (the proposed method).
#
# Sweeps the LoRA boost_stop (rungs climbed up the capacity ladder by sensitive
# layers) and top_m (sensitive layers per module), over 1-bit and 2-bit HQQ
# backbones, for the three KurtBoost llama models. Trained LoRA checkpoints are
# saved under SNAPSHOT_DIR.
#
# The fp16 and HQQ+ baselines live in separate scripts:
#   experiment-llama-sft-fp16.sh
#   experiment-llama-sft-hqqplus.sh
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

BOOST_STOPS="${BOOST_STOPS:-1 2 3}"
BOOST_TOP_MS="${BOOST_TOP_MS:-1}"

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

for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        EXP_NAME="hqq-plus-sft-kurtboost-${BOOST_STOP}-${BOOST_TOP_M}"
        log_file="$LOG_DIR/bench-sft-${EXP_NAME}-$(date +%Y%m%d%H%M%S).log"
        echo "========= ${EXP_NAME} ========="
        python "$REPO_DIR/src/cli.py" sft \
            --experiment-name "$EXP_NAME" \
            --algorithm kurtboost \
            --model ${MODELS} \
            --nbits ${NBITS} \
            --group-size "$GROUP_SIZE" \
            --boost-stop "$BOOST_STOP" \
            --top-m-layer "$BOOST_TOP_M" \
            --lr "$LR" \
            --n-epochs "$N_EPOCHS" \
            --max-tokens "$MAX_TOKENS" \
            --result-dir "$RESULT_DIR" \
            --snapshot-dir "$SNAPSHOT_DIR" \
            2>&1 | tee -a "$log_file"
        exit_code=${PIPESTATUS[0]}
        if [ "$exit_code" -ne 0 ]; then
            echo "Run ${EXP_NAME} failed (exit ${exit_code})!"
            exit "$exit_code"
        fi
    done
done

echo "========= All KurtBoost runs complete. Results under ${RESULT_DIR} ========="
