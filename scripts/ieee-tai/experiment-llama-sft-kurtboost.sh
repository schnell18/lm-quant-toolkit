#!/bin/bash
#
# HQQ+ SFT benchmark with KurtBoost-guided LoRA allocation.
#
# Sweeps the LoRA boost_stop (rungs climbed up the capacity ladder by sensitive
# layers) and top_m (sensitive layers per module), over 1-bit and 2-bit HQQ
# backbones, for the three KurtBoost llama models. The uniform-LoRA control does
# not depend on boost_stop/top_m, so it is run once up front.
#
# Runs the standalone harness src/lm_quant_toolkit/eval/bench_sft.py (not cli.py).

set -u

# --- configuration ----------------------------------------------------------
#

BASE_DIR="/fdata/llm/ieee-tai/hqqplus"
RESULT_DIR="$BASE_DIR/results"
LOG_DIR="$BASE_DIR/logs"

NBITS="${NBITS:-1 2}"          # backbone bit-widths (include 1-bit)
GROUP_SIZE="${GROUP_SIZE:-8}"

BOOST_STOPS="${BOOST_STOPS:-1 2}"
BOOST_TOP_MS="${BOOST_TOP_MS:-1}"

# SFT hyper-parameters (defaults mirror hqq_plus.py)
LR="${LR:-1e-5}"
N_EPOCHS="${N_EPOCHS:-2}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

# Use cached dataset to speed up wikitext loading.
export HF_DATASETS_OFFLINE=1

# Resolve repo paths so the script works from any cwd.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
export PYTHONPATH="$REPO_DIR/src:${PYTHONPATH:-}"

mkdir -p "$RESULT_DIR"
mkdir -p "$LOG_DIR"

run_bench() {
    # args: experiment_name variant [extra args...]
    local exp_name="$1"; shift
    local variant="$1"; shift
    local log_file="$LOG_DIR/bench-sft-${exp_name}-$(date +%Y%m%d%H%M%S).log"
    echo "========= ${exp_name} (variant=${variant}) ========="
    python "$REPO_DIR/src/cli.py" sft \
        --experiment-name "$exp_name" \
        --variant "$variant" \
        --nbits ${NBITS} \
        --group-size "$GROUP_SIZE" \
        --lr "$LR" \
        --n-epochs "$N_EPOCHS" \
        --max-tokens "$MAX_TOKENS" \
        --result-dir "$RESULT_DIR/$exp_name" \
        "$@" \
        2>&1 | tee -a "$log_file"
    local exit_code=${PIPESTATUS[0]}
    if [ "$exit_code" -ne 0 ]; then
        echo "Run ${exp_name} failed (exit ${exit_code})!"
        exit "$exit_code"
    fi
}

# --- uniform control (boost_stop-independent) -------------------------------
run_bench "hqq-plus-sft-uniform" uniform

# --- KurtBoost sweep --------------------------------------------------------
for BOOST_STOP in $BOOST_STOPS; do
    for BOOST_TOP_M in $BOOST_TOP_MS; do
        EXP_NAME="kurtboost-${BOOST_STOP}-${BOOST_TOP_M}"
        run_bench "$EXP_NAME" kurtboost \
            --boost-stop "$BOOST_STOP" \
            --top-m-layer "$BOOST_TOP_M"
    done
done

echo "========= All HQQ+ SFT runs complete. Results under ${RESULT_DIR} ========="
