#!/bin/bash

BUDGETS="4.13 4.25 4.51"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"

ATTEMPT="debug-sensi-cfg-wise-7b5"
EXP_BASE_NAME=$ATTEMPT
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}

# Use cached dataset to speedup wikitext, c4 ppl evaluation
export HF_DATASETS_OFFLINE=1

weight_algo=sensi-milp

log_file="logs/bench-$(date +%Y%m%d%H%M%S).log"

mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT

MODELS="0"
EXP_NAME="${ATTEMPT}"
echo "=========Run perplexity evaluation on Llama-2-7b========="
python -m pdb ../src/cli.py llm \
  --task eval_ppl \
  --model $MODELS \
  --algo mxq \
  --weight-algo $weight_algo \
  --config ${BUDGETS} \
  --experiment-name "${EXP_NAME}_ppl" \
  --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
  --result-dir=$RESULT_DIR \
  2>&1 \
  | tee -a $log_file
