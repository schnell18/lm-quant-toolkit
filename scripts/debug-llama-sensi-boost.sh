#!/bin/bash

BUDGETS="3.13 3.25 3.51 4.13 4.25 4.51"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"

BOOST_STOP=2
ATTEMPT="sensi-boost-debug"
EXP_BASE_NAME=$ATTEMPT
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}

# Use cached dataset to speedup wikitext, c4 ppl evaluation
export HF_DATASETS_OFFLINE=1

weight_algo=sensi-boost

log_file="logs/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"

mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT

MODELS="0 1 2"
# MODELS="0"
EXP_NAME="${ATTEMPT}"
echo "=========Run perplexity evaluation========="
python -m pdb ../src/cli.py llm \
  --task eval_ppl \
  --model $MODELS \
  --algo mxq \
  --weight-algo $weight_algo \
  --boost-stop $BOOST_STOP \
  --config ${BUDGETS} \
  --experiment-name "${EXP_NAME}_ppl" \
  --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
  --result-dir=$RESULT_DIR \
  2>&1 \
  | tee -a $log_file
EXIT_CODE=$?
if [ $EXIT_CODE -ne 0 ]; then
  echo "Perplexity evaluation failed!"
  exit $EXIT_CODE
fi
