#!/bin/bash


# 100%, 99%, 98%, 97%, 96%, 95% of [3.51, 4.25]
# MXQ2=(minibatch 3.51 3.47 3.44 3.40 3.37 3.33 4.25 4.21 4.17 4.12 4.08 4.04)
MXQ2=(minibatch 6.89 5.72 5.02 4.51 4.25 4.21 4.17 4.13 4.11 4.07 3.95 3.87 3.83 3.65 3.51 3.25 3.19 3.15 3.13 3.11 3.07)
MXQ_BATCHES=(MXQ2)
declare -n MXQ_BATCH

ATTEMPT="sensi-milp-mini2"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"
EXP_BASE_NAME="${ATTEMPT}"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"
# Setup data files directories for reporting
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT
mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT

weight_algo=sensi-milp

for MXQ_BATCH in "${MXQ_BATCHES[@]}"; do
  batch_name=${MXQ_BATCH[@]:0:1}
  EXP_NAME="${EXP_BASE_NAME}-${batch_name}"
  mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
  mkdir -p $RESULT_DIR/${EXP_NAME}_stor
  log_file="logs/bench-${EXP_NAME}-$(date +%Y%m%d%H%M%S).log"

  # echo "=========Delete quantized models of batch ${batch_name}========="
  # find $QUANT_SNAPSHOT_DIR/$ATTEMPT -maxdepth 1 -type d | xargs rm -fr

  OLD_DIR=$(pwd)
  cd $RESULT_DIR/$EXP_BASE_NAME
  if [ ! -d pdfs ]; then
      mkdir pdfs
  fi

  # plot configuration allocations for 3 * 12 MXQ combinations
  MODELS="Llama-2-7b-hf Llama-2-13b-hf Meta-Llama-3-8B"
  BUDGETS=${MXQ_BATCH[@]:1}
  for model in $MODELS; do
      for budget in $BUDGETS; do
          $OLD_DIR/../data-vis/plot-mxq-allocation.R \
            -m $model \
            -b $budget \
            --fnorm_data_dir $OLD_DIR/../src/data \
            --attempt1 mxq1 \
            --attempt2 $ATTEMPT \
            --quant_cfg_allot_file data/quant-cfg-allocation.csv
      done
  done
  cd $OLD_DIR
done
