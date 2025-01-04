#!/bin/bash

# BUDGETS="2.13 2.25 2.51 3.13 3.25 3.51 4.13 4.25 4.51"
BUDGETS="4.13 4.25 4.51"
RESULT_DIR="/fdata/llm/mxq/results"

TAILS=(tail-prioritized 2.00 4.00)
# MXQ_BATCHES=(HEADS TAILS)
MXQ_BATCHES=(TAILS)
declare -n MXQ_BATCH

EXP_BASE_NAME="tail-factor-search"
mkdir -p $RESULT_DIR/$EXP_BASE_NAME

for MXQ_BATCH in "${MXQ_BATCHES[@]}"; do
  weight_algo=${MXQ_BATCH[@]:0:1}
  for factor in ${MXQ_BATCH[@]:1}; do
      ATTEMPT="${weight_algo}_${factor/./_}"
      EXP_NAME="${ATTEMPT}"
      log_file="logs/bench-$(date +%Y%m%d%H%M%S).log"

      mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
      mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT

      # python -m pdb ../src/cli.py dump \
      python ../src/cli.py dump \
        --type quant_config_sim \
        --model 0 1 2 \
        --weight-algo $weight_algo \
        --factor $factor \
        --budget ${BUDGETS} \
        --output-file="$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/${EXP_NAME}.csv"\
        2>&1 \
        | tee -a $log_file

  done

done

OLD_DIR=$(pwd)
cd $RESULT_DIR/$EXP_BASE_NAME
if [ ! -d pdfs/allot ]; then
    mkdir -p pdfs/allot
fi
$OLD_DIR/../data-vis/combine.R \
    --baseline_data_dir $OLD_DIR/../data-vis/data \
    --mxq_data_dir data
# plot configuration allocations for 3 * 12 MXQ combinations
MODELS="Llama-2-7b-hf Llama-2-13b-hf Meta-Llama-3-8B"
for model in $MODELS; do
    for budget in $BUDGETS; do
        $OLD_DIR/../data-vis/plot-mxq-allocation.R \
          -m $model \
          -b $budget \
          --attempt1 tail-prioritized_2_00 \
          --attempt2 tail-prioritized_4_00 \
          --fnorm_data_dir $OLD_DIR/../src/data \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv
    done
done
