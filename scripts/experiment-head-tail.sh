#!/bin/bash

HEADS=(head-prioritized 1.05 1.10 1.15 1.20)
TAILS=(tail-prioritized 1.05 1.10 1.15 1.20)
# MXQ_BATCHES=(HEADS TAILS)
MXQ_BATCHES=(TAILS)
declare -n MXQ_BATCH

BUDGETS="2.13 2.25 2.51 3.13 3.25 3.51 4.13 4.25 4.51"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"

# --config ${MXQ_BATCH[@]:1}

EXP_BASE_NAME="head-tail-prioritized"
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT
for MXQ_BATCH in "${MXQ_BATCHES[@]}"; do
  weight_algo=${MXQ_BATCH[@]:0:1}
  for factor in ${MXQ_BATCH[@]:1}; do
      ATTEMPT="${weight_algo}_${factor/./_}"
      EXP_NAME="${ATTEMPT}"
      log_file="logs/bench-$(date +%Y%m%d%H%M%S).log"

      mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
      mkdir -p $RESULT_DIR/${EXP_NAME}_ppl

      echo "=========Run perplexity evaluation on batch ${EXP_NAME}========="
      python ../src/cli.py llm \
          --task eval_ppl \
          --model 0 1 2 \
          --algo mxq \
          --weight-algo $weight_algo \
          --factor $factor \
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
      echo "=========Collect perplexity evaluation result on batch ${EXP_NAME}========="
      find $RESULT_DIR/${EXP_NAME}_ppl \
          -name "result-*.csv" \
          -printf '%T@ %p\n' \
          | sort -n \
          | tail -1 \
          | cut -d' '  -f2 \
          | xargs -i cp {} $RESULT_DIR/$EXP_BASE_NAME/data/ppl/mxq/$ATTEMPT

      echo "=========Dump quantization configs on batch ${EXP_NAME}========="
      mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT
      python ../src/cli.py dump \
          --type quant_config \
          --model 0 1 2 \
          --budget ${BUDGETS} \
          --attempt $ATTEMPT \
          --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
          --output_file "$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/quant-allot-${EXP_NAME}.csv" \
          2>&1 \
          | tee -a $log_file

      echo "=========Run memory evaluation on batch ${EXP_NAME}========="
      algo=mxq
      model_ids="0 1 2"
      for m in $model_ids; do
          for cfg in ${BUDGETS}; do
              python ../src/cli.py llm \
                  --model $m \
                  --algo ${algo} \
                  --config ${cfg} \
                  --task eval_model_storage \
                  --experiment-name "${EXP_NAME}_stor" \
                  --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
                  --result-dir=$RESULT_DIR \
                  2>&1 \
                  | tee -a $log_file
          done
      done
      echo "=========Collect memory evaluation result on batch ${EXP_NAME}========="
      find $RESULT_DIR/${EXP_NAME}_stor \
          -name "result-*.csv" \
          -printf '%T@ %p\n' \
          | sort -n \
          | tail -1 \
          | cut -d' '  -f2 \
          | xargs -i cp {} $RESULT_DIR/$EXP_BASE_NAME/data/stor/mxq/$ATTEMPT
  done


  # echo "=========Delete quantized models of batch ${batch_name}========="
  # find $QUANT_SNAPSHOT_DIR/$ATTEMPT -maxdepth 1 -type d | xargs rm -fr
done

OLD_DIR=$(pwd)
cd $RESULT_DIR/$EXP_BASE_NAME
if [ ! -d pdfs ]; then
    mkdir pdfs
fi
$OLD_DIR/../data-vis/combine.R \
    --baseline_data_dir $OLD_DIR/../data-vis/data \
    --mxq_data_dir data
$OLD_DIR/../data-vis/plot-mxq-paired.R data/combined.csv
$OLD_DIR/../data-vis/plot-mem-consumption.R data/combined.csv
$OLD_DIR/../data-vis/plot-quant-speed.R data/combined.csv
$OLD_DIR/../data-vis/gen-table-mxq-llm.R data/combined.csv
pdflatex table.tex

# plot configuration allocations for 3 * 12 MXQ combinations
MODELS="Llama-2-7b-hf Llama-2-13b-hf Meta-Llama-3-8B"
for model in $MODELS; do
    for budget in $BUDGETS; do
        $OLD_DIR/../data-vis/plot-mxq-allocation.R -m $model -b $budget \
          --fnorm_data_dir $OLD_DIR/../src/data \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv
    done
done
