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

  echo "=========Run perplexity evaluation on batch ${batch_name}========="
  python ../src/cli.py llm \
      --task eval_ppl \
      --model 0 1 2 \
      --algo mxq \
      --weight-algo ${weight_algo} \
      --config ${MXQ_BATCH[@]:1} \
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
  echo "=========Collect perplexity evaluation result on batch ${batch_name}========="
  find $RESULT_DIR/${EXP_NAME}_ppl \
      -name "result-*.csv" \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' '  -f2 \
      | xargs -i cp {} $RESULT_DIR/$EXP_BASE_NAME/data/ppl/mxq/$ATTEMPT

  echo "=========Dump quantization configs on batch ${batch_name}========="
  mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT
  python ../src/cli.py dump \
      --type quant_config \
      --model 0 1 2 \
      --budget ${MXQ_BATCH[@]:1} \
      --attempt $ATTEMPT \
      --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
      --output-file "$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/quant-allot-${EXP_NAME}.csv" \
      2>&1 \
      | tee -a $log_file

  echo "=========Run memory evaluation on batch ${batch_name}========="
  algo=mxq
  model_ids="0 1 2"
  for m in $model_ids; do
      for cfg in "${MXQ_BATCH[@]:1}"; do
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
  echo "=========Collect memory evaluation result on batch ${batch_name}========="
  find $RESULT_DIR/${EXP_NAME}_stor \
      -name "result-*.csv" \
      -printf '%T@ %p\n' \
      | sort -n \
      | tail -1 \
      | cut -d' '  -f2 \
      | xargs -i cp {} $RESULT_DIR/$EXP_BASE_NAME/data/stor/mxq/$ATTEMPT


  # echo "=========Delete quantized models of batch ${batch_name}========="
  # find $QUANT_SNAPSHOT_DIR/$ATTEMPT -maxdepth 1 -type d | xargs rm -fr

  OLD_DIR=$(pwd)
  cd $RESULT_DIR/$EXP_BASE_NAME
  if [ ! -d pdfs ]; then
      mkdir pdfs
  fi
  $OLD_DIR/../data-vis/combine.R \
      --baseline_data_dir $OLD_DIR/../data-vis/data \
      --mxq_data_dir data
  $OLD_DIR/../data-vis/plot-ppl-mem.R -d data/combined.csv
  $OLD_DIR/../data-vis/plot-mem-consumption.R -d data/combined.csv
  $OLD_DIR/../data-vis/plot-quant-speed.R -d data/combined.csv
  $OLD_DIR/../data-vis/gen-table-mxq-llm.R --csv_file data/combined.csv --attempt $ATTEMPT
  cd pdfs
  pdflatex table.tex
  cd ..

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
