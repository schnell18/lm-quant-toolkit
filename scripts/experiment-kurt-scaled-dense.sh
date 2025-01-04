#!/bin/bash

# MXQ1=(2bit 2.13 2.15 2.17 2.19 2.21 2.23 2.25 2.27 2.29 2.31 2.45 2.47 2.49 2.51 2.53 2.55 2.57)
# MXQ2=(3bit 3.07 3.09 3.11 3.13 3.15 3.17 3.19 3.21 3.23 3.25 3.27 3.29 3.31 3.33 3.35 3.37 3.39 3.41 3.42 3.43 3.45 3.47 3.49 3.51 3.53 3.55 3.57 3.59 3.61 3.63 3.65 3.67 3.69 3.71 3.73 3.75 3.77 3.79 3.81 3.83 3.85 3.87 3.89 3.91 3.93 3.95 3.97 3.99)
# Removed 4.99 which is unsolvable with kurt-scaled scheme
MXQ4=(4bit 4.01 4.03 4.05 4.07 4.09 4.11 4.13 4.15 4.17 4.19 4.21 4.23 4.25 4.27 4.29 4.31 4.33 4.35 4.37 4.39 4.41 4.43 4.45 4.47 4.49 4.51 4.53 4.55 4.57 4.59 4.61 4.63 4.65 4.67 4.69 4.71 4.73 4.75 4.77 4.79 4.81 4.83 4.85 4.87 4.89 4.91 4.93 4.95 4.97)
# Remove 5.01 5.02, 5.04 5.10 5.32 which is unsolvable with kurt-scaled scheme
#MXQ5=(5bit 5.00 5.03 5.06 5.08 5.09 5.12 5.14 5.16 5.18 5.20 5.22 5.24 5.26 5.28 5.30 5.33 5.34 5.36 5.38 5.40 5.42 5.44 5.46 5.48 5.50 5.52 5.54 5.56 5.58 5.60 5.62 5.64 5.66 5.68 5.70 5.72 5.74 5.76 5.78 5.80 5.82 5.84 5.86 5.88 5.90 5.92 5.94 5.96 5.98)
#MXQ6=(6bit 6.00 6.02 6.03 6.04 6.05 6.06 6.07 6.09 6.11 6.13 6.15 6.17 6.19 6.21 6.23 6.25 6.27 6.29 6.31 6.33 6.35 6.37 6.39 6.41 6.43 6.45 6.47 6.49 6.51 6.53 6.55 6.57 6.59 6.61 6.63 6.65 6.68 6.69 6.71 6.72 6.75 6.77 6.78 6.81 6.83 6.86 6.87 6.89 6.92 6.95 6.96 6.98)
# MXQ7=(78bit 7.01 7.02 7.04 7.06 7.08 7.10 7.13 7.14 7.16 7.19 7.20 7.22 7.24 7.26 7.29 7.30 7.32 7.34 7.36 7.38 7.41 7.42 7.44 7.47 7.48 7.50 7.53 7.54 7.56 7.57 7.60 7.62 7.63 7.66 7.68 7.70 7.72 7.74 7.76 8.13 8.25 8.51)
MXQ_BATCHES=(MXQ4)
declare -n MXQ_BATCH

ATTEMPT="kurt-scaled"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"
EXP_BASE_NAME="${ATTEMPT}-dense"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"
# Setup data files directories for reporting
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT
mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT


for MXQ_BATCH in "${MXQ_BATCHES[@]}"; do
  batch_name=${MXQ_BATCH[@]:0:1}
  EXP_NAME="${EXP_BASE_NAME}-${batch_name}"
  log_file="logs/bench-${batch_name}-$(date +%Y%m%d%H%M%S).log"

  echo "=========Run perplexity evaluation on batch ${batch_name}========="
  python ../src/cli.py llm \
      --task eval_ppl \
      --model 0 1 2 \
      --algo mxq \
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
      --output_file "$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/quant-allot-${EXP_NAME}.csv" \
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
BUDGETS="2.13 2.25 2.51 3.13 3.25 3.51 4.13 4.25 4.51"
for model in $MODELS; do
    for budget in $BUDGETS; do
        $OLD_DIR/../data-vis/plot-mxq-allocation.R -m $model -b $budget \
          --fnorm_data_dir $OLD_DIR/../src/data \
          --attempt1 mxq1 \
          --attempt2 $ATTEMPT \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv
    done
done
