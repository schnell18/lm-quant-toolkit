#!/bin/bash

BUDGETS="4.25"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"

FACTOR=2
ATTEMPT="sensi-milp-debug-lowbits9"
EXP_BASE_NAME=$ATTEMPT
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}

# Use cached dataset to speedup wikitext, c4 ppl evaluation
export HF_DATASETS_OFFLINE=1

weight_algo=sensi-milp

log_file="logs/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"

# MODELS="0 1 2"
MODELS="0"
EXP_NAME="${ATTEMPT}"

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
  --model $MODELS \
  --budget ${BUDGETS} \
  --attempt $ATTEMPT \
  --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
  --output-file "$RESULT_DIR/$EXP_BASE_NAME/data/allot/mxq/$ATTEMPT/quant-allot-${EXP_NAME}.csv" \
  2>&1 \
  | tee -a $log_file

echo "=========Run memory evaluation on batch ${EXP_NAME}========="
algo=mxq
model_ids=$MODELS
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

# echo "=========Delete quantized models of batch ${batch_name}========="
# find $QUANT_SNAPSHOT_DIR/$ATTEMPT -maxdepth 1 -type d | xargs rm -fr

OLD_DIR=$(pwd)
cd $RESULT_DIR/$EXP_BASE_NAME
if [ ! -d pdfs/allot ]; then
    mkdir -p pdfs/allot
fi
$OLD_DIR/../data-vis/combine.R \
    --baseline_data_dir $OLD_DIR/../data-vis/data \
    --mxq_data_dir data
$OLD_DIR/../data-vis/plot-mxq-paired.R data/combined.csv
$OLD_DIR/../data-vis/plot-mem-consumption.R data/combined.csv
$OLD_DIR/../data-vis/plot-quant-speed.R data/combined.csv
$OLD_DIR/../data-vis/gen-table-mxq-llm.R --csv_file data/combined.csv --attempt $ATTEMPT
pdflatex table.tex

# plot configuration allocations for 3 * 12 MXQ combinations
MODELS="Llama-2-7b-hf"
for model in $MODELS; do
    for budget in $BUDGETS; do
        $OLD_DIR/../data-vis/plot-mxq-allocation.R \
          -m $model \
          -b $budget \
          --attempt1 $ATTEMPT \
          --attempt2 mxq1 \
          --baseline_data_dir $OLD_DIR/../src/data \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv
    done
done