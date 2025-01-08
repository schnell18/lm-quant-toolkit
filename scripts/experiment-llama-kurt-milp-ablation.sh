#!/bin/bash

BUDGETS="6.89 5.72 5.02 4.51 4.25 4.21 4.17 4.13 4.11 4.07 3.95 3.87 3.83 3.65 3.51 3.25 3.19 3.15 3.13 3.11 3.07"
RESULT_DIR="/fdata/llm/mxq/results"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"

# Use cached dataset to speedup wikitext, c4 ppl evaluation
export HF_DATASETS_OFFLINE=1
weight_algo=kurt-milp
MODELS="0 1 2"
MODEL_NAMES="Llama-2-7b-hf Llama-2-13b-hf Meta-Llama-3-8B"


# for SensiMiLP ablation test, all topm values are equivalent

ATTEMPT="kurt-milp-abl"
EXP_BASE_NAME=$ATTEMPT
EXP_NAME="${ATTEMPT}"

log_file="logs/bench-${ATTEMPT}-$(date +%Y%m%d%H%M%S).log"

mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,qnt,stor}

mkdir -p $QUANT_SNAPSHOT_DIR/$ATTEMPT
mkdir -p $RESULT_DIR/${EXP_NAME}_ppl
mkdir -p $RESULT_DIR/$EXP_BASE_NAME/data/{ppl,stor}/mxq/$ATTEMPT

echo "=========Run perplexity evaluation========="
python ../src/cli.py llm \
  --task eval_ppl \
  --model $MODELS \
  --algo mxq \
  --weight-algo $weight_algo \
  --ablation \
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
if [ ! -d pdfs ]; then
    mkdir pdfs
fi
$OLD_DIR/../data-vis/combine.R \
    --baseline_data_dir $OLD_DIR/../data-vis/data \
    --mxq_data_dir data
$OLD_DIR/../data-vis/plot-ppl-mem.R -d data/combined.csv
$OLD_DIR/../data-vis/gen-table-mxq-llm.R --csv_file data/combined.csv --attempt $ATTEMPT
cd pdfs
pdflatex table.tex
cd ..

cd $OLD_DIR
