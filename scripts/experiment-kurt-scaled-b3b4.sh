#!/bin/bash

ATTEMPT="kurt-scaled"
RESULT_DIR="/fdata/llm/mxq/results"
MXQ_CONFIGS="3.13 3.25 3.51 4.13 4.25 4.51"
EXP_NAME="${ATTEMPT}-b3b4"
QUANT_SNAPSHOT_DIR="/fdata/llm/mxq/snapshots"


# Setup data files directories for reporting
mkdir -p $RESULT_DIR/$EXP_NAME/data/{ppl,qnt,stor}

python ../src/cli.py llm \
    --task eval_ppl \
    --model 0 1 2 \
    --algo mxq \
    --config $MXQ_CONFIGS \
    --experiment-name "${EXP_NAME}_ppl" \
    --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
    --result-dir=$RESULT_DIR \
    2>&1 \
    | tee logs/bench-$(date +%Y%m%d%H%M%S).log

mkdir -p $RESULT_DIR/$EXP_NAME/data/ppl/mxq/$ATTEMPT
find $RESULT_DIR/${EXP_NAME}_ppl \
    -name "result-*.csv" \
    -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' '  -f2 \
    | xargs -i cp {} $RESULT_DIR/$EXP_NAME/data/ppl/mxq/$ATTEMPT

python ../src/cli.py dump \
    --type quant_config \
    --model 0 1 2 \
    --budget $MXQ_CONFIGS \
    --attempt $ATTEMPT \
    --quant-snapshot-dir=$QUANT_SNAPSHOT_DIR \
    --output_file "$RESULT_DIR/$EXP_NAME/data/mxq-quant-cfgs-${EXP_NAME}.csv"

algo=mxq
model_ids="0 1 2"
for m in $model_ids; do
    for cfg in $MXQ_CONFIGS; do
        python ../src/cli.py llm \
            --model $m \
            --algo ${algo} \
            --config ${cfg} \
            --task eval_model_storage \
            --experiment-name "${EXP_NAME}_stor" \
            --quant-snapshot-dir="$QUANT_SNAPSHOT_DIR/$ATTEMPT" \
            --result-dir=$RESULT_DIR \
            2>&1 \
            | tee logs/bench-$(date +%Y%m%d%H%M%S).log
    done
done

mkdir -p $RESULT_DIR/$EXP_NAME/data/stor/mxq/$ATTEMPT
find $RESULT_DIR/${EXP_NAME}_stor \
    -name "result-*.csv" \
    -printf '%T@ %p\n' \
    | sort -n \
    | tail -1 \
    | cut -d' '  -f2 \
    | xargs -i cp {} $RESULT_DIR/$EXP_NAME/data/stor/mxq/$ATTEMPT

PWD=$(pwd)
cd $RESULT_DIR/$EXP_NAME
if [ ! -d pdfs ]; then
    mkdir pdfs
fi
~/study/aut-study/master-thesis/data-vis/combine.R \
    --baseline_data_dir ~/study/aut-study/master-thesis/data-vis/data \
    --mxq_data_dir data
~/study/aut-study/master-thesis/data-vis/plot-mxq-llm.R data/combined.csv
~/study/aut-study/master-thesis/data-vis/plot-mem-consumption.R data/combined.csv
~/study/aut-study/master-thesis/data-vis/plot-quant-speed.R data/combined.csv
~/study/aut-study/master-thesis/data-vis/gen-table-mxq-llm.R data/combined.csv
pdflatex table.tex
