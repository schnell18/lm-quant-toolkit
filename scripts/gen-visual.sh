#!/bin/bash

OLD_DIR=$(pwd)
cd ../data-vis/
if [ ! -d pdfs/latex ]; then
    mkdir -p pdfs/latex
fi
if [ ! -d pdfs/allot ]; then
    mkdir -p pdfs/allot
fi

./plot-mxq-paired.R data/combined.csv
./plot-mem-consumption.R data/combined.csv
./plot-quant-speed.R data/combined.csv

# plot configuration allocations for 3 * 12 MXQ combinations
MODELS="Llama-2-7b-hf Llama-2-13b-hf Meta-Llama-3-8B"
BUDGETS="2.13 2.25 2.51 3.13 3.25 3.51 4.13 4.25 4.51"
for model in $MODELS; do
    for budget in $BUDGETS; do
        ./plot-mxq-allocation.R -m $model -b $budget \
          --baseline_data_dir data \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv \
          --output_dir pdfs/allot
    done
done


# plot FNorm vs Kurtosis
MODELS="Llama-2-7b-hf Llama-2-13b-hf Meta-Llama-3-8B"
for model in $MODELS; do
    ./plot-fnorm-kurt.R -m $model
done

cd pdfs/latex
../../gen-table-mxq-llm.R ../../data/combined.csv
pdflatex table.tex
cd ../..

