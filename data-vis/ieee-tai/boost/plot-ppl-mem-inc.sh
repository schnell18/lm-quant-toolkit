#!/bin/bash
#
TOOLKIT_DIR="../../.."

$TOOLKIT_DIR/data-vis/plot-ppl-mem-inc.R \
  --type sensi-vs-ablation \
  --combined_csv_file data/combined.csv \
  --allot_csv_file data/quant-cfg-allocation.csv

$TOOLKIT_DIR/data-vis/plot-ppl-mem-inc.R \
  --type kurt-vs-ablation \
  --combined_csv_file data/combined.csv \
  --allot_csv_file data/quant-cfg-allocation.csv

$TOOLKIT_DIR/data-vis/plot-ppl-mem-inc.R \
  --type sensi-vs-kurt \
  --combined_csv_file data/combined.csv \
  --allot_csv_file data/quant-cfg-allocation.csv
