#!/bin/bash

split -a 1 -n 5 data-Llama-2-7b-chat-hf-vLLM.csv data-
cat data-a data-b data-c data-d > train.csv
head -1 data-Llama-2-7b-chat-hf-vLLM.csv > test.csv
cat data-e >> test.csv
rm -f data-a data-b data-c data-d data-e
