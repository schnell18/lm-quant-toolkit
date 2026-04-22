#!/bin/bash

MODELS="Qwen/Qwen3.5-2B Qwen/Qwen3.5-4B Qwen/Qwen3.5-9B"

mkdir -p /tmp/fnorm-dump
python ../../src/dump.py fnorm \
    --model $MODELS \
    --output-dir /tmp/fnorm-dump
