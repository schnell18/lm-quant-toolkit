#!/bin/bash

# check if python is installed
which python > /dev/null
if [[ $? -ne 0 ]]; then
    echo "python is not installed!"
    exit 1
fi

OLD_DIR=$(pwd)

# install the patched dependencies from source
# clone patched AutoGPTQ from https://github.com/schnell18/AutoGPTQ.git
# git clone https://github.com/schnell18/AutoGPTQ.git .deps/AutoGPTQ
# cd .deps/AutoGPTQ
# uv pip install -e .
# cd $OLD_DIR


# clone patched hqq from https://github.com/schnell18/hqq.git
if [ ! -d .deps/hqq ]; then
    git clone https://github.com/schnell18/hqq.git .deps/hqq
fi
cd .deps/hqq
uv pip install -e .
cd $OLD_DIR

# clone patched lm-evaluation-harness from https://github.com/schnell18/lm-evaluation-harness.git
if [ ! -d .deps/lm-evaluation-harness ]; then
    git clone https://github.com/schnell18/lm-evaluation-harness.git .deps/lm-evaluation-harness
fi
cd .deps/lm-evaluation-harness
uv pip install -e .
cd $OLD_DIR

# clone patched lm-evaluation-harness from https://github.com/schnell18/CLIP_benchmark.git
if [ ! -d .deps/CLIP_benchmark ]; then
    git clone https://github.com/schnell18/CLIP_benchmark.git .deps/CLIP_benchmark
fi
cd .deps/CLIP_benchmark
git pull
uv pip install -e .
cd $OLD_DIR


# perform an editable install of the lm-quant-toolkit project
# to install all dependencies that are published on PyPI
uv pip install -e .

