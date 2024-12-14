#!/bin/bash

# check if python is installed
which python > /dev/null
if [[ $? -ne 0 ]]; then
    echo "python is not installed!"
    exit 1
fi

# check if miniconda is installed
if [[ ! -d ~/miniconda3 ]]; then
    echo "miniconda is not installed!"
    exit 2
fi

# check the current activated env
if [[ $CONDA_DEFAULT_ENV == "base" ]]; then
    echo "Please switch to an environment other than base!"
    exit 3
fi

OLD_DIR=$(pwd)

# perform an editable install of the lm-quant-toolkit project
# to install all dependencies that are published on PyPI
pip install -e .

# install the patched dependencies from source
# clone patched AutoGPTQ from https://github.com/schnell18/AutoGPTQ.git
git clone https://github.com/schnell18/AutoGPTQ.git .deps/AutoGPTQ
cd .deps/AutoGPTQ
pip install -e .
cd $OLD_DIR


# clone patched hqq from https://github.com/schnell18/hqq.git
git clone https://github.com/schnell18/hqq.git .deps/hqq
cd .deps/hqq
pip install -e .
cd $OLD_DIR

# clone patched lm-evaluation-harness from https://github.com/schnell18/lm-evaluation-harness.git
git clone https://github.com/schnell18/lm-evaluation-harness.git .deps/lm-evaluation-harness
cd .deps/lm-evaluation-harness
pip install -e .
cd $OLD_DIR

# clone patched lm-evaluation-harness from https://github.com/schnell18/CLIP_benchmark.git
git clone https://github.com/schnell18/CLIP_benchmark.git .deps/CLIP_benchmark
cd .deps/CLIP_benchmark
pip install -e .
cd $OLD_DIR
