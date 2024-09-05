# lm-quant-toolkit

A library and toolkit for Large language Model quantization research.

Large Language Models (LLMs) have demonstrated significant capabilities in
intelligent activities such as natural language comprehension, content
generation, and knowledge retrieval. However, training and deploying these
models require substantial computation resources, setting up a significant
barrier for developing AI applications and conducting research. Various model
compression techniques have been developed to address the demanding
computational resource issue. Nonetheless, there has been limited exploration
into high-level quantization strategy orthogonal to existing methods to offer
better flexibility of balancing the trade-off between memory usage and
accuracy.

## Requirements

* Python 3.7 over
* CUDA 12.x
* AutoAWQ
* AutoGPTQ
* HQQ

## Features

* Oraginze complex quantization experiments
* Collect experiments process data
* Run perplexity, [Open LLM LeaderBoard benchmarks][1]
* Create fnorm dataset
* Extract data from original/quantized models

## Setup

```bash
conda create -n lm-quant-toolkit
conda activate lm-quant-toolkit
pip install -e .
```

## Usage

### Run quantization harness

The `bench` tool is designed to run long running LLM quantization and evaluation
experiments. It collects the perplexity, Open LLM LeaderBoard scores durations
and GPU memory consumptions under various experiment settings. These experiment
data are recorded in .csv format to ease further analysis and reporting.

Run the following comand to
quantize LLMs:

```bash
python src/lm_quant_toolkit/eval/bench.py
```


### Prepare FNorm dataset for MXQ quantization

The MXQ quantization method discovers optimal quantization configuration such as
group size, bit width for large number of weight matricies so that it can
produce quantized model with minimal accuracy degradation while confines to
fixed memory budget. All possible meta data about quantization accuracy,
measured in Frobenius Norm, under given bit width, group size should be
collected in advance. The `fnorm` tool is designed to serve this purpose.
Run the following comand to produce fnorm dataset:

```bash
python src/lm_quant_toolkit/prep/fnorm.py
```

This program supports the following Llama models:

- meta-llama/Llama-2-7b-hf
- meta-llama/Llama-2-13b-hf
- meta-llama/Meta-Llama-3-8B
- meta-llama/Meta-Llama-3-70B
- meta-llama/Llama-2-70b-hf
- meta-llama/Meta-Llama-3-70B-Instruct
- meta-llama/Meta-Llama-3.1-405B-Instruct


[1]: https://huggingface.co/docs/leaderboards/leaderboards/intro
