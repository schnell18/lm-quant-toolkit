# Overview

The **lm-quant-toolkit** is a suite of tools to facilitate large neural network
quantization research. It includes a quantization harness tool to drive
quantization experiments on large language models and vision models, to collect
and summarize experiment data for further analysis. It also includes tool to
prepare experiment meta data and visualization tools to interpret experiment
results. Specifically, lm-quant-toolkit consists of:

- LLM quantization harness tool
- ViT quantization harness tool
- FNorm Metadata Preparation Tool
- Kurtosis Metrics Measuring Tool
- Sensitivity Score Measuring Tool
- Calibration Dataset Generation Tool
- Visualization Tools

## Citation

~~~~
@inproceedings{zhang2025mxq,
  title = {A Mixed Quantization Approach for Data-Free Quantization of LLMs},
  author = {Feng Zhang and Yanbin Liu and Weihua Li and Xiaodan Wang and Quan Bai},
  year = {2025},
  url = {https://openreview.net/forum?id=M3Y74vmsMcY},
}
~~~~

## Setup test harness

Most tools are implemented in Python and are extensively tested under the
Python 3.11.9. The visualization tools are implemented in R. The usages of
these tools are elaborated in the following sections. This section describes
how to setup the lm-quant-toolkit and the companion visualization tools.

The Python tools dependend on Python libraries such as transformers, datasets,
numpy, PyTorch etc. A few Python libraries are patched to support MXQ.
Specifically, required patched dependencies include AutoGPTQ (for CUDA 12.5
compatibility), HQQ (support MXQ extension), lm_eval (for end-to-end LLM
performance evaluation), clip_benchmark (for vision model evaluation). These
dependencies are installed automatically as part of setup process. To setup the
Python tools, follow this procedure:

- Ensure Python and miniconda are installed
- Create a Python virtual enivonrment using Python 3.11.9 and activate this enivonrment
- Clone the lm-quant-toolkit project from [the lm-quant-toolkit project][2]
- Run the script setup-harness.sh under the root directory of the lm-quant-toolkit project

Or simply use the convenient script `setup-harness.sh` included this project.

## Setup visualization tools

The visualization tools are R scripts to transform, aggregate and visualize
experiment results. They are wrapped in bash scripts to automate the whole
experiment loop, which consists of model quantization, perplex evaluation,
memory consumption test and experiment report generation. The R visualization
scripts can also be used separately. To setup the visualization tools, please
follow this procedure:

- Ensure a recent version of R, for instance R 4.4.1, is installed.
- Optionally, RStudio could be installed to extend and trouble shoot the
visualization tools in an intuitive enivonrment.
- Install the third-party packages required by the visualization tools by
running the script `setup-visualization.sh` under the root directory of the
lm-quant-toolkit project.

# Quantization tool usage

## LLM Quantization Harness Tool

This tool executes various quantization tasks and runs diverse evaluation
benchmarks such as perplexity, GPU memory usage, quantized model storage. It
also supports end-to-end LLM performance evaluation through the integration
with the `lm-eval` tool. This harness tool works with various state-of-art
quantization methods such as GPTQ, AWQ, BitsAndBytes and HQQ, which enables a
fair comparison between the proposed methods and the state-of-art baselines.
Furthermore, it facilitates the complex and time-consuming benchmarking tasks
by offering resumption from failed subtasks, aggregate subtask's evaluation
results. Lastly, this tool provides declarative CLI interface to ease complex
experiment automation through shell scripting.

## FNorm Metadata Preparation Tool

This tool calculates the Frobenius norms, a.k.a FNorm, of the quantization
errors of all weight matricies inside a particular large language model. The
FNorm meta-data are crucial to the MXQ quantization scheme as it guides MXQ to
allocate optimal quantization configurations. This tool accepts a list of
Hugging Face-compliant model identifiers. The output of this tool is a series
of .csv files under specified directory. Each file contains the Frobenius
norms for the 12 quantization configurations.

The tool is implemented in Python and provides a convenient CLI interface to
enable shell scripting. It is located separately in the `dump.py` file
under the `src` folder in the `lm-quant-toolkit` project, which helps
to reduce unnecessary dependencies. A typical usage is demonstrated in the code
snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"
MODELS="meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Llama-3.1-8B"
mkdir -p /tmp/fnorm-dump
python $TOOLKIT_DIR/src/dump.py fnorm \
    --model $MODELS \
    --output-dir /tmp/fnorm-dump
~~~~


## Kurtosis Metrics Measuring Tool

This tool calculates the Kurtosis metrics of weight matricies layer-by-layer
inside a particular large language model. The Kurtosis metrcis are crucial to
identify sensitive layers to improve the accuracy of MXQ quantization. This
tool accepts a list of Hugging Face-compliant model identifiers. The output of
this tool is a series of .csv files under specified directory. Each file
contains the Kurtosis metrics for corresponding models.

The tool is implemented in Python and provides a convenient CLI interface to
enable shell scripting. It is included in the `dump.py` file under the
`src` folder in the `lm-quant-toolkit` project. A typical usage is
demonstrated in the code snippet as follows:

~~~~bash
#!/bin/bash

MODELS="meta-llama/Llama-2-7b-hf meta-llama/Llama-2-13b-hf meta-llama/Meta-Llama-3-8B"
mkdir -p /tmp/kurtosis-dump
python ../src/dump.py kurtosis \
    --model $MODELS \
    --output-dir /tmp/kurtosis-dump
~~~~

This code snippet demonstrates dumping the kurtosis metrics for the three Llama
models into the `/tmp/kurtosis-dump` directory.

## Sensitivity Score Measuring Tool

This tool calculates the sensitivity score of each layer of a particular large
language model. The sensitivity score are crucial to identify sensitive layers
to improve the accuracy of MXQ quantization. This tool accepts a list of
Hugging Face-compliant model identifiers. The output of this tool is a series
of .csv files, each contains the sensitivity score for corresponding model.
These files are crucial inputs to guide the SensiBoost and Sensitivity-based
MiLP.

The tool is implemented in Python and provides a convenient CLI interface to
enable shell scripting. It is compatible with any transformer-based LLMs with
an implementation of the popular Hugging Face transformers library. It is
located separately in the `dump.py` file under the `src` folder in
the `lm-quant-toolkit` project, which helps to reduce unnecessary
dependencies. A typical usage is demonstrated in the code snippet as follows:


~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"
RESULT_BASE_DIR="/data/llm/mxq/results"
CALIB_DATASETS="bos pileval wikitext c4"
CONFIGS="b2g128 b2g64 b2g32 b3g128 b3g64 b3g32 b4g128 b4g64 b4g32 b8g128 b8g64 b8g32"
MODELS="Qwen/Qwen2.5-7B Qwen/Qwen2.5-Coder-7B Qwen/Qwen2.5-Coder-7B-Instruct Qwen/Qwen2.5-Math-7B"

EXP_NAME=sensi_qwen25
RESULT_DIR=$RESULT_BASE_DIR/$EXP_NAME
mkdir -p $RESULT_DIR/data

for DS in $CALIB_DATASETS; do
    for CFG in $CONFIGS; do
        for MODEL in $MODELS; do
            SHORT_ID=$(echo $MODEL | cut -d/ -f2)
            OUT_FILE="${RESULT_DIR}/data/qwen25-sensi-${SHORT_ID}-${CFG}-${DS}.csv"
            python $TOOLKIT_DIR/src/dump.py sensi \
                --model $MODEL \
                --config $CFG \
                --calib-dataset $DS \
                --output-file $OUT_FILE
        done
    done
done
~~~~

The code snippet demonstrates how to calculate the sensitivity scores for a
series of Qwen2.5 models using 4 calibration datasets under 12 bit budgets.

## Calibration Dataset Generation Tool

This tool generates a small synthensized dataset named branch of science
(denoted as BoS, published on Hugging Face), which includes a few hundred of
textual defintions for science, art and business topics such as Mathematics,
Physics, Chemstry, Law, Music and Journalism etc. The dataset is intended to
validate if the sensitivity property generalize to diverse datasets.

The tool generates an initial dataset in .csv format which requires further
processing. The output of this tool is random due to the generative nature of
LLM. This tool requires a Llama-2-7B model being served with an OpenAI
compatible RESTful API endpoint. User can either use a hosted API endpoint or
deploy a local instance by following the instruction at the end of this section.

Once the API endpoint is secured, run the following script to generate the BoS dataset:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"

$TOOLKIT_DIR/utils/generate.py \
  --model="meta-llama/Llama-2-7b-chat-hf" \
  --variant="vLLM" \
  --topic-file=topics-l1.txt \
  --trace
~~~~
Lastly, find the result in the csv files under current directory.

### Local API endpoint
To deploy a local API endpoint using vLLM, create a virtual environment using
`conda` as follows:

~~~~bash
conda create -n vllm python=3.11 -y
conda activate vllm
pip install vllm==0.6.4.post1
~~~~
Then configure and launch the API server
~~~~bash
#!/bin/bash

vllm serve meta-llama/Llama-2-7b-chat-hf --dtype auto --api-key token-abc123
~~~~
Watch the output vLLm to make sure it starts up successfully.

# Visualization Tool usage

The visualization tools facilitate visualizing the experiment results and the
weight distribution, and generating insights of the latent features to quantize
LLMs more efficiently. Most visualization tools are implemented in R and
leverages the open-source plot libraries such as ggplot2, circlize, ggbreak,
ggmagnify. They provide CLI interface to simplify integaration with the
quantization harness tool.

These CLI tools support diverse options to allow user specify input dataset,
select particular model or approach to plot. To get help on these specific CLI
options, type `./plot_xxx.R --help` on command line prompt. For instance,
to get help on the MXQ allocation visualization tool, you may run command as
follows:

~~~~bash
./plot-mxq-allocation.R --help
Usage: ./plot-mxq-allocation.R [options]

Options:
        -h, --help
                Show this help message and exit

        -m CHARACTER, --model=CHARACTER
                Model ID

        -b DOUBLE, --budget=DOUBLE
                Bit Budget

        -d CHARACTER, --baseline_data_dir=CHARACTER
                Data directory of baseline results

        -q CHARACTER, --quant_cfg_allot_file=CHARACTER
                The combined quant config allocation csv file

        --attempt1=CHARACTER
                The first attempt to plot

        --attempt2=CHARACTER
                The second attempt to plot

        --fnorm
                Display FNorm value in the bar chart
~~~~

## Weight Distribution Visualization Tool

This tool enables visualizing layer-wised weight distribution of large language
models. It is implemented as an R script, which provides a convenient CLI
interface to enable shell scripting. Given a weight distribution metrics csv
file, it produces a pdf file under the `pdfs` with 3x3 sub-plots of column
digrams for the 9 modules in the Llama family models.

The tool is named `plot-wdist-llm.R` and located under the
`data-vis` folder in the `lm-quant-toolkit` project. A typical usage
is demonstrated in the code snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"

$TOOLKIT_DIR/data-vis/plot-wdist-llm.R -m Llama-2-7b-hf
~~~~

## Perplexity vs Bit Budget Visualization Tool

This tool enables visualizing the relationship between perplexity and bit
budget for diverse MXQ experiments against their baselines. The generated
diagram shows how memory reduction affects perplexity, which facilitates
memory-accuracy trade-off.

The tool is named `plot-ppl-mem.R` and located under the `data-vis`
folder in the `lm-quant-toolkit` project. It accepts a csv file containing
the perplexity metrics of MXQ and its baselines. The output are series of PDF
files corresponding to the models defined in the input file, which are placed
under the `pdfs` subfolder. A typical usage is demonstrated in the code
snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"

$TOOLKIT_DIR/data-vis/plot-ppl-mem.R -d data/combined.csv
~~~~

## Quantization Speed Comparison Visualization Tool

This tool generates column digrams to explore the quantization speed among
various approaches. The tool is also implemented as an R script, which provides a
convenient CLI interface to enable shell scripting. The tool is named
`plot-quant-speed.R` and located under the `data-vis` folder in the
`lm-quant-toolkit` project. Given a combined perplexity metrics csv file,
it produces a column digrams with x-axis in log-scale. Similar to other tools,
the PDF file is placed under the `pdfs` subfolder. A typical usage is
demonstrated in the code snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"

$TOOLKIT_DIR/data-vis/plot-quant-speed.R -d data/combined.csv
~~~~

## GPU Memory Usage Visualization Tool

This tool generates column digrams to present the actual GPU memory consumption
of LLMs quantized by diverse methods. The tool is also implemented as an R script,
which provides a convenient CLI interface to enable shell scripting. The tool
is named `plot-mem-consumption.R` and located under the `data-vis`
folder in the `lm-quant-toolkit` project. Given a combined perplexity
metrics csv file, it produces a column digrams of GPU memory usage in
Giga-byte. Similar to other tools, the PDF file is placed under the `pdfs`
subfolder. A typical usage is demonstrated in the code snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"

$TOOLKIT_DIR/data-vis/plot-mem-consumption.R -d data/combined.csv
~~~~

## Quantization Configuration Allocation Visualization Tool

This tool offers insights into the way MXQ and its variants allocate bit budget
to modules and layers. The variants, a.k.a. attempt, to include in the plot are
configurable. A maximium of 4 variants can be plotted in a circular layout
thanks to plot library circlize \citep{zuguang_2014}.
The first input expected by the tool is a combined quantization configuration
allocation csv file which should include experiment outcome for diverse methods
such as HQQ and MXQ. The second parameter is the directory where Frobenius
norms csv files are located. The third parameter is the perplexity score csv
file. The tool produces circos digram in PDF format.

The tool is named `plot-circos-allot.R` and located under the
`data-vis` folder in the `lm-quant-toolkit` project. A typical usage
is demonstrated in the code snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="../../.."

MODELS="
Llama-3-7b-hf
Llama-3-13b-hf
Meta-Llama-3-8B
"
BUDGETS="4.25 3.51"

STOP=2
TOPM=2
for MODEL in $MODELS; do
    for BUDGET in $BUDGETS; do
        $TOOLKIT_DIR/data-vis/plot-circos-allot.R \
          --model $MODEL \
          --budget $BUDGET \
          --fnorm_data_dir $TOOLKIT_DIR/src/data/ \
          --ppl_csv_file data/combined.csv \
          --quant_cfg_allot_file data/quant-cfg-allocation.csv \
          --attempt1 sensi-boost-${STOP}-${TOPM} \
          --attempt2 kurt-boost-${STOP}-${TOPM} \
          --attempt3 hqq\
          --attempt4 mxq1
    done
done
~~~~

This code snippet demonstrates how to generate a quant config allocation
comparison diagram to examine the nuanced difference between the SensiBoost and
kurtBoost approaches, with a stop of 2 and top-{m} 2, as well as the HQQ and
MXQ baselines.

## SensiBoost/KurtBoost Win-Tie-Loss Visualization Tool

This tool enables qualitative analysis of effectiveness of the proposed
SensiBoost and KurtBoost methods. It is implemented as an R script, which
provides a conventional CLI interface to ease automation.
Given a combined perplexity metrics csv file, it produces a series of column
digrams in PDF format. The csv file should include experiment outcome for
SensiBoost, KurtBoost, the ablation tests or baseline such as HQQ and MXQ. The
name experiment, a.k.a. attempt, should follow the pattern
`<method>-<stop>-<top m>`.

This tool is included in the `lm-quant-toolkit` under `data-vis`
folder. A typical usage is demonstrated in the code snippet as follows:

~~~~bash
#!/bin/bash

TOOLKIT_DIR="$HOME/work/lm-quant-toolkit"

$TOOLKIT_DIR/data-vis/plot-win-tie-loss.R -f data/combined.csv
~~~~

[1]: https://huggingface.co/docs/leaderboards/leaderboards/intro
[2]: https://github.com/schnell18/lm-quant-toolkit.git
