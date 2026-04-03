# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**lm-quant-toolkit** is a research toolkit for large neural network quantization experiments, associated with the paper "A Mixed Quantization Approach for Data-Free Quantization of LLMs" (Zhang et al., 2025). It supports quantization of LLMs and Vision Transformers (ViT) using multiple methods (HQQ, MXQ, GPTQ, AWQ, BitsAndBytes, FP16), evaluates perplexity/memory/storage, and generates metadata (Frobenius norms, kurtosis, sensitivity scores) used to guide the MXQ mixed-precision allocation algorithm.

## Setup

```bash
# Python tools (requires Python 3.11.9 + miniconda)
./setup-harness.sh      # installs patched HQQ, lm-eval, CLIP_benchmark forks + editable package

# R visualization tools (requires R >= 4.4.1)
./setup-visualization.sh
```

Patched dependencies are installed into `.deps/` from custom forks — do not replace with upstream versions.

## Commands

### Run experiments
```bash
# LLM quantization
python src/cli.py llm --model <hf_model_id> --algo <algo> --config <cfg> --task <task> --result-dir <dir>

# ViT quantization
python src/cli.py vit --model <hf_model_id> --algo <algo> --config <cfg> --result-dir <dir>
```

### Prepare metadata
```bash
python src/dump.py fnorm     --model <hf_model_id> --output-dir <dir>
python src/dump.py kurtosis  --model <hf_model_id> --output-dir <dir>
python src/dump.py sensi     --model <hf_model_id> --config <cfg> --output-file <csv>
```

### Tests and linting
```bash
tox -e py311          # run tests with coverage
tox -e flake8         # PEP8 style
tox -e pycodestyle    # code style (verbose)
tox -e pydocstyle     # docstring style

# Run a single test file
pytest -v tests/path/to/test_file.py
```

## Architecture

### Entry points
- **`src/cli.py`** — Main CLI with `llm` and `vit` subcommands; delegates to `eval/bench.py` and `eval/bench_vit.py`
- **`src/dump.py`** — Metadata CLI with `fnorm`, `kurtosis`, `sensi` subcommands

### Package structure (`src/lm_quant_toolkit/`)

| Module | Purpose |
|--------|---------|
| `adapter/` | Quantization adapters: `mxq.py`, `hqq.py`, `autogptq.py`, `autoawq.py`, `bnb.py`, `fp16.py`; `vit/` for vision models |
| `eval/` | Experiment harnesses (`bench.py` ~2000 lines, `bench_vit.py`), perplexity eval, lm-eval integration |
| `prep/` | Metadata calculation: `fnorm.py` (Frobenius norms), `sensitivity.py`, `wdist.py` (kurtosis) |
| `misc/` | Quantization config simulation (`quant_sim.py`), weight allocation (`qweight.py`) |
| `utils/` | HF Hub model registry (`hub.py`), safetensors helpers |

### Adapter pattern
All quantization adapters expose:
```python
create_<method>_model(model_id, quant_config, config_id, load_quantized, save_dir)
quantize_<method>_model(model, tokenizer, quant_config, model_id, config_id, save_dir)
```

### Quantization configurations
- **HQQ_CONFIGS** — 12 standard configs named `b<bits>g<group>` (e.g., `b4g64`): bit depths 2/3/4/8, group sizes 16/32/64/128
- **MXQ_CONFIGS** — Mixed-precision bit budgets (2.48–5.00 bits/param), uses HQQ with dynamic per-layer allocation guided by FNorm metadata
- BNB, GPTQ, AWQ each have their own config dicts in `eval/common.py`

### Experiment flow (`bench.py`)
1. Parse CLI → iterate (model, algo, config) combinations
2. Load model via adapter → quantize → evaluate (perplexity, GPU memory, storage, optionally lm-eval leaderboard tasks)
3. Save partial CSV per config; combine at end
4. Resume support via checkpoint CSVs (`partial-<algo>-<model>-<config>.csv`)

### Model registry (`utils/hub.py`)
`LLAMA_MODELS` and `VIT_OPENCLIP_MODELS` dicts map model IDs to metadata (layer counts, base directories). Add new models here when extending support.

### Visualization
~60 R scripts in `data-vis/` for perplexity plots, circos allocation diagrams, memory analysis, win-tie-loss comparisons. Results are CSV files; scripts aggregate with `combine.R` before plotting.

### Supporting scripts
- `scripts/` — ~200 bash scripts automating experiment pipelines and metadata dumps
- `utils/generate.py` — calibration dataset generation (Branch of Science / BoS)
