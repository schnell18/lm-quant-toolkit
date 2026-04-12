import copy
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from gptqmodel import QuantizeConfig as GPTQQuantConfig
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig
from lm_eval.models.huggingface import HFLM
from transformers import BitsAndBytesConfig

from lm_quant_toolkit.adapter.awq import (
    create_autoawq_model,
    quantize_autoawq_model,
)
from lm_quant_toolkit.adapter.bnb import create_bnb_model, quantize_bnb_model
from lm_quant_toolkit.adapter.fp16 import create_fp16_model
from lm_quant_toolkit.adapter.gptq import (
    create_gptq_model,
    quantize_gptq_model,
)
from lm_quant_toolkit.adapter.hqq import create_hqq_model, quantize_hqq_model
from lm_quant_toolkit.adapter.mxq import create_mxq_model, quantize_mxq_model
from lm_quant_toolkit.eval.common import (
    _dump_cuda_mem_snapshot,
    _reset_peak_memory_stats,
    cleanup,
    combine_metrics,
    get_memory_metrics,
    get_mxq_quant_meta_data_file,
    persist_progress,
    save_partial_metric,
)
from lm_quant_toolkit.eval.lmeval import eval_llm_perf
from lm_quant_toolkit.eval.perplexity import eval_ppls

ALL_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
    "meta-llama/Llama-3.1-8B",
]

MXQ_CONFIGS = [
    (
        f"{bits:.2f}".replace(".", "_"),
        HQQQuantConfig(mixed=True, budget=bits, quant_scale=True),
    )
    for bits in [5.00, 4.75, 4.50, 4.25, 4.01, 3.76, 3.50, 3.00, 2.75, 2.48]
]

BNB_CONFIGS = [
    (
        "b4g64",
        BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype="float16",
            bnb_4bit_use_double_quant=True,
        ),
    ),
    (
        "b8g128",
        BitsAndBytesConfig(
            load_in_8bit=True,
            bnb_4bit_compute_dtype="float16",
        ),
    ),
]


AUTOAWQ_CONFIGS = [
    ("b4g32", {"w_bit": 4, "q_group_size": 32, "zero_point": True, "version": "GEMM"}),
    ("b4g64", {"w_bit": 4, "q_group_size": 64, "zero_point": True, "version": "GEMM"}),
    (
        "b4g128",
        {"w_bit": 4, "q_group_size": 128, "zero_point": True, "version": "GEMM"},
    ),
    # 3-bit not supported by AutoAWQ right now
    # ("b3g64", {"w_bit": 3, "q_group_size": 64, "zero_point": True, 'version':'gemv_fast'}),
    # ("b3g128", {"w_bit": 3, "q_group_size": 128, "zero_point": True, 'version':'gemv_fast'}),
]

GPTQ_CONFIGS = [
    (
        "b8g32",
        GPTQQuantConfig(bits=8, group_size=32, damp_percent=0.01, desc_act=False),
    ),
    (
        "b8g64",
        GPTQQuantConfig(bits=8, group_size=64, damp_percent=0.01, desc_act=False),
    ),
    (
        "b8g128",
        GPTQQuantConfig(bits=8, group_size=128, damp_percent=0.01, desc_act=False),
    ),
    (
        "b4g32",
        GPTQQuantConfig(bits=4, group_size=32, damp_percent=0.01, desc_act=False),
    ),
    (
        "b4g64",
        GPTQQuantConfig(bits=4, group_size=64, damp_percent=0.01, desc_act=False),
    ),
    (
        "b4g128",
        GPTQQuantConfig(bits=4, group_size=128, damp_percent=0.01, desc_act=False),
    ),
    (
        "b3g32",
        GPTQQuantConfig(bits=3, group_size=32, damp_percent=0.01, desc_act=False),
    ),
    (
        "b3g64",
        GPTQQuantConfig(bits=3, group_size=64, damp_percent=0.01, desc_act=False),
    ),
    (
        "b3g128",
        GPTQQuantConfig(bits=3, group_size=128, damp_percent=0.01, desc_act=False),
    ),
]


def gen_experiment_items(models, tasks):
    dikts = []
    for algo, spec in tasks.items():
        configs = spec["configs"]
        for config in configs:
            for model_id in models:
                dikts.append(
                    {
                        "model": model_id,
                        "cfg": config[0],
                        "task_type": spec["type"],
                        "algo": algo,
                    }
                )
    return pd.DataFrame(dikts)


def _setup_fn(algo, spec):
    match algo:
        case "fp16":
            spec["create_fn"] = create_fp16_model
            spec["quantize_fn"] = None
        case "awq":
            spec["create_fn"] = create_autoawq_model
            spec["quantize_fn"] = quantize_autoawq_model
        case "gptq":
            spec["create_fn"] = create_gptq_model
            spec["quantize_fn"] = quantize_gptq_model
        case "hqq":
            spec["create_fn"] = create_hqq_model
            spec["quantize_fn"] = quantize_hqq_model
        case "bnb":
            spec["create_fn"] = create_bnb_model
            spec["quantize_fn"] = quantize_bnb_model
        case "mxq":
            spec["create_fn"] = create_mxq_model
            spec["quantize_fn"] = quantize_mxq_model
        case _:
            raise ValueError(f"Invalid algo: {algo}")


def _load_todo_tasks(result_dir, experiment_name, models, tasks):
    df_all = gen_experiment_items(models, tasks)
    progress_path = os.path.join(result_dir, experiment_name, "progress.csv")
    if Path(progress_path).exists():
        df_saved = pd.read_csv(progress_path)
        df_all = df_all.merge(
            df_saved, how="left", on=["model", "cfg", "task_type", "algo"]
        )
        # filter already processed repos, equivalent to SQL is null
        df_todo = df_all.query("status != status or status != 1")
    else:
        df_all["status"] = 0
        df_all["completion_time"] = ""
        df_todo = df_all
    print("*" * 72)
    print("Sub-task list:")
    print(df_all)
    cnt_todo, cnt_tot = len(df_todo), len(df_all)
    print(f"Todo:{cnt_todo}, Done: {cnt_tot - cnt_todo}, Total: {cnt_tot}")
    if cnt_todo == 0:
        print("Tasks completed!")
    print("*" * 72)
    return df_all, df_todo, progress_path


def _quant_model(
    model, tokenizer, cfg, quant_dir, config, algo, model_id, quant_fn, kwargs
):
    # avoid interventions between models
    quant_config = copy.deepcopy(config[1])
    if algo == "mxq":
        ok, metric_fp = get_mxq_quant_meta_data_file(model_id)
        if not ok:
            raise ValueError(
                f"Quantization meta data file: {metric_fp} doesn't exists!"
            )
        quant_config["quant_metrics_file"] = metric_fp
        quant_config["weight_algo"] = kwargs.get("weight_algo", None)
        quant_config["boost_layers"] = kwargs.get("boost_layers", None)
        quant_config["decline_layers"] = kwargs.get("decline_layers", None)
        quant_config["boost_stop"] = kwargs.get("boost_stop", None)
        quant_config["decline_stop"] = kwargs.get("decline_stop", None)
        quant_config["ablation"] = kwargs.get("ablation", None)
        quant_config["top_m_layer"] = kwargs.get("top_m_layer", None)
        quant_config["factor"] = kwargs.get("factor", None)
    return quant_fn(
        model,
        tokenizer,
        quant_config,
        model_id,
        cfg,
        quant_dir,
    )


def do_expermient(
    experiment_name,
    models,
    tasks,
    quant_dir="snapshots",
    result_dir="results",
    log_dir="logs",
    track_cuda_memory=False,
    **kwargs,
):
    df_all, df_todo, progress_path = _load_todo_tasks(
        result_dir, experiment_name, models, tasks
    )
    if len(df_todo) == 0:
        return

    LLM_PERF_TASKS = ["eval_gpga_diamond_zeroshot", "eval_gsm8k"]
    df_todo = df_todo.sort_values(by=["model", "cfg"], ascending=False)
    for idx, row in df_todo.iterrows():
        model_id, task_type = row["model"], row["task_type"]
        algo, cfg = row["algo"], row["cfg"]
        spec = tasks[algo]
        _setup_fn(algo, spec)
        config = [c for c in spec["configs"] if c[0] == cfg][0]
        quant_fn = spec["quantize_fn"]
        metric = _init_metrics(model_id, algo, config)
        print("*" * 72)
        if task_type == "quant":
            print(f"Quantizing {algo} on {model_id} w/ config: {cfg}...")
        elif task_type == "eval_ppl":
            print(f"Evaluating {algo} PPL on {model_id} w/ config: {cfg}...")
        elif task_type in LLM_PERF_TASKS:
            print(f"Evaluating {algo} benchmarks on {model_id} w/ config: {cfg}...")
        else:
            print(
                f"Evaluating {algo} model storage metrics on {model_id} w/ config: {
                    cfg
                }..."
            )
        print("*" * 72)

        if track_cuda_memory:
            torch.cuda.memory._record_memory_history()
        _reset_peak_memory_stats()

        # create model for perplexity or downstream task eval
        create_fn = spec["create_fn"]
        model, tokenizer, quantized, model_file_size = create_fn(
            model_id, config[1], cfg, quant_fn is not None, quant_dir
        )
        if task_type == "quant":
            if not quantized and quant_fn:
                model, duration, model_file_size = _quant_model(
                    model,
                    tokenizer,
                    cfg,
                    quant_dir,
                    config,
                    algo,
                    model_id,
                    quant_fn,
                    kwargs,
                )
                metric["quant_duration"] = duration
        elif task_type == "eval_model_storage":
            allot, reserved = get_memory_metrics()
            metric["load_mem_allot"] = allot
            metric["load_mem_reserved"] = reserved
            metric["model_storage_size"] = model_file_size
        elif task_type == "eval_ppl":
            # Evaluate the quantized model
            metric = eval_ppls(model, tokenizer, metric)
            metric["ppl_mem_allot"], metric["ppl_mem_reserved"] = get_memory_metrics()
        elif task_type in LLM_PERF_TASKS:
            # Wrap the model into HFLM
            model = HFLM(pretrained=model)
            task = task_type.split("_", 1)[1]
            metric = eval_llm_perf(
                experiment_name,
                task,
                model,
                metric,
                result_dir,
            )
            metric[f"{task_type}_mem_allot"], metric[f"{task_type}_mem_reserved"] = (
                get_memory_metrics()
            )
        else:
            raise ValueError(f"Invalid task_type: {task_type}")
        cleanup(model)
        if track_cuda_memory:
            _dump_cuda_mem_snapshot(experiment_name, model_id, algo, result_dir)
        save_partial_metric(experiment_name, algo, model_id, cfg, metric, result_dir)
        df_all.loc[
            (df_all["model"] == model_id)
            & (df_all["cfg"] == cfg)
            & (df_all["algo"] == algo)
            & (df_all["task_type"] == task_type),
            ["status", "completion_time"],
        ] = 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        persist_progress(df_all, progress_path)
    # combine metrics
    combine_metrics(experiment_name, result_dir)


def _init_metrics(model_id, algo, config):
    return {
        "model": model_id.split("/")[1],
        "algo": algo,
        "config": config[0],
        "config_detail": str(config[1]).replace("\n", ""),
        "quant_duration": 0,
        "model_storage_size": 0,
        "load_mem_allot": 0,
        "load_mem_reserved": 0,
        "ppl_mem_allot": 0,
        "ppl_mem_reserved": 0,
        "leaderboard_mem_allot": 0,
        "leaderboard_mem_reserved": 0,
        "ppl_wikitext": 0,
        "ppl_c4": 0,
        "duration_wikitext": 0,
        "duration_c4": 0,
    }


def do_expermient_fdata(
    experiment_name,
    models,
    tasks,
    track_cuda_memory=False,
):
    do_expermient(
        experiment_name,
        models,
        tasks,
        quant_dir="/fdata/llm/mxq/snapshots",
        result_dir="/fdata/llm/mxq/results",
        log_dir="/fdata/llm/mxq/logs",
        track_cuda_memory=track_cuda_memory,
    )


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )


if __name__ == "__main__":
    # os.environ['HF_DATASETS_OFFLINE'] = '1'

    max_threads = str(min(8, os.cpu_count()))
    os.environ["OMP_NUM_THREADS"] = max_threads
    os.environ["OPENBLAS_NUM_THREADS"] = max_threads
    os.environ["MKL_NUM_THREADS"] = max_threads
    os.environ["VECLIB_MAXIMUM_THREADS"] = max_threads
    os.environ["NUMEXPR_NUM_THREADS"] = max_threads
    os.environ["NUMEXPR_MAX_THREADS"] = max_threads
    os.environ["HF_HOME"] = "/data/hugginface/"

    main()
