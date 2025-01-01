import copy
import logging
import os

# from adapter.awq import create_awq_model
# from adapter.awq import quantize_awq_model
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from auto_gptq import BaseQuantizeConfig as GPTQQuantConfig
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig
from transformers import BitsAndBytesConfig

from lm_quant_toolkit.adapter.autoawq import (
    create_autoawq_model,
    quantize_autoawq_model,
)
from lm_quant_toolkit.adapter.autogptq import (
    create_autogptq_model,
    quantize_autogptq_model,
)
from lm_quant_toolkit.adapter.bnb import create_bnb_model, quantize_bnb_model
from lm_quant_toolkit.adapter.fp16 import create_fp16_model
from lm_quant_toolkit.adapter.hqq import create_hqq_model, quantize_hqq_model
from lm_quant_toolkit.adapter.mxq import create_mxq_model, quantize_mxq_model
from lm_quant_toolkit.eval.common import (
    HQQ_CONFIGS,
    _dump_cuda_mem_snapshot,
    _reset_peak_memory_stats,
    cleanup,
    combine_metrics,
    get_memory_metrics,
    get_mxq_quant_meta_data_file,
    persist_progress,
    save_partial_metric,
)
from lm_quant_toolkit.eval.leaderboard import eval_llm_leaderboard
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
            spec["create_fn"] = create_autogptq_model
            spec["quantize_fn"] = quantize_autogptq_model
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
    if cnt_todo == 0:
        return

    df_todo = df_todo.sort_values(by=["model", "cfg"], ascending=False)
    for idx, row in df_todo.iterrows():
        model_id = row["model"]
        algo = row["algo"]
        task_type = row["task_type"]
        cfg = row["cfg"]
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
        elif task_type == "eval_leaderboard":
            print(
                f"Evaluating {algo} LLM Leaderboard benchmarks on {model_id} w/ config: {cfg}..."
            )
        else:
            print(
                f"Evaluating {algo} model storage metrics on {model_id} w/ config: {cfg}..."
            )
        print("*" * 72)

        if track_cuda_memory:
            torch.cuda.memory._record_memory_history()
        _reset_peak_memory_stats()
        if task_type != "eval_leaderboard":
            create_fn = spec["create_fn"]
            model, tokenizer, quantized, model_file_size = create_fn(
                model_id, config[1], cfg, quant_fn is not None, quant_dir
            )

            if not quantized and quant_fn:
                # avoid interventions between models
                quant_config = copy.deepcopy(config[1])
                if algo == "mxq":
                    ok, metric_fp = get_mxq_quant_meta_data_file(model_id)
                    if not ok:
                        print(
                            f"Quantization meta data file: {metric_fp} doesn't exists!"
                        )
                        return
                    quant_config["quant_metrics_file"] = metric_fp
                    quant_config["weight_algo"] = kwargs.get("weight_algo", None)
                    quant_config["boost_layers"] = kwargs.get("boost_layers", None)
                    quant_config["decline_layers"] = kwargs.get("decline_layers", None)
                    quant_config["boost_stop"] = kwargs.get("boost_stop", None)
                    quant_config["decline_stop"] = kwargs.get("decline_stop", None)
                    quant_config["ablation"] = kwargs.get("ablation", None)
                    quant_config["top_m_layer"] = kwargs.get("top_m_layer", None)
                    quant_config["factor"] = kwargs.get("factor", None)
                model, duration, model_file_size = quant_fn(
                    model,
                    tokenizer,
                    quant_config,
                    model_id,
                    cfg,
                    quant_dir,
                )
                metric["quant_duration"] = duration
            if task_type == "eval_model_storage":
                allot, reserved = get_memory_metrics()
                metric["load_mem_allot"] = allot
                metric["load_mem_reserved"] = reserved
                metric["model_storage_size"] = model_file_size

            elif task_type == "eval_ppl":
                # Evaluate the quantized model
                metric = eval_ppls(model, tokenizer, metric)
                metric["ppl_mem_allot"], metric["ppl_mem_reserved"] = (
                    get_memory_metrics()
                )
            if track_cuda_memory:
                _dump_cuda_mem_snapshot(experiment_name, model_id, algo, result_dir)
            cleanup(model)
        else:
            metric = eval_llm_leaderboard(
                experiment_name,
                model_id,
                algo,
                cfg,
                quant_fn is not None,
                metric,
                quant_dir,
                result_dir,
            )
            metric["leaderboard_mem_allot"], metric["leaderboard_mem_reserved"] = (
                get_memory_metrics()
            )
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
        "quant_duration": 0,
        "ppl_wikitext": 0,
        "ppl_c4": 0,
        "duration_wikitext": 0,
        "duration_c4": 0,
        "duration_leaderboard": 0,
        "ifeval": 0,
        "bbh": 0,
        "mathlevel5": 0,
        "gpqa": 0,
        "musr": 0,
        "mmlupro": 0,
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


########################################################################
#  Quantization experiments
########################################################################


def experiment_quant_hqq():
    models = ALL_MODELS
    tasks = {
        "hqq": {
            "type": "quant",
            "configs": HQQ_CONFIGS,
        },
    }
    do_expermient_fdata(
        "quant_hqq",
        models,
        tasks,
    )


def experiment_quant_mxq():
    models = ALL_MODELS
    type = "quant"
    algo = "mxq"
    tasks = {
        algo: {
            "type": type,
            "configs": MXQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}_mxq", models, tasks)


def experiment_quant_awq():
    # models = [ALL_MODELS[0], ALL_MODELS[2]]
    models = [ALL_MODELS[1]]
    type = "quant"
    algo = "awq"
    tasks = {
        algo: {
            "type": type,
            "configs": AUTOAWQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_quant_gptq():
    models = ALL_MODELS
    type = "quant"
    algo = "gptq"
    tasks = {
        algo: {
            "type": type,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_quantize_405B():
    models = [
        "meta-llama/Meta-Llama-3.1-405B-Instruct",
    ]

    tasks = {
        "hqq": {
            "type": "quant",
            "configs": HQQ_CONFIGS[1:2],
        },
    }
    do_expermient(
        "quant_hqq_405B",
        models,
        tasks,
        quant_dir="/data/gqq-eval/snapshots/",
    )


########################################################################
#  Perplexity evaluation experiments
########################################################################


def experiment_ppl_eval_fp16():
    models = ALL_MODELS
    type = "eval_ppl"
    algo = "fp16"
    tasks = {
        algo: {
            "type": type,
            "configs": [
                ("base", {}),
            ],
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_ppl_eval_awq():
    models = [ALL_MODELS[0], ALL_MODELS[2]]
    type = "eval_ppl"
    algo = "awq"
    tasks = {
        algo: {
            "type": type,
            "configs": AUTOAWQ_CONFIGS[1:2],
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_ppl_eval_gptq():
    models = ALL_MODELS
    type = "eval_ppl"
    algo = "gptq"
    tasks = {
        algo: {
            "type": type,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_ppl_eval_hqq():
    models = ALL_MODELS
    type = "eval_ppl"
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


########################################################################
#  Open LLM Leaderboard evaluation experiments
########################################################################


def experiment_llm_leaderboard_fp16():
    models = [ALL_MODELS[0], ALL_MODELS[2]]
    type = "eval_leaderboard"
    algo = "fp16"
    tasks = {
        algo: {
            "type": type,
            "configs": [
                ("base", {}),
            ],
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_llm_leaderboard_autogptq():
    models = ALL_MODELS
    type = "eval_leaderboard"
    algo = "gptq"
    tasks = {
        algo: {
            "type": type,
            "configs": GPTQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_llm_leaderboard_hqq():
    models = ALL_MODELS
    type = "eval_leaderboard"
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_llm_leaderboard_mxq():
    models = ALL_MODELS
    type = "eval_leaderboard"
    algo = "mxq"
    tasks = {
        algo: {
            "type": type,
            "configs": MXQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_llm_leaderboard_autoawq():
    models = ALL_MODELS[0:1]
    type = "eval_leaderboard"
    algo = "awq"
    tasks = {
        algo: {
            "type": type,
            "configs": AUTOAWQ_CONFIGS[0:2],
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


########################################################################
#  Mixed Quant Eval experiments
########################################################################


def experiment_quant_ppl_eval_mxq_comprise():
    models = ALL_MODELS
    equiv_mxq_configs = []
    nbits = [4.06, 4.10, 4.15, 4.19, 4.24, 4.28, 4.33]
    for bits in nbits:
        cfg_name = f"mxq-{str(bits).replace('.', '_')}"
        equiv_mxq_configs.append(
            (cfg_name, HQQQuantConfig(mixed=True, budget=bits, quant_scale=True))
        )
    quant_tasks = {
        "hqq": {
            "type": "quant",
            "configs": equiv_mxq_configs,
        },
    }
    ppl_tasks = {
        "hqq": {
            "type": "eval_ppl",
            "configs": equiv_mxq_configs,
        },
    }
    do_expermient_fdata("quant_mxq_compromise", models, quant_tasks)
    do_expermient_fdata("eval_mxq_compromise", models, ppl_tasks)


########################################################################
#  Misc experiments
########################################################################


def experiment_fp16_llama3_8B_OOM():
    models = ALL_MODELS[-1:]
    type = "eval_ppl"
    algo = "fp16"
    tasks = {
        algo: {
            "type": type,
            "configs": [
                ("base", {}),
            ],
        },
    }
    do_expermient_fdata(
        f"{type}_llama3_8B_OOM_{algo}",
        models,
        tasks,
        track_cuda_memory=True,
    )


def experiment_fp16_vs_hqq_eval_gpu_mem():
    models = ALL_MODELS[-1:]
    type = "eval_ppl"
    algo = "fp16"
    tasks = {
        algo: {
            "type": type,
            "configs": [
                ("base", {}),
            ],
        },
    }
    do_expermient_fdata("experiment_fp16_vs_hqq_eval_gpu_mem", models, tasks)
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS[1:2],
        },
    }
    do_expermient_fdata("experiment_fp16_vs_hqq_eval_gpu_mem", models, tasks)


def experiment_eval_model_storage():
    models = ALL_MODELS
    type = "eval_model_storage"
    tasks = {
        "fp16": {
            "type": type,
            "configs": [
                ("base", {}),
            ],
        },
        "mxq": {
            "type": type,
            "configs": MXQ_CONFIGS,
        },
        "hqq": {
            "type": type,
            "configs": HQQ_CONFIGS,
        },
        "awq": {
            "type": type,
            "configs": AUTOAWQ_CONFIGS,
        },
        "gptq": {"type": type, "configs": GPTQ_CONFIGS},
    }
    for i in range(5):
        do_expermient_fdata(f"eval_model_storge_{i}", models, tasks)


def experiment_eval_ppl_all():
    models = ALL_MODELS
    type = "eval_ppl"
    tasks = {
        "fp16": {
            "type": type,
            "configs": [
                ("base", {}),
            ],
        },
        "mxq": {
            "type": type,
            "configs": MXQ_CONFIGS,
        },
        "hqq": {
            "type": type,
            "configs": HQQ_CONFIGS,
        },
        "awq": {
            "type": type,
            "configs": AUTOAWQ_CONFIGS,
        },
        "gptq": {"type": type, "configs": GPTQ_CONFIGS},
    }
    do_expermient_fdata("experiment_eval_ppl_all", models, tasks)


def experiment_debug_quant_hqq():
    models = [ALL_MODELS[1]]
    type = "eval_model_storage"
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS[1:2],
        },
    }

    do_expermient(
        f"debug_{type}_{algo}",
        models,
        tasks,
        quant_dir="/fdata/llm/mxq/snapshots-debug",
        result_dir="/fdata/llm/mxq/results",
    )


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # experiment_llm_leaderboard_autogptq()
    # experiment_llm_leaderboard_fp16()
    # experiment_llm_leaderboard_hqq()
    # experiment_llm_leaderboard_autoawq()
    # experiment_quant_hqq()
    # experiment_quant_mxq()
    # experiment_quant_awq()
    # experiment_quant_gptq()
    # experiment_ppl_eval_fp16()
    # experiment_ppl_eval_hqq()
    # experiment_ppl_eval_gptq()
    # experiment_ppl_eval_awq()
    # experiment_fp16_llama3_8B_OOM()
    # experiment_fp16_vs_hqq_eval_gpu_mem()
    # experiment_debug_quant_hqq()
    # experiment_eval_model_storage()
    # experiment_eval_ppl_all()
    experiment_debug_quant_hqq()


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
