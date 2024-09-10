import copy
import gc
import glob
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

from lm_quant_toolkit.adapter.autoawq import (
    create_autoawq_model,
    quantize_autoawq_model,
)
from lm_quant_toolkit.adapter.autogptq import (
    create_autogptq_model,
    quantize_autogptq_model,
)
from lm_quant_toolkit.adapter.fp16 import create_fp16_model
from lm_quant_toolkit.adapter.hqq import create_hqq_model, quantize_hqq_model
from lm_quant_toolkit.eval.leaderboard import eval_llm_leaderboard
from lm_quant_toolkit.eval.perplexity import eval_ppls

ALL_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

QUANT_METRICS_FILE_MAP = {
    "meta-llama/Llama-2-7b-hf": "data/fnorm-Llama-2-7b-hf.csv",
    "meta-llama/Llama-2-13b-hf": "data/fnorm-Llama-2-13b-hf.csv",
    "meta-llama/Meta-Llama-3-8B": "data/fnorm-Llama-3-8B.csv",
}

HQQ_CONFIGS = [
    ("b4g32", HQQQuantConfig(nbits=4, group_size=32)),
    ("b4g64", HQQQuantConfig(nbits=4, group_size=64)),
    ("b4g128", HQQQuantConfig(nbits=4, group_size=128)),
    ("b3g32", HQQQuantConfig(nbits=3, group_size=32)),
    ("b3g64", HQQQuantConfig(nbits=3, group_size=64)),
    ("b3g128", HQQQuantConfig(nbits=3, group_size=128)),
    ("b2g16", HQQQuantConfig(nbits=2, group_size=16)),
    ("b2g32", HQQQuantConfig(nbits=2, group_size=32)),
    ("b2g64", HQQQuantConfig(nbits=2, group_size=64)),
    ("mxq-5_00", HQQQuantConfig(mixed=True, budget=5.00, quant_scale=True)),
    ("mxq-4_75", HQQQuantConfig(mixed=True, budget=4.75, quant_scale=True)),
    ("mxq-4_50", HQQQuantConfig(mixed=True, budget=4.50, quant_scale=True)),
    ("mxq-4_25", HQQQuantConfig(mixed=True, budget=4.25, quant_scale=True)),
    ("mxq-4_01", HQQQuantConfig(mixed=True, budget=4.01, quant_scale=True)),
    ("mxq-3_76", HQQQuantConfig(mixed=True, budget=3.76, quant_scale=True)),
    ("mxq-3_50", HQQQuantConfig(mixed=True, budget=3.50, quant_scale=True)),
    ("mxq-3_00", HQQQuantConfig(mixed=True, budget=3.00, quant_scale=True)),
    ("mxq-2_75", HQQQuantConfig(mixed=True, budget=2.75, quant_scale=True)),
    ("mxq-2_48", HQQQuantConfig(mixed=True, budget=2.48, quant_scale=True)),
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

AWQ_CONFIGS = [
    ("b4g32", {"w_bit": 4, "q_group_size": 32, "zero_point": True}),
    ("b4g64", {"w_bit": 4, "q_group_size": 64, "zero_point": True}),
    ("b4g128", {"w_bit": 4, "q_group_size": 128, "zero_point": True}),
    ("b3g32", {"w_bit": 3, "q_group_size": 32, "zero_point": True}),
    ("b3g64", {"w_bit": 3, "q_group_size": 64, "zero_point": True}),
    ("b3g128", {"w_bit": 3, "q_group_size": 128, "zero_point": True}),
]

GPTQ_CONFIGS = [
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


def calc_bits(b1, g1, b2, g2):
    return b1 + 2 * b2 / g1 + 32 / g1 / g2


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


def persist_progress(
    model,
    cfg,
    algo,
    task_type,
    progress_path,
):
    """Save the progress of experiment for resumption."""
    Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
    if not Path(progress_path).exists():
        with open(progress_path, "w") as f:
            f.write("model,cfg,algo,task_type,status,completion_time\n")

    ts_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(progress_path, "a") as f:
        f.write(f"{model},{cfg},{algo},{task_type},1,{ts_str}\n")


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
        case _:
            raise ValueError(f"Invalid algo: {algo}")


def do_expermient(
    experiment_name,
    models,
    tasks,
    quant_dir="snapshots",
    result_dir="results",
    log_dir="logs",
):
    all_df = gen_experiment_items(models, tasks)
    progress_path = os.path.join(result_dir, experiment_name, "progress.csv")
    if Path(progress_path).exists():
        checked_df = pd.read_csv(progress_path)
        df2 = all_df.merge(
            checked_df, how="left", on=["model", "cfg", "task_type", "algo"]
        )
        # filter already processed repos, equivalent to SQL is null
        df2 = df2.query("status != status or status != 1")
    else:
        df2 = all_df
    if df2.empty:
        print("*" * 72)
        print("Tasks completed!")
        print("*" * 72)
        return

    df2 = df2.sort_values(by=["model", "cfg"])
    print("*" * 72)
    print("Sub-task list:")
    print(df2)
    print("*" * 72)
    for idx, row in df2.iterrows():
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
        else:
            print(
                f"Evaluating {algo} LLM Leaderboard benchmarks on {model_id} w/ config: {cfg}..."
            )
        print("*" * 72)

        if task_type != "eval_leaderboard":
            create_fn = spec["create_fn"]
            model, tokenizer, quantized = create_fn(
                model_id, config[1], cfg, quant_fn is not None, quant_dir
            )
            metric["load_mem_allot"], metric["load_mem_reserved"] = get_memory_metrics()

            if not quantized and quant_fn:
                # avoid interventions between models
                quant_config = copy.deepcopy(config[1])
                if cfg.startswith("mxq-") and model_id in QUANT_METRICS_FILE_MAP:
                    quant_config["quant_metrics_file"] = QUANT_METRICS_FILE_MAP[
                        model_id
                    ]
                model, duration = quant_fn(
                    model,
                    tokenizer,
                    quant_config,
                    model_id,
                    cfg,
                    quant_dir,
                )
                # persistent the quantized model
                os.sync()
                metric["quant_duration"] = duration
            # Evaluate the quantized model
            if task_type == "eval_ppl":
                metric = eval_ppls(model, tokenizer, metric)
                metric["ppl_mem_allot"], metric["ppl_mem_reserved"] = (
                    get_memory_metrics()
                )
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
        save_partial_metric(experiment_name, algo, model_id, cfg, metric, result_dir)
        persist_progress(model_id, cfg, algo, task_type, progress_path)
    # combine metrics
    combine_metrics(experiment_name, result_dir)


def _init_metrics(model_id, algo, config):
    return {
        "model": model_id.split("/")[1],
        "algo": algo,
        "config": config[0],
        "config_detail": config[1],
        "quant_duration": 0,
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


def save_partial_metric(experiment_name, algo, model_id, config, metric, result_dir):
    metrics = [metric]
    df = pd.DataFrame(metrics)
    result_dir = os.path.join(result_dir, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    model_short_id = model_id.split("/")[1]
    file_name = f"{result_dir}/partial-{algo}-{model_short_id}-{config}.csv"
    df.to_csv(file_name, index=False)


def combine_metrics(experiment_name, result_dir):
    dfs = []
    iters = glob.iglob(f"{result_dir}/{experiment_name}/partial-*.csv")
    for it in iters:
        df = pd.read_csv(it)
        dfs.append(df)
    combined = pd.concat(dfs)
    ts_str = datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"{result_dir}/{experiment_name}/result-{experiment_name}-{ts_str}.csv"
    Path(file_name).parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(file_name, index=False)


def cleanup(model):
    del model
    torch.cuda.empty_cache()
    gc.collect()


def get_memory_metrics():
    return torch.cuda.memory_allocated(), torch.cuda.memory_reserved()


def do_expermient_fdata(
    experiment_name,
    models,
    tasks,
):
    do_expermient(
        experiment_name,
        models,
        tasks,
        quant_dir="/fdata/llm/mxq/snapshots",
        result_dir="/fdata/llm/mxq/results",
        log_dir="/fdata/llm/mxq/logs",
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
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS[9:],
        },
    }
    do_expermient_fdata(f"{type}_{algo}_mxq", models, tasks)


def experiment_quant_awq():
    models = [ALL_MODELS[0], ALL_MODELS[2]]
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
    models = ALL_MODELS
    type = "eval_ppl"
    algo = "awq"
    tasks = {
        algo: {
            "type": type,
            "configs": AUTOAWQ_CONFIGS[0:2],
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
    models = ALL_MODELS[0:1]
    type = "eval_leaderboard"
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS[1:2],
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
    experiment_ppl_eval_hqq()


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
