import copy
import gc
import glob
import logging
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig

from lm_quant_toolkit.eval.clipbenchmark import (
    eval_linear_probe,
    eval_zeroshot_classification,
)

ALL_MODELS = [
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K",
]

QUANT_METRICS_FILE_MAP = {
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": "data/fnorm-CLIP-ViT-B-32-laion2B-s34B-b79K.csv",
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": "data/fnorm-CLIP-ViT-H-14-laion2B-s32B-b79K.csv",
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": "data/fnorm-CLIP-ViT-L-14-laion2B-s32B-b82K.csv",
}

HQQ_CONFIGS = [
    ("b8g32", HQQQuantConfig(nbits=8, group_size=32)),
    ("b8g64", HQQQuantConfig(nbits=8, group_size=64)),
    ("b8g128", HQQQuantConfig(nbits=8, group_size=128)),
    ("b4g32", HQQQuantConfig(nbits=4, group_size=32)),
    ("b4g64", HQQQuantConfig(nbits=4, group_size=64)),
    ("b4g128", HQQQuantConfig(nbits=4, group_size=128)),
    ("b3g32", HQQQuantConfig(nbits=3, group_size=32)),
    ("b3g64", HQQQuantConfig(nbits=3, group_size=64)),
    ("b3g128", HQQQuantConfig(nbits=3, group_size=128)),
    ("b2g16", HQQQuantConfig(nbits=2, group_size=16)),
    ("b2g32", HQQQuantConfig(nbits=2, group_size=32)),
    ("b2g64", HQQQuantConfig(nbits=2, group_size=64)),
]

MXQ_CONFIGS = [
    (
        f"{bits:.2f}".replace(".", "_"),
        HQQQuantConfig(mixed=True, budget=bits, quant_scale=True),
    )
    for bits in [
        7.80,
        7.72,
        7.64,
        7.56,
        7.48,
        7.40,
        7.32,
        5.00,
        4.95,
        4.90,
        4.86,
        4.82,
        4.78,
        4.73,
        4.00,
        3.96,
        3.92,
        3.00,
    ]
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


def persist_progress(df, progress_path):
    """Save the progress of experiment for resumption."""
    Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(progress_path, index=False)


def _dump_cuda_mem_snapshot(experiment_name, model_id, algo, result_dir):
    mem_fp = f"{result_dir}/{experiment_name}/mem-snapshot-{algo}-{model_id.split('/')[1]}.pickle"
    torch.cuda.memory._dump_snapshot(mem_fp)


def do_expermient(
    experiment_name,
    models,
    tasks,
    quant_dir="snapshots",
    result_dir="results",
    track_cuda_memory=False,
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
        config = [c for c in spec["configs"] if c[0] == cfg][0]
        metric = _init_metrics(model_id, algo, config)
        print("*" * 72)
        # if task_type == "quant":
        #     print(f"Quantizing {algo} on {model_id} w/ config: {cfg}...")
        # elif task_type == "eval_zeroshot_cls":
        if task_type == "eval_zeroshot_cls":
            print(
                f"Evaluating {algo} zero-shot classification on {model_id} w/ config: {cfg}..."
            )
        else:
            print(f"Evaluating {algo} linear-probe on {model_id} w/ config: {cfg}...")
        print("*" * 72)

        if track_cuda_memory:
            torch.cuda.memory._record_memory_history()
        _reset_peak_memory_stats()

        quant_config = None
        if algo != "fp16":
            quant_config = copy.deepcopy(config[1])
            if algo == "mxq" and model_id in QUANT_METRICS_FILE_MAP:
                quant_config["quant_metrics_file"] = QUANT_METRICS_FILE_MAP[model_id]
        if task_type == "eval_zeroshot_cls":
            # avoid interventions between models
            metric = eval_zeroshot_classification(
                metric,
                model_id,
                result_dir,
                quant_dir,
                quant_config,
            )
            (
                metric["zeroshot_mem_allot"],
                metric["zeroshot_mem_reserved"],
            ) = get_memory_metrics()
        else:
            # Make model with different quantization configs don't share the
            # pre-calated features vectors, thus they are evaluated
            # separately.
            feature_root = os.path.join(quant_dir, "features", cfg)
            Path(feature_root).mkdir(parents=True, exist_ok=True)
            metric = eval_linear_probe(
                metric,
                model_id,
                result_dir,
                quant_dir,
                quant_config,
                feature_root,
            )
            (
                metric["linear_probe_mem_allot"],
                metric["linear_probe_mem_reserved"],
            ) = get_memory_metrics()

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
    quant_config = copy.deepcopy(config[1])
    mixed = quant_config.get("mixed", False)
    if mixed:
        quant_config.pop("weight_quant_params", None)
        quant_config.pop("scale_quant_params", None)
        quant_config.pop("zero_quant_params", None)

    return {
        "model": model_id.split("/")[1],
        "algo": algo,
        "config": config[0],
        "config_detail": quant_config,
        "quant_duration": 0,
        "load_mem_allot": 0,
        "load_mem_reserved": 0,
        "zeroshot_mem_allot": 0,
        "zeroshot_mem_reserved": 0,
        "linear_probe_mem_allot": 0,
        "linear_probe_mem_reserved": 0,
        "acc1_zeroshot_cls": 0,
        "acc5_zeroshot_cls": 0,
        "recall_zeroshot_cls": 0,
        "duration_zeroshot_cls": 0,
        "acc1_linear_probe": 0,
        "acc5_linear_probe": 0,
        "recall_linear_probe": 0,
        "duration_linear_probe": 0,
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


def calc_bits(b1, g1, b2=8, g2=128):
    return b1 + 2 * b2 / g1 + 32 / g1 / g2


def get_mxq_bits(reduction_pcts=[3, 5, 8]):
    nbits = []
    for cfg in HQQ_CONFIGS:
        bpp = calc_bits(
            cfg[1]["weight_quant_params"]["nbits"],
            cfg[1]["weight_quant_params"]["group_size"],
            8,
            128,
        )
        nbits.extend([round(bpp * (1 - pct / 100), 2) for pct in reduction_pcts])
    return sorted(list(set(nbits)), reverse=True)


def _reset_peak_memory_stats():
    return torch.cuda.reset_peak_memory_stats()


def get_memory_metrics():
    return torch.cuda.max_memory_allocated(), torch.cuda.max_memory_reserved()


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
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS[9:],
        },
    }
    do_expermient_fdata(f"{type}_{algo}_vit", models, tasks)


########################################################################
#  zero-shot image classification evaluation experiments
########################################################################


def experiment_zeroshot_eval_fp16():
    models = ALL_MODELS
    type = "eval_zeroshot_cls"
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


def experiment_eval_fp16_combined():
    models = ALL_MODELS
    algo = "fp16"
    tasks = {
        algo: {
            "type": "eval_zeroshot_cls",
            "configs": [
                ("base", {}),
            ],
        },
    }
    linear_probe_tasks = {
        algo: {
            "type": "eval_linear_probe",
            "configs": [
                ("base", {}),
            ],
        },
    }
    do_expermient_fdata(f"eval_linear_probe_{algo}2", models, linear_probe_tasks)
    do_expermient_fdata(f"eval_zeroshot_cls_{algo}2", models, tasks)


def experiment_eval_hqq_comprehensive():
    models = ALL_MODELS
    tasks = {
        "hqq": {
            "type": "eval_zeroshot_cls",
            "configs": HQQ_CONFIGS,
        }
    }
    linear_probe_tasks = {
        "hqq": {
            "type": "eval_linear_probe",
            "configs": HQQ_CONFIGS,
        },
    }
    # do_expermient_fdata("eval_lp_hqq_comprehensive4", models, linear_probe_tasks)
    do_expermient_fdata("eval_zs_hqq_comprehensive5", models, tasks)


def experiment_eval_mxq_358_memory_saving():
    nbits = get_mxq_bits()
    # budget 2.07 is infeasible
    nbits = [bit for bit in nbits if bit > 2.07]
    mxq_configs = [
        (
            f"{bits:.2f}".replace(".", "_"),
            HQQQuantConfig(mixed=True, budget=bits, quant_scale=True),
        )
        for bits in nbits
    ]
    models = ALL_MODELS
    tasks = {
        "mxq": {
            "type": "eval_zeroshot_cls",
            "configs": mxq_configs,
        }
    }
    linear_probe_tasks = {
        "mxq": {
            "type": "eval_linear_probe",
            "configs": mxq_configs,
        },
    }
    # do_expermient_fdata("eval_lp_mxq_358_memory_saving4", models, linear_probe_tasks)
    do_expermient_fdata("eval_zs_mxq_358_memory_saving5", models, tasks)


def experiment_eval_mxq_comprehensive():
    models = ALL_MODELS
    zeroshot_tasks = {
        "mxq": {
            "type": "eval_zeroshot_cls",
            "configs": MXQ_CONFIGS,
        },
    }
    linear_probe_tasks = {
        "mxq": {
            "type": "eval_linear_probe",
            "configs": MXQ_CONFIGS,
        },
    }
    do_expermient_fdata("eval_zs_mxq_comprehensive2", models, zeroshot_tasks)
    do_expermient_fdata("eval_lp_mxq_comprehensive2", models, linear_probe_tasks)


def experiment_zeroshot_eval_mxq():
    models = ALL_MODELS
    type = "eval_zeroshot_cls"
    tasks = {
        "mxq": {
            "type": type,
            "configs": MXQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_mxq", models, tasks)


def experiment_eval_mxq_combined():
    models = ALL_MODELS
    equiv_mxq_configs = [
        (
            f"{bits:.2f}".replace(".", "_"),
            HQQQuantConfig(mixed=True, budget=bits, quant_scale=True),
        )
        for bits in [4.06, 4.10, 4.15, 4.19, 4.24, 4.28, 4.33]
    ]
    zeroshot_tasks = {
        "mxq": {
            "type": "eval_zeroshot_cls",
            "configs": equiv_mxq_configs,
        },
    }
    linear_probe_tasks = {
        "mxq": {
            "type": "eval_linear_probe",
            "configs": equiv_mxq_configs,
        },
    }
    do_expermient_fdata("eval_linear_probe_mxq2", models, linear_probe_tasks)
    do_expermient_fdata("eval_zeroshot_cls_mxq2", models, zeroshot_tasks)


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # experiment_quant_hqq()
    # experiment_quant_mxq()
    # experiment_zeroshot_eval_fp16()
    # experiment_zeroshot_eval_mxq()
    # experiment_eval_mxq_combined()
    # experiment_eval_fp16_combined()
    experiment_eval_hqq_comprehensive()
    experiment_eval_mxq_358_memory_saving()


if __name__ == "__main__":
    main()
