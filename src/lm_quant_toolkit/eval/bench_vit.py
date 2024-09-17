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

from lm_quant_toolkit.eval.zeroshot import eval_zeroshot_classification

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


def _dump_cuda_mem_snapshot(experiment_name, model_id, algo, result_dir):
    mem_fp = f"{result_dir}/{experiment_name}/mem-snapshot-{algo}-{model_id.split('/')[1]}.pickle"
    torch.cuda.memory._dump_snapshot(mem_fp)


def do_expermient(
    experiment_name,
    models,
    tasks,
    quant_dir="snapshots",
    result_dir="results",
    log_dir="logs",
    track_cuda_memory=False,
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
        config = [c for c in spec["configs"] if c[0] == cfg][0]
        metric = _init_metrics(model_id, algo, config)
        print("*" * 72)
        # if task_type == "quant":
        #     print(f"Quantizing {algo} on {model_id} w/ config: {cfg}...")
        # elif task_type == "eval_zero_shot":
        if task_type == "eval_zero_shot":
            print(
                f"Evaluating {algo} zero-shot classification on {model_id} w/ config: {cfg}..."
            )
        else:
            print(f"Evaluating {algo} linear-probe on {model_id} w/ config: {cfg}...")
        print("*" * 72)

        if track_cuda_memory:
            torch.cuda.memory._record_memory_history()
        _reset_peak_memory_stats()
        if task_type == "eval_zero_shot":
            if algo == "fp16":
                metric = eval_zeroshot_classification(metric, model_id, False)
            else:
                # avoid interventions between models
                quant_config = copy.deepcopy(config[1])
                if cfg.startswith("mxq-") and model_id in QUANT_METRICS_FILE_MAP:
                    quant_config["quant_metrics_file"] = QUANT_METRICS_FILE_MAP[
                        model_id
                    ]
                metric = eval_zeroshot_classification(
                    metric, model_id, True, quant_config
                )
            metric["zeroshot_mem_allot"], metric["zeroshot_mem_reserved"] = (
                get_memory_metrics()
            )
            if track_cuda_memory:
                _dump_cuda_mem_snapshot(experiment_name, model_id, algo, result_dir)
        elif task_type == "eval_linear_probe":
            # TODO: implement linear probe here later
            pass
        save_partial_metric(experiment_name, algo, model_id, cfg, metric, result_dir)
        persist_progress(model_id, cfg, algo, task_type, progress_path)
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
        "acc1_zeroshot_cls": 0,
        "acc5_zeroshot_cls": 0,
        "recall_zeroshot_cls": 0,
        "duration_zeroshot_cls": 0,
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
    type = "eval_zero_shot"
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


def experiment_zeroshot_eval_hqq():
    models = ALL_MODELS
    type = "eval_zero_shot"
    algo = "hqq"
    tasks = {
        algo: {
            "type": type,
            "configs": HQQ_CONFIGS,
        },
    }
    do_expermient_fdata(f"{type}_{algo}", models, tasks)


def experiment_quant_zeroshot_eval_mxq_comprise():
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
    zeroshot_tasks = {
        "hqq": {
            "type": "eval_zero_shot",
            "configs": equiv_mxq_configs,
        },
    }
    do_expermient_fdata("quant_mxq_compromise", models, quant_tasks)
    # do_expermient_fdata("eval_mxq_compromise", models, zeroshot_tasks)


def main():
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(name)s] %(message)s",
        level=logging.INFO,
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # experiment_quant_hqq()
    # experiment_quant_mxq()
    # experiment_zeroshot_eval_fp16()
    experiment_zeroshot_eval_hqq()
    # experiment_quant_zeroshot_eval_mxq_comprise()


if __name__ == "__main__":
    main()
