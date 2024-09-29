import gc
import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig
from hqq.utils.optimizer import find_optimal_configs

from lm_quant_toolkit.utils.hub import LLAMA_MODELS, VIT_OPENCLIP_MODELS

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


def get_mxq_quant_meta_data_file(model_id):
    short_id = model_id.split("/")[1]
    data_dir = os.path.join(os.path.dirname(__file__), "..", "..", "data")
    fp = os.path.join(data_dir, f"fnorm-{short_id}.csv")
    return os.path.exists(fp), os.path.abspath(fp)


def persist_progress(df, progress_path):
    """Save the progress of experiment for resumption."""
    Path(progress_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(progress_path, index=False)


def save_partial_metric(experiment_name, algo, model_id, config, metric, result_dir):
    metrics = [metric]
    df = pd.DataFrame(metrics)
    result_dir = os.path.join(result_dir, experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    model_short_id = model_id.split("/")[1]
    file_name = f"{result_dir}/partial-{algo}-{model_short_id}-{config}.csv"
    df.to_csv(file_name, index=False)


def _dump_cuda_mem_snapshot(experiment_name, model_id, algo, result_dir):
    mem_fp = f"{result_dir}/{experiment_name}/mem-snapshot-{algo}-{model_id.split('/')[1]}.pickle"
    torch.cuda.memory._dump_snapshot(mem_fp)


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


def plan_eval_bit_budgets(model_arch="ViT", points=5, desc=True):
    bases = [4.51]
    for base in bases:
        ideals, solvables = get_eval_plan(model_arch, base, points, desc)
        print("*" * 72)
        print(f"base: {base}")
        for t in zip(ideals, solvables):
            print(f"ideal: {t[0]:.2f}, solvable: {t[1]:.2f}")
        print("*" * 72)


def get_eval_plan(model_arch, base, points, desc):
    ideals = []
    solvables = []
    for point in range(1, points + 1):
        pct = 100 - point if desc else 100 + point
        tentative = round(base * pct / 100, 2)
        ideals.append(tentative)
        ret = try_solvable(model_arch, tentative, desc)
        if ret is not None:
            solvables.append(ret)
        else:
            solvables.append(0.0)
    return ideals, solvables


def try_solvable(model_arch, bit_budget, desc):
    model_ids = (
        VIT_OPENCLIP_MODELS.keys() if model_arch == "ViT" else LLAMA_MODELS.keys()
    )

    dikt = {}
    feasible_budget = round(bit_budget, 2)
    for model_id in model_ids:
        attempts = 1
        _, fp = get_mxq_quant_meta_data_file(model_id)
        while True:
            try:
                find_optimal_configs(fp, feasible_budget, time_limit=200)
                dikt[model_id] = feasible_budget
                break
            except ValueError:
                print(f"Warning: {feasible_budget:.2f} unsolvable for model {model_id}")
                if attempts > 3:
                    break
                feasible_budget += -0.01 if desc else 0.01
            attempts += 1
    if len(set(dikt.values())) > 1:
        print(dikt)
        for model_id, budget in dikt.items():
            if abs(budget - feasible_budget) > 0.01:
                try:
                    fp = get_mxq_quant_meta_data_file(model_id)
                    find_optimal_configs(fp, feasible_budget, time_limit=200)
                except ValueError:
                    print(
                        f"Warning: {feasible_budget:.2f} unsolvable for model {model_id}"
                    )
                    return None
    return feasible_budget


if __name__ == "__main__":
    plan_eval_bit_budgets(points=5, desc=True)
    plan_eval_bit_budgets(points=10, desc=False)
