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
    ("b8g32", HQQQuantConfig(nbits=8, group_size=32, quant_scale=True)),
    ("b8g64", HQQQuantConfig(nbits=8, group_size=64, quant_scale=True)),
    ("b8g128", HQQQuantConfig(nbits=8, group_size=128, quant_scale=True)),
    ("b4g32", HQQQuantConfig(nbits=4, group_size=32, quant_scale=True)),
    ("b4g64", HQQQuantConfig(nbits=4, group_size=64, quant_scale=True)),
    ("b4g128", HQQQuantConfig(nbits=4, group_size=128, quant_scale=True)),
    ("b3g32", HQQQuantConfig(nbits=3, group_size=32, quant_scale=True)),
    ("b3g64", HQQQuantConfig(nbits=3, group_size=64, quant_scale=True)),
    ("b3g128", HQQQuantConfig(nbits=3, group_size=128, quant_scale=True)),
    ("b2g16", HQQQuantConfig(nbits=2, group_size=16, quant_scale=True)),
    ("b2g32", HQQQuantConfig(nbits=2, group_size=32, quant_scale=True)),
    ("b2g64", HQQQuantConfig(nbits=2, group_size=64, quant_scale=True)),
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


def plan_eval_bit_budgets(
    model_arch="ViT",
    points=5,
    step=1,
    bases=[4.51],
    include_base=False,
):
    for base in bases:
        ideals, solvables = get_eval_plan(model_arch, base, points, step, include_base)
        print("*" * 72)
        print(f"base: {base}")
        for t in zip(ideals, solvables):
            print(f"ideal: {t[0]:.2f}, solvable: {t[1]:.2f}")
        print("*" * 72)


def get_eval_plan(model_arch, base, points, step, include_base):
    ideals = []
    solvables = []
    start = 0 if include_base else 1
    for point in range(start, points + 1):
        tentative = round(base + point * step, 2)
        ideals.append(tentative)
        ret = try_solvable(model_arch, tentative, step)
        if ret is not None:
            solvables.append(ret)
        else:
            solvables.append(0.0)
    return ideals, solvables


def try_solvable(model_arch, bit_budget, step):
    if model_arch == "ViT":
        model_ids = VIT_OPENCLIP_MODELS.keys()
    else:
        model_ids = [
            key for key in LLAMA_MODELS if LLAMA_MODELS[key].get("experiment", False)
        ]

    dikt = {}
    feasible_budget = round(bit_budget, 2)
    for model_id in model_ids:
        attempts = 1
        _, fp = get_mxq_quant_meta_data_file(model_id)
        while True:
            try:
                # find_optimal_configs(fp, feasible_budget, time_limit=200)
                find_optimal_configs(
                    fp,
                    feasible_budget,
                    time_limit=200,
                    weight_algo="sensi-milp",
                )
                dikt[model_id] = feasible_budget
                break
            except ValueError:
                print(f"Warning: {feasible_budget:.2f} unsolvable for model {model_id}")
                if attempts > 3:
                    return None
                feasible_budget += -0.01 if step < 0 else 0.01
            attempts += 1
    if len(set(dikt.values())) > 1:
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


def debug_milp_solvable(model_id, bit_budget):
    _, fp = get_mxq_quant_meta_data_file(model_id)
    configs = find_optimal_configs(fp, bit_budget, time_limit=200)
    print(configs)


def plan_432_bits():
    bases = [4.51, 4.25, 4.13]
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=0.02, bases=bases, include_base=True
    )
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=-0.02, bases=bases, include_base=True
    )
    bases = [3.51, 3.25, 3.13, 2.51, 2.25, 2.13]
    plan_eval_bit_budgets(
        model_arch="llm", points=3, step=0.02, bases=bases, include_base=True
    )
    plan_eval_bit_budgets(
        model_arch="llm", points=3, step=-0.02, bases=bases, include_base=True
    )


def plan_567_bits():
    best_bit_budget = calc_bits(8, 32, 8, 128)
    save_objs = [10, 20, 30, 40]
    bases = [best_bit_budget * (100 - obj) / 100 for obj in save_objs]
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=0.02, bases=bases, include_base=True
    )
    plan_eval_bit_budgets(
        model_arch="llm", points=5, step=-0.02, bases=bases, include_base=True
    )


def fill_budget_gap(start, stop, step=0.02):
    result = []
    d = start + step
    while d < stop:
        result.append(round(d, 2))
        d += step
    return result


def fill_gaps():
    gaps = []
    # gaps.extend(fill_budget_gap(3.57, 4.03))
    # gaps.extend(fill_budget_gap(3.29, 3.45))
    # gaps.extend(fill_budget_gap(6.92, 7.56))
    # gaps.extend(fill_budget_gap(4.61, 5.00))
    # gaps.extend(fill_budget_gap(5.20, 5.86))
    gaps.extend(fill_budget_gap(6.01, 6.70))
    # gaps.extend([4.39, 4.37])
    print(gaps)
    plan_eval_bit_budgets(
        model_arch="llm", points=0, step=0.02, bases=gaps, include_base=True
    )


if __name__ == "__main__":
    # fill_gaps()
    plan_432_bits()
    plan_567_bits()
