import gc
import glob
import os
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig

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
