#!/usr/bin/env python3


import re

import pandas as pd
import torch
from hqq.core.quantize import Quantizer as hQuant
from safetensors.torch import save_file as safe_save
from scipy.stats import kurtosis
from torch import uint8

from lm_quant_toolkit.utils.safetensors import get_tensor, get_tensor_dual


def load_weight(matrix_name, base_dir):
    m = f"{matrix_name}.weight"
    return get_tensor(m, base_dir)


def load_weight_dual(prefix, base_dir, st_file):
    o = f"{prefix}.weight"
    q = f"{prefix}.qweight"
    return get_tensor_dual(o, q, base_dir, st_file)


def dequantize(wq, meta):
    # Zero/Scale packed together
    if "zero_scale" in meta:
        zero_scale = meta["zero_scale"]

        if zero_scale.dtype == uint8:
            meta["zero_q"], meta["scale_q"] = zero_scale[0], zero_scale[1]
        else:
            meta["zero"], meta["scale"] = zero_scale[0], zero_scale[1]

    if meta["quant_zero"]:
        meta["zero"] = hQuant.dequantize(meta["zero_q"], meta["meta_zero"])

    if meta["quant_scale"]:
        meta["scale"] = hQuant.dequantize(meta["scale_q"], meta["meta_scale"])
    return hQuant.dequantize(wq, meta)


def restore_weight(matrix, state_dict):
    key = matrix
    if key in state_dict:
        m_dikt = state_dict[key]
        if "meta" in m_dikt:
            meta_dict = m_dikt["meta"]
            meta_scale_dict = meta_dict.get("meta_scale", None)
            b1 = meta_dict["nbits"]
            g1 = meta_dict["group_size"]
            b2 = meta_scale_dict["nbits"] if meta_scale_dict else 8
            g2 = meta_scale_dict["group_size"] if meta_scale_dict else 128
            quant_config = {
                "b1": b1,
                "g1": g1,
                "b2": b2,
                "g2": g2,
            }
            wq = dequantize(m_dikt["W_q"], meta_dict)
            return wq, quant_config
        else:
            return None, None
    else:
        return None, None


def save_compare_pair(
    base_dir, quant_base_dir, quant_cfg, model_id, layers, output_dir
):
    file_path = f"{quant_base_dir}/{model_id}-{quant_cfg}-hqq/qmodel.pt"
    state_dict = torch.load(file_path, map_location="cpu")

    tensors = {}
    metadata = {}
    # walk thru the linear layers
    # for each layer
    # for each linear module
    # load the original weight
    # load the quantized weight and dequantized
    # save the two matrix into a combined safetensors
    for layer in range(layers):
        matricies = [
            f"model.layers.{layer}.mlp.down_proj",
            f"model.layers.{layer}.mlp.gate_proj",
            f"model.layers.{layer}.mlp.up_proj",
            f"model.layers.{layer}.self_attn.k_proj",
            f"model.layers.{layer}.self_attn.o_proj",
            f"model.layers.{layer}.self_attn.q_proj",
            f"model.layers.{layer}.self_attn.v_proj",
        ]
        for matrix in matricies:
            wq, quant_cfg = restore_weight(matrix, state_dict)
            if wq is None:
                # skip unquantized matrix
                continue
            wo = load_weight(matrix, base_dir)
            tensors[f"{matrix}.weight"] = wo
            tensors[f"{matrix}.qweight"] = wq
            metadata[f"{matrix}.quant_cfg.b1"] = str(quant_cfg["b1"])
            metadata[f"{matrix}.quant_cfg.b2"] = str(quant_cfg["b2"])
            metadata[f"{matrix}.quant_cfg.g1"] = str(quant_cfg["g1"])
            metadata[f"{matrix}.quant_cfg.g2"] = str(quant_cfg["g2"])

    output_fp = f"{output_dir}/{model_id}-cmp.safetensors"
    safe_save(tensors, output_fp, metadata=metadata)


def compare_pair(model_id, layers, output_dir):
    st_file = f"{output_dir}/{model_id}-cmp.safetensors"
    for layer in range(layers):
        matricies = [
            f"model.layers.{layer}.mlp.down_proj",
            f"model.layers.{layer}.mlp.gate_proj",
            f"model.layers.{layer}.mlp.up_proj",
            f"model.layers.{layer}.self_attn.k_proj",
            f"model.layers.{layer}.self_attn.o_proj",
            f"model.layers.{layer}.self_attn.q_proj",
            f"model.layers.{layer}.self_attn.v_proj",
        ]
        for matrix in matricies:
            wo, wq = load_weight_dual(matrix, output_dir, st_file)
            diff = torch.norm(wo - wq).item()
            kurt_peason = kurtosis(
                wo.numpy(), axis=None, fisher=False, bias=True, nan_policy="omit"
            )
            kurt_fisher = kurtosis(
                wo.numpy(), axis=None, fisher=True, bias=True, nan_policy="omit"
            )
            # print(f"{matrix} FNorm Diff: {diff:.5f} Kurtosis: {kurt:.2f}")
            print(f"{matrix},{diff:.5f},{kurt_fisher:.3f},{kurt_peason:.3f}")


def is_linear_module(key):
    self_attns = ["q_proj", "v_proj", "k_proj", "o_proj"]
    mlps = ["gate_proj", "up_proj", "down_proj"]
    modules = self_attns + mlps
    for module in modules:
        if module in key:
            return True
    return False


def extract_quant_config(base_dir, model_id, config, algo="hqq"):
    file_path = f"{base_dir}/{model_id}-{config}-{algo}/qmodel.pt"
    dikt = torch.load(file_path, map_location="cpu")
    quant_configs = {}
    mem_fp16_all_total = 0
    mem_all_total = 0
    mem_quant_total = 0
    # search quantized linear module with meta
    for key in dikt.keys():
        m_dikt = dikt[key]
        if is_linear_module(key):
            if "meta" in m_dikt:
                meta_dict = m_dikt["meta"]
                meta_scale_dict = meta_dict.get("meta_scale", None)
                shape = meta_dict["shape"]
                b1 = meta_dict["nbits"]
                g1 = meta_dict["group_size"]
                b2 = meta_scale_dict["nbits"] if meta_scale_dict else 8
                g2 = meta_scale_dict["group_size"] if meta_scale_dict else 128
                memmb = (
                    (b1 + 2 * b2 / (g1 * g2)) * shape[0] * shape[1] / 8 / 1024 / 1024
                )
                mem_fp16_all_total += shape[0] * shape[1] * 2 / 1024 / 1024
                mem_quant_total += memmb
                mem_all_total += memmb
                quant_configs[key] = {
                    "b1": b1,
                    "g1": g1,
                    "b2": b2,
                    "g2": g2,
                    "memmb": memmb,
                }
        else:
            w = m_dikt["weight"]
            mem_all_total += w.numel() * 2 / 1024 / 1024
            mem_fp16_all_total += w.numel() * 2 / 1024 / 1024
    return quant_configs, mem_quant_total, mem_all_total, mem_fp16_all_total


def get_mem_usage_df(model_ids, confs, base_dir):
    dikts = []
    for model_id in model_ids:
        for conf in confs:
            configs, mem_quant_total, mem_all_total, mem_fp16_all_total = (
                extract_quant_config(base_dir, model_id, conf)
            )
            dikt = {
                "model": model_id.split("/")[1],
                "config": conf,
                "mem_quant_total": mem_quant_total,
                "mem_all_total": mem_all_total,
                "mem_fp16_all_total": mem_fp16_all_total,
            }
            dikts.append(dikt)
    df = pd.DataFrame(dikts)
    return df


if __name__ == "__main__":
    base_dir = "/fdata/llm/mxq/snapshots/"
    model_ids = [
        "meta-llama/Llama-2-7b-hf",
        # "meta-llama/Llama-2-13b-hf",
        # "meta-llama/Meta-Llama-3-8B",
    ]
    confs = [
        "4_51",
        "4_25",
        "4_13",
    ]

    dikt = []
    pat = re.compile(r"model\.layers\.(\d+)\.(.+)")
    for model_id in model_ids:
        for conf in confs:
            configs, mem_quant_total, mem_all_total, mem_fp16_all_total = (
                extract_quant_config(base_dir, model_id, conf, algo="mxq")
            )
            for key, val in configs.items():
                matcher = re.match(pat, key)
                if matcher:
                    layer = matcher.group(1)
                    module = matcher.group(2)
                    # val["model"] = model_id.split("/")[1]
                    val["layer"] = layer
                    val["module"] = module
                    val["bit_budget"] = conf.replace("_", ".")
                    dikt.append(val)
    df = pd.DataFrame(dikt)
    df.to_csv("llama-mxq-cfgs.csv", index=False)


# if __name__ == "__main__":
#     output_dir = "snapshots/cmp"
#     model_id = "meta-llama/Llama-2-7b-hf"
#     model = LLAMA_MODELS[model_id]
#     compare_pair(model_id.split("/")[1], model["layers"], output_dir)


# if __name__ == "__main__":
#     # Llama-2-7b-hf-b3g128-hqq:
#     # Llama-2-7b-hf-b3g32-hqq:
#     # Llama-2-7b-hf-b3g64-hqq:
#     # Llama-2-7b-hf-b4g128-hqq:
#     # Llama-2-7b-hf-b4g32-hqq:
#     # Llama-2-7b-hf-b4g64-hqq:
#
#     quant_cfg = "b4g64"
#     quant_base_dir = "/data/gqq-eval"
#     output_dir = "snapshots/cmp"
#     model_id = "meta-llama/Llama-2-7b-hf"
#     model = LLAMA_MODELS[model_id]
#     if model["base_dir"]:
#         model_base_dir = get_hf_model_storge_base_dir(
#             model_id, hf_hub_dir=LLAMA_MODELS["base_dir"]
#         )
#     else:
#         model_base_dir = get_hf_model_storge_base_dir(model_id)
#     save_compare_pair(
#         model_base_dir,
#         quant_base_dir,
#         quant_cfg,
#         model_id.split("/")[1],
#         model["layers"],
#         output_dir,
#     )
