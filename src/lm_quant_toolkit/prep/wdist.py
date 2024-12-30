import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from lm_quant_toolkit.utils.hub import (
    LLAMA_MODELS,
    VIT_OPENCLIP_MODELS,
    get_hf_model_storge_base_dir,
)
from lm_quant_toolkit.utils.pickle import load_state_dict
from lm_quant_toolkit.utils.safetensors import get_tensor


def calculate_kurtosis_llm(model_id, base_dir, layers, output_dir):
    modules = [
        "self_attn.q_proj",
        "self_attn.k_proj",
        "self_attn.v_proj",
        "self_attn.o_proj",
        "mlp.gate_proj",
        "mlp.down_proj",
        "mlp.up_proj",
    ]
    dikts = []
    for module in modules:
        for layer in range(layers):
            full_name = f"model.layers.{layer}.{module}.weight"
            w = get_tensor(full_name, base_dir)
            param_count = w.numel()
            w = w.flatten().float().numpy()
            kurt_pearson = kurtosis(
                w, axis=None, fisher=False, bias=True, nan_policy="omit"
            )
            dikt = {
                "module": module,
                "layer": layer,
                "param_count": param_count,
                "kurtosis": kurt_pearson,
            }
            dikts.append(dikt)
    df = pd.DataFrame(dikts)
    short_id = model_id.split("/")[1]
    csv_fp = f"{output_dir}/kurtosis-{short_id}.csv"
    df.to_csv(csv_fp, index=False)


def summarize_vit_percentiles(
    model_id,
    model_cfg,
    csv_fp,
    wt_file="open_clip_pytorch_model.bin",
):
    modules = [
        "attn.in_proj_bias",
        "attn.in_proj_weight",
        "attn.out_proj.bias",
        "attn.out_proj.weight",
        "ln_1.bias",
        "ln_1.weight",
        "ln_2.bias",
        "ln_2.weight",
        "mlp.c_fc.bias",
        "mlp.c_fc.weight",
        "mlp.c_proj.bias",
        "mlp.c_proj.weight",
    ]
    dikts = []
    state_dict = load_state_dict(model_id)
    for layer_type, layers in model_cfg.items():
        if layer_type == "vlayers":
            model_type = "vision"
            prefix = "visual.transformer"
        else:
            model_type = "text"
            prefix = "transformer"
        for i, module in enumerate(modules):
            for layer in range(layers):
                full_name = f"{prefix}.resblocks.{layer}.{module}"
                w = state_dict[full_name]
                param_count = w.numel()
                w = w.flatten().float().numpy()
                percentiles = np.percentile(np.abs(w), [0, 99, 99.9, 99.99, 100])
                kurt_pearson = kurtosis(
                    w, axis=None, fisher=False, bias=True, nan_policy="omit"
                )
                dikt = {
                    "type": model_type,
                    "module": module,
                    "param_count": param_count,
                    "layer": layer,
                    "percentile_0": percentiles[0],
                    "percentile_99": percentiles[1],
                    "percentile_999": percentiles[2],
                    "percentile_9999": percentiles[3],
                    "percentile_100": percentiles[4],
                    "kurtosis": kurt_pearson,
                }
                dikts.append(dikt)
    df = pd.DataFrame(dikts)
    df.to_csv(csv_fp, index=False)


def summarize_llama_quantable_params(base_dir, layers, fp, llama2=True):
    modules = {
        "norm": {"layerwise": False, "quant": False},
        "lm_head": {"layerwise": False, "quant": False, "prefix": ""},
        "embed_tokens": {"layerwise": False, "quant": False},
        "input_layernorm": {"layerwise": True, "quant": False},
        "post_attention_layernorm": {"layerwise": True, "quant": False},
        "mlp.down_proj": {"layerwise": True, "quant": True},
        "mlp.gate_proj": {"layerwise": True, "quant": True},
        "mlp.up_proj": {"layerwise": True, "quant": True},
        "self_attn.k_proj": {"layerwise": True, "quant": True},
        "self_attn.o_proj": {"layerwise": True, "quant": True},
        "self_attn.q_proj": {"layerwise": True, "quant": True},
        "self_attn.v_proj": {"layerwise": True, "quant": True},
    }
    if llama2:
        modules["self_attn.rotary_emb"] = {
            "layerwise": True,
            "quant": False,
            "suffix": "inv_freq",
        }
    dikts = []
    for module, data in modules.items():
        suffix = data.get("suffix", "weight")
        if data["layerwise"]:
            for layer in range(layers):
                full_name = f"model.layers.{layer}.{module}.{suffix}"
                w = get_tensor(full_name, base_dir)
                param_count = w.numel()
                dikt = {
                    "module": module,
                    "layer": layer,
                    "param_count": param_count,
                    "quant_count": param_count if data["quant"] else 0,
                }
                dikts.append(dikt)
        else:
            prefix = data.get("prefix", "model")
            if prefix == "":
                full_name = f"{module}.{suffix}"
            else:
                full_name = f"{prefix}.{module}.{suffix}"
            w = get_tensor(full_name, base_dir)
            param_count = w.numel()
            dikt = {
                "module": module,
                "layer": 0,
                "param_count": param_count,
                "quant_count": param_count if data["quant"] else 0,
            }
            dikts.append(dikt)
    df = pd.DataFrame(dikts)
    df.to_csv(fp, index=False)


def summarize_vit_quantable_params(
    model_id,
    csv_fp,
    wt_file="open_clip_pytorch_model.bin",
):
    quantables = [
        "mlp.c_fc.weight",
        "mlp.c_proj.weight",
    ]
    dikts = []
    state_dict = load_state_dict(model_id)
    pat = r"(.*)\.resblocks\.(\d+)\.(.*)"
    for key in state_dict:
        m = re.match(pat, key)
        if m is not None:
            layer = int(m.group(2))
            if "visual" in m.group(1):
                type = "vision"
            elif "transformer" in m.group(1):
                type = "text"
            else:
                type = ""
            module = m.group(3)
        else:
            layer = 0
            module = key
            if "visual" in key:
                type = "vision"
            elif "transformer" in key:
                type = "text"
            else:
                type = ""
        quantable = any([qnt in key for qnt in quantables])
        w = state_dict[key]
        param_count = w.numel()
        dikt = {
            "type": type,
            "module": module,
            "layer": layer,
            "param_count": param_count,
            "quant_count": param_count if quantable else 0,
        }
        dikts.append(dikt)
    df = pd.DataFrame(dikts)
    df.to_csv(csv_fp, index=False)


def aggregate_quantable_parameters(models, base_dir="data", unit=1_000_000_000):
    ret = {}
    for model in models:
        csv_file = f"{base_dir}/quantable-{model}.csv"
        df = pd.read_csv(csv_file)
        total = df["param_count"].sum()
        quant = df["quant_count"].sum()
        pct = quant / total
        ret[model] = (total / unit, quant / unit, pct * 100)
    return ret


def summarize_known_llama_quantable_params(base_dir="data", skip_models=None):
    for model_id, cfg in LLAMA_MODELS.items():
        if skip_models is not None and len(skip_models) > 0:
            if any([skip_model in model_id for skip_model in skip_models]):
                continue
        model_base_dir = get_hf_model_storge_base_dir(
            model_id, cfg.get("base_dir", None)
        )
        layers = cfg["layers"]
        llama2 = cfg.get("llama2", True)
        csv_file = f"{base_dir}/quantable-{model_id.split('/')[1]}.csv"
        summarize_llama_quantable_params(
            model_base_dir, layers, csv_file, llama2=llama2
        )


def summarize_known_vit_quantable_params(base_dir="data", skip_models=None):
    for model_id, cfg in VIT_OPENCLIP_MODELS.items():
        if skip_models is not None and len(skip_models) > 0:
            if any([skip_model in model_id for skip_model in skip_models]):
                continue
        csv_file = f"{base_dir}/quantable-{model_id.split('/')[1]}.csv"
        summarize_vit_quantable_params(model_id, csv_file)


def summarize_vit_wdist(base_dir="data"):
    for model_id, cfg in VIT_OPENCLIP_MODELS.items():
        csv_file = f"data/wdist-{model_id.split('/')[1]}.csv"
        summarize_vit_percentiles(model_id, cfg, csv_file)
        # dump_vit_weight_names(model_id)


if __name__ == "__main__":
    summarize_known_vit_quantable_params()
