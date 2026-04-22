import re

import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from lm_quant_toolkit.utils.hub import (
    LLAMA_MODELS,
    QWEN35_MODELS,
    VIT_OPENCLIP_MODELS,
    get_hf_model_storge_base_dir,
)
from lm_quant_toolkit.utils.pickle import load_state_dict
from lm_quant_toolkit.utils.safetensors import get_tensor


_LLM_QUANT_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
    "mlp.up_proj",
]

_QWEN35_FULL_ATTN_INTERVAL = 4


def _iter_llama_quant_keys(layers):
    for module in _LLM_QUANT_MODULES:
        for layer in range(layers):
            yield module, layer, f"model.layers.{layer}.{module}.weight"


def _iter_qwen35_quant_keys(layers):
    start = _QWEN35_FULL_ATTN_INTERVAL - 1
    step = _QWEN35_FULL_ATTN_INTERVAL
    full_attn_layers = set(range(start, layers, step))
    prefix = "model.language_model.layers"
    for module in _LLM_QUANT_MODULES:
        is_attn = module.startswith("self_attn.")
        for layer in range(layers):
            if is_attn and layer not in full_attn_layers:
                continue
            yield module, layer, f"{prefix}.{layer}.{module}.weight"


def calculate_kurtosis_llm(
    model_id, base_dir, layers, output_dir, family="llama", moe=False
):
    if family == "llama":
        iterator = _iter_llama_quant_keys(layers)
    elif family == "qwen35":
        if moe:
            raise NotImplementedError(
                "Kurtosis for MoE Qwen3.5 variants is not supported yet"
            )
        iterator = _iter_qwen35_quant_keys(layers)
    else:
        raise ValueError(f"Unknown model family: {family}")

    dikts = []
    for module, layer, full_name in iterator:
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


def calculate_kurtosis_vit(model_id, model_cfg, output_dir):
    modules = [
        "mlp.c_fc",
        "mlp.c_proj",
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
                full_name = f"{prefix}.resblocks.{layer}.{module}.weight"
                w = state_dict[full_name]
                w = w.flatten().float().numpy()
                kurt_pearson = kurtosis(
                    w, axis=None, fisher=False, bias=True, nan_policy="omit"
                )
                dikt = {
                    "module": f"{model_type}.{module}",
                    "layer": layer,
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
                percentiles = np.percentile(
                    np.abs(w), [0, 99, 99.9, 99.99, 100])
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


def summarize_qwen35_quantable_params(
    base_dir, layers, vision_layers, fp, moe=False, lm_head_tied=False
):
    """Write a CSV of all parameters in a Qwen3.5 VLM annotated with
       quantability.

    Qwen3.5 is a hybrid VLM:
    - Text decoder: ``model.language_model.layers.{i}``
      - Full-attention layers every 4th (indices 3,7,11,…):
        quantizable q/k/v/o_proj
      - GatedDeltaNet linear-attention layers (all others): not quantized
      - MLP layers (all): quantizable gate/up/down_proj
      - MoE variants use packed:
        ``mlp.experts.gate_up_proj`` / ``mlp.experts.down_proj``
    - Vision encoder: ``model.visual.blocks.{i}``
      - Quantizable: attn.qkv, attn.proj, mlp.linear_fc1/fc2,
        merger.linear_fc1/fc2
    """
    FULL_ATTN_INTERVAL = 4
    full_attn_layers = set(
        range(
            FULL_ATTN_INTERVAL - 1,
            layers,
            FULL_ATTN_INTERVAL,
        )
    )

    dikts = []

    def _record(module, layer, w, quant, typ="text"):
        dikts.append(
            {
                "type": typ,
                "module": module,
                "layer": layer,
                "param_count": w.numel(),
                "quant_count": w.numel() if quant else 0,
            }
        )

    def _t(name):
        return get_tensor(name, base_dir)

    # ── top-level text (non-layerwise)
    _record("embed_tokens", 0, _t(
        "model.language_model.embed_tokens.weight"), False)
    _record("norm", 0, _t("model.language_model.norm.weight"), False)
    if not lm_head_tied:
        _record("lm_head", 0, _t("lm_head.weight"), False)

    # ── layerwise text
    for i in range(layers):
        pfx = f"model.language_model.layers.{i}"
        _record("input_layernorm", i, _t(
            f"{pfx}.input_layernorm.weight"), False)
        _record(
            "post_attention_layernorm",
            i,
            _t(f"{pfx}.post_attention_layernorm.weight"),
            False,
        )

        if i in full_attn_layers:
            # Standard self-attention (quantizable projections)
            for m in ("q_proj", "k_proj", "v_proj", "o_proj"):
                _record(f"self_attn.{m}", i, _t(
                    f"{pfx}.self_attn.{m}.weight"), True)
            # RMS norms inside attention (not quantizable)
            for m in ("q_norm", "k_norm"):
                _record(
                    f"self_attn.{m}", i, _t(
                        f"{pfx}.self_attn.{m}.weight"), False
                )
        else:
            # GatedDeltaNet linear attention — quantized
            for m in ("in_proj_qkv", "in_proj_z", "out_proj", "in_proj_a",
                      "in_proj_b",):
                _record(
                    f"linear_attn.{m}", i, _t(
                        f"{pfx}.linear_attn.{m}.weight"), True
                )
            _record(
                "linear_attn.conv1d", i, _t(
                    f"{pfx}.linear_attn.conv1d.weight"), False
            )
            _record(
                "linear_attn.norm", i, _t(
                    f"{pfx}.linear_attn.norm.weight"), False
            )
            # Learnable scalars stored without .weight suffix
            for m in ("A_log", "dt_bias"):
                _record(f"linear_attn.{m}", i, _t(
                    f"{pfx}.linear_attn.{m}"), False)

        # MLP
        if moe:
            # Packed expert tensors (gate and up fused, no .weight suffix)
            _record(
                "mlp.experts.gate_up_proj",
                i,
                _t(f"{pfx}.mlp.experts.gate_up_proj"),
                True,
            )
            _record(
                "mlp.experts.down_proj", i, _t(
                    f"{pfx}.mlp.experts.down_proj"), True
            )
            for m in ("gate_proj", "up_proj", "down_proj"):
                _record(
                    f"mlp.shared_expert.{m}",
                    i,
                    _t(f"{pfx}.mlp.shared_expert.{m}.weight"),
                    True,
                )
        else:
            for m in ("gate_proj", "up_proj", "down_proj"):
                _record(f"mlp.{m}", i, _t(f"{pfx}.mlp.{m}.weight"), True)

    # ── vision encoder
    # Patch embedding (Conv2d — not quantized)
    _record(
        "visual.patch_embed.proj",
        0,
        _t("model.visual.patch_embed.proj.weight"),
        False,
        "vision",
    )
    try:
        _record(
            "visual.pos_embed",
            0,
            _t("model.visual.pos_embed.weight"),
            False,
            "vision",
        )
    except ValueError:
        # some models use learned positional embeddings stored differently
        pass

    for i in range(vision_layers):
        vpfx = f"model.visual.blocks.{i}"
        # Quantizable: fused QKV, output proj, MLP fc1/fc2
        _record(
            "visual.attn.qkv", i, _t(
                f"{vpfx}.attn.qkv.weight"), False, "vision"
        )
        _record(
            "visual.attn.proj", i, _t(
                f"{vpfx}.attn.proj.weight"), False, "vision"
        )
        _record(
            "visual.mlp.linear_fc1",
            i,
            _t(f"{vpfx}.mlp.linear_fc1.weight"),
            False,
            "vision",
        )
        _record(
            "visual.mlp.linear_fc2",
            i,
            _t(f"{vpfx}.mlp.linear_fc2.weight"),
            False,
            "vision",
        )
        # Layer norms (not quantizable)
        for m in ("norm1", "norm2"):
            _record(f"visual.{m}", i, _t(
                f"{vpfx}.{m}.weight"), False, "vision")

    # Vision-language merger MLP
    for m in ("linear_fc1", "linear_fc2"):
        _record(
            f"visual.merger.{m}",
            0,
            _t(f"model.visual.merger.{m}.weight"),
            False,
            "vision",
        )
    _record(
        "visual.merger.norm",
        0,
        _t("model.visual.merger.norm.weight"),
        False,
        "vision",
    )

    df = pd.DataFrame(dikts)
    df.to_csv(fp, index=False)


def summarize_known_qwen35_quantable_params(base_dir="data", skip_models=None):
    for model_id, cfg in QWEN35_MODELS.items():
        if skip_models is not None and len(skip_models) > 0:
            if any(skip_model in model_id for skip_model in skip_models):
                continue
        model_base_dir = get_hf_model_storge_base_dir(
            model_id, cfg.get("base_dir", None)
        )
        csv_file = f"{base_dir}/quantable-{model_id.split('/')[1]}.csv"
        summarize_qwen35_quantable_params(
            model_base_dir,
            cfg["layers"],
            cfg["vision_layers"],
            csv_file,
            moe=cfg.get("moe", False),
            lm_head_tied=cfg.get("lm_head_tied", False),
        )


def aggregate_quantable_parameters(
    models,
    base_dir="data",
    unit=1_000_000_000,
):
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


def calculate_vit_kurtosis():
    for model_id, cfg in VIT_OPENCLIP_MODELS.items():
        calculate_kurtosis_vit(model_id, cfg, "data")


if __name__ == "__main__":
    calculate_vit_kurtosis()
    # summarize_known_vit_quantable_params()
