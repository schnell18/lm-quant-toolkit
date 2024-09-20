import os
from timeit import default_timer as timer

import pandas as pd
import torch
from hqq.core.quantize import Quantizer as hQuant

from lm_quant_toolkit.utils.hub import (
    LLAMA_MODELS,
    VIT_OPENCLIP_MODELS,
    get_hf_model_storge_base_dir,
)
from lm_quant_toolkit.utils.safetensors import get_tensor


def quant_hqq(tensor, nbits, group_size=64, optimize=True):
    wq, meta = hQuant.quantize(
        tensor, nbits=nbits, group_size=group_size, optimize=optimize
    )
    return hQuant.dequantize(wq, meta)


def calc_fnorm_vit(
    state_dict,
    model_type,
    prefix,
    layer,
    module,
    suffix,
    nbits1,
    gsizes1,
    nbits2,
    gsizes2,
):
    dikts = []
    matrix_name = f"{prefix}.{layer}.{module}.{suffix}"
    w = state_dict[matrix_name]
    params = w.numel()
    for nbit1 in nbits1:
        for gsize1 in gsizes1:
            for nbit2 in nbits2:
                for gsize2 in gsizes2:
                    wq_hqq = quant_hqq(w, nbits=nbit1, group_size=gsize1, optimize=True)
                    norm_hqq = torch.norm(w - wq_hqq).item()
                    bpp = nbit1 + 2 * nbit2 / gsize1 + 32 / (gsize1 * gsize2)
                    memmb = bpp * params / 8 / (1024**2)
                    dikt = {
                        "layer": layer,
                        "module": f"{model_type}.{module}",
                        "nbit1": nbit1,
                        "gsize1": gsize1,
                        "nbit2": nbit2,
                        "gsize2": gsize2,
                        "fnorm": norm_hqq,
                        "memmb": memmb,
                        "params": params,
                    }
                    dikts.append(dikt)
    return dikts


def calc_fnorm(
    base_dir, prefix, layer, module, suffix, nbits1, gsizes1, nbits2, gsizes2
):
    dikts = []
    matrix_name = f"{prefix}.{layer}.{module}.{suffix}"
    w = get_tensor(matrix_name, base_dir)
    params = w.numel()
    for nbit1 in nbits1:
        for gsize1 in gsizes1:
            for nbit2 in nbits2:
                for gsize2 in gsizes2:
                    wq_hqq = quant_hqq(w, nbits=nbit1, group_size=gsize1, optimize=True)
                    norm_hqq = torch.norm(w - wq_hqq).item()
                    bpp = nbit1 + 2 * nbit2 / gsize1 + 32 / (gsize1 * gsize2)
                    memmb = bpp * params / 8 / (1024**2)
                    dikt = {
                        "layer": layer,
                        "module": module,
                        "nbit1": nbit1,
                        "gsize1": gsize1,
                        "nbit2": nbit2,
                        "gsize2": gsize2,
                        "fnorm": norm_hqq,
                        "memmb": memmb,
                        "params": params,
                    }
                    dikts.append(dikt)
    return dikts


def calc_fnorm_for_vit_model(model_id, base_dir, layer_cfg):
    # self_attns = ["out_proj"]
    self_attns = []
    mlps = ["c_fc", "c_proj"]
    nbits1 = [2, 3, 4, 8]
    gsizes1 = [32, 64, 128]
    nbits2 = [8]
    gsizes2 = [128]
    prefixes = [
        "visual.transformer.resblocks",
        "transformer.resblocks",
    ]
    suffix = "weight"
    dikts = []
    state_dict = torch.load(
        os.path.join(base_dir, "open_clip_pytorch_model.bin"),
        weights_only=True,
    )
    for prefix in prefixes:
        if "visual" in prefix:
            model_type = "vision"
            layers = layer_cfg["vlayers"]
        else:
            model_type = "text"
            layers = layer_cfg["tlayers"]
        for layer in range(layers):
            for attn in self_attns:
                ds = calc_fnorm_vit(
                    state_dict,
                    model_type,
                    prefix,
                    layer,
                    f"attn.{attn}",
                    suffix,
                    nbits1,
                    gsizes1,
                    nbits2,
                    gsizes2,
                )
                dikts.extend(ds)
            for mlp in mlps:
                ds = calc_fnorm_vit(
                    state_dict,
                    model_type,
                    prefix,
                    layer,
                    f"mlp.{mlp}",
                    suffix,
                    nbits1,
                    gsizes1,
                    nbits2,
                    gsizes2,
                )
                dikts.extend(ds)
    df = pd.DataFrame(dikts)
    file_name = f"data/fnorm-{model_id.split('/')[1]}.csv"
    df.to_csv(
        file_name,
        columns=[
            "layer",
            "module",
            "nbit1",
            "gsize1",
            "nbit2",
            "gsize2",
            "fnorm",
            "memmb",
            "params",
        ],
        index=False,
    )


def calc_fnorm_for_model(model_id, base_dir, layers):
    self_attns = ["q_proj", "v_proj", "k_proj", "o_proj"]
    mlps = ["gate_proj", "up_proj", "down_proj"]
    nbits1 = [2, 3, 4, 8]
    gsizes1 = [32, 64, 128]
    nbits2 = [8]
    gsizes2 = [128]
    prefix = "model.layers"
    suffix = "weight"
    dikts = []
    for layer in range(layers):
        for attn in self_attns:
            ds = calc_fnorm(
                base_dir,
                prefix,
                layer,
                f"self_attn.{attn}",
                suffix,
                nbits1,
                gsizes1,
                nbits2,
                gsizes2,
            )
            dikts.extend(ds)
        for mlp in mlps:
            ds = calc_fnorm(
                base_dir,
                prefix,
                layer,
                f"mlp.{mlp}",
                suffix,
                nbits1,
                gsizes1,
                nbits2,
                gsizes2,
            )
            dikts.extend(ds)

    df = pd.DataFrame(dikts)
    file_name = f"data/fnorm-{model_id.split('/')[1]}.csv"
    df.to_csv(
        file_name,
        columns=[
            "layer",
            "module",
            "nbit1",
            "gsize1",
            "nbit2",
            "gsize2",
            "fnorm",
            "memmb",
            "params",
        ],
        index=False,
    )


def main_vit():
    for model_id, layer_cfg in VIT_OPENCLIP_MODELS.items():
        t1 = timer()
        base_dir = layer_cfg.get("base_dir", None)
        model_base_dir = get_hf_model_storge_base_dir(model_id, base_dir)
        calc_fnorm_for_vit_model(model_id, model_base_dir, layer_cfg)
        t2 = timer()
        print(f"Finished {model_id} metrics calc in {t2 - t1} seconds")


def main():
    for model_id, model in LLAMA_MODELS.items():
        if model_id != "meta-llama/Meta-Llama-3-8B":
            continue
        t1 = timer()
        base_dir = model.get("base_dir", None)
        model_base_dir = get_hf_model_storge_base_dir(model_id, base_dir)
        calc_fnorm_for_model(model_id, model_base_dir, model["layers"])
        t2 = timer()
        print(f"Finished {model_id} metrics calc in {t2 - t1} seconds")


if __name__ == "__main__":
    main_vit()
