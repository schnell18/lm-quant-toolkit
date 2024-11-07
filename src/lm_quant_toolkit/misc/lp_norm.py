import numpy as np
import pandas as pd
import torch
from hqq.core.quantize import Quantizer as hQuant

from lm_quant_toolkit.utils.hub import get_hf_model_storge_base_dir
from lm_quant_toolkit.utils.safetensors import get_tensor


def quant_hqq(tensor, nbits, group_size=64, lp_norm=0.7, optimize=True):
    opt_params = {
        "lp_norm": lp_norm,
        "iters": 100,
        "early_stop": False,
    }

    wq, meta = hQuant.quantize(
        tensor, nbits=nbits, group_size=group_size, optimize=optimize, **opt_params
    )
    return hQuant.dequantize(wq, meta)


if __name__ == "__main__":
    model_id = "meta-llama/Llama-2-7b-hf"
    base_dir = get_hf_model_storge_base_dir(model_id)

    layer = 31
    module = "self_attn.o_proj"
    matrix_name = f"model.layers.{layer}.{module}.weight"
    w = get_tensor(matrix_name, base_dir)

    dick = []
    for lp_norm in np.linspace(0.1, 0.9, 9):
        for b in [3, 4, 8]:
            for g in [32, 64, 128]:
                wq_hqq = quant_hqq(w, nbits=b, lp_norm=lp_norm, group_size=g)
                norm_hqq = torch.norm(w - wq_hqq)
                dick.append(
                    {
                        "model": model_id.split("/")[1],
                        "layer": layer,
                        "module": module,
                        "lp_norm": lp_norm,
                        "b": b,
                        "g": g,
                        "fnorm": norm_hqq.item(),
                    }
                )
                print(f"FNorm HQQ(lp_norm={lp_norm:.2f}): {norm_hqq}")
    df = pd.DataFrame(dick)
    df.to_csv("lp_norm_tuning.csv", index=False)
