import numpy as np
import pandas as pd
from scipy.stats import kurtosis

from lm_quant_toolkit.utils.pickle import load_state_dict

# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.
# transformer.resblocks.


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
    for prefix, layers in model_cfg.items():
        model_type = "vision" if "visual" in prefix else "text"
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


if __name__ == "__main__":
    models = {
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {
            "transformer": 12,
            "visual.transformer": 12,
        },
        "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {
            "transformer": 24,
            "visual.transformer": 32,
        },
        "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {
            "transformer": 12,
            "visual.transformer": 24,
        },
    }
    for model_id, cfg in models.items():
        csv_file = f"data/wdist-{model_id.split('/')[1]}.csv"
        summarize_vit_percentiles(model_id, cfg, csv_file)
        # dump_vit_weight_names(model_id)
