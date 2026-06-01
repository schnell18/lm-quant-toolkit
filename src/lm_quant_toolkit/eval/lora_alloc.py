"""KurtBoost-guided LoRA capacity allocation for HQQ+ (SFT).

This module extends the KurtBoost idea - originally used to allocate *bits* to
sensitive layers (see ``hqq/utils/optimizer.py`` ``_allocate_boost_decline_configs``)
- to allocate *adaptation capacity* (LoRA rank ``r`` and scale ``lora_alpha``) for
the HQQ+ method.

The hypothesis: layers/modules whose weight kurtosis changes abruptly across depth
are harder to recover after aggressive (1-/2-bit) quantization, so giving them more
LoRA capacity should improve the SFT'd model.

Mechanism (mirrors ``boost_cfg(budget, stop)`` in the optimizer): a (layer, module)
pair flagged sensitive climbs ``boost_stop`` rungs up a discrete LoRA capacity
ladder; everyone else stays at the per-module-type base rung taken from
``hqq_plus.py`` (attention r=32, MLP r=8). The ladder uses ``alpha = 2 * r``.
"""

import pandas as pd
import torch

# Reuse the upstream sensitivity detector (re-enabled in hqq/utils/optimizer.py)
# instead of duplicating its logic here.
from hqq.utils.optimizer import identify_sensitive_modules

# LoRA capacity ladder (ascending capacity). Each entry is (r, lora_alpha) with
# alpha = 2 * r. The ladder is the LoRA analogue of ``budget_map`` in
# ``_allocate_boost_decline_configs`` of hqq/utils/optimizer.py.
LORA_LADDER = [
    (4, 8),     # 0
    (8, 16),    # 1
    (16, 32),   # 2
    (32, 64),   # 3
    (64, 128),  # 4
    (128, 256),  # 5
]

ATTN_MODULES = {
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
}
MLP_MODULES = {
    "mlp.gate_proj",
    "mlp.up_proj",
    "mlp.down_proj",
}

# Base rungs, matching the uniform allocation in hqq_plus.py.
ATTN_BASE_R = 32  # ladder index 3 -> (32, 64)
MLP_BASE_R = 8    # ladder index 1 -> (8, 16)


def _ladder_index_for_r(r: int) -> int:
    for i, (rung_r, _) in enumerate(LORA_LADDER):
        if rung_r == r:
            return i
    raise ValueError(f"r={r} is not a rung in LORA_LADDER={LORA_LADDER}")


def boost_cfg(base_idx: int, stop: int):
    """Climb ``stop`` rungs up the ladder from ``base_idx`` (clamped).

    Direct analogue of ``boost_cfg`` in hqq/utils/optimizer.py, but indexing the
    LoRA capacity ladder instead of the quantization budget ladder.
    """
    idx = base_idx + stop
    if idx < 0:
        idx = 0
    elif idx >= len(LORA_LADDER):
        idx = len(LORA_LADDER) - 1
    return LORA_LADDER[idx]


def _peft_config(r, lora_alpha, dropout, train_dtype, train_bias):
    return {
        "lora_type": "default",
        "r": r,
        "lora_alpha": lora_alpha,
        "dropout": dropout,
        "train_dtype": train_dtype,
        "train_bias": train_bias,
    }


def uniform_lora_configs(
    metric_fp,
    attn_base_r=ATTN_BASE_R,
    mlp_base_r=MLP_BASE_R,
    dropout=0.05,
    train_dtype=torch.float32,
    train_bias=True,
):
    """Per-(layer, module) LoRA config with the uniform hqq_plus allocation.

    Returns ``{ "{layer}.{module}": peft_config }`` so it can be applied with the
    same per-layer patch path as the KurtBoost variant (the control group).
    """
    attn_r, attn_a = LORA_LADDER[_ladder_index_for_r(attn_base_r)]
    mlp_r, mlp_a = LORA_LADDER[_ladder_index_for_r(mlp_base_r)]
    df = pd.read_csv(metric_fp)
    cfgs = {}
    n_layers_per_mod = df.groupby(["module"]).layer.nunique().to_dict()
    for module, n_layers in n_layers_per_mod.items():
        is_attn = module in ATTN_MODULES
        r, a = (attn_r, attn_a) if is_attn else (mlp_r, mlp_a)
        for layer in range(n_layers):
            cfgs[f"{layer}.{module}"] = _peft_config(
                r, a, dropout, train_dtype, train_bias
            )
    return cfgs, {}


def allocate_lora_configs(
    metric_fp,
    boost_stop=1,
    top_m=1,
    attn_base_r=ATTN_BASE_R,
    mlp_base_r=MLP_BASE_R,
    dropout=0.05,
    train_dtype=torch.float32,
    train_bias=True,
):
    """KurtBoost-guided per-(layer, module) LoRA allocation.

    Sensitive (layer, module) pairs - identified by abrupt kurtosis jumps - climb
    ``boost_stop`` rungs up :data:`LORA_LADDER` from their module-type base rung;
    all other pairs stay at the base rung. This is the additive-boost variant
    (no decline branch).

    Returns ``({ "{layer}.{module}": peft_config }, module_outliers)``.
    """
    df = pd.read_csv(metric_fp)
    module_outliers = identify_sensitive_modules(
        df, src="kurtosis", top_m=top_m, diff_method="subtract"
    )
    attn_base_idx = _ladder_index_for_r(attn_base_r)
    mlp_base_idx = _ladder_index_for_r(mlp_base_r)

    cfgs = {}
    n_layers_per_mod = df.groupby(["module"]).layer.nunique().to_dict()
    for module, n_layers in n_layers_per_mod.items():
        is_attn = module in ATTN_MODULES
        base_idx = attn_base_idx if is_attn else mlp_base_idx
        base_r, base_a = LORA_LADDER[base_idx]
        boosted_layers = module_outliers.get(module, [])
        for layer in range(n_layers):
            if layer in boosted_layers:
                r, a = boost_cfg(base_idx, boost_stop)
            else:
                r, a = base_r, base_a
            cfgs[f"{layer}.{module}"] = _peft_config(
                r, a, dropout, train_dtype, train_bias
            )
    return cfgs, module_outliers
