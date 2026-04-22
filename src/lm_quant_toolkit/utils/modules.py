"""Shared per-family enumeration of HQQ-quantized weight tensors.

Used by kurtosis and Frobenius-norm preparation so both stay in sync with
HQQ's ``get_linear_tags`` across Llama and Qwen3.5 families.
"""

LLM_QUANT_MODULES = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
    "mlp.gate_proj",
    "mlp.down_proj",
    "mlp.up_proj",
]

# Qwen3.5 interleaves GatedDeltaNet linear-attention layers with standard
# self-attention layers. Only layers at indices stride-1, 2*stride-1, ...
# carry q/k/v/o_proj. HQQ's Qwen35Patch skips the others.
QWEN35_FULL_ATTN_INTERVAL = 4


def _iter_llama_quant_keys(layers):
    for module in LLM_QUANT_MODULES:
        for layer in range(layers):
            yield module, layer, f"model.layers.{layer}.{module}.weight"


def _iter_qwen35_quant_keys(layers):
    start = QWEN35_FULL_ATTN_INTERVAL - 1
    step = QWEN35_FULL_ATTN_INTERVAL
    full_attn_layers = set(range(start, layers, step))
    prefix = "model.language_model.layers"
    for module in LLM_QUANT_MODULES:
        is_attn = module.startswith("self_attn.")
        for layer in range(layers):
            if is_attn and layer not in full_attn_layers:
                continue
            yield module, layer, f"{prefix}.{layer}.{module}.weight"


def iter_llm_quant_keys(family, layers, moe=False):
    """Yield ``(module, layer, tensor_key)`` for every HQQ-quantized weight."""
    if family == "llama":
        yield from _iter_llama_quant_keys(layers)
    elif family == "qwen35":
        if moe:
            raise NotImplementedError(
                "MoE Qwen3.5 variants are not supported yet"
            )
        yield from _iter_qwen35_quant_keys(layers)
    else:
        raise ValueError(f"Unknown model family: {family}")
