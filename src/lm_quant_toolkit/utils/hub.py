import os

from huggingface_hub.constants import HUGGINGFACE_HUB_CACHE

LLAMA_MODELS = {
    "meta-llama/Llama-2-7b-hf": {
        "layers": 32,
        "experiment": True,
    },
    "meta-llama/Llama-2-13b-hf": {
        "layers": 40,
        "experiment": True,
    },
    "meta-llama/Meta-Llama-3-8B": {
        "layers": 32,
        "llama2": False,
        "experiment": True,
    },
    "meta-llama/Llama-3.1-8B": {
        "layers": 32,
        "llama2": False,
        "experiment": False,
    },
    "meta-llama/Meta-Llama-3-70B": {
        "layers": 80,
        "llama2": False,
        "base_dir": "/data/hugginface/hub",
    },
    "meta-llama/Llama-2-70b-hf": {
        "layers": 80,
        "base_dir": "/data/hugginface/hub",
    },
    "meta-llama/Meta-Llama-3-70B-Instruct": {
        "layers": 80,
        "base_dir": "/data/hugginface/hub",
        "llama2": False,
    },
    "meta-llama/Meta-Llama-3.1-405B-Instruct": {
        "layers": 126,
        "base_dir": "/data/hugginface/hub",
        "llama2": False,
    },
}

VIT_OPENCLIP_MODELS = {
    "laion/CLIP-ViT-B-32-laion2B-s34B-b79K": {"vlayers": 12, "tlayers": 12},
    "laion/CLIP-ViT-H-14-laion2B-s32B-b79K": {"vlayers": 32, "tlayers": 24},
    "laion/CLIP-ViT-L-14-laion2B-s32B-b82K": {"vlayers": 24, "tlayers": 12},
}

QWEN35_MODELS = {
    # Dense VLMs (text decoder + vision encoder)
    # vision_layers: number of ViT blocks in model.visual.blocks
    # lm_head_tied: True when lm_head shares weights with embed_tokens
    # (no separate lm_head.weight)
    "Qwen/Qwen3.5-0.8B": {"layers": 24, "vision_layers": 12, "lm_head_tied": True},
    "Qwen/Qwen3.5-2B": {"layers": 24, "vision_layers": 24, "lm_head_tied": True},
    "Qwen/Qwen3.5-4B": {"layers": 32, "vision_layers": 24, "lm_head_tied": True},
    "Qwen/Qwen3.5-9B": {"layers": 32, "vision_layers": 27},
    "Qwen/Qwen3.5-27B": {"layers": 64, "vision_layers": 27},
    # MoE VLMs — expert weights are packed tensors:
    #   mlp.experts.gate_up_proj / mlp.experts.down_proj  (no .weight suffix)
    #   mlp.shared_expert.{gate,up,down}_proj.weight
    "Qwen/Qwen3.5-35B-A3B": {"layers": 40, "vision_layers": 27, "moe": True},
    # "Qwen/Qwen3.5-122B-A10B": {"layers": 48, "vision_layers": 27, "moe": True},
    # "Qwen/Qwen3.5-397B-A17B": {"layers": 60, "vision_layers": 27, "moe": True},
}

SENSITIVITY_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "epfl-llm/meditron-7b",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def get_model_meta(model_id):
    """Return normalized metadata for a model id across known LLM families."""
    if model_id in LLAMA_MODELS:
        cfg = LLAMA_MODELS[model_id]
        return {
            "family": "llama",
            "layers": cfg["layers"],
            "base_dir": cfg.get("base_dir"),
            "moe": False,
            "lm_head_tied": False,
        }
    if model_id in QWEN35_MODELS:
        cfg = QWEN35_MODELS[model_id]
        return {
            "family": "qwen35",
            "layers": cfg["layers"],
            "base_dir": cfg.get("base_dir"),
            "vision_layers": cfg.get("vision_layers"),
            "moe": cfg.get("moe", False),
            "lm_head_tied": cfg.get("lm_head_tied", False),
        }
    raise KeyError(f"Unknown model id: {model_id}")


def resolve_models(model_args, model_list):
    """Resolve --model values to HuggingFace model IDs.

    Each value is treated as a numeric index into model_list when it is an
    integer string (e.g. "0", "2"), preserving backward compatibility with
    existing scripts.  Any value that cannot be parsed as an integer is used
    directly as a HuggingFace model ID (e.g. "meta-llama/Llama-3.1-8B").
    """
    models = []
    for m in model_args:
        try:
            models.append(model_list[int(m)])
        except ValueError:
            models.append(m)
    return models


def get_hf_model_storge_base_dir(model_id, hf_hub_dir=None):
    model_id_x = model_id.replace("/", "--")
    cache_dir = hf_hub_dir if hf_hub_dir else HUGGINGFACE_HUB_CACHE
    hf_model_dir = os.path.join(cache_dir, f"models--{model_id_x}")
    ref_main_fp = os.path.join(hf_model_dir, "refs", "main")
    with open(ref_main_fp) as fh:
        commit_sha = fh.read().strip()
    return os.path.join(hf_model_dir, "snapshots", commit_sha)
