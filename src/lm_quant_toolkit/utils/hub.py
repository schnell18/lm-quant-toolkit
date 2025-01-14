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

SENSITIVITY_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-7b-chat-hf",
    "epfl-llm/meditron-7b",
    "meta-llama/Llama-3.1-8B",
    "meta-llama/Llama-3.1-8B-Instruct",
]


def get_hf_model_storge_base_dir(model_id, hf_hub_dir=None):
    model_id_x = model_id.replace("/", "--")
    cache_dir = hf_hub_dir if hf_hub_dir else HUGGINGFACE_HUB_CACHE
    hf_model_dir = os.path.join(cache_dir, f"models--{model_id_x}")
    ref_main_fp = os.path.join(hf_model_dir, "refs", "main")
    with open(ref_main_fp) as fh:
        commit_sha = fh.read().strip()
    return os.path.join(hf_model_dir, "snapshots", commit_sha)
