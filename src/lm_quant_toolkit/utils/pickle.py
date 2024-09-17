import os

import torch

from .hub import get_hf_model_storge_base_dir


def dump_vit_weight_names(model_id, wt_file="open_clip_pytorch_model.bin"):
    base_dir = get_hf_model_storge_base_dir(model_id)
    state_dict = torch.load(os.path.join(base_dir, wt_file), weights_only=True)
    for name in state_dict:
        print(name)


def load_state_dict(model_id, wt_file="open_clip_pytorch_model.bin"):
    base_dir = get_hf_model_storge_base_dir(model_id)
    wt_file_fp = os.path.join(base_dir, wt_file)
    return torch.load(wt_file_fp, weights_only=True)
