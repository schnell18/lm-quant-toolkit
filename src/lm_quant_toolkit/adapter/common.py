import json
import os


def get_model_storage_size(
    base_dir,
    index_file="model.safetensors.index.json",
    model_file="model.safetensors",
):
    size = 0
    index_file = os.path.join(base_dir, index_file)
    if os.path.exists(index_file):
        # model is split into shards
        with open(index_file, "r") as f:
            index = json.load(f)
            for shard in set(index["weight_map"].values()):
                size += os.path.getsize(os.path.join(base_dir, shard))
    else:
        size = os.path.getsize(os.path.join(base_dir, model_file))
    return size
