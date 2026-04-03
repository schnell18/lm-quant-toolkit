import json
import os

from datasets import load_dataset


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


def prepare_calibration_dataset(tokenizer, n_samples=1024, max_tokens=512):
    data = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split="train",
    ).select(range(n_samples))["text"]
    return data
