import json
import os

from safetensors import safe_open


def get_tensor(matrix_name, base_dir, index_json="model.safetensors.index.json"):
    fp = os.path.join(base_dir, index_json)
    with open(fp, "r") as fh:
        index = json.load(fh)
        try:
            st_file = index["weight_map"][matrix_name]
            mp = os.path.join(base_dir, st_file)
            with safe_open(mp, framework="pt", device="cpu") as f:
                return f.get_tensor(matrix_name)
        except Exception:
            raise ValueError(f"Invalid key {matrix_name}")


def get_tensor2(matrix_name, st_file_fp):
    try:
        with safe_open(st_file_fp, framework="pt", device="cpu") as f:
            return f.get_tensor(matrix_name)
    except Exception:
        raise ValueError("Fail to retrieve tensor keys")


def get_tensor_keys(base_dir, st_file):
    try:
        mp = os.path.join(base_dir, st_file)
        with safe_open(mp, framework="pt", device="cpu") as f:
            return f.keys()
    except Exception:
        raise ValueError("Fail to retrieve tensor keys")


def get_tensor_dual(matrix1, matrix2, base_dir, st_file):
    try:
        mp = os.path.join(base_dir, st_file)
        with safe_open(mp, framework="pt", device="cpu") as f:
            return f.get_tensor(matrix1), f.get_tensor(matrix2)
    except Exception:
        raise ValueError(f"Invalid key {matrix1}")
