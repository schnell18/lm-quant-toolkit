import gc
import os
import random
import time

import torch
import transformers
from datasets import load_dataset
from gptqmodel import GPTQModel
from tqdm import tqdm

from lm_quant_toolkit.adapter.common import get_model_storage_size


def _prepare_calibration_dataset(tokenizer, n_samples=1024, max_tokens=512):
    data = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split=f"train[:{n_samples}]",
    )
    tokenized_data = torch.cat(
        [
            tokenizer(data[i]["text"], return_tensors="pt").input_ids
            for i in tqdm(range(len(data)))
        ],
        axis=-1,
    )

    random.seed(1)
    examples_ids = []
    for _ in range(n_samples):
        i = random.randint(0, tokenized_data.shape[1] - max_tokens - 1)
        j = i + max_tokens
        input_ids = tokenized_data[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        examples_ids.append({"input_ids": input_ids, "attention_mask": attention_mask})

    print("Using " + str(len(examples_ids)) + " samples for calibration.")
    del tokenized_data
    torch.cuda.empty_cache()
    gc.collect()
    return examples_ids


def create_gptq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    model_file_size = 0
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    if load_quantized and os.path.exists(quant_path):
        model = GPTQModel.load(quant_path, device="cuda:0")
        quantized = True
        model_file_size = get_model_storage_size(quant_path)
    else:
        model = GPTQModel.load(model_id, quantize_config=quant_config)
    return model, tokenizer, quantized, model_file_size


def quantize_gptq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    calibration_dataset = _prepare_calibration_dataset(tokenizer)
    model.quantize(calibration_dataset, batch_size=1, calibration_concat_size=0)
    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with GPTQModel")
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    model.save(quant_path)
    os.sync()
    return model, t2 - t1, get_model_storage_size(quant_path)
