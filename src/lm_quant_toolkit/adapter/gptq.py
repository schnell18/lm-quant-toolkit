import os
import time

import transformers
from gptqmodel import GPTQModel

from lm_quant_toolkit.adapter.common import (
    get_model_storage_size,
    prepare_calibration_dataset,
)


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
    calibration_dataset = prepare_calibration_dataset(tokenizer)
    model.quantize(calibration_dataset, batch_size=1, calibration_concat_size=0)
    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with GPTQModel")
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    model.save(quant_path)
    os.sync()
    return model, t2 - t1, get_model_storage_size(quant_path)
