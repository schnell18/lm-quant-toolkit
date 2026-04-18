import os
import time

from hqq.engine.hf import AutoTokenizer as hggAutoTokenizer
from hqq.engine.hf import HQQModelForCausalLM

from lm_quant_toolkit.adapter.common import (
    get_model_storage_size,
)


def create_hqq_model(
    model_id,
    quant_config,
    config_id,
    load_quantized,
    save_dir,
):
    quantized = False
    model_file_size = 0
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    if load_quantized and os.path.exists(quant_path):
        model = HQQModelForCausalLM.from_quantized(quant_path, device="cuda:0")
        tokenizer = hggAutoTokenizer.from_pretrained(model_id)
        quantized = True
        model_file_size = get_model_storage_size(quant_path)
    else:
        model = HQQModelForCausalLM.from_pretrained(model_id)
        tokenizer = hggAutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, quantized, model_file_size


def quantize_hqq_model(
    model,
    tokenizer,
    quant_config,
    model_id,
    config_id,
    save_dir,
):
    model_file_size = 0
    t1 = time.time()
    model.quantize_model(quant_config=quant_config)
    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with HQQ")
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    model.save_quantized(quant_path)
    # persistent the quantized model
    os.sync()
    model_file_size = get_model_storage_size(quant_path)
    return model, t2 - t1, model_file_size
