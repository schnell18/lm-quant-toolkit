import os
import time

from transformers import AutoModelForCausalLM, AutoTokenizer

from lm_quant_toolkit.adapter.common import get_model_storage_size


def create_bnb_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    model_file_size = 0
    quant_path = f"{save_dir}/{model_id}-{config_id}-bnb"
    if load_quantized and os.path.exists(quant_path):
        model = AutoModelForCausalLM.from_pretrained(quant_path)
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        quantized = True
        model_file_size = get_model_storage_size(quant_path)
    else:
        model = None
        tokenizer = AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, quantized, model_file_size


def quantize_bnb_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    model_file_size = 0
    t1 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        model_id, quantization_config=quant_config
    )
    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with BnB")
    quant_path = f"{save_dir}/{model_id}-{config_id}-bnb"
    model.save_pretrained(quant_path)
    # persistent the quantized model
    os.sync()
    model_file_size = get_model_storage_size(quant_path)
    return model, t2 - t1, model_file_size
