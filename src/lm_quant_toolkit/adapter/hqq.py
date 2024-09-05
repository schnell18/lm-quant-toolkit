import os
import time
from hqq.engine.hf import AutoTokenizer as hggAutoTokenizer
from hqq.engine.hf import HQQModelForCausalLM


def create_hqq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    if load_quantized and os.path.exists(quant_path):
        model = HQQModelForCausalLM.from_quantized(quant_path)
        tokenizer = hggAutoTokenizer.from_pretrained(model_id)
        quantized = True
    else:
        model = HQQModelForCausalLM.from_pretrained(model_id)
        tokenizer = hggAutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, quantized


def quantize_hqq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model.quantize_model(quant_config=quant_config)
    t2 = time.time()
    print('Took ' + str(t2 - t1) + ' seconds to quantize the model with HQQ')
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    model.save_quantized(quant_path)
    return model, t2 - t1



