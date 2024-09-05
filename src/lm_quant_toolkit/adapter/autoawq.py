import os
import time
import transformers

from awq import AutoAWQForCausalLM


def create_autoawq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    if load_quantized and os.path.exists(quant_path):
        model = AutoAWQForCausalLM.from_quantized(quant_path, "", fuse_layers=False)
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        quantized = True
        model = model.cuda()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = AutoAWQForCausalLM.from_pretrained(model_id)
    return model, tokenizer, quantized


def quantize_autoawq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    t2 = time.time()
    print('Took ' + str(t2 - t1) + ' seconds to quantize the model with AutoAWQ')
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    return model, t2 - t1


