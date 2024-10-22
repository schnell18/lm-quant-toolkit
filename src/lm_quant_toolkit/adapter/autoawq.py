import os
import time

import torch
import transformers
from awq import AutoAWQForCausalLM
from transformers import AutoConfig

from lm_quant_toolkit.adapter.common import get_model_storage_size


def create_autoawq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    model_file_size = 0
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"

    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False
    if load_quantized and os.path.exists(quant_path):
        model = AutoAWQForCausalLM.from_quantized(
            quant_path,
            device_map="auto",
            offload_state_dict=False,
            config=config,
        )
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        quantized = True
        model_file_size = get_model_storage_size(quant_path)
        model = model.cuda()
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        #     max_memory={0: "18GiB", "cpu": "60GiB"},
        # )
        model = AutoAWQForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            offload_state_dict=False,
            torch_dtype=torch.float16,
            max_memory={0: "18GiB", "cpu": "60GiB"},
            config=config,
        )
    return model, tokenizer, quantized, model_file_size


def quantize_autoawq_model(
    model, tokenizer, quant_config, model_id, config_id, save_dir
):
    t1 = time.time()
    model.quantize(tokenizer, quant_config=quant_config)
    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with AutoAWQ")
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    model.save_quantized(quant_path)
    tokenizer.save_pretrained(quant_path)
    # persistent the quantized model
    os.sync()
    return model, t2 - t1, _get_model_file_size(quant_path)


def _get_model_file_size(quant_path):
    quant_fp_pt = os.path.join(quant_path, "qmodel.pth")
    if os.path.exists(quant_fp_pt):
        return os.path.getsize(quant_fp_pt)
    else:
        return get_model_storage_size(quant_path)
