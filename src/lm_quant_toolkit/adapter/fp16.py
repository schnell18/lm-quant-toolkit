import torch
import transformers
from transformers import AutoModelForCausalLM

from lm_quant_toolkit.adapter.common import get_model_storage_size
from lm_quant_toolkit.utils.hub import get_hf_model_storge_base_dir


def create_fp16_model(model_id, quant_config, config_id, load_quantized, save_dir):
    model_file_size = 0
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    base_dir = get_hf_model_storge_base_dir(model_id)
    model_file_size = get_model_storage_size(base_dir)
    return model, tokenizer, False, model_file_size
