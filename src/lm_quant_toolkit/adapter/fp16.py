import torch
import transformers
from transformers import AutoModelForCausalLM


def create_fp16_model(model_id, quant_config, config_id, load_quantized, save_dir):
    model = AutoModelForCausalLM.from_pretrained(
        model_id, device_map="auto", torch_dtype=torch.float16
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
    return model, tokenizer, False
