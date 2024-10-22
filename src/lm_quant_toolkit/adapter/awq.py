import os
import time

import torch
from accelerate import (
    infer_auto_device_map,
    init_empty_weights,
    load_checkpoint_in_model,
)
from awq.quantize.pre_quant import apply_awq, run_awq
from awq.quantize.quantizer import real_quantize_model_weight
from awq.utils.utils import simple_dispatch_model
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer


def create_awq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"

    tokenizer = AutoTokenizer.from_pretrained(
        model_id, use_fast=False, trust_remote_code=True
    )
    config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
    # Note (Haotian): To avoid OOM after huggingface transformers 4.36.2
    config.use_cache = False
    if load_quantized and os.path.exists(f"{quant_path}/qmodel.pth"):
        with init_empty_weights():
            model = AutoModelForCausalLM.from_config(
                config=config, torch_dtype=torch.float16, trust_remote_code=True
            )
        max_memory = {0: "20GiB", "cpu": "60GiB"}
        # Infer device map
        kwargs = {"max_memory": max_memory} if len(max_memory) else {}
        device_map = infer_auto_device_map(
            model,
            no_split_module_classes=[
                "OPTDecoderLayer",
                "LlamaDecoderLayer",
                "BloomBlock",
                "MPTBlock",
                "DecoderLayer",
            ],
            **kwargs,
        )
        # Load checkpoint in the model
        load_checkpoint_in_model(
            model,
            checkpoint=quant_path,
            device_map=device_map,
            offload_state_dict=False,
        )
        # Dispatch model
        model = simple_dispatch_model(model, device_map=device_map)
        quantized = True
        model.eval()
    else:
        kwargs = {"torch_dtype": torch.float16, "low_cpu_mem_usage": True}
        model = AutoModelForCausalLM.from_pretrained(
            model_id, config=config, trust_remote_code=True, **kwargs
        )
    return model, tokenizer, quantized, 0


def quantize_awq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    nbits = quant_config.pop("w_bit")
    awq_results = run_awq(
        model,
        tokenizer,
        w_bit=nbits,
        q_config=quant_config,
        n_samples=128,
        seqlen=512,
    )
    intermediate_fp = f"{save_dir}/{model_id}-{config_id}-awq/intermediate.pth"
    dirpath = os.path.dirname(intermediate_fp)
    os.makedirs(dirpath, exist_ok=True)
    torch.save(awq_results, intermediate_fp)
    awq_results = torch.load(intermediate_fp, map_location="cpu")
    apply_awq(model, awq_results)
    real_quantize_model_weight(model, w_bit=nbits, q_config=quant_config)

    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with AWQ")
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    quant_fp = os.path.join(quant_path, "qmodel.pth")
    torch.save(model.cpu().state_dict(), quant_fp)
    tokenizer.save_pretrained(quant_path)

    model_file_size = os.path.getsize(quant_fp)
    return model, t2 - t1, model_file_size
