import os
import time

import torch
import transformers
from gptqmodel import GPTQModel, QuantizeConfig
from gptqmodel.quantization import FORMAT, METHOD
from gptqmodel.utils.backend import BACKEND

from lm_quant_toolkit.adapter.common import (
    get_model_storage_size,
    prepare_calibration_dataset,
)

_VERSION_TO_FORMAT = {
    "GEMM": FORMAT.GEMM,
    "GEMV": FORMAT.GEMV,
    "MARLIN": FORMAT.GPTQ,
    "GEMV_FAST": FORMAT.GEMV_FAST,
    "LLM_AWQ": FORMAT.LLM_AWQ,
}


def _awq_config_to_gm(awq_config):
    """
    Convert AutoAWQ-style config dict to a GPTQModel QuantizeConfig for AWQ.
    """
    version = awq_config.get("version", "GEMM").upper()
    fmt = _VERSION_TO_FORMAT.get(version, FORMAT.GEMM)
    qcfg = QuantizeConfig(
        bits=awq_config["w_bit"],
        group_size=awq_config["q_group_size"],
        sym=not awq_config.get("zero_point", True),
        quant_method=METHOD.AWQ,
        format=fmt,
    )
    # Keep linear attention unquantised
    qcfg.dynamic = {
        r"-:model\.language_model\.layers\.\d+\.linear_attn.*": {},
    }
    return qcfg


def create_autoawq_model(
    model_id,
    quant_config,
    config_id,
    load_quantized,
    save_dir,
):
    model_file_size = 0
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id, use_fast=False, trust_remote_code=True
    )
    if load_quantized and os.path.exists(quant_path):
        model = GPTQModel.load(
            quant_path,
            device="cuda:0",
            # backend=BACKEND.AWQ_MARLIN,
            backend=BACKEND.GEMM,
        )
        quantized = True
        model_file_size = get_model_storage_size(quant_path)
    else:
        gm_config = _awq_config_to_gm(quant_config)
        model = GPTQModel.load(model_id, gm_config)
    # ensure bfloat16 is converted to float16 for AWQ GEMM kernel
    model = model.to(torch.float16)
    return model, tokenizer, quantized, model_file_size


def quantize_autoawq_model(
    model, tokenizer, quant_config, model_id, config_id, save_dir
):
    t1 = time.time()
    calibration_dataset = prepare_calibration_dataset(tokenizer)
    model.quantize(calibration_dataset, batch_size=1)
    t2 = time.time()
    print("Quantized model in " + str(t2 - t1) + " seconds w/ GPTQModel (AWQ)")
    quant_path = f"{save_dir}/{model_id}-{config_id}-awq"
    model.save(quant_path)
    tokenizer.save_pretrained(quant_path)
    os.sync()
    return model, t2 - t1, get_model_storage_size(quant_path)
