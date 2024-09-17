import os
import time
from pathlib import Path

from hqq.engine.open_clip import HQQOpenCLIP


def create_hqq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    if load_quantized and os.path.exists(quant_path):
        model = HQQOpenCLIP.from_quantized(quant_path)
        quantized = True
    else:
        model = HQQOpenCLIP.create_model(model_id, device="cpu")
    return model, quantized


def quantize_hqq_model(model, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model.quantize_model(quant_config=quant_config)
    t2 = time.time()
    print("Took " + str(t2 - t1) + " seconds to quantize the model with HQQ")
    quant_path = f"{save_dir}/{model_id}-{config_id}-hqq"
    Path(quant_path).mkdir(parents=True, exist_ok=True)
    model.save_quantized(save_dir=quant_path)
    return model, t2 - t1
