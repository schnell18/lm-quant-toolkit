import gc
import os
import random
import time
import torch
import transformers

from auto_gptq import AutoGPTQForCausalLM
from datasets import load_dataset
from tqdm import tqdm


# Adapted from: https://towardsdatascience.com/4-bit-quantization-with-gptq-36b0f4f02c34
def prepare_model(model, tokenizer, n_samples=1024, max_tokens=512, use_triton=False):
    # Load data and tokenize examples
    data = load_dataset(
        "allenai/c4",
        data_files="en/c4-train.00001-of-01024.json.gz",
        split=f"train[:{n_samples}]"
    )
    # ~536K tokens
    tokenized_data = torch.cat(
        [tokenizer(data[i]['text'], return_tensors='pt').input_ids
            for i in tqdm(range(len(data)))], axis=-1)

    # Format tokenized examples
    random.seed(1)
    examples_ids = []
    for _ in range(n_samples):
        i = random.randint(0, tokenized_data.shape[1] - max_tokens - 1)
        j = i + max_tokens
        input_ids = tokenized_data[:, i:j]
        attention_mask = torch.ones_like(input_ids)
        examples_ids.append({'input_ids': input_ids, 'attention_mask': attention_mask})

    print('Using ' + str(len(examples_ids)) + ' samples for calibration.')
    model.quantize(examples_ids, batch_size=1, use_triton=use_triton)
    # model = model.cuda()
    # with torch.no_grad():
    #     x = model(input_ids.to('cuda'))
    # del examples_ids, x
    del examples_ids
    torch.cuda.empty_cache()
    gc.collect()
    return model


def create_autogptq_model(model_id, quant_config, config_id, load_quantized, save_dir):
    quantized = False
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    if load_quantized and os.path.exists(quant_path):
        model = AutoGPTQForCausalLM.from_quantized(quant_path, device="cuda:0")
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        quantized = True
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(model_id)
        model = AutoGPTQForCausalLM.from_pretrained(model_id, quant_config)
    return model, tokenizer, quantized


def quantize_autogptq_model(model, tokenizer, quant_config, model_id, config_id, save_dir):
    t1 = time.time()
    model = prepare_model(model, tokenizer)
    t2 = time.time()
    print('Took ' + str(t2 - t1) + ' seconds to quantize the model with AutoGPTQ')
    quant_path = f"{save_dir}/{model_id}-{config_id}-gptq"
    model.save_quantized(quant_path, use_safetensors=True)
    return model, t2 - t1


