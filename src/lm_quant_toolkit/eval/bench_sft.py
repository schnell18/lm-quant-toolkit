"""HQQ+ SFT benchmark with KurtBoost-guided LoRA capacity allocation.

Templated on ``bench.py`` (experiment loop + progress/resume + combined CSV) and
on ``hqq/examples/hqq_plus.py`` (1-/2-bit HQQ backbone + LoRA + SFT on wikitext-2
+ perplexity eval). Simplified to a single dataset (wikitext-2-raw-v1) as in
hqq_plus.py.

For each of the three KurtBoost llama models, the backbone is quantized at 1-bit
and 2-bit (group_size=8), then LoRA adapters are attached and SFT-trained on
wikitext-2 train. Three algorithms can be benchmarked per model/backbone:

* ``fp16``      - the un-quantized model, evaluated as-is; quantization and SFT
                  are skipped altogether. The upper-bound baseline.
* ``HQQ+``      - HQQ+ with the fixed hqq_plus allocation (attn r=32, mlp r=8);
                  the quantized-and-recovered control.
* ``kurtboost`` - HQQ+ with more adaptation capacity (LoRA r, alpha) given to
                  layers/modules flagged sensitive by weight kurtosis (see
                  ``lora_alloc.py``); the proposed method.

The model's kurtosis metric file (``src/data/fnorm-<model>.csv``, which carries a
``kurtosis`` column) guides the kurtboost allocation.
"""

import gc
import logging
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import SFTConfig, SFTTrainer

from hqq.core.peft import (
    PeftUtils,
    autoname_modules,
    patch_linear_add_peft,
)
from hqq.core.quantize import BaseQuantizeConfig, HQQBackend, HQQLinear
from hqq.core.utils import cleanup as hqq_cleanup
from hqq.models.base import _QUANT_LAYERS, find_parent, name_to_linear_tag
from hqq.models.hf.base import AutoHQQHFModel

from lm_quant_toolkit.eval.common import (
    combine_metrics,
    get_memory_metrics,
    get_mxq_quant_meta_data_file,
    persist_progress,
    save_partial_metric,
)
from lm_quant_toolkit.eval.lora_alloc import (
    allocate_lora_configs,
    hqqplus_lora_configs,
)

logger = logging.getLogger(__name__)

# The three llama models used by the KurtBoost quantization experiments
# (see scripts/experiment-llama-kurt-boost.sh).
ALL_MODELS = [
    "meta-llama/Llama-2-7b-hf",
    "meta-llama/Llama-2-13b-hf",
    "meta-llama/Meta-Llama-3-8B",
]

# Low-bit HQQ backbones to recover via HQQ+. (nbits, group_size); g=8 matches
# hqq_plus.py.
BACKBONE_CONFIGS = [
    (1, 8),
    (2, 8),
]

ALGORITHMS = ["fp16", "HQQ+", "kurtboost"]

COMPUTE_DTYPE = torch.bfloat16
TRAIN_DTYPE = torch.float32
DEVICE = "cuda:0"


# Wrap model to avoid accelerate issues (from hqq_plus.py).
class WrappedModel(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, *args, **kwargs):
        return self.model.forward(*args, **kwargs)

    def train(self):
        self.model.train()

    def eval(self):
        self.model.eval()

    def parameters(self):
        return self.model.parameters()

    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.model, name)


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def _full_name_to_key(name: str) -> str:
    """``model.layers.5.self_attn.q_proj`` -> ``5.self_attn.q_proj``.
    Return empty string if no layer number is found.

    Matches the ``{layer}.{module}`` keys produced by lora_alloc.py.
    """
    nums = [p for p in name.split(".") if p.isnumeric()]
    if len(nums) == 0:
        return ""
    layer = nums[0]
    return f"{layer}.{name_to_linear_tag(name)}"


def add_lora_per_layer(model, lora_configs, base_class=None, verbose=True):
    """Attach LoRA with a *per-(layer, module)* config.

    ``PeftUtils.add_lora`` keys its config by ``name_to_linear_tag`` which strips
    the layer index, so it can only assign one config per module-type. This
    replicates ``patch_linearlayers`` but keys on the full module name, enabling
    the per-layer KurtBoost allocation.
    """
    base_class = PeftUtils.get_base_class(model, base_class)
    base_class.setup_model(model)

    # Freeze base weights; only LoRA params train.
    for param in model.parameters():
        param.requires_grad = False

    ignore_tags = base_class.get_ignore_layers(model)
    tmp_mapping = {}
    for name, module in model.named_modules():
        if (type(module) in _QUANT_LAYERS) and (name not in ignore_tags):
            tmp_mapping[name] = module

    for name in tqdm(tmp_mapping, disable=not verbose):
        key = _full_name_to_key(name)
        if key == "":
            continue
        patch_param = lora_configs.get(key, None)
        setattr(
            find_parent(model, name),
            name.split(".")[-1],
            patch_linear_add_peft(tmp_mapping[name], patch_param),
        )

    autoname_modules(model)
    model.peft_config = lora_configs
    hqq_cleanup()


def build_lora_configs(algorithm, metric_fp, boost_stop, top_m):
    if algorithm == "HQQ+":
        return hqqplus_lora_configs(metric_fp, train_dtype=TRAIN_DTYPE)
    elif algorithm == "kurtboost":
        return allocate_lora_configs(
            metric_fp,
            boost_stop=boost_stop,
            top_m=top_m,
            train_dtype=TRAIN_DTYPE,
        )
    raise ValueError(f"Invalid algorithm: {algorithm}")


def _checkpoint_tag(model_id, nbits, gsize, algorithm, boost_stop, top_m):
    short = model_id.split("/")[1]
    tag = f"{short}-{nbits}b_g{gsize}-{algorithm}"
    if algorithm == "kurtboost":
        tag += f"-bs{boost_stop}-tm{top_m}"
    return tag


def _save_checkpoint(
    model, snapshot_dir, model_id, nbits, gsize, algorithm, boost_stop, top_m
):
    """Persist the trained LoRA weights (+ peft config) for this cell."""
    Path(snapshot_dir).mkdir(parents=True, exist_ok=True)
    tag = _checkpoint_tag(model_id, nbits, gsize, algorithm, boost_stop, top_m)
    fp = os.path.join(snapshot_dir, f"{tag}.lora.pt")
    PeftUtils.save_lora_weights(model, fp)
    logger.info("Saved LoRA checkpoint to %s", fp)
    return fp


# Adapted from hqq_plus.py / huggingface transformers v4.2.2 perplexity recipe.
def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True):
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    encodings = tokenizer("\n\n".join(dataset["text"]), return_tensors="pt")
    encodings["input_ids"] = encodings["input_ids"].to(DEVICE)

    lls, t = [], []
    for i in tqdm(range(0, encodings["input_ids"].size(1), stride), disable=not verbose):
        begin_loc = max(i + stride - max_length, 0)
        end_loc = min(i + stride, encodings["input_ids"].size(1))
        trg_len = end_loc - i
        input_ids = encodings["input_ids"][:, begin_loc:end_loc]
        target_ids = input_ids.clone()
        target_ids[:, :-trg_len] = -100  # ignore context

        t1 = time.time()
        with torch.no_grad():
            log_likelihood = model(input_ids, labels=target_ids).loss * trg_len
        torch.cuda.synchronize()
        t2 = time.time()
        t.append((t2 - t1))
        lls.append(log_likelihood)
        del input_ids, target_ids

    ppl = np.round(float(torch.exp(torch.stack(lls).sum() / end_loc)), 4)
    pred_time = np.round(np.mean(t), 3)
    del encodings
    cleanup()
    return {"perplexity": ppl, "prediction_time": pred_time}


def _train_sft(model, tokenizer, sft_kwargs):
    tokenizer.pad_token = tokenizer.unk_token
    tokenizer.padding_side = "right"
    tokenizer.add_bos_token = False
    tokenizer.add_eos_token = False

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")

    training_args = SFTConfig(
        output_dir=sft_kwargs.get("output_dir", "."),
        per_device_train_batch_size=sft_kwargs.get("batch_size", 1),
        gradient_accumulation_steps=sft_kwargs.get("grad_acc", 1),
        learning_rate=sft_kwargs.get("lr", 1e-5),
        logging_steps=sft_kwargs.get("logging_st", 1),
        num_train_epochs=sft_kwargs.get("n_epochs", 2),
        max_steps=sft_kwargs.get("max_steps", -1),
        remove_unused_columns=False,
        bf16=True,
        max_grad_norm=1.0,
        save_steps=10000000,
        lr_scheduler_type="cosine",
        max_length=sft_kwargs.get("max_tokens", 1024),
        dataset_text_field="text",
        packing=True,
    )

    trainer = SFTTrainer(
        model=WrappedModel(model),
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=None,
        args=training_args,
    )
    model.is_parallelizable = False
    trainer.is_model_parallel = False
    trainer.place_model_on_device = False
    model.train()
    t1 = time.time()
    trainer.train()
    return time.time() - t1


def run_one(
    model_id,
    nbits,
    gsize,
    algorithm,
    boost_stop,
    top_m,
    sft_kwargs,
    snapshot_dir=None,
):
    """Run one experiment cell and return its metrics.

    For ``algorithm == "fp16"`` the model is loaded and evaluated as-is:
    quantization, LoRA and SFT are all skipped (upper-bound baseline). For the
    HQQ+ algorithms (``HQQ+``/``kurtboost``) the backbone is quantized, LoRA
    is attached per the chosen allocation, SFT-trained, and (optionally) the
    LoRA weights are checkpointed under ``snapshot_dir``.
    """
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=COMPUTE_DTYPE,
        attn_implementation="sdpa",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    n_boosted = 0
    train_duration = 0.0
    if algorithm == "fp16":
        # Baseline: no quantization, no SFT.
        model = model.to(DEVICE)
    else:
        ok, metric_fp = get_mxq_quant_meta_data_file(model_id)
        if not ok:
            raise ValueError(f"Kurtosis metric file not found: {metric_fp}")

        quant_config = BaseQuantizeConfig(
            nbits=nbits, group_size=gsize, quant_scale=False, quant_zero=False, axis=0
        )
        AutoHQQHFModel.quantize_model(
            model,
            quant_config=quant_config,
            compute_dtype=COMPUTE_DTYPE,
            device=DEVICE,
        )

        lora_configs, module_outliers = build_lora_configs(
            algorithm, metric_fp, boost_stop, top_m
        )
        n_boosted = sum(len(v) for v in module_outliers.values())
        logger.info(
            "algorithm=%s boosted (layer,module) pairs=%d outliers=%s",
            algorithm,
            n_boosted,
            module_outliers,
        )

        add_lora_per_layer(model, lora_configs)
        HQQLinear.set_backend(HQQBackend.ATEN_BACKPROP)
        model.config.use_cache = False

        train_duration = _train_sft(model, tokenizer, sft_kwargs)

        tokenizer.add_bos_token = True
        tokenizer.add_eos_token = False
        PeftUtils.cast_lora_weights(model, dtype=COMPUTE_DTYPE)

        if snapshot_dir:
            _save_checkpoint(
                model,
                snapshot_dir,
                model_id,
                nbits,
                gsize,
                algorithm,
                boost_stop,
                top_m,
            )

    ppl_res = eval_wikitext2(model, tokenizer)
    mem_allot, mem_reserved = get_memory_metrics()

    metric = {
        "model": model_id.split("/")[1],
        "algorithm": algorithm,
        "nbits": nbits if algorithm != "fp16" else "",
        "group_size": gsize if algorithm != "fp16" else "",
        "boost_stop": boost_stop if algorithm == "kurtboost" else "",
        "top_m": top_m if algorithm == "kurtboost" else "",
        "n_boosted_pairs": n_boosted,
        "ppl_wikitext": ppl_res["perplexity"],
        "pred_time_wikitext": ppl_res["prediction_time"],
        "train_duration": np.round(train_duration, 2),
        "ppl_mem_allot": mem_allot,
        "ppl_mem_reserved": mem_reserved,
    }

    del model
    cleanup()
    return metric


def gen_experiment_items(models, backbones, algorithms):
    dikts = []
    for model_id in models:
        for algorithm in algorithms:
            if algorithm == "fp16":
                # fp16 is independent of the quantization backbone, so it is
                # evaluated once per model (nbits/group_size are unused: 0).
                dikts.append(
                    {
                        "model": model_id,
                        "nbits": 0,
                        "group_size": 0,
                        "algorithm": algorithm,
                    }
                )
            else:
                for nbits, gsize in backbones:
                    dikts.append(
                        {
                            "model": model_id,
                            "nbits": nbits,
                            "group_size": gsize,
                            "algorithm": algorithm,
                        }
                    )
    return pd.DataFrame(dikts)


def _load_todo_tasks(result_dir, experiment_name, models, backbones, algorithms):
    df_all = gen_experiment_items(models, backbones, algorithms)
    progress_path = os.path.join(result_dir, experiment_name, "progress.csv")
    if Path(progress_path).exists():
        df_saved = pd.read_csv(progress_path)
        df_all = df_all.merge(
            df_saved, how="left", on=["model", "nbits", "group_size", "algorithm"]
        )
        df_todo = df_all.query("status != status or status != 1")
    else:
        df_all["status"] = 0
        df_all["completion_time"] = ""
        df_todo = df_all
    print("*" * 72)
    print(df_all)
    cnt_todo, cnt_tot = len(df_todo), len(df_all)
    print(f"Todo:{cnt_todo}, Done: {cnt_tot - cnt_todo}, Total: {cnt_tot}")
    print("*" * 72)
    return df_all, df_todo, progress_path


def do_experiment(
    experiment_name,
    models,
    backbones,
    algorithms,
    boost_stop=1,
    top_m=1,
    result_dir="results",
    snapshot_dir=None,
    sft_kwargs=None,
):
    sft_kwargs = sft_kwargs or {}
    df_all, df_todo, progress_path = _load_todo_tasks(
        result_dir, experiment_name, models, backbones, algorithms
    )
    if len(df_todo) == 0:
        print("Tasks completed!")
        return

    for _, row in df_todo.iterrows():
        model_id = row["model"]
        nbits, gsize, algorithm = row["nbits"], row["group_size"], row["algorithm"]
        print("*" * 72)
        if algorithm == "fp16":
            print(f"FP16 baseline: {model_id}")
            cfg = "fp16"
        else:
            print(f"HQQ+ SFT: {model_id} backbone={nbits}b/g{gsize} algo={algorithm}")
            cfg = f"{int(nbits)}b_g{int(gsize)}_{algorithm}"
        print("*" * 72)
        _release_gpu()

        metric = run_one(
            model_id,
            int(nbits),
            int(gsize),
            algorithm,
            boost_stop,
            top_m,
            sft_kwargs,
            snapshot_dir=snapshot_dir,
        )
        save_partial_metric(
            experiment_name, "hqq_plus", model_id, cfg, metric, result_dir
        )
        df_all.loc[
            (df_all["model"] == model_id)
            & (df_all["nbits"] == nbits)
            & (df_all["group_size"] == gsize)
            & (df_all["algorithm"] == algorithm),
            ["status", "completion_time"],
        ] = 1, datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        persist_progress(df_all, progress_path)

    combine_metrics(experiment_name, result_dir)


def _release_gpu():
    cleanup()
    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
