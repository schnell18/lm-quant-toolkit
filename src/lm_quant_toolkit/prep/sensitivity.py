import functools
import gc
import inspect
import os
import re
import time
from collections import defaultdict
from typing import List, Union

import pandas as pd
import torch
import torch.nn as nn
from datasets import load_dataset
from hqq.core.quantize import Quantizer as hQuant
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.llama.modeling_llama import LlamaDecoderLayer


def quant_hqq(tensor, nbits, group_size=64, optimize=True):
    wq, meta = hQuant.quantize(
        tensor, nbits=nbits, group_size=group_size, optimize=optimize
    )
    return hQuant.dequantize(wq, meta)


def get_named_linears(module):
    return {name: m for name, m in module.named_modules() if isinstance(m, nn.Linear)}


def get_layers_for_scaling(module: LlamaDecoderLayer, input_feat, module_kwargs):
    layers = []

    # attention input
    layers.append(
        dict(
            part="attn_in",
            prev_op=module.input_layernorm,
            layers=[
                module.self_attn.q_proj,
                module.self_attn.k_proj,
                module.self_attn.v_proj,
            ],
            inp=input_feat["self_attn.q_proj"],
            module2inspect=module.self_attn,
            kwargs=module_kwargs,
        )
    )

    # attention out
    # Please refer to https://github.com/mit-han-lab/llm-awq/pull/67#issue-1850622696
    if module.self_attn.v_proj.weight.shape == module.self_attn.o_proj.weight.shape:
        layers.append(
            dict(
                part="attn_out",
                prev_op=module.self_attn.v_proj,
                layers=[module.self_attn.o_proj],
                inp=input_feat["self_attn.o_proj"],
            )
        )

    # linear 1
    layers.append(
        dict(
            part="mlp_gate",
            prev_op=module.post_attention_layernorm,
            layers=[module.mlp.gate_proj, module.mlp.up_proj],
            inp=input_feat["mlp.gate_proj"],
            module2inspect=module.mlp,
        )
    )

    # linear 2
    layers.append(
        dict(
            part="mlp_down",
            prev_op=module.mlp.up_proj,
            layers=[module.mlp.down_proj],
            inp=input_feat["mlp.down_proj"],
        )
    )

    return layers


def clear_memory(weight=None):
    if weight is not None:
        del weight
    gc.collect()
    torch.cuda.empty_cache()


def get_best_device(idx=None):
    if os.environ.get("USE_CPU_FOR_SENSITIVITY", None) == "1":
        return "cpu"
    if torch.cuda.is_available():
        if idx is None:
            return "cuda:0"
        else:
            return "cuda:" + str(idx % torch.cuda.device_count())
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


def get_calib_dataset(
    data: Union[str, List[str], List[List[int]]] = "pileval",
    tokenizer=None,
    n_samples=512,
    block_size=512,
    split="train",
    text_column="text",
):
    if isinstance(data, str):
        if data == "pileval":
            dataset = load_dataset("mit-han-lab/pile-val-backup", split="validation")
        elif data == "wikitext":
            dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="validation")
        elif data == "c4":
            dataset = load_dataset(
                "allenai/c4",
                data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
                split="validation",
                download_mode="reuse_dataset_if_exists",
            )
        else:
            dataset = load_dataset(data, split=split)

        dataset = dataset.shuffle(seed=42)

    elif isinstance(data, list):
        if isinstance(data[0], str):
            dataset = [{text_column: text} for text in data]
        elif isinstance(data[0][0], int):
            dataset = data
        else:
            raise NotImplementedError(
                "Either pass a string to a huggingface dataset or a list"
                "that is preprocessed with one sample of text per element"
                " or a list of list of int for tokenized words."
            )
    else:
        raise NotImplementedError(
            "Either pass a string to a huggingface dataset or a list"
            "that is preprocessed with one sample of text per element"
            " or a list of list of int for tokenized words."
        )

    samples = []
    n_run = 0
    for data in dataset:
        if isinstance(data, list):
            line_encoded = data
        else:
            line = data[text_column]
            line = line.strip()
            line_encoded = tokenizer.encode(line)
        if len(line_encoded) > 512:
            continue
        sample = torch.tensor([line_encoded])
        if sample.numel() == 0:
            continue
        samples.append(sample)
        n_run += 1
        if n_run == n_samples:
            break
    # now concatenate all samples and split according to block size
    cat_samples = torch.cat(samples, dim=1)
    n_split = cat_samples.shape[1] // block_size
    return [
        cat_samples[:, i * block_size : (i + 1) * block_size] for i in range(n_split)
    ]


class SensitiveLayerFinder:
    def __init__(
        self,
        model,
        model_name,
        w_bit,
        group_size,
        tokenizer,
        calib_data="pileval",
        split="train",
        text_column="text",
    ) -> None:
        self.model = model
        self.model_name = model_name
        self.tokenizer = tokenizer
        self.w_bit = w_bit
        self.group_size = group_size
        self.calib_data = calib_data
        self.split = split
        self.text_column = text_column
        self.layers, self.module_kwargs, self.inps = self.init_quant()

    @torch.no_grad()
    def measure(self, csv_fp):
        dikts = []
        cfg = f"b{self.w_bit}g{self.group_size}"
        for i in tqdm(
            range(len(self.layers)), desc=f"{self.model_name}-{cfg}-{self.calib_data}"
        ):
            # Move module and inputs to correct device
            common_device = next(self.layers[i].parameters()).device
            if common_device is None or str(common_device) == "cpu":
                best_device = get_best_device(i)
                self.layers[i] = self.layers[i].to(best_device)
                common_device = next(self.layers[i].parameters()).device

            if self.module_kwargs.get("position_ids") is not None:
                self.module_kwargs["position_ids"] = self.module_kwargs[
                    "position_ids"
                ].to(common_device)

            if self.module_kwargs.get("attention_mask") is not None:
                self.module_kwargs["attention_mask"] = self.module_kwargs[
                    "attention_mask"
                ].to(common_device)

            self.inps = self.inps.to(common_device)
            named_linears = get_named_linears(self.layers[i])
            input_feat = self._get_input_feat(self.layers[i], named_linears)
            clear_memory()

            module_config = get_layers_for_scaling(
                self.layers[i], input_feat, self.module_kwargs
            )

            for layer in module_config:
                part = layer.pop("part", "Unknown")
                mse = self._measure_layer_sensitivity(self.layers[i], **layer)
                dikts.append(
                    {
                        "dataset": self.calib_data,
                        "part": part,
                        "model": self.model_name,
                        "nbits": self.w_bit,
                        "group_size": self.group_size,
                        "layer": i,
                        "sensitivity": mse,
                    }
                )

            del module_config
            del input_feat
            clear_memory()
        return dikts

    def _measure_layer_sensitivity(
        self,
        module,
        prev_op,
        layers: List[nn.Linear],
        inp: torch.Tensor,
        module2inspect=None,
        kwargs={},
    ):
        if module2inspect is None:
            assert len(layers) == 1
            module2inspect = layers[0]

        # Put x on the right device
        inp = inp.to(next(module2inspect.parameters()).device)

        module_kwargs = self._sanitize_kwargs(kwargs, module2inspect)

        fp16_output = module2inspect(inp, **module_kwargs)
        if isinstance(fp16_output, tuple):
            fp16_output = fp16_output[0]

        # Quantize the weights
        for fc in layers:
            # call HQQ to quantize
            fc.weight.data = quant_hqq(fc.weight.data, self.w_bit, self.group_size)

        # W * X
        int_w_output = module2inspect(inp, **module_kwargs)
        if isinstance(int_w_output, tuple):
            int_w_output = int_w_output[0]

        # compute mean squared error (L2 norm)
        mse = (fp16_output - int_w_output).float().pow(2).mean().item()
        del fp16_output
        del int_w_output
        clear_memory()
        return mse

    def init_quant(self, n_samples=128, seqlen=512):
        modules = self.model.model.layers
        samples = get_calib_dataset(
            data=self.calib_data,
            tokenizer=self.tokenizer,
            n_samples=n_samples,
            block_size=seqlen,
            split=self.split,
            text_column=self.text_column,
        )
        samples = torch.cat(samples, dim=0)

        inps = []
        layer_kwargs = {}

        best_device = get_best_device()
        modules[0] = modules[0].to(best_device)
        self.model.model.embed_tokens = self.model.model.embed_tokens.to(best_device)

        # get input and kwargs to layer 0
        # with_kwargs is only supported in PyTorch 2.0
        # use this Catcher hack for now
        class Catcher(nn.Module):
            def __init__(self, module):
                super().__init__()
                self.module = module

            def forward(self, *args, **kwargs):
                # assume first input to forward is hidden states
                if len(args) > 0:
                    hidden_states = args[0]
                    del args
                else:
                    first_key = list(kwargs.keys())[0]
                    hidden_states = kwargs.pop(first_key)

                inps.append(hidden_states)
                layer_kwargs.update(kwargs)
                raise ValueError  # early exit to break later inference

        # patch layer 0 to catch input and kwargs
        modules[0] = Catcher(modules[0])
        try:
            self.model(samples.to(next(self.model.parameters()).device))
        except ValueError:  # work with early exit
            pass
        modules[0] = modules[0].module  # restore

        # Update the layer kwargs with `prepare_inputs_for_generation` method
        # that takes care of everything to avoid unexpected errors.
        layer_kwargs = self.model.prepare_inputs_for_generation(samples, **layer_kwargs)
        # Pop the input_ids as they are not needed at all.
        layer_kwargs.pop("input_ids")

        del samples
        inps = inps[0]

        modules[0] = modules[0].cpu()
        self.model.model.embed_tokens = self.model.model.embed_tokens.to("cpu")

        clear_memory()

        if layer_kwargs.get("attention_mask") is not None:
            layer_kwargs["attention_mask"] = layer_kwargs["attention_mask"].to(
                best_device
            )

        return modules, layer_kwargs, inps

    def _get_input_feat(self, layer, named_linears):
        # firstly, get input features of all linear layers
        def cache_input_hook(m, x, y, name, feat_dict):
            x = x[0]
            x = x.detach().cpu()
            feat_dict[name].append(x)

        input_feat = defaultdict(list)
        handles = []

        for name in named_linears:
            handles.append(
                named_linears[name].register_forward_hook(
                    functools.partial(cache_input_hook, name=name, feat_dict=input_feat)
                )
            )
        self.inps = self.inps.to(next(layer.parameters()).device)  # in case multi-gpu
        # get output as next layer's input

        # Sanitize the kwargs in case we use transformers version that contains
        # kwargs that are not handled by the module.
        # Useful for trust_remote_code models.
        module_kwargs = self._sanitize_kwargs(self.module_kwargs, layer)

        self.inps = layer(self.inps, **module_kwargs)[0]
        for h in handles:
            h.remove()
        # now solve for scaling and clipping
        input_feat = {k: torch.cat(v, dim=0) for k, v in input_feat.items()}

        return input_feat

    def _sanitize_kwargs(self, inputs_kwargs, module):
        """
        Remove the arguments that are not supported in the module's
        forward pass to avoid breaking behaviour between different versions
        of transformers.

        Args:
            inputs_kwargs (`dict`):
                The input dictionary to pass to the model layer
            module (`torch.nn.Module`):
                Target module to quantize.
        """
        module_signature = inspect.signature(module.forward).parameters
        sanitized_kwargs = {}
        for k, v in inputs_kwargs.items():
            if k in module_signature:
                sanitized_kwargs[k] = v
        return sanitized_kwargs


def measure_sensitivity(models, cfgs, calib_datasets, csv_fp):
    pat = re.compile(r"b(\d)g(\d+)")
    bgs = []
    for cfg in cfgs:
        m = re.match(pat, cfg)
        if m:
            bgs.append((int(m.group(1)), int(m.group(2))))
    dikts = []
    for ds in calib_datasets:
        for bg in bgs:
            for model_path in models:
                short_name = model_path.split("/")[1]
                model = AutoModelForCausalLM.from_pretrained(
                    model_path,
                    torch_dtype=torch.float16,
                    device_map="auto",
                    offload_state_dict=False,
                    max_memory={0: "18GiB", "cpu": "60GiB"},
                )
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                finder = SensitiveLayerFinder(
                    model,
                    short_name,
                    bg[0],
                    bg[1],
                    tokenizer,
                    ds,
                )
                dikts.extend(finder.measure(csv_fp))
                clear_memory()
                time.sleep(2)

    df = pd.DataFrame(dikts)
    df.to_csv(csv_fp, index=False)


# with profile(
#     activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True
# ) as prof:
#     finder.identify(csv_fp)
# print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))
