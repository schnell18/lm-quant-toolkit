import gc
import time

import numpy as np
import torch
from datasets import load_dataset
from tqdm import tqdm


def cleanup():
    torch.cuda.empty_cache()
    gc.collect()


def eval_ptb(model, tokenizer, max_length=1024, stride=512, verbose=True):
    dataset = load_dataset("ptb_text_only", "penn_treebank", split="test")
    return eval_ppl(
        "ptb",
        model,
        tokenizer,
        dataset,
        text_column="sentence",
        max_length=max_length,
        stride=stride,
        verbose=verbose,
    )


def eval_c4(model, tokenizer, max_length=1024, stride=512, verbose=True):
    dataset = load_dataset(
        "allenai/c4",
        data_files={"validation": "en/c4-validation.00000-of-00008.json.gz"},
        split="validation",
        download_mode="reuse_dataset_if_exists",
    )
    # pick first 1100
    dataset = dataset[:1100]
    return eval_ppl(
        "C4",
        model,
        tokenizer,
        dataset,
        text_column="text",
        max_length=max_length,
        stride=stride,
        verbose=verbose,
    )


def eval_wikitext2(model, tokenizer, max_length=1024, stride=512, verbose=True):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="test")
    return eval_ppl(
        "wikitext",
        model,
        tokenizer,
        dataset,
        text_column="text",
        max_length=max_length,
        stride=stride,
        verbose=verbose,
    )


# Adapted from https://huggingface.co/transformers/v4.2.2/perplexity.html
def eval_ppl(
    ds_type,
    model,
    tokenizer,
    dataset,
    text_column="text",
    max_length=1024,
    stride=512,
    verbose=True,
):
    model.eval()
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.add_eos_token = False

    encodings = tokenizer("\n\n".join(dataset[text_column]), return_tensors="pt")

    encodings["input_ids"] = encodings["input_ids"].to("cuda")

    lls, t = [], []
    for i in tqdm(
        range(0, encodings["input_ids"].size(1), stride),
        desc=ds_type,
        disable=not verbose,
    ):
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
    if verbose:
        print(f"{ds_type} perplexity: {ppl}, time: {pred_time} sec")

    del encodings
    cleanup()

    return ppl, pred_time


def eval_ppls(model, tokenizer, metric):
    ppl_wikitext, duration_wikitext = eval_wikitext2(model, tokenizer, verbose=True)
    ppl_c4, duration_c4 = eval_c4(model, tokenizer, verbose=True)
    metric["ppl_wikitext"] = ppl_wikitext
    metric["ppl_c4"] = ppl_c4
    metric["duration_wikitext"] = duration_wikitext
    metric["duration_c4"] = duration_c4
    return metric
