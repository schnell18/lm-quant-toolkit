import json
import os
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from lm_eval import evaluator

HIGHER_IS_BETTER_SYMBOLS = {
    True: "↑",
    False: "↓",
}


def make_table(result_dict, column: str = "results", sort_results: bool = False):
    """Generate table of results."""
    from pytablewriter import LatexTableWriter, MarkdownTableWriter

    if column == "results":
        column_name = "Tasks"
    elif column == "groups":
        column_name = "Groups"

    all_headers = [
        column_name,
        "Version",
        "Filter",
        "n-shot",
        "Metric",
        "",
        "Value",
        "",
        "Stderr",
    ]

    md_writer = MarkdownTableWriter()
    latex_writer = LatexTableWriter()
    md_writer.headers = all_headers
    latex_writer.headers = all_headers

    values = []

    keys = result_dict[column].keys()
    if sort_results:
        # sort entries alphabetically by task or group name.
        # NOTE: we default here to false, because order matters
        # for multi-level table printing a la mmlu.
        # sorting here would mess that up
        keys = sorted(keys)
    for k in keys:
        dic = result_dict[column][k]
        version = result_dict["versions"].get(k, "    N/A")
        n = str(result_dict.get("n-shot", " ").get(k, " "))
        higher_is_better = result_dict.get("higher_is_better", {}).get(k, {})

        if "alias" in dic:
            k = dic.pop("alias")

        metric_items = dic.items()
        metric_items = sorted(metric_items)

        for (mf), v in metric_items:
            m, _, f = mf.partition(",")
            if m.endswith("_stderr"):
                continue

            hib = HIGHER_IS_BETTER_SYMBOLS.get(higher_is_better.get(m), "")

            v = "%.4f" % v if isinstance(v, float) else v

            if m + "_stderr" + "," + f in dic:
                se = dic[m + "_stderr" + "," + f]
                se = "   N/A" if se == "N/A" else "%.4f" % se
                values.append([k, version, f, n, m, hib, v, "±", se])
            else:
                values.append([k, version, f, n, m, hib, v, "", ""])
            k = ""
            version = ""
    md_writer.value_matrix = values
    latex_writer.value_matrix = values

    # todo: make latex table look good
    # print(latex_writer.dumps())

    return md_writer.dumps()


def handle_non_serializable(o):
    if isinstance(o, np.int64) or isinstance(o, np.int32):
        return int(o)
    elif isinstance(o, set):
        return list(o)
    else:
        return str(o)


def eval_llm_perf(
    experiment_name,
    task,
    model,
    metric,
    result_dir,
    verbosity="INFO",
):
    t1 = time.time()
    results = evaluator.simple_evaluate(
        model=model,
        tasks=task,
        # num_fewshot=args.num_fewshot,
        batch_size="auto:16",
        max_batch_size=16,
        # use_cache=True,
        # check_integrity=True,
        write_out=False,
        log_samples=True,
        system_instruction=None,
        # apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=False,
        # gen_kwargs=args.gen_kwargs,
        verbosity=verbosity,
        predict_only=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    t2 = time.time()

    if results is not None:
        timestamp = datetime.now().strftime("%Y-%m-%d%_H%M%S")
        lm_eval_result_fp = os.path.join(
            result_dir,
            experiment_name,
            f"results-{task}-{timestamp}.json",
        )
        Path(lm_eval_result_fp).parent.mkdir(parents=True, exist_ok=True)

        results.pop("samples", None)
        with open(lm_eval_result_fp, "w") as fh:
            json.dump(
                results,
                fh,
                indent=2,
                default=handle_non_serializable,
                ensure_ascii=False,
            )
        print(make_table(results))

    (
        metric,
        score,
    ) = _cal_eval_score(task, results)
    metric[metric] = score
    metric[f"duration_{task}"] = t2 - t1
    return metric


def _cal_eval_score(task, results):
    if task == "gsm8k":
        return "gsm8k", results["gsm8k"]["exact_match,flexible-extract"]
    elif task == "gpqa_diamond_zeroshot":
        return "gpqa", results["gpqa_diamond_zeroshot"]["acc,none"]
    else:
        return "unknown", 0
