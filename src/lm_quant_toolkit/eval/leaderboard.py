import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
from lm_eval import evaluator
from lm_eval.tasks import TaskManager

from lm_quant_toolkit.utils.hub import get_hf_model_storge_base_dir

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


def eval_llm_leaderboard(
    experiment_name,
    model_id,
    quant_method,
    confg_name,
    quantized,
    metric,
    quant_base_dir,
    result_dir,
    verbosity="INFO",
):
    if quantized:
        quant_dir = os.path.join(
            quant_base_dir, f"{model_id}-{confg_name}-{quant_method.lower()}"
        )
        # prepare the quantized model by copying tokenizer files
        _prepare_tokenizer_files(model_id, quant_dir)
        model_args = f"pretrained={quant_dir},quant_method={quant_method}"
    else:
        model_args = f"pretrained={model_id}"

    t1 = time.time()
    task_manager = TaskManager(verbosity)
    results = evaluator.simple_evaluate(
        model="hf",
        model_args=model_args,
        tasks="leaderboard",
        # num_fewshot=args.num_fewshot,
        batch_size="auto:16",
        max_batch_size=16,
        device="cuda:0",
        # use_cache=True,
        # check_integrity=True,
        write_out=False,
        log_samples=True,
        system_instruction=None,
        # apply_chat_template=args.apply_chat_template,
        fewshot_as_multiturn=False,
        # gen_kwargs=args.gen_kwargs,
        task_manager=task_manager,
        verbosity=verbosity,
        predict_only=False,
        random_seed=0,
        numpy_random_seed=1234,
        torch_random_seed=1234,
        fewshot_random_seed=1234,
    )
    t2 = time.time()

    if results is not None:
        lm_eval_result_fp = os.path.join(
            result_dir,
            experiment_name,
            f"{model_id.split('/')[1]}-{confg_name}-{quant_method.lower()}-results.json",
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

    ifeval, bbh, mathlevel5, gpqa, musr, mmlupro = _cal_leaderboard_score(results)
    metric["ifeval"] = ifeval
    metric["bbh"] = bbh
    metric["mathlevel5"] = mathlevel5
    metric["gpqa"] = gpqa
    metric["musr"] = musr
    metric["mmlupro"] = mmlupro
    metric["duration_leaderboard"] = t2 - t1
    return metric


def _cal_leaderboard_score(results):
    ifeval = _cal_leaderboard_ifeval_score(results)
    bbh = _cal_leaderboard_bbh(results)
    mathlevel5 = _cal_leaderboard_mathlevel5(results)
    gpqa = _cal_leaderboard_gpqa(results)
    musr = _cal_leaderboard_musr(results)
    mmlupro = _cal_leaderboard_mmlu_pro(results)
    return ifeval, bbh, mathlevel5, gpqa, musr, mmlupro


def _cal_leaderboard_gpqa(results):
    res_gpqa = results["results"]["leaderboard_gpqa"]
    value = res_gpqa.get("acc_norm,none", None)
    # aggregate data from sub tasks
    if value is None:
        subtask_names = results["group_subtasks"]["leaderboard_gpqa"]
        metrics = {}
        for name in subtask_names:
            metrics[name] = len(results["configs"][name]["doc_to_choice"])
        scores = []
        for metric, choices in metrics.items():
            value = results["results"][metric]["acc_norm,none"]
            scores.append(_cal_normalized_score(value, 1 / choices, 1.0))
        return sum(scores) / len(scores)

    else:
        return _cal_normalized_score(value, 1 / 4, 1.0)


def _cal_leaderboard_mathlevel5(results):
    res_math_hard = results["results"]["leaderboard_math_hard"]
    value = res_math_hard.get("exact_match,none", None)
    # aggregate data from sub tasks
    if value is None:
        subtask_names = results["group_subtasks"]["leaderboard_math_hard"]
        scores = []
        for metric in subtask_names:
            value = results["results"][metric]["exact_match,none"]
            scores.append(_cal_normalized_score(value, 0, 1.0))
        return sum(scores) / len(scores)
    else:
        return _cal_normalized_score(value, 0, 1.0)


def _cal_leaderboard_bbh(results):
    subtask_names = results["group_subtasks"]["leaderboard_bbh"]
    metrics = {}
    for name in subtask_names:
        metrics[name] = len(results["configs"][name]["doc_to_choice"])
    scores = []
    for metric, choices in metrics.items():
        value = results["results"][metric]["acc_norm,none"]
        scores.append(_cal_normalized_score(value, 1 / choices, 1.0))
    return sum(scores) / len(scores)


# noqa refer to: https://huggingface.co/docs/leaderboards/open_llm_leaderboard/normalization#example-normalizing-musr-scores
def _cal_leaderboard_musr(results):
    metrics = {
        "leaderboard_musr_murder_mysteries": 2,
        "leaderboard_musr_object_placements": 5,
        "leaderboard_musr_team_allocation": 3,
    }
    scores = []
    for metric, choices in metrics.items():
        value = results["results"][metric]["acc_norm,none"]
        scores.append(_cal_normalized_score(value, 1 / choices, 1.0))
    return sum(scores) / len(scores)


def _cal_leaderboard_mmlu_pro(results):
    value = results["results"]["leaderboard_mmlu_pro"]["acc,none"]
    return _cal_normalized_score(value, 0.1, 1.0)


def _cal_leaderboard_ifeval_score(results):
    scores = []
    value1 = results["results"]["leaderboard_ifeval"]["inst_level_strict_acc,none"]
    value2 = results["results"]["leaderboard_ifeval"]["prompt_level_strict_acc,none"]
    scores.append(_cal_normalized_score(value1, 0, 1.0))
    scores.append(_cal_normalized_score(value2, 0, 1.0))
    return sum(scores) / len(scores)


def _cal_normalized_score(value, lower_bound, higher_bound=1.0):
    if value < lower_bound:
        return 0
    return 100 * (value - lower_bound) / (higher_bound - lower_bound)


def _prepare_tokenizer_files(model_id, quant_dir):
    files = [
        "config.json",
        "tokenizer.model",
        "tokenizer.json",
        "special_tokens_map.json",
        "tokenizer_config.json",
    ]

    base_dir = get_hf_model_storge_base_dir(model_id)
    for f in files:
        src_fp = os.path.join(base_dir, f)
        if os.path.exists(src_fp):
            dst_fp = os.path.join(quant_dir, f)
            shutil.copyfile(src_fp, dst_fp, follow_symlinks=True)


def test_run(fp):
    with open(fp) as fh:
        results = json.load(fh)
        ifeval, bbh, mathlevel5, gpqa, musr, mmlupro = _cal_leaderboard_score(results)
        avg = (ifeval + bbh + mathlevel5 + gpqa + musr + mmlupro) / 6
        print(f"avg={avg:.2f}")
        print(f"ifeval={ifeval:.2f}")
        print(f"bbh={bbh:.2f}")
        print(f"mathlevel5={mathlevel5:.2f}")
        print(f"gpqa={gpqa:.2f}")
        print(f"musr={musr:.2f}")
        print(f"mmlupro={mmlupro:.2f}")


if __name__ == "__main__":
    # noqa result3.json is a copy of https://huggingface.co/datasets/open-llm-leaderboard/meta-llama__Llama-2-7b-hf-details/raw/main/meta-llama__Llama-2-7b-hf/results_2024-06-16T18-52-55.970021.json
    test_run("results/fp16_leaderboard/Llama-2-7b-hf-base-fp16-results.json")
    test_run("logs/result3.json")
