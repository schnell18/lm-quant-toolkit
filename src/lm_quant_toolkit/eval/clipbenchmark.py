import os
import sys
import time
from pathlib import Path

from clip_benchmark.cli import get_parser_args, run


def eval_clip_benchmark(
    task,
    model_id,
    result_dir,
    quant_dir,
    quant_config,
    additional_args=[],
):
    # build args for clip_benchmark
    model_type = "open_clip_hqq" if quant_config else "open_clip"
    comps = model_id.split("/")
    elems = comps[1].split("-")
    model_name = "-".join(elems[1:4])
    pretrained = "-".join(elems[4:])
    extra_args = ""
    if quant_config:
        mixed = quant_config.get("mixed", None)
        if mixed:
            budget = quant_config["budget"]
            qmf = quant_config["quant_metrics_file"]
            weight_algo = quant_config["weight_algo"]
            boost_stop = quant_config["boost_stop"]
            top_m_layer = quant_config["top_m_layer"]
            extra_args = f"budget={budget},quant_metrics_file={qmf},weight_algo={weight_algo},boost_stop={boost_stop},top_m_layer={top_m_layer}"
        else:
            b = quant_config["weight_quant_params"]["nbits"]
            g = quant_config["weight_quant_params"]["group_size"]
            extra_args = f"nbits={b},group_size={g}"
    wds_cache_dir = os.path.join(quant_dir, "wds-dataset")
    Path(wds_cache_dir).mkdir(parents=True, exist_ok=True)
    out_file = f"benchmark_{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}-{model_type}.json"
    out_fp = os.path.join(result_dir, out_file)
    args_str = [
        "eval",
        f"--task={task}",
        f"--model_type={model_type}",
        "--dataset=wds/imagenet1k",
        "--dataset_root=https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main",
        f"--output={out_fp}",
        f"--wds_cache_dir={wds_cache_dir}",
        f"--extra_args={extra_args}",
    ]
    if len(additional_args) > 0:
        args_str.extend(additional_args)
    t1 = time.time()

    # clear the command line args for main python program
    # so that they are not passed to clip_benchmark CLI
    sys.argv = [sys.argv[0]]
    parser, _ = get_parser_args()
    cli_args = parser.parse_args(args_str)
    cli_args.dataset = cli_args.dataset[0]
    if any(["--train_split" in arg for arg in additional_args]):
        cli_args.train_split = cli_args.train_split[0]
    cli_args.model = model_name
    cli_args.pretrained = pretrained
    result_dict = run(cli_args)
    t2 = time.time()
    result_dict["__lm_quant_tk_duration"] = t2 - t1
    return result_dict


def eval_zeroshot_classification(metric, model_id, result_dir, quant_dir, quant_config):
    result_dict = eval_clip_benchmark(
        "zeroshot_classification",
        model_id,
        result_dir,
        quant_dir,
        quant_config,
    )
    metric["acc1_zeroshot_cls"] = result_dict["metrics"]["acc1"]
    metric["acc5_zeroshot_cls"] = result_dict["metrics"]["acc5"]
    metric["recall_zeroshot_cls"] = result_dict["metrics"]["mean_per_class_recall"]
    metric["duration_zeroshot_cls"] = result_dict["__lm_quant_tk_duration"]
    return metric


def eval_linear_probe(
    metric, model_id, result_dir, quant_dir, quant_config, feature_root
):
    additional_args = [
        "--batch_size=512",
        "--fewshot_lr=0.1",
        "--fewshot_epochs=20",
        "--train_split=train",
        "--test_split=test",
        f"--feature_root={feature_root}",
    ]
    result_dict = eval_clip_benchmark(
        "linear_probe",
        model_id,
        result_dir,
        quant_dir,
        quant_config,
        additional_args,
    )
    metric["acc1_linear_probe"] = result_dict["metrics"]["lp_acc1"]
    metric["acc5_linear_probe"] = result_dict["metrics"]["lp_acc5"]
    metric["recall_linear_probe"] = result_dict["metrics"]["lp_mean_per_class_recall"]
    metric["duration_linear_probe"] = result_dict["__lm_quant_tk_duration"]
    return metric
