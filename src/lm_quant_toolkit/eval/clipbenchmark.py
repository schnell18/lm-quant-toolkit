import time

from clip_benchmark.cli import get_parser_args, run


def eval_clip_benchmark(
    task,
    model_id,
    quantized,
    quant_config,
    additional_args,
):
    # build args for clip_benchmark
    model_type = "open_clip_hqq" if quantized else "open_clip"
    comps = model_id.split("/")
    elems = comps[1].split("-")
    model_name = "-".join(elems[1:4])
    pretrained = "-".join(elems[4:])
    extra_args = ""
    if quantized:
        mixed = quant_config.get("mixed", None)
        if mixed:
            budget = quant_config["budget"]
            qmf = quant_config["quant_metrics_file"]
            extra_args = f"budget={budget},quant_metrics_file={qmf}"
        else:
            b = quant_config["weight_quant_params"]["nbits"]
            g = quant_config["weight_quant_params"]["group_size"]
            extra_args = f"nbits={b},group_size={g}"
    args_str = [
        "eval",
        f"--task={task}",
        f"--model_type={model_type}",
        "--dataset=wds/imagenet1k",
        "--dataset_root=https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main",
        f"--output=benchmark_{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}-{model_type}.json",
        f"--extra_args={extra_args}",
    ]
    if additional_args is not None and len(additional_args) > 0:
        args_str.extend(additional_args)
    t1 = time.time()
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


def eval_zeroshot_classification(metric, model_id, quantized, quant_config):
    result_dict = eval_clip_benchmark(
        "zeroshot_classification",
        model_id,
        quantized,
        quant_config,
        None,
    )
    metric["acc1_zeroshot_cls"] = result_dict["metrics"]["acc1"]
    metric["acc5_zeroshot_cls"] = result_dict["metrics"]["acc5"]
    metric["recall_zeroshot_cls"] = result_dict["metrics"]["mean_per_class_recall"]
    metric["duration_zeroshot_cls"] = result_dict["__lm_quant_tk_duration"]
    return metric


def eval_linear_probe(metric, model_id, quantized, quant_config):
    additional_args = [
        "--batch_size=64",
        "--fewshot_lr=0.1",
        "--fewshot_epochs=20",
        "--train_split=train",
        "--test_split=test",
    ]

    result_dict = eval_clip_benchmark(
        "linear_probe",
        model_id,
        quantized,
        quant_config,
        additional_args,
    )
    metric["acc1_linear_probe"] = result_dict["metrics"]["lp_acc1"]
    metric["acc5_linear_probe"] = result_dict["metrics"]["lp_acc5"]
    metric["recall_linear_probe"] = result_dict["metrics"]["lp_mean_per_class_recall"]
    metric["duration_linear_probe"] = result_dict["__lm_quant_tk_duration"]
    return metric
