import time

from clip_benchmark.cli import get_parser_args, run


def eval_zeroshot_classification(metric, model_id, quantized, quant_config):
    # build args for clip_benchmark
    model_type = "open_clip_hqq" if quantized else "open_clip"
    comps = model_id.split("/")
    elems = comps[1].split("-")
    model_name = "-".join(elems[1:4])
    pretrained = "-".join(elems[4:])
    extra_args = ""
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
        f"--model_type={model_type}",
        "--dataset=wds/imagenet1k",
        "--dataset_root=https://huggingface.co/datasets/clip-benchmark/wds_{dataset_cleaned}/tree/main",
        f"--output=benchmark_{{dataset}}_{{pretrained}}_{{model}}_{{language}}_{{task}}-{model_type}.json",
        f"--extra_args={extra_args}",
    ]

    t1 = time.time()
    parser, _ = get_parser_args()
    cli_args = parser.parse_args(args_str)
    cli_args.dataset = cli_args.dataset[0]
    cli_args.model = model_name
    cli_args.pretrained = pretrained
    result_dict = run(cli_args)
    t2 = time.time()
    metric["acc1_zeroshot_cls"] = result_dict["metrics"]["acc1"]
    metric["acc5_zeroshot_cls"] = result_dict["metrics"]["acc5"]
    metric["recall_zeroshot_cls"] = result_dict["metrics"]["mean_per_class_recall"]
    metric["duration_zeroshot_cls"] = t2 - t1
    return metric
