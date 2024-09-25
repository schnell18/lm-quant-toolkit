"""Console script for lm-quant-toolkit."""

import argparse
import sys

from lm_quant_toolkit.eval.bench import (
    ALL_MODELS,
    AUTOAWQ_CONFIGS,
    GPTQ_CONFIGS,
    MXQ_CONFIGS,
    do_expermient,
)
from lm_quant_toolkit.eval.bench_vit import ALL_MODELS as ALL_VIT_MODELS
from lm_quant_toolkit.eval.bench_vit import MXQ_CONFIGS as VIT_MXQ_CONFIGS
from lm_quant_toolkit.eval.bench_vit import do_expermient as do_expermient_vit
from lm_quant_toolkit.eval.common import HQQ_CONFIGS


def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_llm = subparsers.add_parser("llm", help="Evaluate Language Model")
    parser_llm.set_defaults(which="llm")

    parser_llm.add_argument(
        "--model",
        type=str,
        nargs="+",
        default="1",
        help="Model to evaluate",
    )

    parser_llm.add_argument(
        "--algo",
        type=str,
        choices=[
            "fp16",
            "hqq",
            "mxq",
            "gptq",
            "awq",
        ],
        nargs="+",
        default=None,
        help="Algorithm to evaluate",
    )

    parser_llm.add_argument(
        "--config",
        type=str,
        default=None,
        nargs="+",
        help="Algorithm specific configuration to evaluate",
    )

    parser_llm.add_argument(
        "--task",
        type=str,
        default=None,
        choices=[
            "quant",
            "eval_model_storage",
            "eval_ppl",
        ],
        help="Task to evaluate on.",
    )

    parser_llm.add_argument(
        "--track-cuda-memory",
        action="store_true",
        default=False,
        help="Whether to dump CUDA memory snapshot",
    )

    parser_llm.add_argument(
        "--quant-snapshot-dir",
        default=None,
        type=str,
        help="directory to where quantized snapshots are stored",
    )

    parser_llm.add_argument(
        "--result-dir",
        default=None,
        type=str,
        help="directory to where evaluation results are stored",
    )

    parser_llm.add_argument(
        "--experiment-name",
        default=None,
        type=str,
        help="name of the experiment",
    )

    parser_vit = subparsers.add_parser("vit", help="Evaluate ViT models")
    parser_vit.set_defaults(which="vit")
    parser_vit.add_argument(
        "--model",
        type=str,
        nargs="+",
        default="1",
        help="Model to evaluate",
    )

    parser_vit.add_argument(
        "--algo",
        type=str,
        choices=[
            "fp16",
            "hqq",
            "mxq",
            "gptq",
            "awq",
        ],
        nargs="+",
        default=None,
        help="Algorithm to evaluate",
    )

    parser_vit.add_argument(
        "--config",
        type=str,
        default=None,
        nargs="+",
        help="Algorithm specific configuration to evaluate",
    )

    parser_vit.add_argument(
        "--task",
        type=str,
        default=None,
        choices=[
            "eval_linear_probe",
            "eval_zeroshot_cls",
        ],
        help="Task to evaluate on.",
    )

    parser_vit.add_argument(
        "--track-cuda-memory",
        action="store_true",
        default=False,
        help="Whether to dump CUDA memory snapshot",
    )

    parser_vit.add_argument(
        "--quant-snapshot-dir",
        default=None,
        type=str,
        help="directory to where quantized snapshots are stored",
    )

    parser_vit.add_argument(
        "--result-dir",
        default=None,
        type=str,
        help="directory to where evaluation results are stored",
    )

    parser_vit.add_argument(
        "--experiment-name",
        default=None,
        type=str,
        help="name of the experiment",
    )

    args = parser.parse_args()
    return parser, args


def _get_configs(algos, config_names):
    algo_configs = {}
    for algo in algos:
        match algo:
            case "fp16":
                algo_configs[algo] = [("base", {})]
            case "hqq":
                algo_configs[algo] = [
                    cfg for cfg in HQQ_CONFIGS if cfg[0] in config_names
                ]
            case "mxq":
                algo_configs[algo] = [
                    cfg for cfg in MXQ_CONFIGS if cfg[0] in config_names
                ]
            case "awq":
                algo_configs[algo] = [
                    cfg for cfg in AUTOAWQ_CONFIGS if cfg[0] in config_names
                ]
            case "gptq":
                algo_configs[algo] = [
                    cfg for cfg in GPTQ_CONFIGS if cfg[0] in config_names
                ]

    return algo_configs


def _get_vit_configs(algos, config_names):
    algo_configs = {}
    for algo in algos:
        match algo:
            case "fp16":
                algo_configs[algo] = [("base", {})]
            case "hqq":
                if config_names is None:
                    algo_configs[algo] = HQQ_CONFIGS
                else:
                    algo_configs[algo] = [
                        cfg for cfg in HQQ_CONFIGS if cfg[0] in config_names
                    ]
            case "mxq":
                if config_names is None:
                    algo_configs[algo] = VIT_MXQ_CONFIGS
                else:
                    algo_configs[algo] = [
                        cfg for cfg in VIT_MXQ_CONFIGS if cfg[0] in config_names
                    ]
    return algo_configs


def main():
    parser, base = get_parser_args()
    print(base)
    if not hasattr(base, "which"):
        parser.print_help()
        return
    if base.which == "llm":
        main_llm(base)
    elif base.which == "vit":
        main_vit(base)


def main_llm(args):
    # if len(args.algo) > 1 and args.config is not None:
    #     print("When config is specified, you can only evaluate one algorithm")
    #     return
    configs = _get_configs(args.algo, args.config)
    indicies = [int(m) for m in args.model]
    models = [ALL_MODELS[i] for i in indicies]
    tasks = {algo: {"type": args.task, "configs": configs[algo]} for algo in args.algo}
    experiment_name = args.experiment_name
    if experiment_name is None or len(experiment_name) < 3:
        algo_str = "-".join(args.algo)
        cfg_str = "-".join(args.config)
        experiment_name = f"{args.task}-{algo_str}-{cfg_str}"
    do_expermient(
        experiment_name,
        models,
        tasks,
        quant_dir=args.quant_snapshot_dir,
        result_dir=args.result_dir,
        track_cuda_memory=args.track_cuda_memory,
    )


def main_vit(args):
    configs = _get_vit_configs(args.algo, args.config)
    indicies = [int(m) for m in args.model]
    models = [ALL_VIT_MODELS[i] for i in indicies]
    tasks = {algo: {"type": args.task, "configs": configs[algo]} for algo in args.algo}
    experiment_name = args.experiment_name
    if experiment_name is None or len(experiment_name) < 3:
        algo_str = "-".join(args.algo)
        cfg_str = "-".join(args.config)
        experiment_name = f"{args.task}-{algo_str}-{cfg_str}"
    do_expermient_vit(
        experiment_name,
        models,
        tasks,
        quant_dir=args.quant_snapshot_dir,
        result_dir=args.result_dir,
        track_cuda_memory=args.track_cuda_memory,
    )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
