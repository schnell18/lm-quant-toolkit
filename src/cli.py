#!/usr/bin/env python
"""Console script for lm-quant-toolkit."""

import argparse
import sys

from hqq.core.quantize import BaseQuantizeConfig as HQQQuantConfig

from lm_quant_toolkit.eval.bench import (
    ALL_MODELS,
    AUTOAWQ_CONFIGS,
    BNB_CONFIGS,
    GPTQ_CONFIGS,
    MXQ_CONFIGS,
    do_expermient,
)
from lm_quant_toolkit.eval.bench_vit import ALL_MODELS as ALL_VIT_MODELS
from lm_quant_toolkit.eval.bench_vit import MXQ_CONFIGS as VIT_MXQ_CONFIGS
from lm_quant_toolkit.eval.bench_vit import do_expermient as do_expermient_vit
from lm_quant_toolkit.eval.common import HQQ_CONFIGS
from lm_quant_toolkit.misc.quant_sim import dump_mxq_configs, dump_mxq_objectives
from lm_quant_toolkit.misc.qweight import dump_quant_allocation


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
            "bnb",
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
            "eval_leaderboard",
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

    parser_llm.add_argument(
        "--weight-algo",
        default=None,
        type=str,
        help="Apply weighted F Norm for MiLP objective, None or `kurt-scaled`",
    )

    parser_llm.add_argument(
        "--boost-layer",
        nargs="+",
        default=None,
        type=int,
        help="Layers to increase memory budget",
    )

    parser_llm.add_argument(
        "--decline-layer",
        nargs="+",
        default=None,
        type=int,
        help="Layers to decrease memory budget",
    )

    parser_llm.add_argument(
        "--boost-stop",
        default=None,
        type=int,
        help="stops to increase",
    )

    parser_llm.add_argument(
        "--decline-stop",
        default=None,
        type=int,
        help="stops to decrease",
    )

    parser_llm.add_argument(
        "--factor",
        default=2.0,
        type=float,
        help="factor to apply",
    )

    parser_llm.add_argument(
        "--top-m-layer",
        default=1,
        type=int,
        help="The top m most sensitive layers to assign extra memory. 0 means all layers.",
    )

    parser_llm.add_argument(
        "--ablation",
        dest="ablation",
        action="store_true",
        help="Enable ablation mode",
    )
    parser_llm.add_argument(
        "--no-ablation",
        dest="ablation",
        action="store_false",
        help="Disable ablation mode",
    )
    parser_llm.set_defaults(ablation=False)

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
            "bnb",
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

    parser_vit.add_argument(
        "--weight-algo",
        default=None,
        type=str,
        help="Apply weighted F Norm for MiLP objective, None or `kurt-scaled`",
    )

    parser_vit.add_argument(
        "--boost-stop",
        default=None,
        type=int,
        help="stops to increase",
    )

    parser_vit.add_argument(
        "--decline-stop",
        default=None,
        type=int,
        help="stops to decrease",
    )

    parser_vit.add_argument(
        "--factor",
        default=2.0,
        type=float,
        help="factor to apply",
    )

    parser_vit.add_argument(
        "--top-m-layer",
        default=1,
        type=int,
        help="The top m most sensitive layers to assign extra memory. 0 means all layers.",
    )

    parser_vit.add_argument(
        "--ablation",
        dest="ablation",
        action="store_true",
        help="Enable ablation mode",
    )
    parser_vit.add_argument(
        "--no-ablation",
        dest="ablation",
        action="store_false",
        help="Disable ablation mode",
    )

    parser_dump = subparsers.add_parser("dump", help="Dump MXQ meta data")
    parser_dump.set_defaults(which="dump")

    parser_dump.add_argument(
        "--type",
        type=str,
        default=None,
        choices=[
            "objective",
            "quant_config",
            "quant_config_sim",
        ],
        help="Type of data to dump.",
    )
    parser_dump.add_argument(
        "--model",
        type=str,
        nargs="+",
        default="1",
        help="Model to evaluate",
    )
    parser_dump.add_argument(
        "--budget",
        type=str,
        default=None,
        nargs="+",
        help="Bit budgets",
    )
    parser_dump.add_argument(
        "--output-file",
        type=str,
        default="mxq-objectives.csv",
        help="Output file location",
    )
    parser_dump.add_argument(
        "--quant-snapshot-dir",
        default=None,
        type=str,
        help="directory to where quantized snapshots are stored",
    )
    parser_dump.add_argument(
        "--attempt",
        default=None,
        type=str,
        nargs="+",
        help="Experiment attempts",
    )
    parser_dump.add_argument(
        "--weight-algo",
        default=None,
        type=str,
        help="Apply weighted F Norm for MiLP objective, None or `kurt-scaled`",
    )
    parser_dump.add_argument(
        "--factor",
        default=None,
        type=float,
        help="Factor to apply to the prioritized weights",
    )
    parser_dump.add_argument(
        "--config",
        default=None,
        type=str,
        nargs="+",
        help="bit-group configurations",
    )
    parser_dump.add_argument(
        "--calib-dataset",
        default=None,
        type=str,
        nargs="+",
        help="calibration dataset(s) to use",
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
                if config_names is None:
                    algo_configs[algo] = HQQ_CONFIGS
                else:
                    algo_configs[algo] = [
                        cfg for cfg in HQQ_CONFIGS if cfg[0] in config_names
                    ]
            case "mxq":
                if config_names is None:
                    algo_configs[algo] = MXQ_CONFIGS
                else:
                    algo_configs[algo] = [
                        (
                            f"{bits:.2f}".replace(".", "_"),
                            HQQQuantConfig(mixed=True, budget=bits, quant_scale=True),
                        )
                        for bits in [float(cfg) for cfg in config_names]
                    ]
            case "awq":
                if config_names is None:
                    algo_configs[algo] = AUTOAWQ_CONFIGS
                else:
                    algo_configs[algo] = [
                        cfg for cfg in AUTOAWQ_CONFIGS if cfg[0] in config_names
                    ]
            case "gptq":
                if config_names is None:
                    algo_configs[algo] = GPTQ_CONFIGS
                else:
                    algo_configs[algo] = [
                        cfg for cfg in GPTQ_CONFIGS if cfg[0] in config_names
                    ]
            case "bnb":
                if config_names is None:
                    algo_configs[algo] = BNB_CONFIGS
                else:
                    algo_configs[algo] = [
                        cfg for cfg in BNB_CONFIGS if cfg[0] in config_names
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
                        (
                            f"{bits:.2f}".replace(".", "_"),
                            HQQQuantConfig(mixed=True, budget=bits, quant_scale=True),
                        )
                        for bits in [float(cfg) for cfg in config_names]
                    ]
    return algo_configs


def main():
    parser, base = get_parser_args()
    print(base)
    if not hasattr(base, "which"):
        parser.print_help()
        return 2
    try:
        if base.which == "llm":
            main_llm(base)
        elif base.which == "vit":
            main_vit(base)
        elif base.which == "dump":
            main_dump(base)
    except Exception as e:
        print(e)
        return 1
    return 0


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

    kwargs = {
        "weight_algo": args.weight_algo,
        "boost_layers": args.boost_layer,
        "decline_layers": args.decline_layer,
        "boost_stop": args.boost_stop,
        "decline_stop": args.decline_stop,
        "top_m_layer": args.top_m_layer,
        "ablation": args.ablation,
        "factor": args.factor,
    }
    do_expermient(
        experiment_name,
        models,
        tasks,
        quant_dir=args.quant_snapshot_dir,
        result_dir=args.result_dir,
        track_cuda_memory=args.track_cuda_memory,
        **kwargs,
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
    kwargs = {
        "weight_algo": args.weight_algo,
        "boost_stop": args.boost_stop,
        "decline_stop": args.decline_stop,
        "top_m_layer": args.top_m_layer,
        "ablation": args.ablation,
        "factor": args.factor,
    }
    do_expermient_vit(
        experiment_name,
        models,
        tasks,
        quant_dir=args.quant_snapshot_dir,
        result_dir=args.result_dir,
        track_cuda_memory=args.track_cuda_memory,
        **kwargs,
    )


def main_dump(args):
    if args.type == "objective":
        budgets = args.budget
        csv_fp = args.output_file
        indicies = [int(m) for m in args.model]
        models = [ALL_MODELS[i] for i in indicies]
        dump_mxq_objectives(models, budgets, csv_fp=csv_fp)
    elif args.type == "quant_config":
        quant_dir = args.quant_snapshot_dir
        attempts = args.attempt
        if "hqq" in attempts:
            budgets = args.budget
            algo = "hqq"
        else:
            budgets = [
                f"{bits:.2f}".replace(".", "_")
                for bits in [float(cfg) for cfg in args.budget]
            ]
            algo = "mxq"
        csv_fp = args.output_file
        indicies = [int(m) for m in args.model]
        models = [ALL_MODELS[i] for i in indicies]
        dump_quant_allocation(
            quant_dir,
            models,
            budgets,
            csv_fp=csv_fp,
            attempts=attempts,
            algo=algo,
        )
    elif args.type == "quant_config_sim":
        budgets = [bits for bits in [float(cfg) for cfg in args.budget]]
        algo = "mxq"
        csv_fp = args.output_file
        indicies = [int(m) for m in args.model]
        models = [ALL_MODELS[i] for i in indicies]
        dump_mxq_configs(
            models,
            budgets,
            csv_fp=csv_fp,
            weight_algo=args.weight_algo,
            factor=args.factor,
        )


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
