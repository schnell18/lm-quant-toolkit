#!/usr/bin/env python
"""Console script for lm-quant-toolkit."""

import argparse
import sys
from timeit import default_timer as timer

from lm_quant_toolkit.prep.fnorm import calc_fnorm_for_model
from lm_quant_toolkit.prep.sensitivity import measure_sensitivity
from lm_quant_toolkit.prep.wdist import calculate_kurtosis_llm
from lm_quant_toolkit.utils.hub import (
    LLAMA_MODELS,
    get_hf_model_storge_base_dir,
)


def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_sensi = subparsers.add_parser(
        "sensi", help="Evaluate and dump sensitivity data"
    )
    parser_sensi.set_defaults(which="sensi")
    parser_sensi.add_argument(
        "--model",
        type=str,
        nargs="+",
        default="1",
        help="Model to evaluate",
    )
    parser_sensi.add_argument(
        "--quant-method",
        type=str,
        choices=[
            "hqq",
            "rtn",
            "bnb",
        ],
        default="hqq",
        help="Output file location",
    )
    parser_sensi.add_argument(
        "--output-file",
        type=str,
        default="sensi.csv",
        help="Output file location",
    )
    parser_sensi.add_argument(
        "--config",
        default=None,
        type=str,
        nargs="+",
        help="bit-group configurations",
    )
    parser_sensi.add_argument(
        "--calib-dataset",
        default=None,
        type=str,
        nargs="+",
        help="calibration dataset(s) to use",
    )

    parser_fnorm = subparsers.add_parser("fnorm", help="Evaluate and dump FNorm data")
    parser_fnorm.set_defaults(which="fnorm")
    parser_fnorm.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="Model to evaluate",
    )
    parser_fnorm.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory",
    )

    parser_kurt = subparsers.add_parser(
        "kurtosis", help="Evaluate and dump model kurtosis data"
    )
    parser_kurt.set_defaults(which="kurtosis")
    parser_kurt.add_argument(
        "--model",
        type=str,
        nargs="+",
        help="Model to evaluate",
    )
    parser_kurt.add_argument(
        "--output-dir",
        type=str,
        default="data",
        help="Output directory",
    )

    args = parser.parse_args()
    return parser, args


def main():
    parser, base = get_parser_args()
    print(base)
    if not hasattr(base, "which"):
        parser.print_help()
        return 2
    try:
        if base.which == "sensi":
            main_sensi(base)
        elif base.which == "fnorm":
            main_fnorm(base)
        elif base.which == "kurtosis":
            main_kurt(base)
    except Exception as e:
        print(e)
        return 1
    return 0


def main_sensi(args):
    csv_fp = args.output_file
    models = args.model
    cfgs = args.config
    calib_ds = args.calib_dataset
    quant_method = args.quant_method
    measure_sensitivity(models, quant_method, cfgs, calib_ds, csv_fp)


def main_fnorm(args):
    if not args.model or len(args.model) < 1:
        raise ValueError("At least one model is required")
    output_dir = args.output_dir
    for model_id in args.model:
        model = LLAMA_MODELS[model_id]
        if not model:
            raise ValueError(f"Unsupported model: {model_id}")

        t1 = timer()
        base_dir = model.get("base_dir", None)
        model_base_dir = get_hf_model_storge_base_dir(model_id, base_dir)
        calc_fnorm_for_model(
            model_id,
            model_base_dir,
            model["layers"],
            output_dir,
        )
        t2 = timer()
        print(f"Finished {model_id} Frobenius norm metrics calc in {t2 - t1} seconds")


def main_kurt(args):
    if not args.model or len(args.model) < 1:
        raise ValueError("At least one model is required")
    output_dir = args.output_dir
    for model_id in args.model:
        model = LLAMA_MODELS[model_id]
        if not model:
            raise ValueError(f"Unsupported model: {model_id}")

        t1 = timer()
        base_dir = model.get("base_dir", None)
        model_base_dir = get_hf_model_storge_base_dir(model_id, base_dir)
        calculate_kurtosis_llm(
            model_id,
            model_base_dir,
            model["layers"],
            output_dir,
        )
        t2 = timer()
        print(f"Finished {model_id} Kurtosis metrics calc in {t2 - t1} seconds")


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
