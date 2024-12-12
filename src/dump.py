#!/usr/bin/env python
"""Console script for lm-quant-toolkit."""

import argparse
import sys

from lm_quant_toolkit.prep.sensitivity import measure_sensitivity


def get_parser_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers()

    parser_dump = subparsers.add_parser(
        "dump", help="Evaluate and dump sensitivity data"
    )
    parser_dump.set_defaults(which="dump")

    parser_dump.add_argument(
        "--type",
        type=str,
        default=None,
        choices=[
            "sensitivity",
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
    parser_dump.add_argument(
        "--output-file",
        type=str,
        default="sensi.csv",
        help="Output file location",
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


def main():
    parser, base = get_parser_args()
    print(base)
    if not hasattr(base, "which"):
        parser.print_help()
        return 2
    try:
        if base.which == "dump":
            main_dump(base)
    except Exception as e:
        print(e)
        return 1
    return 0


def main_dump(args):
    if args.type == "sensitivity":
        csv_fp = args.output_file
        models = args.model
        cfgs = args.config
        calib_ds = args.calib_dataset
        quant_method = args.quant_method
        measure_sensitivity(models, quant_method, cfgs, calib_ds, csv_fp)


if __name__ == "__main__":
    sys.exit(main())  # pragma: no cover
