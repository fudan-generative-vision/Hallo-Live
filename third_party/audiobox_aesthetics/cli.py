# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import json
from functools import partial
import logging
import itertools
from pathlib import Path

# import submitit
try:
    from .infer import initialize_predictor, load_dataset, main_predict
except ImportError:
    from infer import initialize_predictor, load_dataset, main_predict

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")


def parse_args():
    parser = argparse.ArgumentParser("CLI for audiobox-aesthetics inference")
    parser.add_argument(
        "--input_file",
        type=str,
        default="/inspire/ssd/project/sais-bio/public/lijiaye/t2av/self-forcing/audiobox-aesthetics/input.jsonl",
    )
    parser.add_argument("--ckpt", type=str)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--remote", action="store_true", default=False, help="Set true to run via SLURM")

    # remote == True
    parser.add_argument("--slurm-gpu", default=1, type=int, help="Slurm GPU")
    parser.add_argument("--slurm-cpu", default=10, type=int, help="Slurm CPU")
    parser.add_argument("--job-dir", default="/tmp", type=str, help="Slurm job directory")
    parser.add_argument("--partition", default="learn", type=str, help="Slurm partition")
    parser.add_argument("--qos", default="", type=str, help="Slurm QOS")
    parser.add_argument("--account", default="", type=str, help="Slurm account")
    parser.add_argument("--comment", default="", type=str, help="Slurm job comment")
    parser.add_argument(
        "--constraint",
        default="",
        type=str,
        help="Slurm constraint eg.: ampere80gb For using A100s or volta32gb for using V100s.",
    )
    parser.add_argument("--exclude", default="", type=str, help="Exclude certain nodes from the slurm job.")
    parser.add_argument("--array", default=100, type=int, help="Slurm max array parallelism")
    parser.add_argument("--chunk", default=1000, type=int, help="chunk size per instance")
    return parser.parse_args()


def app():
    args = parse_args()
    metadata = load_dataset(args.input_file, 0, 2**64)
    fn_wrapped = partial(
        main_predict,
        batch_size=args.batch_size,
        ckpt="/inspire/ssd/project/sais-bio/public/lijiaye/t2av/self-forcing/ckpt/checkpoint.pt",
    )
    outputs = fn_wrapped(metadata)
    print(outputs)
    first_output = json.loads(outputs[0]) if isinstance(outputs[0], str) else outputs[0]
    print(first_output["CE"])
    # print("\n".join(str(x) for x in outputs))


if __name__ == "__main__":
    """
    Example usage:
    
    Single node
    > python cli.py input.jsonl --batch-size 100 > output.jsonl

    Multi-node via SLURM
    > python cli.py input.jsonl --batch-size 100 --remote --array 5 --job-dir $HOME/slurm_logs/ --chunk 1000 > output.jsonl
    """

    app()
