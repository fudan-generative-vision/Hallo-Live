#!/usr/bin/env python3
"""
Extract the top-level generator_ema entry from a PyTorch checkpoint.

Usage:
    python tools/extract_generator_ema.py --ckpt checkpoint_model_002000/model.pt
    python tools/extract_generator_ema.py --ckpt checkpoint_model_002000/model.pt --output generator_ema_bf16.pt
"""

import argparse
from pathlib import Path

import torch


def parse_args():
    parser = argparse.ArgumentParser(
        description="Extract the top-level `generator_ema` entry from a PyTorch checkpoint."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the source checkpoint, for example checkpoint_model_002000/model.pt",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Output path. Defaults to `<checkpoint_dir>/generator_ema_bf16.pt`.",
    )
    return parser.parse_args()


def convert_state_dict_to_bf16(state_dict: dict[str, object]) -> tuple[dict[str, object], int]:
    converted_state_dict = {}
    converted_tensor_count = 0

    for key, value in state_dict.items():
        if torch.is_tensor(value) and torch.is_floating_point(value):
            converted_state_dict[key] = value.to(dtype=torch.bfloat16)
            converted_tensor_count += 1
        else:
            converted_state_dict[key] = value

    return converted_state_dict, converted_tensor_count


def main():
    args = parse_args()

    checkpoint_path = args.ckpt.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    output_path = (
        args.output.expanduser().resolve()
        if args.output is not None
        else checkpoint_path.parent / "generator_ema_bf16.pt"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading checkpoint metadata from: {checkpoint_path}")
    state = torch.load(checkpoint_path, map_location="cpu", weights_only=True, mmap=True)

    if "generator_ema" in state:
        generator_ema = state["generator_ema"]
    elif "generator" in state:
        generator_ema = state["generator"]
    else:
        available_keys = ", ".join(state.keys()) if isinstance(state, dict) else type(state).__name__
        raise KeyError(f"`generator_ema` and `generator` not found in checkpoint. Available keys: {available_keys}")

    if not isinstance(generator_ema, dict):
        raise TypeError(f"`generator_ema` should be a dict, got: {type(generator_ema).__name__}")

    print(f"Found `generator_ema` with {len(generator_ema)} tensors/entries.")
    print("Converting floating-point tensors to bfloat16...")
    generator_ema_bf16, converted_tensor_count = convert_state_dict_to_bf16(generator_ema)
    print(f"Converted {converted_tensor_count} floating-point tensors to bfloat16.")
    generator_ema_bf16 = {k.replace("_fsdp_wrapped_module.", ""): v for k, v in generator_ema_bf16.items()}
    print(f"Saving to: {output_path}")
    torch.save(generator_ema_bf16, output_path)
    print("Done.")


if __name__ == "__main__":
    main()
