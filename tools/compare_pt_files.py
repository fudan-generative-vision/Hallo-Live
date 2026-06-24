#!/usr/bin/env python3
"""
Compare two PyTorch .pt files exactly.

By default this script loads both files with ``torch.load`` and recursively
compares the deserialized contents. Use ``--mode bytes`` when you need a strict
bit-for-bit file comparison instead.

Examples:
    python tools/compare_pt_files.py a.pt b.pt
    python tools/compare_pt_files.py a.pt b.pt --mode bytes
    python tools/compare_pt_files.py a.pt b.pt --mode both --unsafe-full-load
"""

import argparse
import hashlib
import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch


MAX_DIFFERENCES = 20


@dataclass(frozen=True)
class Difference:
    path: str
    message: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare two PyTorch .pt files exactly.")
    parser.add_argument("left", type=Path, help="Path to the first .pt file.")
    parser.add_argument("right", type=Path, help="Path to the second .pt file.")
    parser.add_argument(
        "--mode",
        choices=("content", "bytes", "both"),
        default="content",
        help=(
            "Comparison mode. 'content' compares torch.load() results recursively; "
            "'bytes' compares file bytes; 'both' requires both to match."
        ),
    )
    parser.add_argument(
        "--max-differences",
        type=int,
        default=MAX_DIFFERENCES,
        help="Maximum number of content differences to print.",
    )
    parser.add_argument(
        "--unsafe-full-load",
        action="store_true",
        help=(
            "Allow full pickle deserialization for checkpoints with custom Python objects. "
            "Use only for trusted files."
        ),
    )
    return parser.parse_args()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def compare_file_bytes(left_path: Path, right_path: Path) -> list[Difference]:
    differences: list[Difference] = []

    left_size = left_path.stat().st_size
    right_size = right_path.stat().st_size
    if left_size != right_size:
        differences.append(Difference("$file", f"file size differs: {left_size} != {right_size}"))

    left_hash = sha256_file(left_path)
    right_hash = sha256_file(right_path)
    if left_hash != right_hash:
        differences.append(Difference("$file", f"sha256 differs: {left_hash} != {right_hash}"))

    return differences


def load_pt_file(path: Path, unsafe_full_load: bool) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=not unsafe_full_load)
    except TypeError:
        # Older torch versions do not support weights_only.
        if unsafe_full_load:
            return torch.load(path, map_location="cpu")
        raise
    except Exception as exc:
        hint = ""
        if not unsafe_full_load:
            hint = (
                "\nRetry with `--unsafe-full-load` if the file is trusted "
                "and contains custom Python objects."
            )
        raise RuntimeError(f"Failed to load {path}: {exc}{hint}") from exc


def compare_content(left: Any, right: Any, max_differences: int) -> list[Difference]:
    differences: list[Difference] = []
    compare_value(left, right, "$", differences, max(1, max_differences))
    return differences


def add_difference(differences: list[Difference], max_differences: int, path: str, message: str) -> None:
    if len(differences) < max_differences:
        differences.append(Difference(path, message))


def limit_reached(differences: list[Difference], max_differences: int) -> bool:
    return len(differences) >= max_differences


def compare_value(left: Any, right: Any, path: str, differences: list[Difference], max_differences: int) -> None:
    if limit_reached(differences, max_differences):
        return

    if torch.is_tensor(left) or torch.is_tensor(right):
        compare_tensor(left, right, path, differences, max_differences)
        return

    if isinstance(left, Mapping) or isinstance(right, Mapping):
        compare_mapping(left, right, path, differences, max_differences)
        return

    if is_sequence(left) or is_sequence(right):
        compare_sequence(left, right, path, differences, max_differences)
        return

    if type(left) is not type(right):
        add_difference(differences, max_differences, path, f"type differs: {type(left).__name__} != {type(right).__name__}")
        return

    if isinstance(left, float):
        if not floats_equal(left, right):
            add_difference(differences, max_differences, path, f"value differs: {left!r} != {right!r}")
        return

    try:
        equal = left == right
    except Exception as exc:
        add_difference(differences, max_differences, path, f"cannot compare {type(left).__name__}: {exc}")
        return

    if isinstance(equal, bool):
        if not equal:
            add_difference(differences, max_differences, path, f"value differs: {left!r} != {right!r}")
        return

    add_difference(
        differences,
        max_differences,
        path,
        f"comparison for {type(left).__name__} did not return a bool; add a custom comparator if needed",
    )


def compare_tensor(
    left: Any, right: Any, path: str, differences: list[Difference], max_differences: int
) -> None:
    if not torch.is_tensor(left) or not torch.is_tensor(right):
        add_difference(differences, max_differences, path, f"type differs: {type(left).__name__} != {type(right).__name__}")
        return

    left_meta = tensor_metadata(left)
    right_meta = tensor_metadata(right)
    if left_meta != right_meta:
        add_difference(differences, max_differences, path, f"tensor metadata differs: {left_meta} != {right_meta}")
        return

    if left.layout != torch.strided:
        left_cmp = left.coalesce() if left.is_sparse else left
        right_cmp = right.coalesce() if right.is_sparse else right
        if not torch.equal(left_cmp, right_cmp):
            add_difference(differences, max_differences, path, "tensor values differ")
        return

    if not torch.equal(left, right):
        mismatch = first_tensor_mismatch(left, right)
        add_difference(differences, max_differences, path, f"tensor values differ{mismatch}")


def tensor_metadata(tensor: torch.Tensor) -> tuple[str, tuple[int, ...], str, str, bool, tuple[int, ...] | None]:
    stride = tuple(tensor.stride()) if tensor.layout == torch.strided else None
    return (
        type(tensor).__name__,
        tuple(tensor.shape),
        str(tensor.dtype),
        str(tensor.layout),
        bool(tensor.requires_grad),
        stride,
    )


def first_tensor_mismatch(left: torch.Tensor, right: torch.Tensor) -> str:
    if left.numel() == 0:
        return ""

    mismatch_mask = left != right
    if mismatch_mask.dtype is torch.bool and mismatch_mask.any().item():
        flat_index = mismatch_mask.reshape(-1).nonzero()[0].item()
        index = unravel_index(flat_index, tuple(left.shape))
        left_value = left.reshape(-1)[flat_index].item()
        right_value = right.reshape(-1)[flat_index].item()
        return f" at {index}: {left_value!r} != {right_value!r}"

    return ""


def unravel_index(flat_index: int, shape: tuple[int, ...]) -> tuple[int, ...]:
    if not shape:
        return ()

    result = []
    for size in reversed(shape):
        result.append(flat_index % size)
        flat_index //= size
    return tuple(reversed(result))


def compare_mapping(
    left: Any, right: Any, path: str, differences: list[Difference], max_differences: int
) -> None:
    if not isinstance(left, Mapping) or not isinstance(right, Mapping):
        add_difference(differences, max_differences, path, f"type differs: {type(left).__name__} != {type(right).__name__}")
        return

    if type(left) is not type(right):
        add_difference(differences, max_differences, path, f"mapping type differs: {type(left).__name__} != {type(right).__name__}")
        if limit_reached(differences, max_differences):
            return

    left_keys = list(left.keys())
    right_keys = list(right.keys())
    if left_keys != right_keys:
        compare_key_lists(left_keys, right_keys, path, differences, max_differences)
        if limit_reached(differences, max_differences):
            return

    for key in left_keys:
        if key in right:
            compare_value(left[key], right[key], join_path(path, key), differences, max_differences)
            if limit_reached(differences, max_differences):
                return


def compare_key_lists(
    left_keys: list[Any], right_keys: list[Any], path: str, differences: list[Difference], max_differences: int
) -> None:
    left_counter = Counter(left_keys)
    right_counter = Counter(right_keys)
    missing = list((left_counter - right_counter).elements())
    extra = list((right_counter - left_counter).elements())

    if missing:
        add_difference(differences, max_differences, path, f"keys only in left: {format_limited_list(missing)}")
    if extra:
        add_difference(differences, max_differences, path, f"keys only in right: {format_limited_list(extra)}")
    if not missing and not extra:
        add_difference(differences, max_differences, path, "key order differs")


def compare_sequence(
    left: Any, right: Any, path: str, differences: list[Difference], max_differences: int
) -> None:
    if not is_sequence(left) or not is_sequence(right):
        add_difference(differences, max_differences, path, f"type differs: {type(left).__name__} != {type(right).__name__}")
        return

    if type(left) is not type(right):
        add_difference(differences, max_differences, path, f"sequence type differs: {type(left).__name__} != {type(right).__name__}")
        if limit_reached(differences, max_differences):
            return

    if len(left) != len(right):
        add_difference(differences, max_differences, path, f"length differs: {len(left)} != {len(right)}")
        if limit_reached(differences, max_differences):
            return

    for index, (left_value, right_value) in enumerate(zip(left, right)):
        compare_value(left_value, right_value, f"{path}[{index}]", differences, max_differences)
        if limit_reached(differences, max_differences):
            return


def is_sequence(value: Any) -> bool:
    return isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray))


def floats_equal(left: float, right: float) -> bool:
    if math.isnan(left) and math.isnan(right):
        return True
    return left == right


def format_limited_list(values: list[Any], limit: int = 10) -> str:
    shown = ", ".join(repr(value) for value in values[:limit])
    if len(values) > limit:
        shown += f", ... ({len(values) - limit} more)"
    return shown


def join_path(path: str, key: Any) -> str:
    if isinstance(key, str) and key.isidentifier():
        return f"{path}.{key}"
    return f"{path}[{key!r}]"


def print_differences(title: str, differences: list[Difference]) -> None:
    print(f"{title}: different")
    for difference in differences:
        print(f"  {difference.path}: {difference.message}")


def main() -> int:
    args = parse_args()
    left_path = args.left.expanduser().resolve()
    right_path = args.right.expanduser().resolve()

    for path in (left_path, right_path):
        if not path.is_file():
            raise FileNotFoundError(f"File not found: {path}")

    all_same = True

    if args.mode in {"bytes", "both"}:
        byte_differences = compare_file_bytes(left_path, right_path)
        if byte_differences:
            all_same = False
            print_differences("Byte comparison", byte_differences)
        else:
            print("Byte comparison: identical")

    if args.mode in {"content", "both"}:
        left_obj = load_pt_file(left_path, args.unsafe_full_load)
        right_obj = load_pt_file(right_path, args.unsafe_full_load)
        content_differences = compare_content(left_obj, right_obj, args.max_differences)
        if content_differences:
            all_same = False
            print_differences("Content comparison", content_differences)
        else:
            print("Content comparison: identical")

    return 0 if all_same else 1


if __name__ == "__main__":
    raise SystemExit(main())
