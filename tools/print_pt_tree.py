#!/usr/bin/env python3
"""
Print the tree structure of a PyTorch .pt file without materializing tensor data.

Features:
- Loads checkpoints in metadata mode with ``map_location="meta"`` and ``mmap=True``.
- Expands flat state_dict keys like ``model.blocks.0.attn.q.weight`` into a tree.
- Compresses repeated numeric children with identical structure, e.g. ``blocks [0..29]``.
- Prints tensor shape and dtype at every tensor leaf.

Example:
    python tools/print_pt_tree.py --ckpt /path/to/model.pt
"""

import argparse
from collections import OrderedDict
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch

MISSING = object()
COMPRESS_NUMERIC_CHILDREN_AT = 5
SHOW_TENSOR_COUNT_AT = 100


@dataclass
class Node:
    name: str
    actual_type: str | None = None
    value: Any = MISSING
    children: OrderedDict[str, "Node"] = field(default_factory=OrderedDict)
    tensor_count: int = 0
    signature: Any = None


@dataclass(frozen=True)
class CompressionInfo:
    start: int
    end: int
    count: int
    representative: Node


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Print the tree structure of a .pt file without loading tensor weights into memory."
    )
    parser.add_argument(
        "--ckpt",
        type=Path,
        required=True,
        help="Path to the .pt checkpoint file.",
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


def load_checkpoint_metadata(checkpoint_path: Path, unsafe_full_load: bool) -> object:
    try:
        return torch.load(
            checkpoint_path,
            map_location="meta",
            mmap=True,
            weights_only=not unsafe_full_load,
        )
    except Exception as exc:
        hint = ""
        if not unsafe_full_load:
            hint = (
                "\nRetry with `--unsafe-full-load` if the checkpoint is trusted "
                "and contains custom Python objects."
            )
        raise RuntimeError(f"Failed to load checkpoint metadata: {exc}{hint}") from exc


def is_dotted_mapping(mapping: Mapping[Any, Any]) -> bool:
    if not mapping:
        return False

    keys = list(mapping.keys())
    if not all(isinstance(key, str) for key in keys):
        return False

    dot_key_count = sum("." in key for key in keys)
    if dot_key_count == len(keys):
        return True

    return len(keys) >= 10 and dot_key_count / len(keys) >= 0.8


def object_to_node(name: str, value: object, actual_type: str | None = None) -> Node:
    node = Node(name=name, actual_type=actual_type or type(value).__name__)

    if torch.is_tensor(value):
        node.value = value
        return node

    if isinstance(value, Mapping):
        if is_dotted_mapping(value):
            node.actual_type = type(value).__name__
            for flat_key, child_value in value.items():
                insert_flat_key(node, str(flat_key).split("."), child_value)
            return node

        for child_name, child_value in value.items():
            child_name_str = str(child_name)
            node.children[child_name_str] = object_to_node(child_name_str, child_value)
        return node

    if isinstance(value, list):
        for index, child_value in enumerate(value):
            child_name = f"[{index}]"
            node.children[child_name] = object_to_node(child_name, child_value)
        return node

    if isinstance(value, tuple):
        for index, child_value in enumerate(value):
            child_name = f"[{index}]"
            node.children[child_name] = object_to_node(child_name, child_value)
        return node

    node.value = value
    return node


def insert_flat_key(root: Node, parts: list[str], value: object) -> None:
    node = root
    for part in parts[:-1]:
        if part not in node.children:
            node.children[part] = Node(name=part)
        node = node.children[part]
    leaf_name = parts[-1]
    node.children[leaf_name] = object_to_node(leaf_name, value)


def finalize_tree(node: Node) -> None:
    if torch.is_tensor(node.value):
        node.tensor_count = 1
        node.signature = ("tensor", tuple(node.value.shape), str(node.value.dtype))
        return

    if node.children:
        for child in node.children.values():
            finalize_tree(child)
        node.tensor_count = sum(child.tensor_count for child in node.children.values())
        node.signature = (
            "node",
            tuple((child_name, child.signature) for child_name, child in node.children.items()),
        )
        return

    node.tensor_count = 0
    node.signature = ("value", type(node.value).__name__, stable_value_signature(node.value))


def stable_value_signature(value: object) -> str:
    if value is MISSING:
        return "MISSING"
    if isinstance(value, str):
        return value
    return repr(value)


def maybe_compress_numeric_children(node: Node) -> CompressionInfo | None:
    if len(node.children) < COMPRESS_NUMERIC_CHILDREN_AT:
        return None

    child_names = list(node.children.keys())
    if not child_names or not all(name.isdigit() for name in child_names):
        return None

    indexes = [int(name) for name in child_names]
    sorted_indexes = sorted(indexes)
    expected = list(range(sorted_indexes[0], sorted_indexes[-1] + 1))
    if sorted_indexes != expected:
        return None

    children = list(node.children.values())
    first_signature = children[0].signature
    if any(child.signature != first_signature for child in children[1:]):
        return None

    representative = children[0]
    return CompressionInfo(
        start=sorted_indexes[0],
        end=sorted_indexes[-1],
        count=len(children),
        representative=representative,
    )


def render_tree(root: Node) -> list[str]:
    lines = [format_root_label(root)]
    children = list(root.children.values())
    for index, child in enumerate(children):
        is_last = index == len(children) - 1
        lines.extend(render_node(child, prefix="", is_last=is_last))
    return lines


def render_node(node: Node, prefix: str, is_last: bool) -> list[str]:
    connector = "└── " if is_last else "├── "
    compression = maybe_compress_numeric_children(node)
    lines = [f"{prefix}{connector}{format_node_label(node, compression)}"]

    child_prefix = prefix + ("    " if is_last else "│   ")
    if compression is not None:
        representative_children = list(compression.representative.children.values())
        for index, child in enumerate(representative_children):
            child_is_last = index == len(representative_children) - 1
            lines.extend(render_node(child, child_prefix, child_is_last))
        return lines

    children = list(node.children.values())
    for index, child in enumerate(children):
        child_is_last = index == len(children) - 1
        lines.extend(render_node(child, child_prefix, child_is_last))
    return lines


def format_root_label(root: Node) -> str:
    if root.actual_type:
        return root.actual_type
    return root.name


def format_node_label(node: Node, compression: CompressionInfo | None) -> str:
    if compression is not None:
        representative = compression.representative
        return (
            f"{node.name} [{compression.start}..{compression.end}] "
            f"({compression.count} entries, each {representative.tensor_count} tensors)"
        )

    if torch.is_tensor(node.value):
        return f"{node.name} {format_tensor_suffix(node.value)}"

    if node.children:
        suffix_parts: list[str] = []
        if node.actual_type and node.actual_type != "dict":
            suffix_parts.append(node.actual_type)
        if node.tensor_count >= SHOW_TENSOR_COUNT_AT or node.actual_type in {"OrderedDict", "list", "tuple"}:
            suffix_parts.append(f"{node.tensor_count} tensors")
        if suffix_parts:
            return f"{node.name} ({', '.join(suffix_parts)})"
        return node.name

    return f"{node.name} {format_scalar_suffix(node.value)}"


def format_tensor_suffix(tensor: torch.Tensor) -> str:
    return f"{format_shape(tuple(tensor.shape))} {tensor.dtype}"


def format_shape(shape: tuple[int, ...]) -> str:
    if not shape:
        return "()"
    if len(shape) == 1:
        return f"({shape[0]},)"
    return "(" + ", ".join(str(dim) for dim in shape) + ")"


def format_scalar_suffix(value: object) -> str:
    if value is None:
        return "None"
    if isinstance(value, str):
        shortened = value if len(value) <= 80 else value[:77] + "..."
        return repr(shortened)
    if isinstance(value, (bool, int, float)):
        return repr(value)
    return f"<{type(value).__name__}>"


def main() -> None:
    args = parse_args()
    checkpoint_path = args.ckpt.expanduser().resolve()
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = load_checkpoint_metadata(checkpoint_path, args.unsafe_full_load)
    root = object_to_node(type(checkpoint).__name__, checkpoint, actual_type=type(checkpoint).__name__)
    finalize_tree(root)

    for line in render_tree(root):
        print(line)


if __name__ == "__main__":
    main()
