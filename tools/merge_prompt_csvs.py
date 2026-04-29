"""
Merge multiple prompt CSV files and remove duplicate prompts.

Usage:
    python tools/merge_prompt_csvs.py
    python tools/merge_prompt_csvs.py prompts/a.csv prompts/b.csv --output prompts/merged.csv
"""

import argparse
import csv
from pathlib import Path


DEFAULT_GLOB = "prompts/synthesize_new_*.csv"
DEFAULT_OUTPUT = Path("prompts/synthesize_new_merged.csv")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Merge multiple prompt CSV files and remove duplicate prompts."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        type=Path,
        help=(
            "Input CSV files. If omitted, the script uses the default glob "
            f"`{DEFAULT_GLOB}`."
        ),
    )
    parser.add_argument(
        "--glob",
        dest="glob_patterns",
        action="append",
        default=[],
        help=(
            "Glob pattern for additional input files. Can be specified multiple "
            "times."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help=f"Output CSV path. Default: {DEFAULT_OUTPUT}",
    )
    parser.add_argument(
        "--column",
        default="text_prompt",
        help="Prompt column name. Default: text_prompt",
    )
    return parser.parse_args()


def resolve_input_paths(inputs: list[Path], glob_patterns: list[str]) -> list[Path]:
    resolved_paths: list[Path] = []
    seen_paths: set[Path] = set()

    for path in inputs:
        normalized = path.resolve()
        if not path.exists():
            raise FileNotFoundError(f"Input file does not exist: {path}")
        if normalized not in seen_paths:
            resolved_paths.append(path)
            seen_paths.add(normalized)

    patterns = glob_patterns or ([DEFAULT_GLOB] if not inputs else [])
    for pattern in patterns:
        matched_paths = sorted(Path().glob(pattern))
        if not matched_paths:
            raise FileNotFoundError(f"No files matched glob pattern: {pattern}")
        for path in matched_paths:
            normalized = path.resolve()
            if normalized not in seen_paths:
                resolved_paths.append(path)
                seen_paths.add(normalized)

    if not resolved_paths:
        raise ValueError("No input CSV files were provided.")

    return resolved_paths


def detect_prompt_column(fieldnames: list[str] | None, expected_column: str, path: Path) -> str:
    if not fieldnames:
        raise ValueError(f"{path} is empty or missing a header row.")

    if expected_column in fieldnames:
        return expected_column

    if len(fieldnames) == 1:
        return fieldnames[0]

    raise ValueError(
        f"{path} does not contain column `{expected_column}`. "
        f"Available columns: {fieldnames}"
    )


def load_unique_prompts(input_paths: list[Path], column: str) -> tuple[list[str], int]:
    unique_prompts: list[str] = []
    seen_prompts: set[str] = set()
    duplicate_count = 0

    for path in input_paths:
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            prompt_column = detect_prompt_column(reader.fieldnames, column, path)

            for row in reader:
                prompt = (row.get(prompt_column) or "").strip()
                if not prompt:
                    continue

                if prompt in seen_prompts:
                    duplicate_count += 1
                    continue

                seen_prompts.add(prompt)
                unique_prompts.append(prompt)

    return unique_prompts, duplicate_count


def write_prompts(output_path: Path, prompts: list[str], column: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[column])
        writer.writeheader()
        for prompt in prompts:
            writer.writerow({column: prompt})


def main() -> None:
    args = parse_args()
    input_paths = resolve_input_paths(args.inputs, args.glob_patterns)
    prompts, duplicate_count = load_unique_prompts(input_paths, args.column)
    write_prompts(args.output, prompts, args.column)

    print(f"Merged {len(input_paths)} files into {args.output}")
    print(f"Unique prompts: {len(prompts)}")
    print(f"Removed duplicates: {duplicate_count}")


if __name__ == "__main__":
    main()
