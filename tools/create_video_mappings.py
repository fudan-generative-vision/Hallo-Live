"""
Create a JSON mapping from video filename to text prompt.

Usage:
    python tools/create_video_mappings.py
    python tools/create_video_mappings.py --csv prompts/synthesize_new_20000.csv --video_dir ode_data/ode_fusion_20000/videos --output_path ode_data/ode_fusion_20000/video_prompt_mappings.json
"""

import os
import csv
import json
import re
import hashlib
import argparse


def format_prompt_for_filename(text: str) -> str:
    """
    Convert prompt to a filename-safe string.
    Uses hash value to ensure uniqueness and avoid collisions between different prompts.
    """

    # Remove anything inside <...>
    no_tags = re.sub(r"<.*?>", "", text)
    # Replace spaces and slashes with underscores
    safe = no_tags.replace(" ", "_").replace("/", "_")
    # Generate hash of the prompt (first 8 chars) to ensure uniqueness
    prompt_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    # Truncate to 42 chars, plus 8-char hash, total 50 chars
    truncated = safe[:42] if len(safe) > 42 else safe
    return f"{truncated}_{prompt_hash}"


def create_video_mappings(csv_file_path, video_dir_path, output_json_path):
    """
    Read prompts from a CSV file and videos from a directory,
    then generate a JSON mapping: video filename -> text prompt.
    """

    # 1. Read all prompts from the CSV file
    prompts = []
    try:
        with open(csv_file_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Strip possible leading/trailing whitespace
                prompts.append(row["text_prompt"].strip())
    except FileNotFoundError:
        print(f"Error: CSV file not found: {csv_file_path}")
        return

    video_filenames = {}
    for prompt in prompts:
        video_filename = f"{format_prompt_for_filename(prompt)}_512x992_103.mp4"
        video_filenames[video_filename] = prompt

    # 2. List all mp4 files under the video directory
    try:
        video_files = [f for f in os.listdir(video_dir_path) if f.endswith(".mp4")]
    except FileNotFoundError:
        print(f"Error: Video directory not found: {video_dir_path}")
        return

    mapping = {}

    print(f"Loaded {len(prompts)} text prompts and {len(video_files)} video files. Starting matching...")

    # 3. Iterate through video files for matching
    for filename in video_files:
        if filename in video_filenames:
            mapping[filename] = video_filenames[filename]

    # 4. Save mapping as a JSON file
    with open(output_json_path, "w", encoding="utf-8") as f:
        json.dump(mapping, f, indent=2, ensure_ascii=False)

    print(f"Done! Successfully created {len(mapping)} mapping entries. Result saved to: {output_json_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Read prompts from a CSV file and videos from a directory, "
            "then generate a JSON mapping: video filename -> text prompt."
        )
    )

    parser.add_argument(
        "--csv",
        dest="csv_path",
        default="prompts/synthesize_new_20000.csv",
        help="Path to the CSV file containing prompts",
    )
    parser.add_argument(
        "--video_dir",
        dest="video_dir",
        default="ode_data/ode_fusion_20000/videos",
        help="Path to the directory that contains video files",
    )
    parser.add_argument(
        "--output_path",
        dest="output_path",
        default="ode_data/ode_fusion_20000/video_prompt_mappings.json",
        help="Output path for the generated JSON mapping file",
    )

    args = parser.parse_args()

    create_video_mappings(args.csv_path, args.video_dir, args.output_path)
