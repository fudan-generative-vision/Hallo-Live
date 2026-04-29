#!/usr/bin/env python3
"""
Compare same-named video files in two directories.

Usage:
    python tools/compare_matching_videos.py --dir-a path/to/a --dir-b path/to/b --match-mode relative --no-recursive

The comparison is based on decoded content instead of container bytes:
- video streams are decoded to RGB24 and compared frame-by-frame with timestamps
- audio streams are decoded to PCM F32LE and compared block-by-block with timestamps

That means two files are treated as identical only when their decoded pixels,
decoded audio samples, and stream timing all match.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable


VIDEO_EXTENSIONS = {
    ".mp4",
    ".mov",
    ".mkv",
    ".avi",
    ".webm",
    ".m4v",
    ".mpg",
    ".mpeg",
}

IGNORED_FRAMEMD5_PREFIXES = (
    "#format:",
    "#version:",
    "#software:",
    "#hash:",
)


@dataclass
class CompareResult:
    matched: bool
    reasons: list[str] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare same-named video files in two directories by decoded video "
            "pixels and decoded audio samples."
        )
    )
    parser.add_argument(
        "--dir-a",
        type=Path,
        default=Path("temp_ode_gen"),
        help="First directory. Default: %(default)s",
    )
    parser.add_argument(
        "--dir-b",
        type=Path,
        default=Path("temp_data_gen"),
        help="Second directory. Default: %(default)s",
    )
    parser.add_argument(
        "--match-mode",
        choices=("basename", "relative"),
        default="basename",
        help=(
            "How to pair files between the two directories. "
            "'basename' matches by file name only. "
            "'relative' matches by relative path. Default: %(default)s"
        ),
    )
    parser.add_argument(
        "--no-recursive",
        action="store_true",
        help="Only scan the top level of each directory.",
    )
    return parser.parse_args()


def ensure_command_exists(command: str) -> None:
    if shutil.which(command) is None:
        raise SystemExit(f"Missing required command: {command}")


def iter_video_files(root: Path, recursive: bool) -> Iterable[Path]:
    if recursive:
        iterator = root.rglob("*")
    else:
        iterator = root.glob("*")

    for path in iterator:
        if path.is_file() and path.suffix.lower() in VIDEO_EXTENSIONS:
            yield path


def collect_files(root: Path, recursive: bool, match_mode: str) -> dict[str, list[Path]]:
    files: dict[str, list[Path]] = {}
    for path in iter_video_files(root, recursive=recursive):
        if match_mode == "basename":
            key = path.name
        else:
            key = path.relative_to(root).as_posix()
        files.setdefault(key, []).append(path)
    return files


def run_json_command(cmd: list[str]) -> dict:
    completed = subprocess.run(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip() or "unknown error"
        raise RuntimeError(f"Command failed: {' '.join(cmd)}\n{stderr}")
    return json.loads(completed.stdout)


def get_stream_counts(video_path: Path) -> tuple[int, int]:
    data = run_json_command(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "json",
            str(video_path),
        ]
    )
    streams = data.get("streams", [])
    video_count = sum(1 for stream in streams if stream.get("codec_type") == "video")
    audio_count = sum(1 for stream in streams if stream.get("codec_type") == "audio")
    return video_count, audio_count


def normalize_framemd5_line(line: str) -> str | None:
    stripped = line.strip()
    if not stripped:
        return None
    if any(stripped.startswith(prefix) for prefix in IGNORED_FRAMEMD5_PREFIXES):
        return None
    if stripped.startswith("#"):
        return stripped

    parts = [part.strip() for part in stripped.split(",")]
    return ",".join(parts)


def framemd5_digest(video_path: Path, stream_kind: str, stream_index: int) -> str:
    if stream_kind == "v":
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(video_path),
            "-map",
            f"0:v:{stream_index}",
            "-fps_mode",
            "passthrough",
            "-pix_fmt",
            "rgb24",
            "-f",
            "framemd5",
            "-hash",
            "sha256",
            "-",
        ]
    elif stream_kind == "a":
        cmd = [
            "ffmpeg",
            "-v",
            "error",
            "-nostdin",
            "-i",
            str(video_path),
            "-map",
            f"0:a:{stream_index}",
            "-c:a",
            "pcm_f32le",
            "-f",
            "framemd5",
            "-hash",
            "sha256",
            "-",
        ]
    else:
        raise ValueError(f"Unsupported stream kind: {stream_kind}")

    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )

    assert process.stdout is not None
    digest = hashlib.sha256()
    for line in process.stdout:
        normalized = normalize_framemd5_line(line)
        if normalized is None:
            continue
        digest.update(normalized.encode("utf-8"))
        digest.update(b"\n")

    stderr = process.stderr.read() if process.stderr is not None else ""
    return_code = process.wait()
    if return_code != 0:
        error_message = stderr.strip() or "unknown ffmpeg error"
        raise RuntimeError(
            f"ffmpeg failed for {video_path} stream {stream_kind}:{stream_index}\n"
            f"{error_message}"
        )

    return digest.hexdigest()


def compare_pair(path_a: Path, path_b: Path) -> CompareResult:
    reasons: list[str] = []

    video_count_a, audio_count_a = get_stream_counts(path_a)
    video_count_b, audio_count_b = get_stream_counts(path_b)

    if video_count_a != video_count_b:
        reasons.append(
            f"video stream count differs: {path_a.name}={video_count_a}, "
            f"{path_b.name}={video_count_b}"
        )
    if audio_count_a != audio_count_b:
        reasons.append(
            f"audio stream count differs: {path_a.name}={audio_count_a}, "
            f"{path_b.name}={audio_count_b}"
        )

    for stream_index in range(min(video_count_a, video_count_b)):
        digest_a = framemd5_digest(path_a, "v", stream_index)
        digest_b = framemd5_digest(path_b, "v", stream_index)
        if digest_a != digest_b:
            reasons.append(f"video stream {stream_index} differs after RGB24 decode")

    for stream_index in range(min(audio_count_a, audio_count_b)):
        digest_a = framemd5_digest(path_a, "a", stream_index)
        digest_b = framemd5_digest(path_b, "a", stream_index)
        if digest_a != digest_b:
            reasons.append(f"audio stream {stream_index} differs after PCM decode")

    return CompareResult(matched=not reasons, reasons=reasons)


def format_path_list(paths: list[Path]) -> str:
    return ", ".join(str(path) for path in paths)


def main() -> int:
    args = parse_args()
    ensure_command_exists("ffmpeg")
    ensure_command_exists("ffprobe")

    dir_a = args.dir_a.resolve()
    dir_b = args.dir_b.resolve()
    recursive = not args.no_recursive

    if not dir_a.is_dir():
        raise SystemExit(f"Directory does not exist: {dir_a}")
    if not dir_b.is_dir():
        raise SystemExit(f"Directory does not exist: {dir_b}")

    files_a = collect_files(dir_a, recursive=recursive, match_mode=args.match_mode)
    files_b = collect_files(dir_b, recursive=recursive, match_mode=args.match_mode)

    keys_a = set(files_a)
    keys_b = set(files_b)
    common_keys = sorted(keys_a & keys_b)
    only_a = sorted(keys_a - keys_b)
    only_b = sorted(keys_b - keys_a)

    print(f"Directory A: {dir_a}")
    print(f"Directory B: {dir_b}")
    print(f"Match mode: {args.match_mode}")
    print(f"Recursive scan: {recursive}")
    print()

    if only_a:
        print("Only in directory A:")
        for key in only_a:
            print(f"  - {key}")
        print()

    if only_b:
        print("Only in directory B:")
        for key in only_b:
            print(f"  - {key}")
        print()

    if not common_keys:
        print("No matching video names were found.")
        return 1

    matched_count = 0
    different_count = 0
    skipped_count = 0

    print("Comparing matching files:")
    for key in common_keys:
        candidates_a = files_a[key]
        candidates_b = files_b[key]

        if len(candidates_a) != 1 or len(candidates_b) != 1:
            skipped_count += 1
            print(f"[SKIP] {key}")
            if len(candidates_a) != 1:
                print(f"  A candidates: {format_path_list(candidates_a)}")
            if len(candidates_b) != 1:
                print(f"  B candidates: {format_path_list(candidates_b)}")
            continue

        path_a = candidates_a[0]
        path_b = candidates_b[0]

        try:
            result = compare_pair(path_a, path_b)
        except RuntimeError as exc:
            skipped_count += 1
            print(f"[ERROR] {key}")
            for line in str(exc).splitlines():
                print(f"  {line}")
            continue

        if result.matched:
            matched_count += 1
            print(f"[MATCH] {key}")
        else:
            different_count += 1
            print(f"[DIFF]  {key}")
            for reason in result.reasons:
                print(f"  - {reason}")

    print()
    print("Summary:")
    print(f"  matched:   {matched_count}")
    print(f"  different: {different_count}")
    print(f"  skipped:   {skipped_count}")
    print(f"  only in A: {len(only_a)}")
    print(f"  only in B: {len(only_b)}")

    return 0 if different_count == 0 and skipped_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
