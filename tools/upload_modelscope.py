#!/usr/bin/env python3
"""
Upload a single model file to ModelScope.

Usage:
    python tools/upload_modelscope.py --token <token> --model_id user/repo --model_name "Model Name" --model_path path/to/model.pt
"""

import argparse
import os
from pathlib import Path

from modelscope.hub.api import HubApi
from modelscope.hub.constants import Licenses, ModelVisibility


def upload_modelscope(
    token: str,
    model_id: str,
    model_name: str,
    model_path: str,
    path_in_repo: str | None = None,
    commit_message: str = "upload checkpoint",
):
    api = HubApi()
    api.login(token)

    local_model_path = Path(model_path).expanduser().resolve()
    if not local_model_path.is_file():
        raise FileNotFoundError(f"Model file not found: {local_model_path}")

    if path_in_repo is None:
        path_in_repo = local_model_path.name

    # --- 1. Check whether the repo exists and create it if needed ---
    try:
        api.get_model(model_id=model_id)
        print(f"Repo {model_id} already exists. Skipping repo creation.")
    except Exception:
        print(f"Creating new repo: {model_id}...")
        api.create_model(
            model_id=model_id,
            visibility=ModelVisibility.PUBLIC,
            license=Licenses.MIT,
            chinese_name=model_name,
        )

    # --- 2. Upload a single file ---
    print(f"Uploading local file: {local_model_path}")
    print(f"Remote path: {path_in_repo}")
    api.upload_file(
        repo_id=model_id,
        path_or_fileobj=str(local_model_path),
        path_in_repo=path_in_repo,
        commit_message=commit_message,
    )
    print("Upload completed.")


def parse_args():
    parser = argparse.ArgumentParser(description="Upload a single model file to ModelScope.")
    parser.add_argument("--token", default=os.getenv("MODELSCOPE_TOKEN"), help="ModelScope access token.")
    parser.add_argument("--model_id", required=True, help="Target repo id, for example `user_name/repo_name`.")
    parser.add_argument("--model_name", required=True, help="Chinese repo name used when creating a new repo.")
    parser.add_argument("--model_path", required=True, help="Local path to a specific model file to upload.")
    parser.add_argument(
        "--path_in_repo",
        default=None,
        help="Target file path inside the remote repo. Defaults to the local file name.",
    )
    parser.add_argument(
        "--commit_message",
        default="upload checkpoint",
        help="Commit message used for the upload.",
    )
    args = parser.parse_args()

    if not args.token:
        parser.error("`--token` is required, or set the MODELSCOPE_TOKEN environment variable.")

    return args


def main():
    args = parse_args()
    upload_modelscope(
        token=args.token,
        model_id=args.model_id,
        model_name=args.model_name,
        model_path=args.model_path,
        path_in_repo=args.path_in_repo,
        commit_message=args.commit_message,
    )


if __name__ == "__main__":
    main()
