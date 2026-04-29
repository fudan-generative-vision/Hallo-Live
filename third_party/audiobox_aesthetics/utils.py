# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
from pathlib import Path
from typing import Optional

import requests
from tqdm.rich import tqdm

DEFAULT_HF_REPO = "facebook/audiobox-aesthetics"
DEFAULT_CKPT_FNAME = "checkpoint.pt"
DEFAULT_S3_URL = "https://dl.fbaipublicfiles.com/audiobox-aesthetics/checkpoint.pt"

logging = logging.getLogger(__name__)

def download_file(url, destination):
    """Download a file from a URL with a progress bar."""
    try:
        # Stream the request to handle large files
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            # Get the total file size from the headers
            total_size = int(response.headers.get("content-length", 0))
            # Open the file in binary write mode
            with open(destination, "wb") as f:
                # Use tqdm to create a progress bar
                with tqdm(
                    total=total_size,
                    unit="B",
                    unit_scale=True,
                    desc=os.path.basename(destination),
                ) as pbar:
                    # Iterate over the response in chunks
                    for chunk in response.iter_content(chunk_size=1024):
                        # Write each chunk to the file
                        f.write(chunk)
                        # Update the progress bar
                        pbar.update(len(chunk))
        logging.info(f"File has been downloaded and saved to '{destination}'.")
    except requests.exceptions.RequestException as e:
        logging.error(f"An error occurred while downloading the file: {e}")


def load_model(checkpoint_pth: Optional[str]) -> str:
    if checkpoint_pth is not None and Path(checkpoint_pth).exists():
        # if user has downloaded the model to local directly
        return checkpoint_pth
    else:
        # if user hasn't downloaded the model, we redirect them to huggingface repo
        try:
            import huggingface_hub
        except ImportError:
            raise ImportError(
                "Please install it using 'pip install huggingface_hub'."
                f"Otherwise, download checkpoint directly from {DEFAULT_S3_URL}"
            )

        cached_file = huggingface_hub.hf_hub_download(
            DEFAULT_HF_REPO, DEFAULT_CKPT_FNAME
        )
        logging.info(f"Load ckpt from {cached_file}")
        return cached_file
