#!/usr/bin/env bash

# Create the conda environment for the project
conda create -n hallolive python=3.10 -y
conda activate hallolive

# Install ffmpeg CLI command
conda install -y -c conda-forge ffmpeg

# Install the Python packages listed in the project requirements file.
pip install -r requirements.txt
pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
