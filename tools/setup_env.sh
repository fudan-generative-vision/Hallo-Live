#!/usr/bin/env bash
conda create -n hallolive python=3.10 -y
conda activate hallolive
pip install -r requirements.txt

pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
