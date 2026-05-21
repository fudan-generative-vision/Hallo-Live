#!/usr/bin/env bash

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 \
	--rdzv_backend=c10d train.py \
	--config_path configs/dual_stream_dmd_5B.yaml \
	--exp_name dmd_5B_data_32k \
	--disable_wandb

python tools/occupy_gpu.py
