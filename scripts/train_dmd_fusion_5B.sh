#!/usr/bin/env bash

torchrun --nnodes=1 --nproc_per_node=8 --rdzv_id=5235 \
	--rdzv_backend=c10d train.py \
	--config_path configs/dmd_fusion_5B.yaml \
	--exp_name dmd_5B_data_30k \
	--disable_wandb

python tools/occupy_gpu.py
