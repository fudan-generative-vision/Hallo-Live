#!/usr/bin/env bash

torchrun --nnodes=1 --nproc_per_node=4 \
	--rdzv_backend=c10d train.py \
	--config_path configs/ode_init_fusion.yaml \
	--exp_name ode_init_fusion_5b_strict_17000_block_30_weight

python tools/occupy_gpu.py
