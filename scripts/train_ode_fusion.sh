#!/usr/bin/env bash

torchrun --nnodes=1 --nproc_per_node=4 \
	--rdzv_backend=c10d train.py \
	--config_path configs/ode_init_fusion.yaml \
	--exp_name ode_init_5B_data_32k \
	--disable_wandb

python tools/occupy_gpu.py
