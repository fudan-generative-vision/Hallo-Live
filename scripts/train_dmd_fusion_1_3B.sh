#!/usr/bin/env bash

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nnodes=1 --nproc_per_node=2 --rdzv_id=5235 \
    --rdzv_backend=c10d train.py \
    --config_path configs/dmd_fusion_1_3B.yaml \
    --exp_name dmd_1_3B \
    --disable_wandb
