#!/usr/bin/env bash

USER_DIR=/inspire/hdd/project/chineseculture/public/chunyu
cd ${USER_DIR}/code/Hallo-Live

export PATH=${USER_DIR}/miniconda3/envs/hallolive/bin:${PATH}
export LD_LIBRARY_PATH=${USER_DIR}/libGL:${LD_LIBRARY_PATH}

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0    # Ensure InfiniBand is active
export NCCL_NET_GDR_LEVEL=2 # Enable GPU Direct RDMA

CONFIG_PATH="configs/dmd_fusion_5B.yaml"
EXP_NAME="dmd_5B_data_32k"

source tools/utils.sh

if [ "$PET_NODE_RANK" -eq 0 ]; then
    print_env_info
fi

# Execute training
torchrun --nproc_per_node=$PET_NPROC_PER_NODE \
    --nnodes=$PET_NNODES \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    train.py \
    --config_path $CONFIG_PATH \
    --exp_name $EXP_NAME

# python tools/occupy_gpu.py
