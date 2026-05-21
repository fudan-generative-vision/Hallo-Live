#!/usr/bin/env bash

# Check IP address of this machine:
# $ hostname -I | awk '{print $1}'

# ========== Core configuration (keep consistent across all nodes) ==========
# Recommended to set via environment variables or edit here directly
MASTER_ADDR=${MASTER_ADDR:-"10.246.3.11"}
MASTER_PORT=${MASTER_PORT:-29500}
NNODES=${NNODES:-4} # Default to 4; can also be passed in externally
NPROC_PER_NODE=${NPROC_PER_NODE:-8}

CONFIG_PATH="configs/dual_stream_dmd_5B.yaml"
SAVE_CKPT_DIR="checkpoints/dual_stream_dmd_5b_strict"
# ===============================================

# Flexible NODE_RANK resolution:
# Priority 1: first command-line argument (e.g. ./train.sh 0)
# Priority 2: environment variable $RANK
# Priority 3: auto-detect based on IP (fallback only)

if [ -n "$1" ]; then
    NODE_RANK=$1
elif [ -n "$RANK" ]; then
    NODE_RANK=$RANK
else
    # Auto-detection logic
    MY_IP=$(hostname -I | awk '{print $1}')
    if [ "$MY_IP" = "$MASTER_ADDR" ]; then
        NODE_RANK=0
    else
        echo "Error: Cannot determine NODE_RANK. Please provide it as an argument."
        echo "Usage: $0 [RANK]"
        exit 1
    fi
fi

echo "---------------------------------------"
echo "Master Addr: $MASTER_ADDR"
echo "World Size:  $NNODES Nodes"
echo "Local Rank:  $NODE_RANK"
echo "---------------------------------------"

# Launch training
torchrun \
    --nproc_per_node=$NPROC_PER_NODE \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr="$MASTER_ADDR" \
    --master_port=$MASTER_PORT \
    train.py \
    --config_path $CONFIG_PATH \
    --save_ckpt_dir $SAVE_CKPT_DIR

# If training fails, run the GPU occupancy script
python tools/occupy_gpu.py
