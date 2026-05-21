#!/usr/bin/env bash

USER_DIR=/inspire/hdd/project/chineseculture/public/chunyu
cd ${USER_DIR}/code/self-forcing

export PATH=${USER_DIR}/miniconda3/envs/self_forcing/bin:${PATH}
export LD_LIBRARY_PATH=${USER_DIR}/libGL:${LD_LIBRARY_PATH}

export NCCL_DEBUG=INFO
export NCCL_IB_DISABLE=0    # Ensure InfiniBand is active
export NCCL_NET_GDR_LEVEL=2 # Enable GPU Direct RDMA

OUTPUT_DIR="ode_data/dual_stream_ode_26000_fix"
TEXT_PROMPT="prompts/synthesize_new_26000.csv"

source tools/utils.sh

if [ "$PET_NODE_RANK" -eq 0 ]; then
    print_env_info
fi

torchrun --nproc_per_node=$PET_NPROC_PER_NODE \
    --nnodes=$PET_NNODES \
    --node_rank=$PET_NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --master_port=$MASTER_PORT \
    -m hallolive.utils.sample_ode_data \
    --config-file configs/sample_ode_data.yaml \
    --output-dir $OUTPUT_DIR \
    --text-prompt $TEXT_PROMPT

if [ "$PET_NODE_RANK" -eq 0 ]; then
    python tools/create_video_mappings.py \
        --csv $TEXT_PROMPT \
        --video_dir "${OUTPUT_DIR}/videos" \
        --output_path "${OUTPUT_DIR}/video_prompt_mappings.json"

    python -m hallolive.utils.create_lmdb --data_path "${OUTPUT_DIR}/pt_files" --lmdb_path "${OUTPUT_DIR}/lmdb"
fi

# python tools/occupy_gpu.py
