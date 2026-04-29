#!/usr/bin/env bash

MODEL_DIR=/path/to/your/model_dir  # Set this to your model directory

torchrun --nnodes=1 --nproc_per_node=1 inference.py \
    --config_path configs/dmd_fusion_5B.yaml \
	--model_dir $MODEL_DIR \
    --output_folder output/inference \
    --generator_ckpt ${MODEL_DIR}/Hallo-Live/hallolive_dit.pt \
    --data_path prompts/fusion_eval_25.csv \
    --use_ema --profile
