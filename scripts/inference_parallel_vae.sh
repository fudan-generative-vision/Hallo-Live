#!/usr/bin/env bash

MODEL_DIR=/inspire/hdd/project/chineseculture/public/chunyu/model  # Set this to your model directory

torchrun --nnodes=1 --nproc_per_node=1 inference.py \
    --config_path configs/dmd_fusion_5B.yaml \
	--model_dir $MODEL_DIR \
    --output_folder output/demo_videos \
    --generator_ckpt ${MODEL_DIR}/Hallo-Live/hallolive_dit.pt \
    --data_path prompts/demo_prompts.csv \
    --parallel_vae_decode \
    --parallel_vae_decode_full_audio \
    --vae_decode_device cuda:1 \
    --use_ema --profile
