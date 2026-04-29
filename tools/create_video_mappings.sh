#!/usr/bin/env bash

CSV_PATH=prompts/synthetic_prompts_17k.csv
VIDEO_DIR=output/dmd_5B_data_17000_block_30_full_attn/step_2000_prompt_17000
OUTPUT_PATH=output/dmd_5B_data_17000_block_30_full_attn/step_2000_prompt_17000/video_prompt_mappings.json

python tools/create_video_mappings.py \
    --csv "${CSV_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_path "${OUTPUT_PATH}"
