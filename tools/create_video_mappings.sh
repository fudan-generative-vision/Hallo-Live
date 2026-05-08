#!/usr/bin/env bash

CSV_PATH=prompts/data/synthetic_prompts_32k.csv
VIDEO_DIR=ode_data/ode_data_30k/videos
OUTPUT_PATH=ode_data/ode_data_30k/video_prompt_mappings.json

python tools/create_video_mappings.py \
    --csv "${CSV_PATH}" \
    --video_dir "${VIDEO_DIR}" \
    --output_path "${OUTPUT_PATH}"
