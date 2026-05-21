#!/usr/bin/env bash

OUTPUT_DIR="ode_data/ode_data_32k"
TEXT_PROMPT="prompts/data/synthetic_prompts_32k.csv"

torchrun --nnodes 1 --nproc_per_node 8 -m hallolive.utils.sample_ode_data \
	--config-file configs/sample_ode_data.yaml \
	--output-dir $OUTPUT_DIR \
	--text-prompt $TEXT_PROMPT

python tools/create_video_mappings.py \
	--csv $TEXT_PROMPT \
	--video_dir "${OUTPUT_DIR}/videos" \
	--output_path "${OUTPUT_DIR}/video_prompt_mappings.json"

python -m hallolive.utils.create_lmdb --data_path "${OUTPUT_DIR}/pt_files" --lmdb_path "${OUTPUT_DIR}/lmdb"

python tools/occupy_gpu.py
