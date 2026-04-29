#!/usr/bin/env bash
python tools/upload_modelscope.py \
    --token TOKEN \
    --model_id MODEL_ID \
    --model_name MODEL_NAME \
    --model_path /path/to/model \
    --path_in_repo file_path \
    --commit_message "upload checkpoint"
