#!/usr/bin/env bash

# Usage:
#   bash tools/download_models.sh inference
#   bash tools/download_models.sh train
#   bash tools/download_models.sh reward

set -euo pipefail

MODEL_DIR=/inspire/hdd/project/chineseculture/public/chunyu/model
TASK="${1:-}"

if [[ "$TASK" == "inference" ]]; then
  # Checkpoints required by inference.py.
  hf download fudan-generative-ai/Hallo-Live hallolive_dit.pt --local-dir "$MODEL_DIR/Hallo-Live"
  hf download Wan-AI/Wan2.1-T2V-1.3B models_t5_umt5-xxl-enc-bf16.pth --local-dir "$MODEL_DIR/Wan2.1-T2V-1.3B"
  hf download Wan-AI/Wan2.1-T2V-1.3B --include "google/umt5-xxl/*" --local-dir "$MODEL_DIR/Wan2.1-T2V-1.3B"
  hf download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$MODEL_DIR/Wan2.2-TI2V-5B"
  hf download hkchengrex/MMAudio ext_weights/v1-16.pth ext_weights/best_netG.pt --local-dir "$MODEL_DIR/MMAudio"
elif [[ "$TASK" == "train" ]]; then
  # Checkpoints required by train.py.
  hf download chetwinlow1/Ovi model.safetensors --local-dir "$MODEL_DIR/Ovi"
elif [[ "$TASK" == "reward" ]]; then
  # Optional reward checkpoints used when enable_rl_reward is true.
  hf download KlingTeam/VideoReward --local-dir "$MODEL_DIR/VideoReward"
  hf download Qwen/Qwen2-VL-2B-Instruct --local-dir "$MODEL_DIR/Qwen2-VL-2B-Instruct"
  hf download facebook/audiobox-aesthetics --local-dir "$MODEL_DIR/audiobox-aesthetics"
  hf download ByteDance/LatentSync-1.6 auxiliary/syncnet_v2.model auxiliary/sfd_face.pth --local-dir "$MODEL_DIR/LatentSync-1.6"
else
  echo "Usage: bash tools/download_models.sh {inference|train|reward}" >&2
  exit 1
fi
