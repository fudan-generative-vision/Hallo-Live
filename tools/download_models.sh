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
  huggingface-cli download --resume-download fudan-generative-ai/Hallo-Live hallolive_dit.pt --local-dir "$MODEL_DIR/Hallo-Live" --local-dir-use-symlinks False
  huggingface-cli download --resume-download Wan-AI/Wan2.1-T2V-1.3B --include "Wan2.1_VAE.pth" "models_t5_umt5-xxl-enc-bf16.pth" "google/umt5-xxl/*" --local-dir "$MODEL_DIR/Wan2.1-T2V-1.3B" --local-dir-use-symlinks False
  huggingface-cli download --resume-download Wan-AI/Wan2.2-TI2V-5B Wan2.2_VAE.pth --local-dir "$MODEL_DIR/Wan2.2-TI2V-5B" --local-dir-use-symlinks False
  huggingface-cli download --resume-download hkchengrex/MMAudio ext_weights/v1-16.pth ext_weights/best_netG.pt --local-dir "$MODEL_DIR/MMAudio" --local-dir-use-symlinks False
elif [[ "$TASK" == "train" ]]; then
  # Checkpoints required by train.py.
  huggingface-cli download --resume-download chetwinlow1/Ovi model.safetensors --local-dir "$MODEL_DIR/Ovi" --local-dir-use-symlinks False
elif [[ "$TASK" == "reward" ]]; then
  # Optional reward checkpoints used when enable_rl_reward is true.
  huggingface-cli download --resume-download KlingTeam/VideoReward --local-dir "$MODEL_DIR/VideoReward" --local-dir-use-symlinks False
  huggingface-cli download --resume-download Qwen/Qwen2-VL-2B-Instruct --local-dir "$MODEL_DIR/Qwen2-VL-2B-Instruct" --local-dir-use-symlinks False
  huggingface-cli download --resume-download facebook/audiobox-aesthetics --local-dir "$MODEL_DIR/audiobox-aesthetics" --local-dir-use-symlinks False
  huggingface-cli download --resume-download ByteDance/LatentSync-1.6 auxiliary/syncnet_v2.model auxiliary/sfd_face.pth --local-dir "$MODEL_DIR/LatentSync-1.6" --local-dir-use-symlinks False
else
  echo "Usage: bash tools/download_models.sh {inference|train|reward}" >&2
  exit 1
fi
