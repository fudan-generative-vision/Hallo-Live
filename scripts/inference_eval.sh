#!/usr/bin/env bash

# Checkpoint steps to evaluate. For example: STEPS=(0500 1000 1500 2000).
STEPS=(1500)

# Experiments to evaluate. For multiple experiments: EXP_NAMES=("exp_a" "exp_b")
EXP_NAMES=("dmd_5B_data_32k")

PROMPT_PATH="prompts/data/eval_prompts_30.csv"
OUTPUT_DIR="eval_prompts_30"

# Detect the number of visible GPUs
NUM_GPUS=$(nvidia-smi --list-gpus 2>/dev/null | wc -l)
echo "Using ${NUM_GPUS} GPU(s) for inference"

# Run inference for every requested experiment and checkpoint step.
for EXP_NAME in "${EXP_NAMES[@]}"; do
	for STEP in "${STEPS[@]}"; do
		echo "========================================"
		echo "Running evaluation for EXP_NAME: ${EXP_NAME}, Step: ${STEP}"
		echo "========================================"

		# Launch distributed inference on all visible GPUs.
		torchrun --nnodes=1 --nproc_per_node=${NUM_GPUS} inference.py \
			--config_path checkpoints/${EXP_NAME}/*.yaml \
			--output_folder output/${EXP_NAME}/${OUTPUT_DIR}_step_${STEP} \
			--generator_ckpt checkpoints/${EXP_NAME}/checkpoint_model_00${STEP}/model.pt \
			--data_path "${PROMPT_PATH}" \
			--use_ema --profile

		# Build a filename-to-prompt mapping json for later evaluation
		python tools/create_video_mappings.py \
			--csv "${PROMPT_PATH}" \
			--video_dir output/${EXP_NAME}/${OUTPUT_DIR}_step_${STEP} \
			--output_path output/${EXP_NAME}/${OUTPUT_DIR}_step_${STEP}/video_prompt_mappings.json
	done
done

python tools/occupy_gpu.py
