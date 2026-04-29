# Copyright (c) 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import subprocess
from tqdm import tqdm
import random


def inference_video_from_fileslist(
    video_fileslist: str,
    audio_fileslist: str,
    output_dir: str,
    unet_config_path: str,
    ckpt_path: str,
    guidance_scale: float,
    seed: int = 42,
):
    with open(video_fileslist, "r", encoding="utf-8") as file:
        video_paths = [line.strip() for line in file.readlines()]

    with open(audio_fileslist, "r", encoding="utf-8") as file:
        audio_paths = [line.strip() for line in file.readlines()]

    random.seed(seed)

    output_dir = f"{output_dir}__{seed}"
    os.makedirs(output_dir, exist_ok=True)

    random.shuffle(video_paths)
    random.shuffle(audio_paths)

    min_length = min(len(video_paths), len(audio_paths))

    video_paths = video_paths[:min_length]
    audio_paths = audio_paths[:min_length]

    random.shuffle(video_paths)
    random.shuffle(audio_paths)

    for index, video_path in tqdm(enumerate(video_paths), total=len(video_paths)):
        audio_path = audio_paths[index]
        video_name = os.path.basename(video_path)[:-4]
        audio_name = os.path.basename(audio_path)[:-4]
        video_out_path = os.path.join(output_dir, f"{video_name}__{audio_name}.mp4")
        inference_command = (
            f"python -m scripts.inference --enable_deepcache --guidance_scale {guidance_scale} --unet_config_path {unet_config_path} "
            f"--video_path {video_path} --audio_path {audio_path} --video_out_path {video_out_path} --inference_ckpt_path {ckpt_path}"
        )
        subprocess.run(inference_command, shell=True)


if __name__ == "__main__":
    video_fileslist = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/video_fileslist.txt"
    audio_fileslist = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/fileslist/audio_fileslist.txt"
    output_dir = "/mnt/bn/maliva-gen-ai-v2/chunyu.li/inference_videos_results"

    unet_config_path = "configs/unet/stage2_512.yaml"
    ckpt_path = "checkpoints/latentsync_unet.pt"
    guidance_scale = 1.5

    seed = 42

    inference_video_from_fileslist(
        video_fileslist, audio_fileslist, output_dir, unet_config_path, ckpt_path, guidance_scale, seed
    )
