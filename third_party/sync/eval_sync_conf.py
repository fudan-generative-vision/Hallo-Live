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

import argparse
import os
import tqdm
from statistics import fmean
import torch

from .syncnet import SyncNetEval
from .syncnet_detect import SyncNetDetector

CRED = "\033[91m"
CEND = "\033[0m"


def red_text(text: str):
    return f"{CRED}{text}{CEND}"


def syncnet_eval(
    syncnet, syncnet_detector, video_path, temp_dir, detect_results_dir="detect_results"
):
    syncnet_detector(video_path=video_path, min_track=50)
    crop_videos = os.listdir(os.path.join(detect_results_dir, "crop"))
    if crop_videos == []:
        raise Exception(red_text(f"Face not detected in {video_path}"))
    av_offset_list = []
    conf_list = []
    for video in crop_videos:
        av_offset, _, conf = syncnet.evaluate(
            video_path=os.path.join(detect_results_dir, "crop", video),
            temp_dir=temp_dir,
        )
        av_offset_list.append(av_offset)
        conf_list.append(conf)
    av_offset = int(fmean(av_offset_list))
    conf = fmean(conf_list)
    print(
        f"Input video: {video_path}\nSyncNet confidence: {conf:.2f}\nAV offset: {av_offset}"
    )
    return av_offset, conf


def main():
    parser = argparse.ArgumentParser(description="SyncNet")
    parser.add_argument(
        "--initial_model",
        type=str,
        default="checkpoints/auxiliary/syncnet_v2.model",
        help="",
    )
    parser.add_argument("--video_path", type=str, default=None, help="")
    parser.add_argument("--videos_dir", type=str, default="/root/processed")
    parser.add_argument("--temp_dir", type=str, default="temp", help="")

    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    syncnet = SyncNetEval(device=device)
    syncnet.loadParameters(args.initial_model)

    syncnet_detector = SyncNetDetector(
        device=device, detect_results_dir="detect_results"
    )

    if args.video_path is not None:
        syncnet_eval(syncnet, syncnet_detector, args.video_path, args.temp_dir)
    else:
        sync_conf_list = []
        video_names = sorted(
            [f for f in os.listdir(args.videos_dir) if f.endswith(".mp4")]
        )
        for video_name in tqdm.tqdm(video_names):
            try:
                _, conf = syncnet_eval(
                    syncnet,
                    syncnet_detector,
                    os.path.join(args.videos_dir, video_name),
                    args.temp_dir,
                )
                sync_conf_list.append(conf)
            except Exception as e:
                print(e)
        print(f"The average sync confidence is {fmean(sync_conf_list):.02f}")


if __name__ == "__main__":
    main()
