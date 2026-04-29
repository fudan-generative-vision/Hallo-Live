from tqdm import tqdm
import numpy as np
import argparse
import torch
import lmdb
import glob
import os

from hallolive.utils.lmdb import store_arrays_to_lmdb


def process_data_dict(data_dict, seen_prompts):
    output_dict = {}

    all_videos = []
    all_audios = []
    all_prompts = []

    prompt = data_dict["prompt"]
    video = data_dict["video_ode"]
    audio = data_dict["audio_ode"]

    if prompt not in seen_prompts:
        seen_prompts.add(prompt)
        video = video.half().numpy()
        audio = audio.half().numpy()
        all_videos.append(video)
        all_audios.append(audio)
        all_prompts.append(prompt)

    if len(all_videos) == 0:
        return {"video_latents": np.array([]), "audio_latents": np.array([]), "prompts": np.array([])}

    all_videos = np.concatenate(all_videos, axis=0)
    all_audios = np.concatenate(all_audios, axis=0)

    output_dict["video_latents"] = all_videos
    output_dict["audio_latents"] = all_audios
    output_dict["prompts"] = np.array(all_prompts)

    return output_dict


def main():
    """
    Aggregate all ode pairs inside a folder into a lmdb dataset.
    Each pt file should contain a (key, value) pair representing a
    video's ODE trajectories.
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, required=True, help="path to ode pairs")
    parser.add_argument("--lmdb_path", type=str, required=True, help="path to lmdb")

    args = parser.parse_args()

    all_files = sorted(glob.glob(os.path.join(args.data_path, "*.pt")))

    # figure out the maximum map size needed
    total_array_size = 5000000000000  # adapt to your need, set to 5TB by default

    env = lmdb.open(args.lmdb_path, map_size=total_array_size * 2)

    counter = 0

    seen_prompts = set()  # for deduplication

    for index, file in tqdm(enumerate(all_files)):
        # read from disk
        data_dict = torch.load(file)

        data_dict = process_data_dict(data_dict, seen_prompts)

        # write to lmdb file
        store_arrays_to_lmdb(env, data_dict, start_index=counter)
        counter += len(data_dict["prompts"])

    # save each entry's shape to lmdb
    with env.begin(write=True) as txn:
        for key, val in data_dict.items():
            array_shape = np.array(val.shape)
            array_shape[0] = counter

            shape_key = f"{key}_shape".encode()
            shape_str = " ".join(map(str, array_shape))
            txn.put(shape_key, shape_str.encode())


if __name__ == "__main__":
    main()
