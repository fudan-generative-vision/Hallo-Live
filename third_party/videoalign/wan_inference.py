import ast
import json
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"
import pdb
from collections.abc import Mapping
import pandas as pd

import torch

from .data import DataConfig
from .prompt_template import build_prompt
from .train_reward import create_model_and_processor
from .utils import ModelConfig, PEFTLoraConfig, TrainingConfig
from .utils import load_model_from_checkpoint
from .vision_process import process_vision_info, process_wanvideo_tensor

import numpy as np
from PIL import Image
import imageio


def load_configs_from_json(config_path):
    with open(config_path, "r") as f:
        config_dict = json.load(f)

    # del config_dict["training_args"]["_n_gpu"]
    del config_dict["data_config"]["meta_data"]
    del config_dict["data_config"]["data_dir"]

    return (
        config_dict["data_config"],
        None,
        config_dict["model_config"],
        config_dict["peft_lora_config"],
        config_dict["inference_config"] if "inference_config" in config_dict else None,
    )


class VideoVLMRewardInference:
    def __init__(
        self,
        load_from_pretrained,
        load_from_pretrained_step=-1,
        qwen_vl_dir="Qwen/Qwen2-VL-2B-Instruct",
        device="cuda",
        dtype=torch.bfloat16,
    ):
        config_path = os.path.join(load_from_pretrained, "model_config.json")
        data_config, _, model_config, peft_lora_config, inference_config = load_configs_from_json(config_path)
        data_config = DataConfig(**data_config)
        model_config = ModelConfig(**model_config)
        peft_lora_config = PEFTLoraConfig(**peft_lora_config)

        training_args = TrainingConfig(
            load_from_pretrained=load_from_pretrained,
            load_from_pretrained_step=load_from_pretrained_step,
            gradient_checkpointing=False,
            disable_flash_attn2=False,
            bf16=True if dtype == torch.bfloat16 else False,
            fp16=True if dtype == torch.float16 else False,
            output_dir="",
        )

        model, processor, peft_config = create_model_and_processor(
            model_config=model_config,
            peft_lora_config=peft_lora_config,
            training_args=training_args,
            qwen_vl_dir=qwen_vl_dir,
        )

        self.device = device

        model, checkpoint_step = load_model_from_checkpoint(model, load_from_pretrained, load_from_pretrained_step)
        model.eval()

        self.model = model
        self.processor = processor

        self.model.to(self.device)

        self.data_config = data_config

        self.inference_config = inference_config

    def _norm(self, reward):
        if self.inference_config is None:
            return reward
        else:
            reward["VQ"] = (reward["VQ"] - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
            reward["MQ"] = (reward["MQ"] - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
            reward["TA"] = (reward["TA"] - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
            return reward

    def _pad_sequence(self, sequences, attention_mask, max_len, padding_side="right"):
        """
        Pad the sequences to the maximum length.
        """
        assert padding_side in ["right", "left"]
        if sequences.shape[1] >= max_len:
            return sequences, attention_mask

        pad_len = max_len - sequences.shape[1]
        padding = (0, pad_len) if padding_side == "right" else (pad_len, 0)

        sequences_padded = torch.nn.functional.pad(
            sequences, padding, "constant", self.processor.tokenizer.pad_token_id
        )
        attention_mask_padded = torch.nn.functional.pad(attention_mask, padding, "constant", 0)

        return sequences_padded, attention_mask_padded

    def _prepare_input(self, data):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        if isinstance(data, Mapping):
            return type(data)({k: self._prepare_input(v) for k, v in data.items()})
        elif isinstance(data, (tuple, list)):
            return type(data)(self._prepare_input(v) for v in data)
        elif isinstance(data, torch.Tensor):
            kwargs = {"device": self.device}
            ## TODO: Maybe need to add dtype
            # if self.is_deepspeed_enabled and (torch.is_floating_point(data) or torch.is_complex(data)):
            #     # NLP models inputs are int/uint and those get adjusted to the right dtype of the
            #     # embedding. Other models such as wav2vec2's inputs are already float and thus
            #     # may need special handling to match the dtypes of the model
            #     kwargs.update({"dtype": self.accelerator.state.deepspeed_plugin.hf_ds_config.dtype()})
            return data.to(**kwargs)
        return data

    def _prepare_inputs(self, inputs):
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs = self._prepare_input(inputs)
        if len(inputs) == 0:
            raise ValueError
        return inputs

    def prepare_batch_from_frames(self, video_tensors, prompts, fps=None, num_frames=None, max_pixels=None):
        """
        Args:
            video_tensors: List[torch.Tensor], shape [T, C, H, W]
            prompts: List[str]
        """
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels

        chat_data = [
            [
                {
                    "role": "user",
                    "content": [
                        {"type": "video", "video": "file://dummy_path"},
                        {
                            "type": "text",
                            "text": build_prompt(
                                prompt, self.data_config.eval_dim, self.data_config.prompt_template_type
                            ),
                        },
                    ],
                }
            ]
            for prompt in prompts
        ]

        processed_videos = []
        for tensor in video_tensors:
            processed = process_wanvideo_tensor(tensor)
            processed_videos.append(processed)
        video_tensors = processed_videos
        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=None,
            videos=video_tensors,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )

        return self._prepare_inputs(batch)

    def reward_from_frames(self, video_tensors, prompts, use_norm=True):
        batch = self.prepare_batch_from_frames(video_tensors, prompts)
        rewards = self.model(**batch, return_dict=True)["logits"]  # [B, 3]

        if use_norm:
            vq = (rewards[:, 0] - self.inference_config["VQ_mean"]) / self.inference_config["VQ_std"]
            mq = (rewards[:, 1] - self.inference_config["MQ_mean"]) / self.inference_config["MQ_std"]
            ta = (rewards[:, 2] - self.inference_config["TA_mean"]) / self.inference_config["TA_std"]
        else:
            vq = rewards[:, 0]
            mq = rewards[:, 1]
            ta = rewards[:, 2]

        overall = (1 * mq + 1 * vq + 1 * ta) / 3

        return {"VQ": vq, "MQ": mq, "TA": ta, "Overall": overall}

    def prepare_batch(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None):
        fps = self.data_config.fps if fps is None else fps
        num_frames = self.data_config.num_frames if num_frames is None else num_frames
        max_pixels = self.data_config.max_frame_pixels if max_pixels is None else max_pixels
        # print(video_paths)
        if num_frames is None:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_paths,
                                "max_pixels": max_pixels,
                                "fps": fps,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompts, self.data_config.eval_dim, self.data_config.prompt_template_type
                                ),
                            },
                        ],
                    }
                ]
            ]
        else:
            chat_data = [
                [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "video",
                                "video": video_path,
                                "max_pixels": max_pixels,
                                "nframes": num_frames,
                                "sample_type": self.data_config.sample_type,
                            },
                            {
                                "type": "text",
                                "text": build_prompt(
                                    prompt, self.data_config.eval_dim, self.data_config.prompt_template_type
                                ),
                            },
                        ],
                    }
                ]
                for video_path, prompt in zip(video_paths, prompts)
            ]
        # print(chat_data)
        image_inputs, video_inputs = process_vision_info(chat_data)

        batch = self.processor(
            text=self.processor.apply_chat_template(chat_data, tokenize=False, add_generation_prompt=True),
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs={"do_rescale": True},
        )
        batch = self._prepare_inputs(batch)
        return batch

    def reward(self, video_paths, prompts, fps=None, num_frames=None, max_pixels=None, use_norm=True):
        """
        Inputs:
            video_paths: List[str], B paths of the videos.
            prompts: List[str], B prompts for the videos.
            eval_dims: List[str], N evaluation dimensions.
            fps: float, sample rate of the videos. If None, use the default value in the config.
            num_frames: int, number of frames of the videos. If None, use the default value in the config.
            max_pixels: int, maximum pixels of the videos. If None, use the default value in the config.
            use_norm: bool, whether to rescale the output rewards
        Outputs:
            Rewards: List[dict], N + 1 rewards of the B videos.
        """
        assert fps is None or num_frames is None, "fps and num_frames cannot be set at the same time."
        # print(f"video path :     {video_paths}")
        batch = self.prepare_batch(video_paths, prompts, fps, num_frames, max_pixels)
        rewards = self.model(return_dict=True, **batch)["logits"]

        rewards = [{"VQ": reward[0].item(), "MQ": reward[1].item(), "TA": reward[2].item()} for reward in rewards]
        for i in range(len(rewards)):
            if use_norm:
                rewards[i] = self._norm(rewards[i])
            rewards[i]["Overall"] = rewards[i]["VQ"] + rewards[i]["MQ"] + rewards[i]["TA"]

        return rewards


if __name__ == "__main__":
    load_from_pretrained = "models/VideoReward"
    device = "cuda:0"
    dtype = torch.bfloat16

    inferencer = VideoVLMRewardInference(load_from_pretrained, device=device, dtype=dtype)

    video_paths = ["test/video1.mp4", "test/video2.mp4", "test/video3.mp4"]

    prompts = [
        "A stylish woman strolls down a bustling Tokyo street, the warm glow of neon lights and animated city signs casting vibrant reflections. She wears a sleek black leather jacket paired with a flowing red dress and black boots, her black purse slung over her shoulder. Sunglasses perched on her nose and a bold red lipstick add to her confident, casual demeanor. The street is damp and reflective, creating a mirror-like effect that enhances the colorful lights and shadows. Pedestrians move about, adding to the lively atmosphere. The scene is captured in a dynamic medium shot with the woman walking slightly to one side, highlighting her graceful strides.",
        "A stunning mid-afternoon landscape photograph with a low camera angle, showcasing several giant wooly mammoths treading through a snowy meadow. Their long, wooly fur gently billows in the brisk wind as they move, creating a sense of natural movement. Snow-covered trees and dramatic snow-capped mountains loom in the distance, adding to the majestic setting. Wispy clouds and a high sun cast a warm glow over the scene, enhancing the serene and awe-inspiring atmosphere. The depth of field brings out the detailed textures of the mammoths and the snowy environment, capturing every nuance of these prehistoric giants in breathtaking clarity.",
        "A movie trailer in a classic cinematic style, featuring the adventurous journey of a 30-year-old space man wearing a vibrant red wool knitted motorcycle helmet. The scene unfolds against a vast blue sky and a desolate salt desert landscape. Shot on 35mm film, the trailer showcases vivid and rich colors, capturing the hero as he navigates through the harsh terrain with determination. His helmet glints under the sun, adding to the dramatic effect. The background is a mix of sweeping desert vistas and distant horizons, with the occasional shimmer of light reflecting off the salt flats. A dynamic medium shot with a sweeping overhead angle, emphasizing the hero's resilience and the vastness of his adventure.",
    ]
    with torch.no_grad():
        rewards = inferencer.reward(video_paths, prompts, use_norm=True)
        print(rewards)

    def load_video_frames(video_path, num_frames=81):
        try:
            reader = imageio.get_reader(video_path)
            frames = []
            for i, frame in enumerate(reader):
                if i >= num_frames:
                    break
                img = Image.fromarray(frame)
                frames.append(np.array(img))

            # Convert to tensor [T, H, W, C] -> [T, C, H, W]
            tensor = torch.tensor(np.stack(frames)).permute(0, 3, 1, 2)
            print(tensor.shape)
            return tensor
        except Exception as e:
            print(f"fail: {video_path}, error: {e}")
            return torch.zeros(num_frames, 3, 224, 224, dtype=torch.uint8)

    video_tensors = [load_video_frames(path) for path in video_paths]

    with torch.no_grad():
        rewards = inferencer.reward_from_frames(video_tensors, prompts, use_norm=True)
        print(rewards)
