import os
import subprocess
from dataclasses import dataclass
from statistics import fmean
from typing import Optional, Sequence

import torch

from hallolive.ovi.utils.io_utils import save_video
from third_party.audiobox_aesthetics.infer import initialize_predictor
from third_party.sync.syncnet import SyncNetEval
from third_party.sync.syncnet_detect import SyncNetDetector
from third_party.videoalign.wan_inference import VideoVLMRewardInference


CRED = "\033[91m"
CEND = "\033[0m"


def red_text(text: str) -> str:
    return f"{CRED}{text}{CEND}"


def syncnet_eval(syncnet, syncnet_detector, video_path, temp_dir, detect_results_dir="detect_results"):
    syncnet_detector(video_path=video_path, min_track=50)
    crop_videos = os.listdir(os.path.join(detect_results_dir, "crop"))

    if crop_videos == []:
        print(red_text(f"Face not detected in {video_path}"))
        return None, 0

    av_offset_list = []
    conf_list = []
    for video in crop_videos:
        av_offset, _, conf = syncnet.evaluate(
            video_path=os.path.join(detect_results_dir, "crop", video), temp_dir=temp_dir
        )
        av_offset_list.append(av_offset)
        conf_list.append(conf)

    av_offset = int(fmean(av_offset_list))
    conf = fmean(conf_list)
    print(f"Input video: {video_path}\nSyncNet confidence: {conf:.2f}\nAV offset: {av_offset}")
    return av_offset, conf


@dataclass
class RewardOutputs:
    sync: Optional[float]
    audio_score: Optional[float]
    ta_reward: Optional[torch.Tensor]
    vq_reward: Optional[torch.Tensor]
    mq_reward: Optional[torch.Tensor]
    video_pred_path: str
    output_audio_path: str


class MultimodalRewardEvaluator:
    def __init__(
        self,
        model_dir: str,
        temp_root: str = "temp",
        reward_model_cpu_offload: bool = True,
        reward_types: Optional[Sequence[str]] = None,
    ):
        self.reward_types = reward_types
        self.use_videoalign = "videoalign" in self.reward_types
        self.use_audiobox = "audiobox" in self.reward_types
        self.use_sync = "sync" in self.reward_types

        self.temp_root = os.path.abspath(temp_root)
        os.makedirs(self.temp_root, exist_ok=True)

        self.rank = torch.distributed.get_rank()
        self.paths = self._rank_paths(self.rank)

        self.device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.reward_model_cpu_offload = reward_model_cpu_offload

        self.syncnet = None
        self.syncnet_detector = None
        self.audiobox_predictor = None
        self.videoalign_inferencer = None

        if self.use_sync:
            syncnet_ckpt = os.path.join(model_dir, "LatentSync-1.6/auxiliary/syncnet_v2.model")
            syncnet_detector_ckpt = os.path.join(model_dir, "LatentSync-1.6/auxiliary/sfd_face.pth")
            self.syncnet = SyncNetEval(device=self.device)
            self.syncnet.loadParameters(syncnet_ckpt)
            self.syncnet_detector = SyncNetDetector(
                ckpt_path=syncnet_detector_ckpt,
                device=self.device,
                detect_results_dir=self.paths["detect_results_dir"],
            )

        if self.use_audiobox:
            audiobox_ckpt = os.path.join(model_dir, "audiobox-aesthetics", "checkpoint.pt")
            self.audiobox_predictor = initialize_predictor(audiobox_ckpt)
            self.audiobox_predictor.model.requires_grad_(False)
            self._move_audiobox_to("cpu" if self.reward_model_cpu_offload else self.device)

        if self.use_videoalign:
            videoreward_model_dir = os.path.join(model_dir, "VideoReward")
            qwen_vl_dir = os.path.join(model_dir, "Qwen2-VL-2B-Instruct")
            self.videoalign_inferencer = VideoVLMRewardInference(
                videoreward_model_dir,
                qwen_vl_dir=qwen_vl_dir,
                device="cpu" if self.reward_model_cpu_offload else self.device,
            )
            self.videoalign_inferencer.model.requires_grad_(False)
            if self.reward_model_cpu_offload:
                self._move_videoalign_to("cpu")

    def _rank_paths(self, rank: int) -> dict:
        rank_dir = os.path.join(self.temp_root, f"rank_{rank}")
        video_dir = os.path.join(rank_dir, "outputs")
        sync_dir = os.path.join(rank_dir, "syncnet")

        for path in (rank_dir, video_dir, sync_dir):
            os.makedirs(path, exist_ok=True)

        exp_name = os.path.basename(self.temp_root)

        video_pred_path = os.path.abspath(os.path.join(video_dir, f"{exp_name}_video_{rank}.mp4"))
        return {
            "video_pred_path": video_pred_path,
            "output_audio_path": os.path.splitext(video_pred_path)[0] + ".wav",
            "detect_results_dir": os.path.join(sync_dir, "detect_results"),
            "syncnet_temp_dir": os.path.join(sync_dir, "eval_tmp"),
        }

    def _save_prediction_video(
        self, video_pred_path: str, video_decoded: torch.Tensor, audio_decoded: torch.Tensor
    ) -> None:
        video_numpy = video_decoded[0].detach().cpu().float().numpy()
        audio_numpy = audio_decoded[0].squeeze().detach().cpu().float().numpy()

        save_video(video_pred_path, video_numpy, audio_numpy, fps=24, sample_rate=16000)

    def _compute_sync_score(self, video_pred_path: str, syncnet_temp_dir: str) -> float:
        _, sync = syncnet_eval(
            self.syncnet,
            self.syncnet_detector,
            video_pred_path,
            syncnet_temp_dir,
            detect_results_dir=self.paths["detect_results_dir"],
        )
        return sync

    @staticmethod
    def _extract_audio(video_pred_path: str, output_audio_path: str) -> None:
        command = [
            "ffmpeg",
            "-y",
            "-nostdin",
            "-loglevel", "error",
            "-i", video_pred_path,
            "-vn",
            "-acodec", "pcm_s16le",
            "-ar", "16000",
            "-ac", "1",
            output_audio_path,
        ]  # fmt:skip
        subprocess.run(command, check=True, stdin=subprocess.DEVNULL)

    def _move_audiobox_to(self, device: str) -> None:
        target_device = torch.device(device)
        self.audiobox_predictor.device = target_device
        self.audiobox_predictor.model.to(target_device)

    def _compute_audio_score(self, output_audio_path: str) -> float:
        with torch.inference_mode():
            if self.reward_model_cpu_offload:
                self._move_audiobox_to(self.device)
            outputs = self.audiobox_predictor.forward([{"path": output_audio_path}])
            first_output = outputs[0]
            audio_score = sum(
                first_output[key] for key in ("CE", "CU", "PQ")
            )  # Production Complexity is not an indicator of audio quality
            if self.reward_model_cpu_offload:
                self._move_audiobox_to("cpu")

        if self.reward_model_cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return audio_score

    def _move_videoalign_to(self, device: str) -> None:
        self.videoalign_inferencer.device = device
        self.videoalign_inferencer.model.to(device)

    def _compute_videoalign_reward(
        self, video_pred_path: str, prompt: str
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        with torch.inference_mode():
            if self.reward_model_cpu_offload:
                self._move_videoalign_to(self.device)
            reward = self.videoalign_inferencer.reward(video_pred_path, [prompt], use_norm=True)
            ta_reward = torch.tensor(reward[0]["TA"], device=self.device)
            vq_reward = torch.tensor(reward[0]["VQ"], device=self.device)
            mq_reward = torch.tensor(reward[0]["MQ"], device=self.device)
            if self.reward_model_cpu_offload:
                self._move_videoalign_to("cpu")

        if self.reward_model_cpu_offload and torch.cuda.is_available():
            torch.cuda.empty_cache()

        return ta_reward, vq_reward, mq_reward

    def evaluate(self, video_decoded: torch.Tensor, audio_decoded: torch.Tensor, prompt: str) -> RewardOutputs:
        self._save_prediction_video(self.paths["video_pred_path"], video_decoded, audio_decoded)
        sync = None
        audio_score = None
        ta_reward = None
        vq_reward = None
        mq_reward = None

        if self.use_sync:
            sync = self._compute_sync_score(self.paths["video_pred_path"], self.paths["syncnet_temp_dir"])

        if self.use_audiobox:
            self._extract_audio(self.paths["video_pred_path"], self.paths["output_audio_path"])
            audio_score = self._compute_audio_score(self.paths["output_audio_path"])

        if self.use_videoalign:
            ta_reward, vq_reward, mq_reward = self._compute_videoalign_reward(self.paths["video_pred_path"], prompt)

        return RewardOutputs(
            sync=sync,
            audio_score=audio_score,
            ta_reward=ta_reward,
            vq_reward=vq_reward,
            mq_reward=mq_reward,
            video_pred_path=self.paths["video_pred_path"],
            output_audio_path=self.paths["output_audio_path"],
        )
