import types
from typing import List, Optional
import torch
import os
import json
from torch import nn

from hallolive.utils.scheduler import SchedulerInterface, FlowMatchScheduler
from hallolive.wan.modules.tokenizers import HuggingfaceTokenizer
from hallolive.wan.modules.model import RegisterTokens, GanAttentionBlock
from hallolive.wan.modules.vae import _video_vae
from hallolive.wan.modules.t5 import umt5_xxl
from hallolive.ovi.modules.fusion import FusionModel
from hallolive.ovi.modules.causal_fusion import CausalFusionModel


DEFAULT_MODEL_DIR = "model"
DEFAULT_WAN_MODEL_NAME = "Wan2.1-T2V-1.3B"


def _wan_model_dir(model_dir: str, model_name: str) -> str:
    return os.path.join(model_dir, model_name)


def init_fusion_model(
    video_config: str,
    audio_config: str,
    is_causal=False,
    meta_init=False,
    enable_cross_attention=True,
    future_audio_frames=None,
):
    assert os.path.exists(video_config), f"{video_config} does not exist"
    assert os.path.exists(audio_config), f"{audio_config} does not exist"

    with open(video_config) as f:
        video_config = json.load(f)

    with open(audio_config) as f:
        audio_config = json.load(f)

    if meta_init:
        with torch.device("meta"):
            if is_causal:
                fusion_model = CausalFusionModel(
                    video_config, audio_config, enable_cross_attention, future_audio_frames=future_audio_frames
                )
            else:
                fusion_model = FusionModel(video_config, audio_config)
    else:
        if is_causal:
            fusion_model = CausalFusionModel(
                video_config, audio_config, enable_cross_attention, future_audio_frames=future_audio_frames
            )
        else:
            fusion_model = FusionModel(video_config, audio_config)

    return fusion_model, video_config, audio_config


class WanTextEncoder(torch.nn.Module):
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, model_name: str = DEFAULT_WAN_MODEL_NAME) -> None:
        super().__init__()
        model_dir = _wan_model_dir(model_dir, model_name)

        self.text_encoder = (
            umt5_xxl(encoder_only=True, return_tokenizer=False, dtype=torch.float32, device=torch.device("cpu"))
            .eval()
            .requires_grad_(False)
        )
        self.text_encoder.load_state_dict(
            torch.load(
                os.path.join(model_dir, "models_t5_umt5-xxl-enc-bf16.pth"), map_location="cpu", weights_only=False
            )
        )

        self.tokenizer = HuggingfaceTokenizer(
            name=os.path.join(model_dir, "google", "umt5-xxl"), seq_len=512, clean="whitespace"
        )

    @property
    def device(self):
        # Assume we are always on GPU
        return torch.cuda.current_device()

    def forward(self, text_prompts: List[str]) -> dict:
        ids, mask = self.tokenizer(text_prompts, return_mask=True, add_special_tokens=True)
        ids = ids.to(self.device)
        mask = mask.to(self.device)
        seq_lens = mask.gt(0).sum(dim=1).long()
        context = self.text_encoder(ids, mask)

        for u, v in zip(context, seq_lens):
            u[v:] = 0.0  # set padding to 0.0

        return {"prompt_embeds": context}


class WanVAEWrapper(torch.nn.Module):
    def __init__(self, model_dir: str = DEFAULT_MODEL_DIR, model_name: str = DEFAULT_WAN_MODEL_NAME):
        super().__init__()
        mean = [
            -0.7571,
            -0.7089,
            -0.9113,
            0.1075,
            -0.1745,
            0.9653,
            -0.1517,
            1.5508,
            0.4134,
            -0.0715,
            0.5517,
            -0.3632,
            -0.1922,
            -0.9497,
            0.2503,
            -0.2921,
        ]
        std = [
            2.8184,
            1.4541,
            2.3275,
            2.6558,
            1.2196,
            1.7708,
            2.6052,
            2.0743,
            3.2687,
            2.1526,
            2.8652,
            1.5579,
            1.6382,
            1.1253,
            2.8251,
            1.9160,
        ]
        self.mean = torch.tensor(mean, dtype=torch.float32)
        self.std = torch.tensor(std, dtype=torch.float32)

        # init model
        self.model = (
            _video_vae(pretrained_path=os.path.join(_wan_model_dir(model_dir, model_name), "Wan2.1_VAE.pth"), z_dim=16)
            .eval()
            .requires_grad_(False)
        )

    def encode_to_latent(self, pixel: torch.Tensor) -> torch.Tensor:
        # pixel: [batch_size, num_channels, num_frames, height, width]
        device, dtype = pixel.device, pixel.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        output = [self.model.encode(u.unsqueeze(0), scale).float().squeeze(0) for u in pixel]
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output

    def decode_to_pixel(self, latent: torch.Tensor, use_cache: bool = False) -> torch.Tensor:
        # from [batch_size, num_frames, num_channels, height, width]
        # to [batch_size, num_channels, num_frames, height, width]
        zs = latent.permute(0, 2, 1, 3, 4)
        if use_cache:
            assert latent.shape[0] == 1, "Batch size must be 1 when using cache"

        device, dtype = latent.device, latent.dtype
        scale = [self.mean.to(device=device, dtype=dtype), 1.0 / self.std.to(device=device, dtype=dtype)]

        if use_cache:
            decode_function = self.model.cached_decode
        else:
            decode_function = self.model.decode

        output = []
        for u in zs:
            output.append(decode_function(u.unsqueeze(0), scale).float().clamp_(-1, 1).squeeze(0))
        output = torch.stack(output, dim=0)
        # from [batch_size, num_channels, num_frames, height, width]
        # to [batch_size, num_frames, num_channels, height, width]
        output = output.permute(0, 2, 1, 3, 4)
        return output


class FusionDiffusionWrapper(torch.nn.Module):
    def __init__(
        self,
        model_name="Wan2.1-T2V-1.3B",
        timestep_shift=8.0,
        is_causal=False,
        local_attn_size=-1,
        sink_size=0,
        video_config: str = None,
        audio_config: str = None,
        enable_cross_attention=True,
        future_audio_frames=None,
        meta_init=False,
    ):
        super().__init__()
        self.meta_init = meta_init

        if is_causal:
            self.model, video_config, audio_config = init_fusion_model(
                video_config,
                audio_config,
                is_causal=True,
                meta_init=meta_init,
                enable_cross_attention=enable_cross_attention,
                future_audio_frames=future_audio_frames,
            )
        else:
            self.model, video_config, audio_config = init_fusion_model(
                video_config, audio_config, is_causal=False, meta_init=meta_init
            )

        self.model.eval()
        self.video_config = video_config
        self.audio_config = audio_config

        # For non-causal diffusion, all frames share the same timestep
        self.uniform_timestep = not is_causal
        self.is_causal = is_causal

        self.scheduler = FlowMatchScheduler(shift=timestep_shift, sigma_min=0.0, extra_one_step=True)
        self.scheduler.set_timesteps(1000, training=True)

        self.video_seq_len = 14880  # [1, 30, 48, 32, 62]
        self.audio_seq_len = 150  # [1, 150, 20]

        self.post_init()

    def enable_gradient_checkpointing(self) -> None:
        self.model.enable_gradient_checkpointing()

    def adding_cls_branch(self, atten_dim=1536, num_class=4, time_embed_dim=0) -> None:
        # NOTE: This is hard coded for WAN2.1-T2V-1.3B for now!!!!!!!!!!!!!!!!!!!!
        self._cls_pred_branch = nn.Sequential(
            # Input: [B, 384, 21, 60, 104]
            nn.LayerNorm(atten_dim * 3 + time_embed_dim),
            nn.Linear(atten_dim * 3 + time_embed_dim, 1536),
            nn.SiLU(),
            nn.Linear(atten_dim, num_class),
        )
        self._cls_pred_branch.requires_grad_(True)
        num_registers = 3
        self._register_tokens = RegisterTokens(num_registers=num_registers, dim=atten_dim)
        self._register_tokens.requires_grad_(True)

        gan_ca_blocks = []
        for _ in range(num_registers):
            block = GanAttentionBlock()
            gan_ca_blocks.append(block)
        self._gan_ca_blocks = nn.ModuleList(gan_ca_blocks)
        self._gan_ca_blocks.requires_grad_(True)
        # self.has_cls_branch = True

    def _convert_flow_pred_to_x0(
        self, flow_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert flow matching's prediction to x0 prediction.
        flow_pred: the prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = noise - x0
        x_t = (1-sigma_t) * x0 + sigma_t * noise
        we have x0 = x_t - sigma_t * pred
        see derivations https://chatgpt.com/share/67bf8589-3d04-8008-bc6e-4cf1a24e2d0e
        """

        # use higher precision for calculations
        original_dtype = flow_pred.dtype
        flow_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(flow_pred.device), [flow_pred, xt, self.scheduler.sigmas, self.scheduler.timesteps]
        )

        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

        if flow_pred.dim() == 2:
            sigma_t = sigmas[timestep_id].reshape(-1, 1)
        else:
            sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)
        x0_pred = xt - sigma_t * flow_pred
        return x0_pred.to(original_dtype)

    @staticmethod
    def _convert_x0_to_flow_pred(
        scheduler, x0_pred: torch.Tensor, xt: torch.Tensor, timestep: torch.Tensor
    ) -> torch.Tensor:
        """
        Convert x0 prediction to flow matching's prediction.
        x0_pred: the x0 prediction with shape [B, C, H, W]
        xt: the input noisy data with shape [B, C, H, W]
        timestep: the timestep with shape [B]

        pred = (x_t - x_0) / sigma_t
        """
        # use higher precision for calculations
        original_dtype = x0_pred.dtype
        x0_pred, xt, sigmas, timesteps = map(
            lambda x: x.double().to(x0_pred.device), [x0_pred, xt, scheduler.sigmas, scheduler.timesteps]
        )
        timestep_id = torch.argmin((timesteps.unsqueeze(0) - timestep.unsqueeze(1)).abs(), dim=1)

        if x0_pred.dim() == 2:
            sigma_t = sigmas[timestep_id].reshape(-1, 1)
        else:
            sigma_t = sigmas[timestep_id].reshape(-1, 1, 1, 1)

        flow_pred = (xt - x0_pred) / sigma_t
        return flow_pred.to(original_dtype)

    def forward(
        self,
        noisy_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        conditional_dict: dict,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        kv_cache: Optional[List[dict]] = None,
        crossattn_cache: Optional[List[dict]] = None,
        video_current_start: Optional[int] = None,
        audio_current_start: Optional[int] = None,
        classify_mode: Optional[bool] = False,
        concat_time_embeddings: Optional[bool] = False,
        clean_x: Optional[torch.Tensor] = None,
        aug_t: Optional[torch.Tensor] = None,
        cache_start: Optional[int] = None,
    ) -> torch.Tensor:
        video_prompt_embeds = conditional_dict["video_prompt_embeds"]
        audio_prompt_embeds = conditional_dict["audio_prompt_embeds"]

        # [B, F] -> [B]
        if self.uniform_timestep:
            video_input_timestep = video_timestep[:, 0]
            audio_input_timestep = audio_timestep[:, 0]
        else:
            video_input_timestep = video_timestep
            audio_input_timestep = audio_timestep

        logits = None
        # X0 prediction
        if kv_cache is not None:
            video_flow_pred, audio_flow_pred = self.model(
                noisy_video.permute(0, 2, 1, 3, 4),
                noisy_audio,
                video_t=video_input_timestep,
                audio_t=audio_input_timestep,
                vid_context=video_prompt_embeds,
                audio_context=audio_prompt_embeds,
                vid_seq_len=self.video_seq_len,
                audio_seq_len=self.audio_seq_len,
                kv_cache=kv_cache,
                crossattn_cache=crossattn_cache,
                video_current_start=video_current_start,
                audio_current_start=audio_current_start,
                cache_start=cache_start,
            )
            video_flow_pred = video_flow_pred.permute(0, 2, 1, 3, 4)
            audio_flow_pred = audio_flow_pred.squeeze(2)

        else:
            if clean_x is not None:
                # teacher forcing
                flow_pred = self.model(
                    noisy_video.permute(0, 2, 1, 3, 4),
                    noisy_audio,
                    video_t=video_input_timestep,
                    audio_t=audio_input_timestep,
                    vid_context=video_prompt_embeds,
                    audio_context=audio_prompt_embeds,
                    vid_seq_len=self.video_seq_len,
                    audio_seq_len=self.audio_seq_len,
                    clean_x=clean_x.permute(0, 2, 1, 3, 4),
                    aug_t=aug_t,
                ).permute(0, 2, 1, 3, 4)
            else:
                if classify_mode:
                    flow_pred, logits = self.model(
                        noisy_video.permute(0, 2, 1, 3, 4),
                        noisy_audio,
                        video_t=video_input_timestep,
                        audio_t=audio_input_timestep,
                        vid_context=video_prompt_embeds,
                        audio_context=audio_prompt_embeds,
                        vid_seq_len=self.video_seq_len,
                        audio_seq_len=self.audio_seq_len,
                        classify_mode=True,
                        register_tokens=self._register_tokens,
                        cls_pred_branch=self._cls_pred_branch,
                        gan_ca_blocks=self._gan_ca_blocks,
                        concat_time_embeddings=concat_time_embeddings,
                    )
                    flow_pred = flow_pred.permute(0, 2, 1, 3, 4)
                else:
                    if self.is_causal:
                        video_flow_pred, audio_flow_pred = self.model(
                            noisy_video.permute(0, 2, 1, 3, 4),
                            noisy_audio,
                            video_t=video_input_timestep,
                            audio_t=audio_input_timestep,
                            vid_context=video_prompt_embeds,
                            audio_context=audio_prompt_embeds,
                            vid_seq_len=self.video_seq_len,
                            audio_seq_len=self.audio_seq_len,
                        )
                    else:
                        video_flow_pred, audio_flow_pred = self.model(
                            [noisy_video.permute(0, 2, 1, 3, 4)[0]],
                            [noisy_audio[0]],
                            t=video_input_timestep,
                            vid_context=[video_prompt_embeds[0]],
                            audio_context=[audio_prompt_embeds[0]],
                            vid_seq_len=self.video_seq_len,
                            audio_seq_len=self.audio_seq_len,
                        )
                        video_flow_pred = video_flow_pred[0].unsqueeze(0)
                        audio_flow_pred = audio_flow_pred[0].unsqueeze(0)

                    video_flow_pred = video_flow_pred.permute(0, 2, 1, 3, 4)
                    audio_flow_pred = audio_flow_pred.squeeze(2)

        video_pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=video_flow_pred.flatten(0, 1),
            xt=noisy_video.flatten(0, 1),
            timestep=video_timestep.flatten(0, 1),
        ).unflatten(0, video_flow_pred.shape[:2])

        audio_pred_x0 = self._convert_flow_pred_to_x0(
            flow_pred=audio_flow_pred.flatten(0, 1),
            xt=noisy_audio.flatten(0, 1),
            timestep=audio_timestep.flatten(0, 1),
        ).unflatten(0, audio_flow_pred.shape[:2])

        if logits is not None:
            return video_flow_pred, audio_flow_pred, video_pred_x0, audio_pred_x0, logits

        return video_flow_pred, audio_flow_pred, video_pred_x0, audio_pred_x0

    def get_scheduler(self) -> SchedulerInterface:
        """
        Update the current scheduler with the interface's static method
        """
        scheduler = self.scheduler
        scheduler.convert_x0_to_noise = types.MethodType(SchedulerInterface.convert_x0_to_noise, scheduler)
        scheduler.convert_noise_to_x0 = types.MethodType(SchedulerInterface.convert_noise_to_x0, scheduler)
        scheduler.convert_velocity_to_x0 = types.MethodType(SchedulerInterface.convert_velocity_to_x0, scheduler)
        self.scheduler = scheduler
        return scheduler

    def post_init(self):
        """
        A few custom initialization steps that should be called after the object is created.
        Currently, the only one we have is to bind a few methods to scheduler.
        We can gradually add more methods here if needed.
        """
        self.get_scheduler()
