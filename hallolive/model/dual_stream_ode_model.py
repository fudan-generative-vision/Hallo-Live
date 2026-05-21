import torch.nn.functional as F
from typing import Tuple
import torch
import os

from hallolive.model.base_model import BaseModel
from hallolive.utils.wan_wrapper import WanTextEncoder, WanVAEWrapper
from hallolive.utils.fusion_wrapper import FusionDiffusionWrapper

from hallolive.ovi.utils.model_loading_utils import init_mmaudio_vae, init_wan_vae_2_2
from hallolive.utils.misc import load_ckpt, safe_load_state_dict


def load_mova_state_dict(model, model_dir):
    mova_ckpt = os.path.join(model_dir, "MOVA-720p", "audio_dit", "diffusion_pytorch_model.safetensors")
    mova_state_dict = load_ckpt(mova_ckpt)
    mova_state_dict = {k.replace(".modulation", ".modulation.modulation"): v for k, v in mova_state_dict.items()}
    mova_state_dict["head.modulation"] = mova_state_dict["head.modulation.modulation"]
    del mova_state_dict["head.modulation.modulation"]

    safe_load_state_dict(model, mova_state_dict)


class DualStreamODEModel(BaseModel):
    def __init__(self, config, device):
        """
        Initialize the ODERegression module.
        This class is self-contained and compute generator losses
        in the forward pass given precomputed ode solution pairs.
        This class supports the ode regression loss for both causal and bidirectional models.
        See Sec 4.3 of CausVid https://arxiv.org/abs/2412.07772 for details
        """
        super().__init__(config, device)

        # Step 1: Initialize all models

        if getattr(config, "generator_ckpt", False):
            generater_state_dict = load_ckpt(config.generator_ckpt)
            if config.generator_ckpt.endswith(".safetensors"):
                # self.generator.model.load_state_dict(generater_state_dict, strict=False)
                safe_load_state_dict(self.generator.model, generater_state_dict)
                if "1_3B" in config.generator_audio_config:
                    load_mova_state_dict(self.generator.model.audio_model, getattr(config, "model_dir", "model"))

            elif config.generator_ckpt.endswith(".pt"):
                self.generator.load_state_dict(generater_state_dict["generator"], strict=True)
            else:
                raise RuntimeError("Only support .safetensors and .pt checkpoints currently")

        self.video_num_frame_per_block = getattr(config, "video_num_frame_per_block", 1)
        self.video_loss_weight = getattr(config, "video_loss_weight", 0.85)
        self.audio_loss_weight = getattr(config, "audio_loss_weight", 0.15)
        self.audio_num_frame_per_block = getattr(config, "audio_num_frame_per_block", 5)

        if self.video_num_frame_per_block > 1:
            self.generator.model.video_model.num_frame_per_block = self.video_num_frame_per_block
            self.generator.model.audio_model.num_frame_per_block = self.audio_num_frame_per_block
            # self.generator.model.video_num_frame_per_block = self.video_num_frame_per_block
            # self.generator.model.audio_num_frame_per_block = self.audio_num_frame_per_block

        self.independent_first_frame = getattr(config, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if config.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()

        # Step 2: Initialize all hyperparameters
        self.timestep_shift = getattr(config, "timestep_shift", 1.0)

    def _initialize_models(self, config, device):
        model_dir = getattr(config, "model_dir", "model")
        self.generator = FusionDiffusionWrapper(
            timestep_shift=config.timestep_shift,
            video_config=config.generator_video_config,
            audio_config=config.generator_audio_config,
            is_causal=True,
            enable_cross_attention=getattr(config, 'enable_cross_attention', True),
            future_audio_frames=getattr(config, "future_audio_frames", 0),
        )
        self.generator.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder(model_dir=model_dir)
        self.text_encoder.requires_grad_(False)

        self.vae = WanVAEWrapper(model_dir=model_dir)
        self.vae.requires_grad_(False)

        self.vae_video = init_wan_vae_2_2(model_dir, rank=device)
        self.vae_video.model.requires_grad_(False).eval()
        self.vae_video.model = self.vae_video.model.bfloat16()

        self.vae_audio = init_mmaudio_vae(model_dir, rank=device)
        self.vae_audio.requires_grad_(False).eval()
        self.vae_audio = self.vae_audio.bfloat16()

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    @torch.no_grad()
    def _prepare_generator_input(
        self, video_ode_latent: torch.Tensor, audio_ode_latent: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Given a tensor containing the whole ODE sampling trajectories,
        randomly choose an intermediate timestep and return the latent as well as the corresponding timestep.
        Input:
            - ode_latent: a tensor containing the whole ODE sampling trajectories [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
        Output:
            - noisy_input: a tensor containing the selected latent [batch_size, num_frames, num_channels, height, width].
            - timestep: a tensor containing the corresponding timestep [batch_size].
        """
        batch_size, num_denoising_steps, video_num_frames, video_num_channels, height, width = (
            video_ode_latent.shape
        )  # [1, 5, 30, 48, 32, 62]

        batch_size, num_denoising_steps, audio_num_frames, audio_num_channels = (
            audio_ode_latent.shape
        )  # [1, 5, 150, 20]

        # Step 1: Randomly choose a timestep for each frame
        video_index = self._get_timestep(
            0,
            len(self.denoising_step_list),
            batch_size,
            video_num_frames,
            self.video_num_frame_per_block,
            uniform_timestep=False,
        )

        audio_index = video_index.reshape(video_index.shape[0], -1, self.video_num_frame_per_block)
        audio_index = audio_index[:, :, 0:1].repeat(1, 1, self.audio_num_frame_per_block)
        audio_index = audio_index.reshape(audio_index.shape[0], -1)

        # audio_index = self._get_timestep(
        #     0,
        #     len(self.denoising_step_list),
        #     batch_size,
        #     audio_num_frames,
        #     self.audio_num_frame_per_block,
        #     uniform_timestep=False,
        # )

        if self.config.i2v:
            video_index[:, 0] = len(self.denoising_step_list) - 1
            audio_index[:, 0] = len(self.denoising_step_list) - 1

        video_noisy_input = torch.gather(
            video_ode_latent,
            dim=1,
            index=video_index.reshape(batch_size, 1, video_num_frames, 1, 1, 1)
            .expand(-1, -1, -1, video_num_channels, height, width)
            .to(self.device),
        ).squeeze(1)

        audio_noisy_input = torch.gather(
            audio_ode_latent,
            dim=1,
            index=audio_index.reshape(batch_size, 1, audio_num_frames, 1)
            .expand(-1, -1, -1, audio_num_channels)
            .to(self.device),
        ).squeeze(1)

        video_timestep = self.denoising_step_list[video_index.to(self.denoising_step_list.device)].to(self.device)

        audio_timestep = self.denoising_step_list[audio_index.to(self.denoising_step_list.device)].to(self.device)

        return video_noisy_input, audio_noisy_input, video_timestep, audio_timestep

    def generator_loss(
        self, video_ode_latent: torch.Tensor, audio_ode_latent: torch.Tensor, conditional_dict: dict
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noisy latents and compute the ODE regression loss.
        Input:
            - ode_latent: a tensor containing the ODE latents [batch_size, num_denoising_steps, num_frames, num_channels, height, width].
            They are ordered from most noisy to clean latents.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - loss: a scalar tensor representing the generator loss.
            - log_dict: a dictionary containing additional information for loss timestep breakdown.
        """
        # Step 1: Run generator on noisy latents
        video_target_latent = video_ode_latent[:, -1]
        audio_target_latent = audio_ode_latent[:, -1]

        video_noisy_input, audio_noisy_input, video_timestep, audio_timestep = self._prepare_generator_input(
            video_ode_latent=video_ode_latent, audio_ode_latent=audio_ode_latent
        )

        video_flow_pred, audio_flow_pred, video_pred, audio_pred = self.generator(
            noisy_video=video_noisy_input,
            noisy_audio=audio_noisy_input,
            conditional_dict=conditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )

        # Step 2: Compute the regression loss
        video_mask = video_timestep != 0
        audio_mask = audio_timestep != 0

        video_loss = F.mse_loss(video_pred[video_mask], video_target_latent[video_mask], reduction="mean")
        audio_loss = F.mse_loss(audio_pred[audio_mask], audio_target_latent[audio_mask], reduction="mean")

        loss = self.video_loss_weight * video_loss + self.audio_loss_weight * audio_loss

        log_dict = {
            "unnormalized_loss": F.mse_loss(video_pred, video_target_latent, reduction="none")
            .mean(dim=[1, 2, 3, 4])
            .detach(),
            "timestep": video_timestep.float().mean(dim=1).detach(),
            "input_video": video_noisy_input.detach(),
            "output_video": video_pred.detach(),
            "input_audio": audio_noisy_input.detach(),
            "output_audio": audio_pred.detach(),
        }

        return loss, log_dict
