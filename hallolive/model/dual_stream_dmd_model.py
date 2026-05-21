import torch.nn.functional as F
from typing import Optional, Tuple
import torch
import os
from hallolive.utils.misc import load_ckpt, safe_load_state_dict, normalize_scalar_across_ranks

from hallolive.model.base_model import SelfForcingModel
from hallolive.utils.wan_wrapper import WanTextEncoder
from hallolive.utils.fusion_wrapper import FusionDiffusionWrapper
from hallolive.ovi.utils.model_loading_utils import init_mmaudio_vae, init_wan_vae_2_2
from hallolive.model.multimodal_reward_evaluator import MultimodalRewardEvaluator


class DualStreamDMDModel(SelfForcingModel):
    def __init__(self, config, device):
        """
        Initialize the DMD (Distribution Matching Distillation) module.
        This class is self-contained and compute generator and fake score losses
        in the forward pass.
        """
        super().__init__(config, device)
        self.use_decoupled_dmd = getattr(config, "use_decoupled_dmd", False)
        self.enable_rl_reward = getattr(config, "enable_rl_reward", False)
        self.reward_beta = getattr(config, "reward_beta", 1.0)
        if self.enable_rl_reward:
            self.reward_types = getattr(config, "reward_types", ["videoalign", "audiobox", "sync"])

        generator_state_dict = None
        if getattr(config, "generator_ckpt", False):
            generator_state_dict = load_ckpt(config.generator_ckpt)
            self.generator.load_state_dict(
                generator_state_dict["generator"] if "generator" in generator_state_dict else generator_state_dict,
                strict=True,
            )
            # self.generator.model.load_state_dict(generater_state_dict, strict=False)

        if getattr(config, "real_score_ckpt", False):
            real_score_state_dict = load_ckpt(config.real_score_ckpt)
            self.real_score.model.load_state_dict(real_score_state_dict, strict=True)

        if generator_state_dict is not None and "critic" in generator_state_dict:
            self.fake_score.load_state_dict(generator_state_dict["critic"], strict=True)
        else:
            if getattr(config, "fake_score_ckpt", False):
                fake_score_state_dict = load_ckpt(config.fake_score_ckpt)
                # self.fake_score.model.load_state_dict(fake_score_state_dict, strict=True)
                safe_load_state_dict(self.fake_score.model, fake_score_state_dict)

        self.video_num_frame_per_block = getattr(config, "video_num_frame_per_block", 1)
        self.audio_num_frame_per_block = getattr(config, "audio_num_frame_per_block", 5)
        self.video_loss_weight = getattr(config, "video_loss_weight", 0.85)
        self.audio_loss_weight = getattr(config, "audio_loss_weight", 0.15)
        self.same_step_across_blocks = getattr(config, "same_step_across_blocks", True)
        self.video_num_training_frames = getattr(config, "video_num_training_frames", 30)
        self.audio_num_training_frames = getattr(config, "audio_num_training_frames", 150)
        self.future_audio_frames = getattr(config, "future_audio_frames", None)

        if self.video_num_frame_per_block > 1:
            self.generator.model.video_model.num_frame_per_block = self.video_num_frame_per_block
            self.generator.model.audio_model.num_frame_per_block = self.audio_num_frame_per_block
            self.generator.model.video_num_frame_per_block = self.video_num_frame_per_block
            self.generator.model.audio_num_frame_per_block = self.audio_num_frame_per_block

        self.independent_first_frame = getattr(config, "independent_first_frame", False)
        if self.independent_first_frame:
            self.generator.model.independent_first_frame = True
        if config.gradient_checkpointing:
            self.generator.enable_gradient_checkpointing()
            self.fake_score.enable_gradient_checkpointing()

        # this will be init later with fsdp-wrapped modules
        self.inference_pipeline = None

        # Step 2: Initialize all dmd hyperparameters
        self.num_train_timestep = config.num_train_timestep
        self.min_step = int(0.02 * self.num_train_timestep)
        self.max_step = int(0.98 * self.num_train_timestep)
        if hasattr(config, "real_guidance_scale"):
            self.real_guidance_scale = config.real_guidance_scale
            self.fake_guidance_scale = config.fake_guidance_scale
        else:
            self.video_real_guidance_scale = config.video_guidance_scale
            self.audio_real_guidance_scale = config.audio_guidance_scale
            self.fake_guidance_scale = 0.0
        self.timestep_shift = getattr(config, "timestep_shift", 1.0)
        self.ts_schedule = getattr(config, "ts_schedule", True)
        self.ts_schedule_max = getattr(config, "ts_schedule_max", False)
        self.min_score_timestep = getattr(config, "min_score_timestep", 0)

        if getattr(self.scheduler, "alphas_cumprod", None) is not None:
            self.scheduler.alphas_cumprod = self.scheduler.alphas_cumprod.to(device)
        else:
            self.scheduler.alphas_cumprod = None

    def _initialize_models(self, config, device):
        model_dir = getattr(config, "model_dir", "model")
        if getattr(config, "enable_rl_reward", False):
            self.reward_evaluator: MultimodalRewardEvaluator = MultimodalRewardEvaluator(
                model_dir=model_dir,
                temp_root=os.path.join(config.reward_temp_dir, config.exp_name),
                reward_model_cpu_offload=config.reward_model_cpu_offload,
                reward_types=getattr(config, "reward_types", ["videoalign", "audiobox", "sync"]),
            )

        self.generator = FusionDiffusionWrapper(
            timestep_shift=config.timestep_shift,
            video_config=config.generator_video_config,
            audio_config=config.generator_audio_config,
            is_causal=True,
            enable_cross_attention=config.enable_cross_attention,
            future_audio_frames=getattr(config, "future_audio_frames", 0),
        )
        self.generator.model.requires_grad_(True)

        self.real_score = FusionDiffusionWrapper(
            timestep_shift=config.timestep_shift,
            video_config=config.real_score_video_config,
            audio_config=config.real_score_audio_config,
            is_causal=False,
        )
        self.real_score.model.requires_grad_(False)

        self.fake_score = FusionDiffusionWrapper(
            timestep_shift=config.timestep_shift,
            video_config=config.fake_score_video_config,
            audio_config=config.fake_score_audio_config,
            is_causal=False,
        )
        self.fake_score.model.requires_grad_(True)

        self.text_encoder = WanTextEncoder(model_dir=model_dir)
        self.text_encoder.requires_grad_(False)

        self.vae_video = init_wan_vae_2_2(model_dir, rank=device)
        self.vae_video.model.requires_grad_(False).eval()
        self.vae_video.model = self.vae_video.model.bfloat16()

        self.vae_audio = init_mmaudio_vae(model_dir, rank=device)
        self.vae_audio.requires_grad_(False).eval()
        self.vae_audio = self.vae_audio.bfloat16()

        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)

    def _compute_kl_grad(
        self,
        noisy_image_or_video: torch.Tensor,
        noisy_audio: torch.Tensor,
        noisy_video_ca: torch.Tensor,
        noisy_audio_ca: torch.Tensor,
        estimated_clean_image_or_video: torch.Tensor,
        estimated_clean_audio: torch.Tensor,
        video_timestep: torch.Tensor,
        audio_timestep: torch.Tensor,
        video_timestep_ca: torch.Tensor,
        audio_timestep_ca: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        normalization: bool = True,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the KL grad (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - noisy_image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - estimated_clean_image_or_video: a tensor with shape [B, F, C, H, W] representing the estimated clean image or video.
            - timestep: a tensor with shape [B, F] containing the randomly generated timestep.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - normalization: a boolean indicating whether to normalize the gradient.
        Output:
            - kl_grad: a tensor representing the KL grad.
            - kl_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Compute the fake score
        _, _, video_pred_fake_cond, audio_pred_fake_cond = self.fake_score(
            noisy_video=noisy_image_or_video,
            noisy_audio=noisy_audio,
            conditional_dict=conditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )

        if self.fake_guidance_scale != 0.0:
            _, _, video_pred_fake_uncond, audio_pred_fake_uncond = self.fake_score(
                noisy_video=noisy_image_or_video,
                noisy_audio=noisy_audio,
                conditional_dict=unconditional_dict,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,
            )
            video_pred_fake = (
                video_pred_fake_cond + (video_pred_fake_cond - video_pred_fake_uncond) * self.fake_guidance_scale
            )
            audio_pred_fake = (
                audio_pred_fake_cond + (audio_pred_fake_cond - audio_pred_fake_uncond) * self.fake_guidance_scale
            )
        else:
            video_pred_fake = video_pred_fake_cond
            audio_pred_fake = audio_pred_fake_cond

        # Step 2: Compute the real score
        # We compute the conditional and unconditional prediction
        # and add them together to achieve cfg (https://arxiv.org/abs/2207.12598)
        _, _, video_pred_real_cond, audio_pred_real_cond = self.real_score(
            noisy_video=noisy_image_or_video,
            noisy_audio=noisy_audio,
            conditional_dict=conditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )

        _, _, video_pred_real_uncond, audio_pred_real_uncond = self.real_score(
            noisy_video=noisy_image_or_video,
            noisy_audio=noisy_audio,
            conditional_dict=unconditional_dict,
            video_timestep=video_timestep,
            audio_timestep=audio_timestep,
        )

        if self.use_decoupled_dmd:
            _, _, video_pred_real_cond_ca, audio_pred_real_cond_ca = self.real_score(
                noisy_video=noisy_video_ca,
                noisy_audio=noisy_audio_ca,
                conditional_dict=conditional_dict,
                video_timestep=video_timestep_ca,
                audio_timestep=audio_timestep_ca,
            )

            _, _, video_pred_real_uncond_ca, audio_pred_real_uncond_ca = self.real_score(
                noisy_video=noisy_video_ca,
                noisy_audio=noisy_audio_ca,
                conditional_dict=unconditional_dict,
                video_timestep=video_timestep_ca,
                audio_timestep=audio_timestep_ca,
            )

        video_pred_real = (
            video_pred_real_uncond + (video_pred_real_cond - video_pred_real_uncond) * self.video_real_guidance_scale
        )
        audio_pred_real = (
            audio_pred_real_uncond + (audio_pred_real_cond - audio_pred_real_uncond) * self.audio_real_guidance_scale
        )

        # Step 3: Compute the DMD gradient (DMD paper eq. 7).
        if self.use_decoupled_dmd:
            video_grad = (
                video_pred_fake_cond
                - video_pred_real_cond
                - (self.video_real_guidance_scale - 1) * (video_pred_real_cond_ca - video_pred_real_uncond_ca)
            )
            audio_grad = (
                audio_pred_fake_cond
                - audio_pred_real_cond
                - (self.audio_real_guidance_scale - 1) * (audio_pred_real_cond_ca - audio_pred_real_uncond_ca)
            )
        else:
            video_grad = video_pred_fake - video_pred_real
            audio_grad = audio_pred_fake - audio_pred_real

        # TODO: Change the normalizer for causal teacher
        if normalization:
            # Step 4: Gradient normalization (DMD paper eq. 8).
            video_p_real = estimated_clean_image_or_video - video_pred_real
            audio_p_real = estimated_clean_audio - audio_pred_real

            video_normalizer = torch.abs(video_p_real).mean(dim=[1, 2, 3, 4], keepdim=True)
            audio_normalizer = torch.abs(audio_p_real).mean(dim=[1, 2], keepdim=True)

            video_grad = video_grad / video_normalizer
            audio_grad = audio_grad / audio_normalizer

        video_grad = torch.nan_to_num(video_grad)
        audio_grad = torch.nan_to_num(audio_grad)

        return (
            video_grad,
            audio_grad,
            {
                "video_dmdtrain_gradient_norm": torch.mean(torch.abs(video_grad)).detach(),
                "audio_dmdtrain_gradient_norm": torch.mean(torch.abs(audio_grad)).detach(),
                "video_timestep": video_timestep.detach(),
                "audio_timestep": audio_timestep.detach(),
            },
        )

    def compute_distribution_matching_loss(
        self,
        image_or_video: torch.Tensor,
        audio: torch.Tensor,
        conditional_dict: dict,
        unconditional_dict: dict,
        video_decoded: Optional[torch.Tensor] = None,
        audio_decoded: Optional[torch.Tensor] = None,
        text_prompts: Optional[list] = None,
        gradient_mask: Optional[torch.Tensor] = None,
        denoised_timestep_from: int = 0,
        denoised_timestep_to: int = 0,
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute the DMD loss (eq 7 in https://arxiv.org/abs/2311.18828).
        Input:
            - image_or_video: a tensor with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - gradient_mask: a boolean tensor with the same shape as image_or_video indicating which pixels to compute loss .
        Output:
            - dmd_loss: a scalar tensor representing the DMD loss.
            - dmd_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        video_original_latent = image_or_video
        audio_original_latent = audio

        batch_size, video_num_frame = image_or_video.shape[:2]
        batch_size, audio_num_frame = audio.shape[:2]

        beta = self.reward_beta if beta is None else beta
        video_reward_scale = 1.0
        audio_reward_scale = 1.0
        reward_log_dict = {}

        if self.enable_rl_reward:
            if video_decoded is None or audio_decoded is None or not text_prompts:
                raise ValueError("RL reward is enabled, but decoded outputs or text prompts are missing.")
            if len(text_prompts) != 1:
                raise ValueError("RL reward currently only supports batch_size=1.")

            reward_outputs = self.reward_evaluator.evaluate(
                video_decoded=video_decoded, audio_decoded=audio_decoded, prompt=text_prompts[0]
            )
            sync_raw = sync_normalized = None
            audio_score_raw = audio_score_normalized = None
            ta_reward = vq_reward = mq_reward = None

            if "sync" in self.reward_types:
                if reward_outputs.sync is None:
                    raise RuntimeError("Sync reward is enabled, but the evaluator did not return sync output.")
                sync_raw, sync_normalized = normalize_scalar_across_ranks(reward_outputs.sync, image_or_video.device)

            if "audiobox" in self.reward_types:
                if reward_outputs.audio_score is None:
                    raise RuntimeError("Audiobox reward is enabled, but the evaluator did not return audio_score.")
                audio_score_raw, audio_score_normalized = normalize_scalar_across_ranks(
                    reward_outputs.audio_score, image_or_video.device
                )

            if "videoalign" in self.reward_types:
                if (
                    reward_outputs.ta_reward is None
                    or reward_outputs.vq_reward is None
                    or reward_outputs.mq_reward is None
                ):
                    raise RuntimeError("VideoAlign reward is enabled, but the evaluator did not return full outputs.")
                ta_reward = reward_outputs.ta_reward
                vq_reward = reward_outputs.vq_reward
                mq_reward = reward_outputs.mq_reward

            if "videoalign" in self.reward_types:
                video_reward_scale = video_reward_scale * (
                    torch.exp(beta * ta_reward) * torch.exp(beta * vq_reward) * torch.exp(beta * mq_reward)
                )
            if "audiobox" in self.reward_types:
                audio_reward_scale = audio_reward_scale * torch.exp(beta * audio_score_normalized)
            if "sync" in self.reward_types:
                video_reward_scale = video_reward_scale * torch.exp(beta * sync_normalized)
                audio_reward_scale = audio_reward_scale * torch.exp(beta * sync_normalized)

            reward_log_dict = {}
            if sync_raw is not None and sync_normalized is not None:
                reward_log_dict["reward_sync_normalized"] = sync_normalized.detach()
                reward_log_dict["reward_sync_raw"] = sync_raw.detach()
            if audio_score_raw is not None and audio_score_normalized is not None:
                reward_log_dict["reward_audio_score_normalized"] = audio_score_normalized.detach()
                reward_log_dict["reward_audio_score_raw"] = audio_score_raw.detach()
            if ta_reward is not None and vq_reward is not None and mq_reward is not None:
                reward_log_dict["reward_ta"] = ta_reward.detach()
                reward_log_dict["reward_vq"] = vq_reward.detach()
                reward_log_dict["reward_mq"] = mq_reward.detach()

            if not isinstance(video_reward_scale, float):
                video_reward_scale = video_reward_scale.detach().float().cpu().item()

            if not isinstance(audio_reward_scale, float):
                audio_reward_scale = audio_reward_scale.detach().float().cpu().item()

        with torch.no_grad():
            # Step 1: Randomly sample timestep based on the given schedule and corresponding noise
            min_timestep = (
                denoised_timestep_to
                if self.ts_schedule and denoised_timestep_to is not None
                else self.min_score_timestep
            )
            max_timestep = (
                denoised_timestep_from
                if self.ts_schedule_max and denoised_timestep_from is not None
                else self.num_train_timestep
            )
            video_timestep = self._get_timestep(
                min_timestep,
                max_timestep,
                batch_size,
                video_num_frame,
                self.video_num_frame_per_block,
                uniform_timestep=True,
            )

            audio_timestep = video_timestep[:, 0:1].repeat(1, audio_num_frame)
            if self.use_decoupled_dmd:
                video_timestep_ca = self._get_timestep(
                    min_timestep,
                    denoised_timestep_from,
                    batch_size,
                    video_num_frame,
                    self.video_num_frame_per_block,
                    uniform_timestep=True,
                )
                audio_timestep_ca = video_timestep_ca[:, 0:1].repeat(1, audio_num_frame)
            else:
                video_timestep_ca = None
                audio_timestep_ca = None

            # audio_timestep = self._get_timestep(
            #     min_timestep,
            #     max_timestep,
            #     batch_size,
            #     audio_num_frame,
            #     self.audio_num_frame_per_block,
            #     uniform_timestep=True,
            # )

            # TODO:should we change it to `timestep = self.scheduler.timesteps[timestep]`?
            if self.timestep_shift > 1:
                video_timestep = (
                    self.timestep_shift
                    * (video_timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (video_timestep / 1000))
                    * 1000
                )
                audio_timestep = (
                    self.timestep_shift
                    * (audio_timestep / 1000)
                    / (1 + (self.timestep_shift - 1) * (audio_timestep / 1000))
                    * 1000
                )
                if self.use_decoupled_dmd:
                    video_timestep_ca = (
                        self.timestep_shift
                        * (video_timestep_ca / 1000)
                        / (1 + (self.timestep_shift - 1) * (video_timestep_ca / 1000))
                        * 1000
                    )
                    audio_timestep_ca = (
                        self.timestep_shift
                        * (audio_timestep_ca / 1000)
                        / (1 + (self.timestep_shift - 1) * (audio_timestep_ca / 1000))
                        * 1000
                    )
                else:
                    video_timestep_ca = None
                    audio_timestep_ca = None

            video_timestep = video_timestep.clamp(self.min_step, self.max_step)
            audio_timestep = audio_timestep.clamp(self.min_step, self.max_step)

            if self.use_decoupled_dmd:
                video_timestep_ca = video_timestep_ca.clamp(self.min_step, self.max_step)
                audio_timestep_ca = audio_timestep_ca.clamp(self.min_step, self.max_step)

            video_noise = torch.randn_like(image_or_video)
            audio_noise = torch.randn_like(audio)

            video_noisy_latent = (
                self.scheduler.add_noise(
                    image_or_video.flatten(0, 1), video_noise.flatten(0, 1), video_timestep.flatten(0, 1)
                )
                .detach()
                .unflatten(0, (batch_size, video_num_frame))
            )

            audio_noisy_latent = (
                self.scheduler.add_noise(audio.flatten(0, 1), audio_noise.flatten(0, 1), audio_timestep.flatten(0, 1))
                .detach()
                .unflatten(0, (batch_size, audio_num_frame))
            )

            if self.use_decoupled_dmd:
                video_noisy_latent_ca = (
                    self.scheduler.add_noise(
                        image_or_video.flatten(0, 1), video_noise.flatten(0, 1), video_timestep_ca.flatten(0, 1)
                    )
                    .detach()
                    .unflatten(0, (batch_size, video_num_frame))
                )
                audio_noisy_latent_ca = (
                    self.scheduler.add_noise(
                        audio.flatten(0, 1), audio_noise.flatten(0, 1), audio_timestep_ca.flatten(0, 1)
                    )
                    .detach()
                    .unflatten(0, (batch_size, audio_num_frame))
                )
            else:
                video_noisy_latent_ca = None
                audio_noisy_latent_ca = None

            # Step 2: Compute the KL grad
            video_grad, audio_grad, dmd_log_dict = self._compute_kl_grad(
                noisy_image_or_video=video_noisy_latent,
                noisy_audio=audio_noisy_latent,
                noisy_video_ca=video_noisy_latent_ca,
                noisy_audio_ca=audio_noisy_latent_ca,
                estimated_clean_image_or_video=video_original_latent,
                estimated_clean_audio=audio_original_latent,
                video_timestep=video_timestep,
                audio_timestep=audio_timestep,
                video_timestep_ca=video_timestep_ca,
                audio_timestep_ca=audio_timestep_ca,
                conditional_dict=conditional_dict,
                unconditional_dict=unconditional_dict,
            )

        if gradient_mask is not None:
            video_dmd_loss = (
                0.5
                * video_reward_scale
                * F.mse_loss(
                    video_original_latent.double()[gradient_mask],
                    (video_original_latent.double() - video_grad.double()).detach()[gradient_mask],
                    reduction="mean",
                )
            )
            audio_dmd_loss = (
                0.5
                * audio_reward_scale
                * F.mse_loss(
                    audio_original_latent.double()[gradient_mask],
                    (audio_original_latent.double() - audio_grad.double()).detach()[gradient_mask],
                    reduction="mean",
                )
            )
        else:
            video_dmd_loss = (
                0.5
                * video_reward_scale
                * F.mse_loss(
                    video_original_latent.double(),
                    (video_original_latent.double() - video_grad.double()).detach(),
                    reduction="mean",
                )
            )
            audio_dmd_loss = (
                0.5
                * audio_reward_scale
                * F.mse_loss(
                    audio_original_latent.double(),
                    (audio_original_latent.double() - audio_grad.double()).detach(),
                    reduction="mean",
                )
            )

        dmd_loss = self.video_loss_weight * video_dmd_loss + self.audio_loss_weight * audio_dmd_loss
        dmd_log_dict.update(reward_log_dict)
        return dmd_loss, dmd_log_dict

    def generator_loss(
        self,
        image_or_video_shape,
        audio_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        text_prompts: Optional[list] = None,
        initial_latent: torch.Tensor = None,
        beta: Optional[float] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and compute the DMD loss.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - generator_log_dict: a dictionary containing the intermediate tensors for logging.
        """
        # Step 1: Unroll generator to obtain fake videos
        (
            video_pred,
            audio_pred,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
            video_decoded,
            audio_decoded,
        ) = self._run_generator(
            image_or_video_shape=image_or_video_shape,
            audio_shape=audio_shape,
            conditional_dict=conditional_dict,
            initial_latent=initial_latent,
        )

        # Step 2: Compute the DMD loss
        dmd_loss, dmd_log_dict = self.compute_distribution_matching_loss(
            image_or_video=video_pred,
            audio=audio_pred,
            video_decoded=video_decoded,
            audio_decoded=audio_decoded,
            text_prompts=text_prompts,
            conditional_dict=conditional_dict,
            unconditional_dict=unconditional_dict,
            gradient_mask=gradient_mask,
            denoised_timestep_from=denoised_timestep_from,
            denoised_timestep_to=denoised_timestep_to,
            beta=beta,
        )

        dmd_log_dict.update({"video_pred": video_pred.detach(), "audio_pred": audio_pred.detach()})

        return dmd_loss, dmd_log_dict

    def critic_loss(
        self,
        image_or_video_shape,
        audio_shape,
        conditional_dict: dict,
        unconditional_dict: dict,
        clean_latent: torch.Tensor,
        initial_latent: torch.Tensor = None,
    ) -> Tuple[torch.Tensor, dict]:
        """
        Generate image/videos from noise and train the critic with generated samples.
        The noisy input to the generator is backward simulated.
        This removes the need of any datasets during distillation.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
        Output:
            - loss: a scalar tensor representing the generator loss.
            - critic_log_dict: a dictionary containing the intermediate tensors for logging.
        """

        # Step 1: Run generator on backward simulated noisy input
        with torch.no_grad():
            video_pred, audio_pred, _, denoised_timestep_from, denoised_timestep_to, _, _ = self._run_generator(
                image_or_video_shape=image_or_video_shape,
                audio_shape=audio_shape,
                conditional_dict=conditional_dict,
                initial_latent=initial_latent,
            )

        # Step 2: Compute the fake prediction
        min_timestep = (
            denoised_timestep_to if self.ts_schedule and denoised_timestep_to is not None else self.min_score_timestep
        )
        max_timestep = (
            denoised_timestep_from
            if self.ts_schedule_max and denoised_timestep_from is not None
            else self.num_train_timestep
        )

        video_critic_timestep = self._get_timestep(
            min_timestep,
            max_timestep,
            image_or_video_shape[0],
            image_or_video_shape[1],
            self.video_num_frame_per_block,
            uniform_timestep=True,
        )

        audio_critic_timestep = video_critic_timestep[:, 0:1].repeat(1, audio_shape[1])

        # audio_critic_timestep = self._get_timestep(
        #     min_timestep,
        #     max_timestep,
        #     audio_shape[0],
        #     audio_shape[1],
        #     self.audio_num_frame_per_block,
        #     uniform_timestep=True,
        # )

        if self.timestep_shift > 1:
            video_critic_timestep = (
                self.timestep_shift
                * (video_critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (video_critic_timestep / 1000))
                * 1000
            )
            audio_critic_timestep = (
                self.timestep_shift
                * (audio_critic_timestep / 1000)
                / (1 + (self.timestep_shift - 1) * (audio_critic_timestep / 1000))
                * 1000
            )

        video_critic_timestep = video_critic_timestep.clamp(self.min_step, self.max_step)
        audio_critic_timestep = audio_critic_timestep.clamp(self.min_step, self.max_step)

        video_critic_noise = torch.randn_like(video_pred)
        audio_critic_noise = torch.randn_like(audio_pred)

        video_noisy_latent = self.scheduler.add_noise(
            video_pred.flatten(0, 1), video_critic_noise.flatten(0, 1), video_critic_timestep.flatten(0, 1)
        ).unflatten(0, image_or_video_shape[:2])

        audio_noisy_latent = self.scheduler.add_noise(
            audio_pred.flatten(0, 1), audio_critic_noise.flatten(0, 1), audio_critic_timestep.flatten(0, 1)
        ).unflatten(0, audio_shape[:2])

        _, _, video_pred_fake, audio_pred_fake = self.fake_score(
            noisy_video=video_noisy_latent,
            noisy_audio=audio_noisy_latent,
            conditional_dict=conditional_dict,
            video_timestep=video_critic_timestep,
            audio_timestep=audio_critic_timestep,
        )

        # Step 3: Compute the denoising loss for the fake critic
        if self.config.denoising_loss_type == "flow":
            from hallolive.utils.fusion_wrapper import FusionDiffusionWrapper

            video_flow_pred_fake = FusionDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=video_pred_fake.flatten(0, 1),
                xt=video_noisy_latent.flatten(0, 1),
                timestep=video_critic_timestep.flatten(0, 1),
            )
            audio_flow_pred_fake = FusionDiffusionWrapper._convert_x0_to_flow_pred(
                scheduler=self.scheduler,
                x0_pred=audio_pred_fake.flatten(0, 1),
                xt=audio_noisy_latent.flatten(0, 1),
                timestep=audio_critic_timestep.flatten(0, 1),
            )

            video_noise_pred_fake = None
            audio_noise_pred_fake = None
        else:
            video_flow_pred_fake = None
            audio_flow_pred_fake = None

            video_noise_pred_fake = self.scheduler.convert_x0_to_noise(
                x0=video_pred_fake.flatten(0, 1),
                xt=video_noisy_latent.flatten(0, 1),
                timestep=video_critic_timestep.flatten(0, 1),
            ).unflatten(0, image_or_video_shape[:2])

            audio_noise_pred_fake = self.scheduler.convert_x0_to_noise(
                x0=audio_pred_fake.flatten(0, 1),
                xt=audio_noisy_latent.flatten(0, 1),
                timestep=audio_critic_timestep.flatten(0, 1),
            ).unflatten(0, audio_shape[:2])

        video_denoising_loss = self.denoising_loss_func(
            x=video_pred.flatten(0, 1),
            x_pred=video_pred_fake.flatten(0, 1),
            noise=video_critic_noise.flatten(0, 1),
            noise_pred=video_noise_pred_fake,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=video_critic_timestep.flatten(0, 1),
            flow_pred=video_flow_pred_fake,
        )

        audio_denoising_loss = self.denoising_loss_func(
            x=audio_pred.flatten(0, 1),
            x_pred=audio_pred_fake.flatten(0, 1),
            noise=audio_critic_noise.flatten(0, 1),
            noise_pred=audio_noise_pred_fake,
            alphas_cumprod=self.scheduler.alphas_cumprod,
            timestep=audio_critic_timestep.flatten(0, 1),
            flow_pred=audio_flow_pred_fake,
        )

        denoising_loss = self.video_loss_weight * video_denoising_loss + self.audio_loss_weight * audio_denoising_loss

        # Step 5: Debugging Log
        critic_log_dict = {
            "video_critic_timestep": video_critic_timestep.detach(),
            "audio_critic_timestep": audio_critic_timestep.detach(),
        }

        return denoising_loss, critic_log_dict
