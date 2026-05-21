from typing import Tuple
from einops import rearrange
from torch import nn
import torch.distributed as dist
import torch

from hallolive.pipeline import DualStreamSelfForcingPipeline
from hallolive.utils.loss import get_denoising_loss


class BaseModel(nn.Module):
    def __init__(self, config, device):
        super().__init__()
        self._initialize_models(config, device)

        self.device = device
        self.config = config
        self.dtype = torch.bfloat16 if config.mixed_precision else torch.float32
        if hasattr(config, "denoising_step_list"):
            self.denoising_step_list = torch.tensor(config.denoising_step_list, dtype=torch.long)
            if config.warp_denoising_step:
                timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
                self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

    def _initialize_models(self, config, device):
        raise NotImplementedError("Subclasses must implement the _initialize_models() method!")

    def _get_timestep(
        self,
        min_timestep: int,
        max_timestep: int,
        batch_size: int,
        num_frame: int,
        num_frame_per_block: int,
        uniform_timestep: bool = False,
    ) -> torch.Tensor:
        """
        Randomly generate a timestep tensor based on the generator's task type. It uniformly samples a timestep
        from the range [min_timestep, max_timestep], and returns a tensor of shape [batch_size, num_frame].
        - If uniform_timestep, it will use the same timestep for all frames.
        - If not uniform_timestep, it will use a different timestep for each block.
        """
        if uniform_timestep:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, 1], device=self.device, dtype=torch.long
            ).repeat(1, num_frame)
            return timestep
        else:
            timestep = torch.randint(
                min_timestep, max_timestep, [batch_size, num_frame], device=self.device, dtype=torch.long
            )
            # make the noise level the same within every block
            if self.independent_first_frame:
                # the first frame is always kept the same
                timestep_from_second = timestep[:, 1:]
                timestep_from_second = timestep_from_second.reshape(
                    timestep_from_second.shape[0], -1, num_frame_per_block
                )
                timestep_from_second[:, :, 1:] = timestep_from_second[:, :, 0:1]
                timestep_from_second = timestep_from_second.reshape(timestep_from_second.shape[0], -1)
                timestep = torch.cat([timestep[:, 0:1], timestep_from_second], dim=1)
            else:
                timestep = timestep.reshape(timestep.shape[0], -1, num_frame_per_block)
                timestep[:, :, 1:] = timestep[:, :, 0:1]
                timestep = timestep.reshape(timestep.shape[0], -1)
            return timestep


class SelfForcingModel(BaseModel):
    def __init__(self, config, device):
        super().__init__(config, device)
        self.denoising_loss_func = get_denoising_loss(config.denoising_loss_type)()

    def _run_generator(
        self, image_or_video_shape, audio_shape, conditional_dict: dict, initial_latent: torch.tensor = None
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Optionally simulate the generator's input from noise using backward simulation
        and then run the generator for one-step.
        Input:
            - image_or_video_shape: a list containing the shape of the image or video [B, F, C, H, W].
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
            - unconditional_dict: a dictionary containing the unconditional information (e.g. null/negative text embeddings, null/negative image embeddings).
            - clean_latent: a tensor containing the clean latents [B, F, C, H, W]. Need to be passed when no backward simulation is used.
            - initial_latent: a tensor containing the initial latents [B, F, C, H, W].
        Output:
            - pred_image: a tensor with shape [B, F, C, H, W].
            - denoised_timestep: an integer
        """
        # Step 1: Sample noise and backward simulate the generator's input
        assert getattr(self.config, "backward_simulation", True), "Backward simulation needs to be enabled"
        if initial_latent is not None:
            conditional_dict["initial_latent"] = initial_latent
        if self.config.i2v:
            video_noise_shape = [image_or_video_shape[0], image_or_video_shape[1] - 1, *image_or_video_shape[2:]]
        else:
            video_noise_shape = image_or_video_shape.copy()
            audio_noise_shape = audio_shape.copy()

        # During training, the number of generated frames should be uniformly sampled from
        # [21, self.num_training_frames], but still being a multiple of self.num_frame_per_block
        min_num_frames = 29 if self.config.independent_first_frame else 30
        max_num_frames = (
            self.num_training_frames - 1 if self.config.independent_first_frame else self.video_num_training_frames
        )
        assert max_num_frames % self.video_num_frame_per_block == 0
        assert min_num_frames % self.video_num_frame_per_block == 0
        max_num_blocks = max_num_frames // self.video_num_frame_per_block
        min_num_blocks = min_num_frames // self.video_num_frame_per_block
        num_generated_blocks = torch.randint(min_num_blocks, max_num_blocks + 1, (1,), device=self.device)
        dist.broadcast(num_generated_blocks, src=0)
        num_generated_blocks = num_generated_blocks.item()
        num_generated_frames = num_generated_blocks * self.video_num_frame_per_block
        if self.config.independent_first_frame and initial_latent is None:
            num_generated_frames += 1
            min_num_frames += 1
        # Sync num_generated_frames across all processes
        video_noise_shape[1] = num_generated_frames

        video_pred, audio_pred, denoised_timestep_from, denoised_timestep_to = self._consistency_backward_simulation(
            video_noise=torch.randn(video_noise_shape, device=self.device, dtype=self.dtype),
            audio_noise=torch.randn(audio_noise_shape, device=self.device, dtype=self.dtype),
            **conditional_dict,
        )
        # Slice last 30 frames
        if video_pred.shape[1] > 30:
            with torch.no_grad():
                # Reencode to get image latent
                latent_to_decode = video_pred[:, :-29, ...]
                # Deccode to video
                pixels = self.vae.decode_to_pixel(latent_to_decode)
                frame = pixels[:, -1:, ...].to(self.dtype)
                frame = rearrange(frame, "b t c h w -> b c t h w")
                # Encode frame to get image latent
                image_latent = self.vae.encode_to_latent(frame).to(self.dtype)
            video_pred_last_30 = torch.cat([image_latent, video_pred[:, -29:, ...]], dim=1)
        else:
            video_pred_last_30 = video_pred

        if num_generated_frames != min_num_frames:
            # Currently, we do not use gradient for the first chunk, since it contains image latents
            gradient_mask = torch.ones_like(video_pred_last_30, dtype=torch.bool)
            if self.config.independent_first_frame:
                gradient_mask[:, :1] = False
            else:
                gradient_mask[:, : self.video_num_frame_per_block] = False
        else:
            gradient_mask = None

        video_pred_last_30 = video_pred_last_30.to(self.dtype)

        with torch.no_grad():
            video = self.vae_video.wrapped_decode(
                rearrange(video_pred_last_30, "b f c h w -> b c f h w")
            )  # [B, C, F, H, W]
            audio = self.vae_audio.wrapped_decode(rearrange(audio_pred, "b f c -> b c f"))

        return (
            video_pred_last_30,
            audio_pred,
            gradient_mask,
            denoised_timestep_from,
            denoised_timestep_to,
            video,
            audio,
        )

    def _consistency_backward_simulation(
        self, video_noise: torch.Tensor, audio_noise: torch.Tensor, **conditional_dict: dict
    ) -> torch.Tensor:
        """
        Simulate the generator's input from noise to avoid training/inference mismatch.
        See Sec 4.5 of the DMD2 paper (https://arxiv.org/abs/2405.14867) for details.
        Here we use the consistency sampler (https://arxiv.org/abs/2303.01469)
        Input:
            - noise: a tensor sampled from N(0, 1) with shape [B, F, C, H, W] where the number of frame is 1 for images.
            - conditional_dict: a dictionary containing the conditional information (e.g. text embeddings, image embeddings).
        Output:
            - output: a tensor with shape [B, T, F, C, H, W].
            T is the total number of timesteps. output[0] is a pure noise and output[i] and i>0
            represents the x0 prediction at each timestep.
        """
        if self.inference_pipeline is None:
            self._initialize_inference_pipeline()

        return self.inference_pipeline.inference_with_trajectory(
            video_noise=video_noise, audio_noise=audio_noise, **conditional_dict
        )

    def _initialize_inference_pipeline(self):
        """
        Lazy initialize the inference pipeline during the first backward simulation run.
        Here we encapsulate the inference code with a model-dependent outside function.
        We pass our FSDP-wrapped modules into the pipeline to save memory.
        """
        self.inference_pipeline = DualStreamSelfForcingPipeline(
            denoising_step_list=self.denoising_step_list,
            scheduler=self.scheduler,
            generator=self.generator,
            video_num_frame_per_block=self.video_num_frame_per_block,
            audio_num_frame_per_block=self.audio_num_frame_per_block,
            future_audio_frames=self.future_audio_frames,
            independent_first_frame=self.config.independent_first_frame,
            same_step_across_blocks=self.config.same_step_across_blocks,
            last_step_only=self.config.last_step_only,
            video_num_max_frames=self.video_num_training_frames,
            audio_num_max_frames=self.audio_num_training_frames,
            context_noise=self.config.context_noise,
        )
