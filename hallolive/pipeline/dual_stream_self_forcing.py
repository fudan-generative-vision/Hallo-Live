from hallolive.utils.fusion_wrapper import FusionDiffusionWrapper
from hallolive.utils.scheduler import SchedulerInterface
from typing import List, Optional
import torch
import torch.distributed as dist


class DualStreamSelfForcingPipeline:
    def __init__(
        self,
        denoising_step_list: List[int],
        scheduler: SchedulerInterface,
        generator: FusionDiffusionWrapper,
        video_num_frame_per_block=3,
        audio_num_frame_per_block=15,
        future_audio_frames: Optional[int] = None,
        independent_first_frame: bool = False,
        same_step_across_blocks: bool = False,
        last_step_only: bool = False,
        video_num_max_frames: int = 30,
        audio_num_max_frames: int = 150,
        context_noise: int = 0,
    ):
        super().__init__()
        self.scheduler = scheduler
        self.generator = generator
        self.denoising_step_list = denoising_step_list
        if self.denoising_step_list[-1] == 0:
            self.denoising_step_list = self.denoising_step_list[:-1]  # remove the zero timestep for inference

        # Wan specific hyperparameters
        self.num_transformer_blocks = 30
        self.video_frame_seq_length = 496
        self.audio_frame_seq_length = 1
        self.video_num_frame_per_block = video_num_frame_per_block
        self.audio_num_frame_per_block = audio_num_frame_per_block
        self.future_audio_frames = int(future_audio_frames)
        if self.future_audio_frames < 0:
            raise ValueError("future_audio_frames must be non-negative")
        self.context_noise = context_noise
        self.i2v = False

        self.kv_cache = None

        self.video_kv_cache_num_heads = generator.video_config["num_heads"]  # 12 for 1.3B, 24 for 5B
        self.video_kv_cache_dim_head = generator.video_config["dim"] // generator.video_config["num_heads"]  # 128

        self.audio_kv_cache_num_heads = generator.audio_config["num_heads"]  # 12 for 1.3B, 24 for 5B
        self.audio_kv_cache_dim_head = generator.audio_config["dim"] // generator.audio_config["num_heads"]  # 128

        self.video_kv_cache_size = video_num_max_frames * self.video_frame_seq_length
        self.audio_kv_cache_size = (
            audio_num_max_frames * self.audio_frame_seq_length + self.future_audio_frames * self.audio_frame_seq_length
        )

        self.independent_first_frame = independent_first_frame
        self.same_step_across_blocks = same_step_across_blocks
        self.last_step_only = last_step_only

    @staticmethod
    def _pad_audio_window_to_length(
        audio_window: torch.Tensor, pad_source: torch.Tensor, target_num_frames: int
    ) -> torch.Tensor:
        while audio_window.shape[1] < target_num_frames:
            remaining_num_frames = target_num_frames - audio_window.shape[1]
            audio_window = torch.cat(
                [audio_window, pad_source[:, : min(remaining_num_frames, pad_source.shape[1])]], dim=1
            )
        return audio_window[:, :target_num_frames]

    def generate_and_sync_list(self, num_blocks, num_denoising_steps, device):
        rank = dist.get_rank() if dist.is_initialized() else 0

        if rank == 0:
            # Generate random indices
            indices = torch.randint(low=0, high=num_denoising_steps, size=(num_blocks,), device=device)
            if self.last_step_only:
                indices = torch.ones_like(indices) * (num_denoising_steps - 1)
        else:
            indices = torch.empty(num_blocks, dtype=torch.long, device=device)

        dist.broadcast(indices, src=0)  # Broadcast the random indices to all ranks
        return indices.tolist()

    def inference_with_trajectory(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        initial_latent: Optional[torch.Tensor] = None,
        return_sim_step: bool = False,
        **conditional_dict,
    ) -> torch.Tensor:
        batch_size, video_num_frames, video_num_channels, height, width = video_noise.shape
        batch_size, audio_num_frames, audio_num_channels = audio_noise.shape
        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert video_num_frames % self.video_num_frame_per_block == 0
            video_num_blocks = video_num_frames // self.video_num_frame_per_block
            audio_num_blocks = audio_num_frames // self.audio_num_frame_per_block
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (video_num_frames - 1) % self.video_num_frame_per_block == 0
            video_num_blocks = (video_num_frames - 1) // self.video_num_frame_per_block
            audio_num_blocks = audio_num_frames // self.audio_num_frame_per_block
        video_num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        video_num_output_frames = video_num_frames + video_num_input_frames  # add the initial latent
        audio_num_output_frames = audio_num_frames  # add the initial latent frames
        video_output = torch.zeros(
            [batch_size, video_num_output_frames, video_num_channels, height, width],
            device=video_noise.device,
            dtype=video_noise.dtype,
        )
        audio_output = torch.zeros(
            [batch_size, audio_num_output_frames, audio_num_channels],
            device=audio_noise.device,
            dtype=audio_noise.dtype,
        )

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache_fusion(batch_size=batch_size, dtype=video_noise.dtype, device=video_noise.device)
        self._initialize_crossattn_cache_fusion(
            batch_size=batch_size, dtype=video_noise.dtype, device=video_noise.device
        )

        # Step 2: Cache context feature
        video_current_start_frame = 0
        audio_current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=video_noise.device, dtype=torch.int64) * 0
            # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
            video_output[:, :1] = initial_latent
            with torch.no_grad():
                self.generator(
                    noisy_image_or_video=initial_latent,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=video_current_start_frame * self.frame_seq_length,
                )
            video_current_start_frame += 1

        # Step 3: Temporal denoising loop
        video_all_num_frames = [self.video_num_frame_per_block] * video_num_blocks
        audio_all_num_frames = [self.audio_num_frame_per_block] * audio_num_blocks
        if self.independent_first_frame and initial_latent is None:
            video_all_num_frames = [1] + video_all_num_frames
        num_denoising_steps = len(self.denoising_step_list)
        exit_flags = self.generate_and_sync_list(
            len(video_all_num_frames), num_denoising_steps, device=video_noise.device
        )
        start_gradient_frame_index = video_num_output_frames - 30

        # for block_index in range(num_blocks):
        for block_index, video_current_num_frames in enumerate(video_all_num_frames):
            audio_current_num_frames = audio_all_num_frames[block_index]
            audio_current_end_frame = audio_current_start_frame + audio_current_num_frames
            future_audio_end_frame = audio_current_end_frame + self.future_audio_frames
            audio_current_window_num_frames = audio_current_num_frames + self.future_audio_frames
            video_noisy_input = video_noise[
                :,
                video_current_start_frame - video_num_input_frames : video_current_start_frame
                + video_current_num_frames
                - video_num_input_frames,
            ]
            audio_history_input = audio_output[:, :audio_current_start_frame]
            audio_current_noisy_input = audio_noise[:, audio_current_start_frame:future_audio_end_frame]
            single_audio_noisy_input = audio_noise[
                :, audio_current_start_frame : audio_current_start_frame + audio_current_num_frames
            ]
            audio_current_noisy_input = self._pad_audio_window_to_length(
                audio_current_noisy_input, single_audio_noisy_input, audio_current_window_num_frames
            )
            audio_noisy_input = torch.cat([audio_history_input, audio_current_noisy_input], dim=1)

            # Step 3.1: Spatial denoising loop
            for index, current_timestep in enumerate(self.denoising_step_list):
                if self.same_step_across_blocks:
                    exit_flag = index == exit_flags[0]
                else:
                    exit_flag = (
                        index == exit_flags[block_index]
                    )  # Only backprop at the randomly selected timestep (consistent across all ranks)
                video_timestep = (
                    torch.ones([batch_size, video_current_num_frames], device=video_noise.device, dtype=torch.int64)
                    * current_timestep
                )
                audio_history_timestep = (
                    torch.ones([batch_size, audio_current_start_frame], device=audio_noise.device, dtype=torch.int64)
                    * self.context_noise
                )
                audio_current_timestep = (
                    torch.ones(
                        [batch_size, audio_current_window_num_frames], device=audio_noise.device, dtype=torch.int64
                    )
                    * current_timestep
                )
                audio_timestep = torch.cat([audio_history_timestep, audio_current_timestep], dim=1)

                if not exit_flag:
                    with torch.no_grad():
                        video_flow_pred, audio_flow_pred, video_pred, audio_pred = self.generator(
                            noisy_video=video_noisy_input,
                            noisy_audio=audio_noisy_input,
                            conditional_dict=conditional_dict,
                            video_timestep=video_timestep,
                            audio_timestep=audio_timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            video_current_start=video_current_start_frame * self.video_frame_seq_length,
                            audio_current_start=0,
                        )
                        next_timestep = self.denoising_step_list[index + 1]
                        video_noisy_input = self.scheduler.add_noise(
                            video_pred.flatten(0, 1),
                            torch.randn_like(video_pred.flatten(0, 1)),
                            next_timestep
                            * torch.ones(
                                [batch_size * video_current_num_frames], device=video_noise.device, dtype=torch.long
                            ),
                        ).unflatten(0, video_pred.shape[:2])
                        audio_current_noisy_input = self.scheduler.add_noise(
                            audio_pred[:, audio_current_start_frame:future_audio_end_frame].flatten(0, 1),
                            torch.randn_like(
                                audio_pred[:, audio_current_start_frame:future_audio_end_frame].flatten(0, 1)
                            ),
                            next_timestep
                            * torch.ones(
                                [int(batch_size * audio_current_window_num_frames)],
                                device=audio_noise.device,
                                dtype=torch.long,
                            ),
                        ).unflatten(0, audio_pred[:, audio_current_start_frame:future_audio_end_frame].shape[:2])
                        audio_noisy_input = torch.cat([audio_history_input, audio_current_noisy_input], dim=1)
                else:
                    # for getting real output
                    # with torch.set_grad_enabled(current_start_frame >= start_gradient_frame_index):
                    if video_current_start_frame < start_gradient_frame_index:
                        with torch.no_grad():
                            video_flow_pred, audio_flow_pred, video_pred, audio_pred = self.generator(
                                noisy_video=video_noisy_input,
                                noisy_audio=audio_noisy_input,
                                conditional_dict=conditional_dict,
                                video_timestep=video_timestep,
                                audio_timestep=audio_timestep,
                                kv_cache=self.kv_cache,
                                crossattn_cache=self.crossattn_cache,
                                video_current_start=video_current_start_frame * self.video_frame_seq_length,
                                audio_current_start=0,
                            )
                    else:
                        video_flow_pred, audio_flow_pred, video_pred, audio_pred = self.generator(
                            noisy_video=video_noisy_input,
                            noisy_audio=audio_noisy_input,
                            conditional_dict=conditional_dict,
                            video_timestep=video_timestep,
                            audio_timestep=audio_timestep,
                            kv_cache=self.kv_cache,
                            crossattn_cache=self.crossattn_cache,
                            video_current_start=video_current_start_frame * self.video_frame_seq_length,
                            audio_current_start=0,
                        )
                    break

            # Step 3.2: record the model's output
            video_output[:, video_current_start_frame : video_current_start_frame + video_current_num_frames] = (
                video_pred
            )
            audio_output[:, audio_current_start_frame:audio_current_end_frame] = audio_pred[
                :, audio_current_start_frame:audio_current_end_frame
            ]

            # Step 3.3: rerun with clean context to update the cache
            video_context_timestep = torch.ones_like(video_timestep) * self.context_noise
            clean_audio_input = audio_pred[:, :future_audio_end_frame]
            audio_context_timestep = (
                torch.ones([batch_size, future_audio_end_frame], device=audio_noise.device, dtype=torch.int64)
                * self.context_noise
            )

            with torch.no_grad():
                self.generator(
                    noisy_video=video_pred,
                    noisy_audio=clean_audio_input,
                    conditional_dict=conditional_dict,
                    video_timestep=video_context_timestep,
                    audio_timestep=audio_context_timestep,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    video_current_start=video_current_start_frame * self.video_frame_seq_length,
                    audio_current_start=0,
                )

            # Step 3.4: update the start and end frame indices
            video_current_start_frame += video_current_num_frames
            audio_current_start_frame = audio_current_end_frame

        # Step 3.5: Return the denoised timestep
        if not self.same_step_across_blocks:
            denoised_timestep_from, denoised_timestep_to = None, None
        elif exit_flags[0] == len(self.denoising_step_list) - 1:
            denoised_timestep_to = 0
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
                ).item()
            )
        else:
            denoised_timestep_to = (
                1000
                - torch.argmin(
                    (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0] + 1].cuda()).abs(), dim=0
                ).item()
            )
            denoised_timestep_from = (
                1000
                - torch.argmin(
                    (self.scheduler.timesteps.cuda() - self.denoising_step_list[exit_flags[0]].cuda()).abs(), dim=0
                ).item()
            )

        if return_sim_step:
            return (video_output, audio_output, denoised_timestep_from, denoised_timestep_to, exit_flags[0] + 1)

        return video_output, audio_output, denoised_timestep_from, denoised_timestep_to

    def _initialize_crossattn_cache_fusion(self, batch_size, dtype, device):
        crossattn_cache = []
        crossattn_cache_classes = ["video", "audio"]

        for _ in range(self.num_transformer_blocks):
            block_crossattn_cache = {}
            for crossattn_cache_class in crossattn_cache_classes:
                if "video" in crossattn_cache_class:
                    kv_cache_num_heads = self.video_kv_cache_num_heads
                    kv_cache_dim_head = self.video_kv_cache_dim_head
                else:
                    kv_cache_num_heads = self.audio_kv_cache_num_heads
                    kv_cache_dim_head = self.audio_kv_cache_dim_head
                block_crossattn_cache[crossattn_cache_class] = {
                    "k": torch.zeros(
                        [batch_size, 512, kv_cache_num_heads, kv_cache_dim_head], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, 512, kv_cache_num_heads, kv_cache_dim_head], dtype=dtype, device=device
                    ),
                    "is_init": False,
                }
            crossattn_cache.append(block_crossattn_cache)
        self.crossattn_cache = crossattn_cache

    def _initialize_kv_cache_fusion(self, batch_size, dtype, device):
        """
        Initialize a Per-GPU KV cache for the Wan model.
        """
        kv_cache = []

        kv_cache_classes = ["video_self", "audio_self", "video_cross", "audio_cross"]

        for _ in range(self.num_transformer_blocks):
            block_kv_cache = {}
            for kv_cache_class in kv_cache_classes:
                if kv_cache_class == "video_self" or kv_cache_class == "audio_cross":
                    kv_cache_num_heads = self.video_kv_cache_num_heads
                    kv_cache_dim_head = self.video_kv_cache_dim_head
                elif kv_cache_class == "audio_self" or kv_cache_class == "video_cross":
                    kv_cache_num_heads = self.audio_kv_cache_num_heads
                    kv_cache_dim_head = self.audio_kv_cache_dim_head
                if "video" in kv_cache_class:
                    kv_cache_size = self.video_kv_cache_size
                elif "audio" in kv_cache_class:
                    kv_cache_size = self.audio_kv_cache_size
                block_kv_cache[kv_cache_class] = {
                    "k": torch.zeros(
                        [batch_size, kv_cache_size, kv_cache_num_heads, kv_cache_dim_head], dtype=dtype, device=device
                    ),
                    "v": torch.zeros(
                        [batch_size, kv_cache_size, kv_cache_num_heads, kv_cache_dim_head], dtype=dtype, device=device
                    ),
                    "global_end_index": torch.tensor([0], dtype=torch.long, device=device),
                    "local_end_index": torch.tensor([0], dtype=torch.long, device=device),
                }
            kv_cache.append(block_kv_cache)

        self.kv_cache = kv_cache  # always store the clean cache
