import time
from concurrent.futures import ThreadPoolExecutor
from typing import List, Optional

import torch

from hallolive.utils.wan_wrapper import WanTextEncoder
from hallolive.utils.fusion_wrapper import FusionDiffusionWrapper
from hallolive.ovi.utils.model_loading_utils import init_mmaudio_vae, init_wan_vae_2_2
from hallolive.ovi.modules.vae2_2 import unpatchify
from einops import rearrange

from hallolive.utils.memory import gpu, get_cuda_free_memory_gb, move_model_to_device_with_memory_preservation


class _ParallelVAEBlockDecoder:
    def __init__(self, vae_video, vae_audio, device: torch.device):
        self.vae_video = vae_video
        self.vae_audio = vae_audio
        self.device = device
        self.executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="parallel-vae-decode")
        self.futures = []
        self.video_first_chunk = True

        with torch.cuda.device(self.device):
            self.stream = torch.cuda.Stream(device=self.device)
            self.vae_video.model.clear_cache()

    def submit(
        self,
        video_latents: Optional[torch.Tensor],
        audio_latents: Optional[torch.Tensor],
        ready_event: Optional[torch.cuda.Event] = None,
    ):
        future = self.executor.submit(self._decode_block, video_latents, audio_latents, ready_event)
        self.futures.append(future)

    def finish(self):
        video_blocks = []
        audio_blocks = []
        try:
            for future in self.futures:
                video_block, audio_block = future.result()
                if video_block is not None:
                    video_blocks.append(video_block)
                if audio_block is not None:
                    audio_blocks.append(audio_block)
        finally:
            self.executor.shutdown(wait=True)
            with torch.cuda.device(self.device):
                self.vae_video.model.clear_cache()

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream):
            video = torch.cat(video_blocks, dim=2) if video_blocks else None
            audio = torch.cat(audio_blocks, dim=-1) if audio_blocks else None
            self.stream.synchronize()
        return video, audio

    def decode_full_audio(self, audio_latents: torch.Tensor) -> torch.Tensor:
        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.inference_mode():
            audio_latents = audio_latents.to(device=self.device, dtype=self.vae_audio.dtype, non_blocking=True)
            audio = self.vae_audio.wrapped_decode(audio_latents.permute(0, 2, 1).contiguous())
            self.stream.synchronize()
            return audio

    def _decode_block(
        self,
        video_latents: Optional[torch.Tensor],
        audio_latents: Optional[torch.Tensor],
        ready_event: Optional[torch.cuda.Event],
    ):
        if ready_event is not None:
            ready_event.synchronize()

        with torch.cuda.device(self.device), torch.cuda.stream(self.stream), torch.inference_mode():
            video = None
            audio = None

            if video_latents is not None:
                video_latents_on_vae = video_latents.to(
                    device=self.device, dtype=self.vae_video.dtype, non_blocking=True
                )
                video = self._decode_video_block(video_latents_on_vae)

            if audio_latents is not None:
                audio_latents_on_vae = audio_latents.to(
                    device=self.device, dtype=self.vae_audio.dtype, non_blocking=True
                )
                audio = self.vae_audio.wrapped_decode(audio_latents_on_vae.permute(0, 2, 1).contiguous())

            self.stream.synchronize()

            return video, audio

    def _decode_video_block(self, video_latents: torch.Tensor) -> torch.Tensor:
        z = video_latents.permute(0, 2, 1, 3, 4).contiguous()
        model = self.vae_video.model
        scale = self.vae_video.scale

        with torch.amp.autocast("cuda", dtype=self.vae_video.dtype):
            if isinstance(scale[0], torch.Tensor):
                z = z / scale[1].view(1, model.z_dim, 1, 1, 1) + scale[0].view(1, model.z_dim, 1, 1, 1)
            else:
                z = z / scale[1] + scale[0]

            x = model.conv2(z)
            outputs = []
            for frame_idx in range(x.shape[2]):
                model._conv_idx = [0]
                outputs.append(
                    model.decoder(
                        x[:, :, frame_idx : frame_idx + 1],
                        feat_cache=model._feat_map,
                        feat_idx=model._conv_idx,
                        first_chunk=self.video_first_chunk,
                    )
                )
                self.video_first_chunk = False

            return unpatchify(torch.cat(outputs, dim=2), patch_size=2).float().clamp_(-1, 1)


class DualStreamCausalInferencePipeline(torch.nn.Module):
    def __init__(self, config, device, generator=None, text_encoder=None, vae=None):
        super().__init__()
        cpu_device = torch.device("cpu")
        self.generator_meta_init = bool(getattr(config, "generator_meta_init", False))
        video_num_frame_per_block = getattr(config, "video_num_frame_per_block", 1)
        audio_num_frame_per_block = getattr(config, "audio_num_frame_per_block", 5)
        future_audio_frames = getattr(config, "future_audio_frames", None)
        self.future_audio_frames = int(future_audio_frames)
        if self.future_audio_frames < 0:
            raise ValueError("future_audio_frames must be non-negative")
        model_dir = getattr(config, "model_dir", "model")

        # Step 1: Initialize all models
        self.generator = FusionDiffusionWrapper(
            timestep_shift=config.timestep_shift,
            video_config=config.generator_video_config,
            audio_config=config.generator_audio_config,
            is_causal=True,
            enable_cross_attention=config.enable_cross_attention,
            future_audio_frames=self.future_audio_frames,
            meta_init=self.generator_meta_init,
        )
        self.generator.model.requires_grad_(False)

        self.text_encoder = WanTextEncoder(model_dir=model_dir)

        # Keep large VAE checkpoints on CPU during construction and move them once later.
        self.vae_video = init_wan_vae_2_2(model_dir, rank=cpu_device)
        self.vae_video.model.requires_grad_(False).eval()

        self.vae_audio = init_mmaudio_vae(model_dir, rank=cpu_device).eval()

        # Step 2: Initialize all causal hyperparmeters
        self.scheduler = self.generator.get_scheduler()
        self.scheduler.timesteps = self.scheduler.timesteps.to(device)
        self.denoising_step_list = torch.tensor(config.denoising_step_list, dtype=torch.long)
        if config.warp_denoising_step:
            timesteps = torch.cat((self.scheduler.timesteps.cpu(), torch.tensor([0], dtype=torch.float32)))
            self.denoising_step_list = timesteps[1000 - self.denoising_step_list]

        self.num_transformer_blocks = 30
        self.video_frame_seq_length = 496
        self.audio_frame_seq_length = 1

        self.kv_cache = None

        self.config = config

        self.video_num_frame_per_block = video_num_frame_per_block
        self.audio_num_frame_per_block = audio_num_frame_per_block
        self.independent_first_frame = getattr(config, "independent_first_frame", False)

        self.video_kv_cache_num_heads = self.generator.video_config["num_heads"]  # 12 for 1.3B, 24 for 5B
        self.video_kv_cache_dim_head = (
            self.generator.video_config["dim"] // self.generator.video_config["num_heads"]
        )  # 128

        self.audio_kv_cache_num_heads = self.generator.audio_config["num_heads"]  # 12 for 1.3B, 24 for 5B
        self.audio_kv_cache_dim_head = (
            self.generator.audio_config["dim"] // self.generator.audio_config["num_heads"]
        )  # 128

        video_num_max_frames = int(config.num_output_frames)
        audio_num_max_frames = int(config.num_output_frames) * 5

        self.video_kv_cache_size = video_num_max_frames * self.video_frame_seq_length
        self.audio_kv_cache_size = (
            audio_num_max_frames * self.audio_frame_seq_length + self.future_audio_frames * self.audio_frame_seq_length
        )

        print(f"KV inference with {self.video_num_frame_per_block} frames per block")

        if self.video_num_frame_per_block > 1:
            self.generator.model.video_model.num_frame_per_block = self.video_num_frame_per_block
            self.generator.model.audio_model.num_frame_per_block = self.audio_num_frame_per_block
            self.generator.model.video_num_frame_per_block = self.video_num_frame_per_block
            self.generator.model.audio_num_frame_per_block = self.audio_num_frame_per_block

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

    @staticmethod
    def resolve_vae_decode_device(
        diffusion_device: torch.device, requested_device: Optional[str] = None
    ) -> torch.device:
        diffusion_index = diffusion_device.index if diffusion_device.index is not None else torch.cuda.current_device()
        if requested_device is not None:
            vae_decode_device = torch.device(requested_device)
        else:
            if torch.cuda.device_count() < 2:
                raise RuntimeError("--parallel_vae_decode requires at least two visible CUDA devices.")
            vae_decode_device = torch.device(f"cuda:{(diffusion_index + 1) % torch.cuda.device_count()}")

        if vae_decode_device.type != "cuda":
            raise ValueError("--vae_decode_device must be a CUDA device when --parallel_vae_decode is enabled.")

        vae_decode_index = (
            vae_decode_device.index if vae_decode_device.index is not None else torch.cuda.current_device()
        )
        if vae_decode_index == diffusion_index:
            raise ValueError("VAE decode device must be different from the diffusion device for parallel decoding.")

        return torch.device(f"cuda:{vae_decode_index}")

    def encode_video_to_latent(self, video: torch.Tensor) -> torch.Tensor:
        if hasattr(self.vae_video, "encode_to_latent"):
            return self.vae_video.encode_to_latent(video)

        latent = self.vae_video.wrapped_encode(video)
        return latent.permute(0, 2, 1, 3, 4).contiguous()

    def inference(
        self,
        video_noise: torch.Tensor,
        audio_noise: torch.Tensor,
        text_prompts: List[str],
        initial_latent: Optional[torch.Tensor] = None,
        return_latents: bool = False,
        profile: bool = False,
        low_memory: bool = False,
        vae_decode_device: Optional[torch.device] = None,
        decode_audio_blocks: bool = True,
    ) -> torch.Tensor:
        """
        Perform inference on the given noise and text prompts.
        Inputs:
            noise (torch.Tensor): The input noise tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
            text_prompts (List[str]): The list of text prompts.
            initial_latent (torch.Tensor): The initial latent tensor of shape
                (batch_size, num_input_frames, num_channels, height, width).
                If num_input_frames is 1, perform image to video.
                If num_input_frames is greater than 1, perform video extension.
            return_latents (bool): Whether to return the latents.
        Outputs:
            video (torch.Tensor): The generated video tensor of shape
                (batch_size, num_output_frames, num_channels, height, width).
                It is normalized to be in the range [0, 1].
        """
        parallel_vae_decode = vae_decode_device is not None
        if parallel_vae_decode and initial_latent is not None:
            raise NotImplementedError(
                "--parallel_vae_decode currently supports the T2V streaming path only. "
                "Run without --parallel_vae_decode for I2V or video-extension inference."
            )

        batch_size, video_num_frames, video_num_channels, height, width = video_noise.shape
        batch_size, audio_num_frames, audio_num_channels = audio_noise.shape

        if not self.independent_first_frame or (self.independent_first_frame and initial_latent is not None):
            # If the first frame is independent and the first frame is provided, then the number of frames in the
            # noise should still be a multiple of num_frame_per_block
            assert video_num_frames % self.video_num_frame_per_block == 0
            video_num_blocks = video_num_frames // self.video_num_frame_per_block
            assert audio_num_frames % self.audio_num_frame_per_block == 0
        else:
            # Using a [1, 4, 4, 4, 4, 4, ...] model to generate a video without image conditioning
            assert (video_num_frames - 1) % self.video_num_frame_per_block == 0
            video_num_blocks = (video_num_frames - 1) // self.video_num_frame_per_block
            assert audio_num_frames % self.audio_num_frame_per_block == 0
        video_num_input_frames = initial_latent.shape[1] if initial_latent is not None else 0
        video_num_output_frames = video_num_frames + video_num_input_frames  # add the initial latent
        audio_num_output_frames = audio_num_frames  # add the initial latent frames

        conditional_dict = self.text_encoder(text_prompts=text_prompts)
        conditional_dict["video_prompt_embeds"] = conditional_dict["prompt_embeds"]
        conditional_dict["audio_prompt_embeds"] = conditional_dict["prompt_embeds"]
        conditional_dict.pop("prompt_embeds")

        if low_memory:
            gpu_memory_preservation = get_cuda_free_memory_gb(gpu) + 5
            move_model_to_device_with_memory_preservation(
                self.text_encoder, target_device=gpu, preserved_memory_gb=gpu_memory_preservation
            )

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

        # Set up profiling if requested
        if profile:
            init_start = torch.cuda.Event(enable_timing=True)
            init_end = torch.cuda.Event(enable_timing=True)
            diffusion_start = torch.cuda.Event(enable_timing=True)
            diffusion_end = torch.cuda.Event(enable_timing=True)
            if not parallel_vae_decode:
                vae_start = torch.cuda.Event(enable_timing=True)
                vae_end = torch.cuda.Event(enable_timing=True)
            block_times = []
            block_start = torch.cuda.Event(enable_timing=True)
            block_end = torch.cuda.Event(enable_timing=True)
            init_start.record()

        # Step 1: Initialize KV cache to all zeros
        self._initialize_kv_cache_fusion(batch_size=batch_size, dtype=video_noise.dtype, device=video_noise.device)
        self._initialize_crossattn_cache_fusion(
            batch_size=batch_size, dtype=video_noise.dtype, device=video_noise.device
        )

        # Step 2: Cache context feature
        video_current_start_frame = 0
        if initial_latent is not None:
            timestep = torch.ones([batch_size, 1], device=video_noise.device, dtype=torch.int64) * 0
            if self.independent_first_frame:
                assert (video_num_input_frames - 1) % self.video_frame_seq_length
                video_num_input_blocks = (video_num_input_frames - 1) // self.video_num_frame_per_block
                # Assume num_input_frames is 1 + self.num_frame_per_block * num_input_blocks
                video_output[:, :1] = initial_latent[:, :1]
                with torch.no_grad():
                    self.generator(
                        noisy_image_or_video=initial_latent[:, :1],
                        conditional_dict=conditional_dict,
                        timestep=timestep * 0,
                        kv_cache=self.kv_cache,
                        crossattn_cache=self.crossattn_cache,
                        current_start=video_current_start_frame * self.frame_seq_length,
                    )
                video_current_start_frame += 1
            else:
                # Assume num_input_frames is self.num_frame_per_block * num_input_blocks
                assert video_num_input_frames % self.video_num_frame_per_block == 0
                video_num_input_blocks = video_num_input_frames // self.video_num_frame_per_block

            for _ in range(video_num_input_blocks):
                current_ref_latents = initial_latent[
                    :, video_current_start_frame : video_current_start_frame + self.video_num_frame_per_block
                ]
                video_output[
                    :, video_current_start_frame : video_current_start_frame + self.video_num_frame_per_block
                ] = current_ref_latents
                self.generator(
                    noisy_image_or_video=current_ref_latents,
                    conditional_dict=conditional_dict,
                    timestep=timestep * 0,
                    kv_cache=self.kv_cache,
                    crossattn_cache=self.crossattn_cache,
                    current_start=video_current_start_frame * self.video_frame_seq_length,
                )
                video_current_start_frame += self.video_num_frame_per_block

        if profile:
            init_end.record()
            torch.cuda.synchronize()
            diffusion_start.record()

        # Step 3: Temporal denoising loop
        video_all_num_frames = [self.video_num_frame_per_block] * video_num_blocks
        audio_all_num_frames = [self.audio_num_frame_per_block] * (audio_num_frames // self.audio_num_frame_per_block)
        if self.independent_first_frame and initial_latent is None:
            video_all_num_frames = [1] + video_all_num_frames
        audio_current_start_frame = 0
        parallel_decoder = None
        if parallel_vae_decode:
            parallel_decoder = _ParallelVAEBlockDecoder(self.vae_video, self.vae_audio, vae_decode_device)

        # Step 3: Temporal denoising loop
        for block_index, video_current_num_frames in enumerate(video_all_num_frames):
            if profile:
                block_start.record()
            video_noisy_input = video_noise[
                :,
                video_current_start_frame - video_num_input_frames : video_current_start_frame
                + video_current_num_frames
                - video_num_input_frames,
            ]
            audio_current_num_frames = audio_all_num_frames[block_index]
            audio_current_end_frame = audio_current_start_frame + audio_current_num_frames
            future_audio_end_frame = audio_current_end_frame + self.future_audio_frames
            audio_current_window_num_frames = audio_current_num_frames + self.future_audio_frames

            # Keep previously generated audio blocks fixed and only denoise the current block.
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
                # print(f"current_timestep: {current_timestep}")
                video_timestep = (
                    torch.ones([batch_size, video_current_num_frames], device=video_noise.device, dtype=torch.int64)
                    * current_timestep
                )
                audio_history_timestep = (
                    torch.ones([batch_size, audio_current_start_frame], device=audio_noise.device, dtype=torch.int64)
                    * self.config.context_noise
                )
                audio_current_timestep = (
                    torch.ones(
                        [batch_size, audio_current_window_num_frames], device=audio_noise.device, dtype=torch.int64
                    )
                    * current_timestep
                )
                audio_timestep = torch.cat([audio_history_timestep, audio_current_timestep], dim=1)

                if index < len(self.denoising_step_list) - 1:
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

            # Step 3.2: record the model's output
            video_output[:, video_current_start_frame : video_current_start_frame + video_current_num_frames] = (
                video_pred
            )
            audio_output[:, audio_current_start_frame:audio_current_end_frame] = audio_pred[
                :, audio_current_start_frame:audio_current_end_frame
            ]

            if parallel_decoder is not None:
                audio_latents_for_decode = (
                    audio_pred[:, audio_current_start_frame:audio_current_end_frame].detach()
                    if decode_audio_blocks
                    else None
                )
                ready_event = torch.cuda.Event()
                ready_event.record(torch.cuda.current_stream(video_noise.device))
                parallel_decoder.submit(
                    video_pred.detach(),
                    audio_latents_for_decode,
                    ready_event=ready_event,
                )

            # Step 3.3: rerun with timestep zero to update KV cache using clean context
            video_context_timestep = torch.ones_like(video_timestep) * self.config.context_noise
            clean_audio_input = audio_pred[:, :future_audio_end_frame]
            audio_context_timestep = (
                torch.ones([batch_size, future_audio_end_frame], device=audio_noise.device, dtype=torch.int64)
                * self.config.context_noise
            )

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

            if profile:
                block_end.record()
                torch.cuda.synchronize()
                block_time = block_start.elapsed_time(block_end)
                block_times.append(block_time)

            # Step 3.4: update the start and end frame indices
            video_current_start_frame += video_current_num_frames
            audio_current_start_frame = audio_current_end_frame

        if profile:
            # End diffusion timing and synchronize CUDA
            diffusion_end.record()
            torch.cuda.synchronize()
            diffusion_time = diffusion_start.elapsed_time(diffusion_end)
            init_time = init_start.elapsed_time(init_end)
            if not parallel_vae_decode:
                vae_start.record()

        # Step 4: Decode the output
        if parallel_decoder is not None:
            torch.cuda.synchronize(video_noise.device)
            vae_wait_start = time.time()
            video, audio = parallel_decoder.finish()
            if audio is None:
                audio = parallel_decoder.decode_full_audio(audio_output)
            vae_drain_time = time.time() - vae_wait_start
        else:
            video = self.vae_video.wrapped_decode(rearrange(video_output, "b f c h w -> b c f h w"))  # [B, C, F, H, W]
            audio = self.vae_audio.wrapped_decode(rearrange(audio_output, "b f c -> b c f"))  # [B, 1, num_samples]

        if profile:
            if parallel_vae_decode:
                total_time = init_time / 1000 + diffusion_time / 1000 + vae_drain_time
                print("Parallel VAE streaming profiling results:")
                print(f"  - Initialization/caching time: {init_time / 1000:.2f} s")
                print(f"  - Diffusion/cache wall time: {diffusion_time / 1000:.2f} s")
                for i, block_time in enumerate(block_times):
                    print(
                        f"    - Block {i} generation time: {block_time / 1000:.2f} s ({100 * block_time / diffusion_time:.2f}% of diffusion)"
                    )
                print(f"  - Final VAE drain wait: {vae_drain_time:.2f} s")
                print(f"  - Total measured time: {total_time:.2f} s")
            else:
                # End VAE timing and synchronize CUDA
                vae_end.record()
                torch.cuda.synchronize()
                vae_time = vae_start.elapsed_time(vae_end)
                total_time = init_time + diffusion_time + vae_time

                print("Profiling results:")
                print(
                    f"  - Initialization/caching time: {init_time / 1000:.2f} s ({100 * init_time / total_time:.2f}%)"
                )
                print(
                    f"  - Diffusion generation time: {diffusion_time / 1000:.2f} s ({100 * diffusion_time / total_time:.2f}%)"
                )

                for i, block_time in enumerate(block_times):
                    print(
                        f"    - Block {i} generation time: {block_time / 1000:.2f} s ({100 * block_time / diffusion_time:.2f}% of diffusion)"
                    )

                print(f"  - VAE decoding time: {vae_time / 1000:.2f} s ({100 * vae_time / total_time:.2f}%)")
                print(f"  - Total time: {total_time / 1000:.2f} s")

        if return_latents:
            return video, audio, video_output, audio_output
        else:
            return video, audio

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
