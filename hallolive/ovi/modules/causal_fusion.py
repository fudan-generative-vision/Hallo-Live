import math

import torch
import torch.nn as nn
from hallolive.ovi.modules.causal_model import WanLayerNorm, CausalWanModel, WanRMSNorm, gradient_checkpointing

from hallolive.ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state
from torch.nn.attention.flex_attention import create_block_mask, BlockMask


class CausalFusionModel(nn.Module):
    def __init__(
        self,
        video_config=None,
        audio_config=None,
        enable_cross_attention=True,
        future_audio_frames=None,
    ):
        super().__init__()
        self.future_audio_frames = None if future_audio_frames is None else int(future_audio_frames)
        has_video = True
        has_audio = True
        if video_config is not None:
            self.video_model = CausalWanModel(**video_config)
        else:
            has_video = False
            self.video_model = None
            print("Warning: No video model is provided!")

        if audio_config is not None:
            self.audio_model = CausalWanModel(**audio_config)
        else:
            has_audio = False
            self.audio_model = None
            print("Warning: No audio model is provided!")

        if has_video and has_audio:
            assert len(self.video_model.blocks) == len(self.audio_model.blocks)
            self.num_blocks = len(self.video_model.blocks)

            self.use_sp = get_sequence_parallel_state()
            if self.use_sp:
                self.sp_size = nccl_info.sp_size
                self.sp_rank = nccl_info.rank_within_group
            if enable_cross_attention:
                self.inject_cross_attention_kv_projections()

        self.init_weights()
        self.gradient_checkpointing = False

        self.block_mask_video_q_audio_kv = None
        self.block_mask_audio_q_video_kv = None

    def enable_gradient_checkpointing(self) -> None:
        self.video_model.set_gradient_checkpointing(True)
        self.audio_model.set_gradient_checkpointing(True)
        self.gradient_checkpointing = True

    # def inject_cross_attention_kv_projections(self):
    #     for vid_block in self.video_model.blocks:
    #         vid_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, vid_block.dim)
    #         vid_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, vid_block.dim)
    #         vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
    #         vid_block.cross_attn.norm_k_fusion = (
    #             WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
    #         )

    #     for audio_block in self.audio_model.blocks:
    #         audio_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, audio_block.dim)
    #         audio_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, audio_block.dim)
    #         audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
    #         audio_block.cross_attn.norm_k_fusion = (
    #             WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()
    #         )

    def inject_cross_attention_kv_projections(self):
        for index in range(len(self.video_model.blocks)):
            vid_block = self.video_model.blocks[index]
            audio_block = self.audio_model.blocks[index]

            vid_block.cross_attn.k_fusion = nn.Linear(audio_block.dim, vid_block.dim)
            vid_block.cross_attn.v_fusion = nn.Linear(audio_block.dim, vid_block.dim)
            vid_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(audio_block.dim, elementwise_affine=True)
            vid_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(vid_block.dim, eps=1e-6) if vid_block.qk_norm else nn.Identity()
            )

            audio_block.cross_attn.k_fusion = nn.Linear(vid_block.dim, audio_block.dim)
            audio_block.cross_attn.v_fusion = nn.Linear(vid_block.dim, audio_block.dim)
            audio_block.cross_attn.pre_attn_norm_fusion = WanLayerNorm(vid_block.dim, elementwise_affine=True)
            audio_block.cross_attn.norm_k_fusion = (
                WanRMSNorm(audio_block.dim, eps=1e-6) if audio_block.qk_norm else nn.Identity()
            )

    def merge_kwargs(self, vid_kwargs, audio_kwargs):
        """
        keys in each kwarg:
        e
        seq_lens
        grid_sizes
        freqs
        context
        context_lens
        """
        merged_kwargs = {}
        for key in vid_kwargs:
            merged_kwargs[f"vid_{key}"] = vid_kwargs[key]
        for key in audio_kwargs:
            merged_kwargs[f"audio_{key}"] = audio_kwargs[key]
        return merged_kwargs

    def single_fusion_cross_attention_forward(
        self,
        cross_attn_block,
        src_seq,
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
        block_mask_cross_attn,
        kv_cache=None,
        crossattn_cache=None,
        src_current_start=0,
        target_current_start=0,
    ):
        return cross_attn_block(
            src_seq,
            src_grid_sizes,
            src_freqs,
            target_seq,
            target_seq_lens,
            target_grid_sizes,
            target_freqs,
            context,
            context_lens,
            block_mask_cross_attn,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            src_current_start=src_current_start,
            target_current_start=target_current_start,
        )

    def single_fusion_cross_attention_ffn_forward(
        self,
        attn_block,
        src_seq,
        src_grid_sizes,
        src_freqs,
        target_seq,
        target_seq_lens,
        target_grid_sizes,
        target_freqs,
        context,
        context_lens,
        src_e,
        block_mask_cross_attn,
        kv_cache=None,
        crossattn_cache=None,
        src_current_start=0,
        target_current_start=0,
    ):
        src_seq = src_seq + self.single_fusion_cross_attention_forward(
            attn_block.cross_attn,
            attn_block.norm3(src_seq),
            src_grid_sizes=src_grid_sizes,
            src_freqs=src_freqs,
            target_seq=target_seq,
            target_seq_lens=target_seq_lens,
            target_grid_sizes=target_grid_sizes,
            target_freqs=target_freqs,
            context=context,
            context_lens=context_lens,
            block_mask_cross_attn=block_mask_cross_attn,
            kv_cache=kv_cache,
            crossattn_cache=crossattn_cache,
            src_current_start=src_current_start,
            target_current_start=target_current_start,
        )

        num_frames, frame_seqlen = src_e[0].shape[1], src_seq.shape[1] // src_e[0].shape[1]

        y = attn_block.ffn(
            (
                attn_block.norm2(src_seq).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + src_e[4])
                + src_e[3]
            ).flatten(1, 2)
        )
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            src_seq = src_seq + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * src_e[5]).flatten(1, 2)
        return src_seq

    def single_fusion_block_forward(
        self,
        vid_block,
        audio_block,
        vid,
        audio,
        vid_e,
        vid_seq_lens,
        vid_grid_sizes,
        vid_freqs,
        vid_context,
        vid_context_lens,
        vid_block_mask,
        audio_e,
        audio_seq_lens,
        audio_grid_sizes,
        audio_freqs,
        audio_context,
        audio_context_lens,
        audio_block_mask,
        kv_cache=None,
        crossattn_cache=None,
        video_current_start=0,
        audio_current_start=0,
    ):
        ## audio modulation
        assert audio_e.dtype == torch.bfloat16
        assert len(audio_e.shape) == 4 and audio_e.size(2) == 6 and audio_e.shape[1] == audio.shape[1], (
            f"{audio_e.shape}, {audio.shape}"
        )
        audio_num_frames, audio_frame_seqlen = audio_e.shape[1], audio.shape[1] // audio_e.shape[1]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            audio_e = audio_block.modulation(audio_e).chunk(6, dim=2)

        # audio_e = audio_e.chunk(6, dim=2)
        assert audio_e[0].dtype == torch.bfloat16

        # audio self-attention
        audio_y = audio_block.self_attn(
            (
                audio_block.norm1(audio).unflatten(dim=1, sizes=(audio_num_frames, audio_frame_seqlen))
                * (1 + audio_e[1])
                + audio_e[0]
            ).flatten(1, 2),
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            audio_block_mask,
            kv_cache["audio_self"] if kv_cache is not None else None,
            audio_current_start,
        )

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # audio = audio + audio_y * audio_e[2].squeeze(2)

            audio = audio + (
                audio_y.unflatten(dim=1, sizes=(audio_num_frames, audio_frame_seqlen)) * audio_e[2]
            ).flatten(1, 2)

        ## video modulation
        # assert len(vid_e.shape) == 4 and vid_e.size(2) == 6 and vid_e.shape[1] == vid.shape[1], (
        #     f"{vid_e.shape}, {vid.shape}"
        # )
        vid_num_frames, vid_frame_seqlen = vid_e.shape[1], vid.shape[1] // vid_e.shape[1]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            vid_e = vid_block.modulation(vid_e).chunk(6, dim=2)  # [4, 30, 6, 3072]

        # video self-attention
        vid_y = vid_block.self_attn(
            (
                vid_block.norm1(vid).unflatten(dim=1, sizes=(vid_num_frames, vid_frame_seqlen)) * (1 + vid_e[1])
                + vid_e[0]
            ).flatten(1, 2),
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            vid_block_mask,
            kv_cache["video_self"] if kv_cache is not None else None,
            video_current_start,
        )  # [4, 14880, 3072]

        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            # vid = vid + vid_y * vid_e[2].squeeze(2)

            vid = vid + (vid_y.unflatten(dim=1, sizes=(vid_num_frames, vid_frame_seqlen)) * vid_e[2]).flatten(1, 2)

        og_audio = audio

        # audio cross-attention
        audio = self.single_fusion_cross_attention_ffn_forward(
            audio_block,
            audio,
            audio_grid_sizes,
            audio_freqs,
            vid,
            vid_seq_lens,
            vid_grid_sizes,
            vid_freqs,
            audio_context,
            audio_context_lens,
            audio_e,
            self.block_mask_audio_q_video_kv,
            kv_cache=kv_cache["video_cross"] if kv_cache is not None else None,
            crossattn_cache=crossattn_cache["audio"] if crossattn_cache is not None else None,
            src_current_start=audio_current_start,
            target_current_start=video_current_start,
        )

        assert not torch.equal(og_audio, audio), "Audio should be changed after cross-attention!"

        # video cross-attention
        vid = self.single_fusion_cross_attention_ffn_forward(
            vid_block,
            vid,
            vid_grid_sizes,
            vid_freqs,
            og_audio,
            audio_seq_lens,
            audio_grid_sizes,
            audio_freqs,
            vid_context,
            vid_context_lens,
            vid_e,
            self.block_mask_video_q_audio_kv,
            kv_cache=kv_cache["audio_cross"] if kv_cache is not None else None,
            crossattn_cache=crossattn_cache["video"] if crossattn_cache is not None else None,
            src_current_start=video_current_start,
            target_current_start=audio_current_start,
        )

        return vid, audio

    def _forward_inference(
        self,
        vid,
        audio,
        video_t,
        audio_t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
        kv_cache: dict = None,
        crossattn_cache: dict = None,
        video_current_start: int = 0,
        audio_current_start: int = 0,
        cache_start: int = 0,
    ):
        r"""
        Run the diffusion model with kv caching.
        See Algorithm 2 of CausVid paper https://arxiv.org/abs/2412.07772 for details.
        This function will be run for num_frame times.
        Process the latent frames one by one (1560 tokens each)

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
            t (Tensor):
                Diffusion timesteps tensor of shape [B]
            context (List[Tensor]):
                List of text embeddings each with shape [L, C]
            seq_len (`int`):
                Maximum sequence length for positional encoding
            clip_fea (Tensor, *optional*):
                CLIP image features for image-to-video mode
            y (List[Tensor], *optional*):
                Conditional video inputs for image-to-video mode, same shape as x

        Returns:
            List[Tensor]:
                List of denoised video tensors with original input shapes [C_out, F, H / 8, W / 8]
        """

        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid,
            t=video_t,
            context=vid_context,
            seq_len=vid_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
            train_mode=False,
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio,
            t=audio_t,
            context=audio_context,
            seq_len=audio_seq_len,
            clip_fea=clip_fea_audio,
            y=None,
            first_frame_is_clean=False,
            train_mode=False,
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]

            if torch.is_grad_enabled() and self.gradient_checkpointing:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[i],
                        "video_current_start": video_current_start,
                        "audio_current_start": audio_current_start,
                    }
                )
            else:
                kwargs.update(
                    {
                        "kv_cache": kv_cache[i],
                        "crossattn_cache": crossattn_cache[i],
                        "video_current_start": video_current_start,
                        "audio_current_start": audio_current_start,
                    }
                )

            vid, audio = gradient_checkpointing(
                enabled=(torch.is_grad_enabled() and self.gradient_checkpointing),
                module=self.single_fusion_block_forward,
                vid_block=vid_block,
                audio_block=audio_block,
                vid=vid,
                audio=audio,
                **kwargs,
            )

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs["grid_sizes"], vid_e, video_t)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs["grid_sizes"], audio_e, audio_t)

        return vid, audio

    def _forward_train(
        self,
        vid,
        audio,
        video_t,
        audio_t,
        vid_context,
        audio_context,
        vid_seq_len,
        audio_seq_len,
        clip_fea=None,
        clip_fea_audio=None,
        y=None,
        first_frame_is_clean=False,
        slg_layer=False,
    ):
        assert clip_fea is None
        assert y is None

        if vid is None or all([x is None for x in vid]):
            assert vid_context is None
            assert vid_seq_len is None
            assert self.audio_model is not None

            return None, self.audio_model(
                x=audio, t=audio_t, context=audio_context, seq_len=audio_seq_len, clip_fea=clip_fea_audio, y=None
            )

        if audio is None or all([x is None for x in audio]):
            assert clip_fea_audio is None
            assert audio_context is None
            assert audio_seq_len is None
            assert self.video_model is not None

            return self.video_model(
                x=vid,
                t=video_t,
                context=vid_context,
                seq_len=vid_seq_len,
                clip_fea=clip_fea,
                y=y,
                first_frame_is_clean=first_frame_is_clean,
            ), None

        audio_frame_seqlen = 1
        audio_num_frames = audio.shape[1] // audio_frame_seqlen
        video_frame_seqlen = (
            vid.shape[-2] * vid.shape[-1] // (self.video_model.patch_size[1] * self.video_model.patch_size[2])
        )
        video_num_frames = vid.shape[2]

        device = next(self.video_model.patch_embedding.parameters()).device

        if self.block_mask_video_q_audio_kv is None:
            self.block_mask_video_q_audio_kv, self.block_mask_audio_q_video_kv = (
                self._prepare_audio_video_crossattn_blockwise_causal_attn_mask(
                    device,
                    video_num_frames=video_num_frames,
                    video_frame_seqlen=video_frame_seqlen,
                    video_num_frame_per_block=self.video_model.num_frame_per_block,
                    audio_num_frames=audio_num_frames,
                    audio_frame_seqlen=audio_frame_seqlen,
                    audio_num_frame_per_block=self.audio_model.num_frame_per_block,
                    future_audio_frames=self.future_audio_frames,
                )
            )

        vid, vid_e, vid_kwargs = self.video_model.prepare_transformer_block_kwargs(
            x=vid,
            t=video_t,
            context=vid_context,
            seq_len=vid_seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
            train_mode=True,
        )

        audio, audio_e, audio_kwargs = self.audio_model.prepare_transformer_block_kwargs(
            x=audio,
            t=audio_t,
            context=audio_context,
            seq_len=audio_seq_len,
            clip_fea=clip_fea_audio,
            y=None,
            first_frame_is_clean=False,
            train_mode=True,
        )

        kwargs = self.merge_kwargs(vid_kwargs, audio_kwargs)

        for i in range(self.num_blocks):
            """
            1 fusion block refers to 1 audio block with 1 video block.
            """
            if slg_layer > 0 and i == slg_layer:
                continue
            vid_block = self.video_model.blocks[i]
            audio_block = self.audio_model.blocks[i]
            vid, audio = gradient_checkpointing(
                enabled=(torch.is_grad_enabled() and self.gradient_checkpointing),
                module=self.single_fusion_block_forward,
                vid_block=vid_block,
                audio_block=audio_block,
                vid=vid,
                audio=audio,
                **kwargs,
            )

        vid = self.video_model.post_transformer_block_out(vid, vid_kwargs["grid_sizes"], vid_e, video_t)
        audio = self.audio_model.post_transformer_block_out(audio, audio_kwargs["grid_sizes"], audio_e, audio_t)

        return vid, audio

    def forward(self, *args, **kwargs):
        if kwargs.get("kv_cache", None) is not None:
            return self._forward_inference(*args, **kwargs)
        else:
            return self._forward_train(*args, **kwargs)

    @staticmethod
    def _prepare_audio_video_crossattn_blockwise_causal_attn_mask(
        device: torch.device | str,
        video_num_frames: int = 30,
        video_frame_seqlen: int = 1560,
        video_num_frame_per_block=1,
        audio_num_frames: int = 30,
        audio_frame_seqlen: int = 5,
        audio_num_frame_per_block=1,
        local_attn_size=-1,
        future_audio_frames=None,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        video_total_length = video_num_frames * video_frame_seqlen
        audio_total_length = audio_num_frames * audio_frame_seqlen

        # we do right padding to get to a multiple of 128
        video_padded_length = math.ceil(video_total_length / 128) * 128 - video_total_length
        audio_padded_length = math.ceil(audio_total_length / 128) * 128 - audio_total_length

        video_ends = torch.zeros(video_total_length + video_padded_length, device=device, dtype=torch.long)
        audio_ends = torch.zeros(audio_total_length + audio_padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        video_frame_indices = torch.arange(
            start=0, end=video_total_length, step=video_frame_seqlen * video_num_frame_per_block, device=device
        )
        audio_frame_indices = torch.arange(
            start=0, end=audio_total_length, step=audio_frame_seqlen * audio_num_frame_per_block, device=device
        )

        assert len(video_frame_indices) == len(audio_frame_indices), (
            f"video_frame_indices length {len(video_frame_indices)} not equals to audio_frame_indices length {len(audio_frame_indices)}"
        )

        for i in range(len(video_frame_indices)):
            video_idx = video_frame_indices[i]
            audio_idx = audio_frame_indices[i]
            video_ends[video_idx : video_idx + video_frame_seqlen * video_num_frame_per_block] = (
                audio_idx + audio_frame_seqlen * audio_num_frame_per_block
            )
            audio_ends[audio_idx : audio_idx + audio_frame_seqlen * audio_num_frame_per_block] = (
                video_idx + video_frame_seqlen * video_num_frame_per_block
            )

        future_audio_kv_tokens = int(future_audio_frames) * audio_frame_seqlen

        def attention_mask_video_q_audio_kv(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return kv_idx < (video_ends[q_idx] + future_audio_kv_tokens)
            else:
                return (kv_idx <= video_ends[q_idx]) & (
                    kv_idx >= (video_ends[q_idx] - local_attn_size * audio_frame_seqlen)
                )

        def attention_mask_audio_q_video_kv(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return kv_idx < audio_ends[q_idx]
            else:
                return (kv_idx <= audio_ends[q_idx]) & (
                    kv_idx >= (audio_ends[q_idx] - local_attn_size * video_frame_seqlen)
                )

        block_mask_video_q_audio_kv = create_block_mask(
            attention_mask_video_q_audio_kv,
            B=None,
            H=None,
            Q_LEN=video_total_length + video_padded_length,
            KV_LEN=audio_total_length + audio_padded_length,
            _compile=False,
            device=device,
        )

        block_mask_audio_q_video_kv = create_block_mask(
            attention_mask_audio_q_video_kv,
            B=None,
            H=None,
            Q_LEN=audio_total_length + audio_padded_length,
            KV_LEN=video_total_length + video_padded_length,
            _compile=False,
            device=device,
        )

        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            print("block_mask_video_q_audio_kv")
            print(block_mask_video_q_audio_kv)
            print("block_mask_audio_q_video_kv")
            print(block_mask_audio_q_video_kv)

        return block_mask_video_q_audio_kv, block_mask_audio_q_video_kv

    def init_weights(self):
        if self.audio_model is not None:
            self.audio_model.init_weights()

        if self.video_model is not None:
            self.video_model.init_weights()

        for name, mod in self.video_model.named_modules():
            if "fusion" in name and isinstance(mod, nn.Linear):
                with torch.no_grad():
                    mod.weight.div_(10.0)

    def set_rope_params(self):
        self.video_model.set_rope_params()
        self.audio_model.set_rope_params()
