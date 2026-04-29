# Copyright 2024-2025 The Alibaba Wan Team Authors. All rights reserved.
import math

import torch
import torch.amp as amp
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.models.modeling_utils import ModelMixin
from .attention import flash_attention
from torch.utils.checkpoint import checkpoint
from hallolive.ovi.distributed_comms.communications import all_gather, all_to_all_4D
from hallolive.wan.modules.attention import attention
from hallolive.ovi.distributed_comms.parallel_states import nccl_info, get_sequence_parallel_state
from torch.nn.attention.flex_attention import create_block_mask, flex_attention, BlockMask
from hallolive.wan.modules.model import (
    WanRMSNorm,
    WanLayerNorm,
    # WAN_CROSSATTENTION_CLASSES,
    MLPProj,
    sinusoidal_embedding_1d,
)

flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")


class WanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, window_size=(-1, -1), qk_norm=True, eps=1e-6, local_attn_size=-1, sink_size=0):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 496

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        # optional sequence parallelism
        # self.world_size = get_world_size()
        self.use_sp = get_sequence_parallel_state()
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, (
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
            )

    # query, key, value function
    def qkv_fn(self, x):
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        q = self.norm_q(self.q(x)).view(b, s, n, d)
        k = self.norm_k(self.k(x)).view(b, s, n, d)
        v = self.v(x).view(b, s, n, d)
        return q, k, v

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, C]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        q, k, v = self.qkv_fn(x)
        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard head")
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = all_to_all_4D(k, scatter_dim=2, gather_dim=1)
            v = all_to_all_4D(v, scatter_dim=2, gather_dim=1)  # [B, L, H/P, C/H]
        x = flash_attention(
            q=rope_apply(q, grid_sizes, freqs),
            k=rope_apply(k, grid_sizes, freqs),
            v=v,
            k_lens=seq_lens,
            window_size=self.window_size,
        )
        if self.use_sp:
            # print(f"[DEBUG SP] Doing all to all to shard sequence")
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)  # [B, L/P, H, C/H]
        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class WanT2VCrossAttention(WanSelfAttention):
    def qkv_fn(self, x, context, crossattn_cache=None):
        b, n, d = x.size(0), self.num_heads, self.head_dim

        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)

        if crossattn_cache is not None:
            if not crossattn_cache["is_init"]:
                crossattn_cache["is_init"] = True
                k = self.norm_k(self.k(context)).view(b, -1, n, d)
                v = self.v(context).view(b, -1, n, d)
                crossattn_cache["k"] = k
                crossattn_cache["v"] = v
            else:
                k = crossattn_cache["k"]
                v = crossattn_cache["v"]
        else:
            k = self.norm_k(self.k(context)).view(b, -1, n, d)
            v = self.v(context).view(b, -1, n, d)

        return q, k, v

    def forward(
        self,
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
        cache_start=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
            crossattn_cache (List[dict], *optional*): Contains the cached key and value tensors for context embedding.
        """
        b, n, d = src_seq.size(0), self.num_heads, self.head_dim
        if hasattr(self, "k_img"):
            ## means is i2v block
            q, k, v, k_img, v_img = self.qkv_fn(src_seq, context, crossattn_cache)
        else:
            ## means is t2v block
            q, k, v = self.qkv_fn(src_seq, context, crossattn_cache)
            k_img = v_img = None

        if self.use_sp:
            q = all_to_all_4D(q, scatter_dim=2, gather_dim=1)
            k = torch.chunk(k, self.sp_size, dim=2)[self.sp_rank]
            v = torch.chunk(v, self.sp_size, dim=2)[self.sp_rank]
            if k_img is not None:
                k_img = torch.chunk(k_img, self.sp_size, dim=2)[self.sp_rank]
            if v_img is not None:
                v_img = torch.chunk(v_img, self.sp_size, dim=2)[self.sp_rank]

        x = flash_attention(q, k, v, k_lens=context_lens)

        if k_img is not None:
            img_x = flash_attention(q, k_img, v_img, k_lens=None)
            x = x + img_x

        if not hasattr(self, "k_fusion"):
            if self.use_sp:
                x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)  # [B, L/P, H, C/H]

            x = x.flatten(2)  # [B, L/P, C]

            x = self.o(x)
            return x

        is_vid = src_grid_sizes.shape[1] > 1
        # compute target attention
        target_seq = self.pre_attn_norm_fusion(target_seq)
        k_target = self.norm_k_fusion(self.k_fusion(target_seq)).view(b, -1, n, d)
        v_target = self.v_fusion(target_seq).view(b, -1, n, d)
        if self.use_sp:
            k_target = all_to_all_4D(k_target, scatter_dim=2, gather_dim=1)  # [B, L, H/P, C/H]
            v_target = all_to_all_4D(v_target, scatter_dim=2, gather_dim=1)  # [B, L, H/P, C/H]

        if kv_cache is None:
            q = rope_apply(q, src_grid_sizes, src_freqs)
            k_target = rope_apply(k_target, target_grid_sizes, target_freqs)

            padded_length_q = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
            padded_length_kv = math.ceil(k_target.shape[1] / 128) * 128 - k_target.shape[1]

            padded_roped_query = torch.cat(
                [
                    q,
                    torch.zeros(
                        [q.shape[0], padded_length_q, q.shape[2], q.shape[3]], device=q.device, dtype=v_target.dtype
                    ),
                ],
                dim=1,
            )

            padded_roped_key = torch.cat(
                [
                    k_target,
                    torch.zeros(
                        [k_target.shape[0], padded_length_kv, k_target.shape[2], k_target.shape[3]],
                        device=k_target.device,
                        dtype=v_target.dtype,
                    ),
                ],
                dim=1,
            )

            padded_v = torch.cat(
                [
                    v_target,
                    torch.zeros(
                        [v_target.shape[0], padded_length_kv, v_target.shape[2], v_target.shape[3]],
                        device=v_target.device,
                        dtype=v_target.dtype,
                    ),
                ],
                dim=1,
            )

            target_x = flex_attention(
                query=padded_roped_query.transpose(2, 1),
                key=padded_roped_key.transpose(2, 1),
                value=padded_v.transpose(2, 1),
                block_mask=block_mask_cross_attn,
            )[:, :, :-padded_length_q].transpose(2, 1)

            # target_x = flash_attention(q, k_target, v_target, k_lens=target_seq_lens)

        else:
            src_frame_seqlen = torch.prod(src_grid_sizes[0][1:]).item()
            target_frame_seqlen = torch.prod(target_grid_sizes[0][1:]).item()
            src_current_start_frame = src_current_start // src_frame_seqlen
            target_current_start_frame = target_current_start // target_frame_seqlen
            roped_query = causal_rope_apply(q, src_grid_sizes, src_freqs, start_frame=src_current_start_frame).type_as(
                v_target
            )
            roped_key = causal_rope_apply(
                k_target, target_grid_sizes, target_freqs, start_frame=target_current_start_frame
            ).type_as(v_target)

            src_current_end = src_current_start + roped_query.shape[1]
            target_current_end = target_current_start + roped_key.shape[1]
            sink_tokens = self.sink_size * target_frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_key.shape[1]
            if (
                self.local_attn_size != -1
                and (target_current_end > kv_cache["global_end_index"].item())
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):
                pass
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = (
                    kv_cache["local_end_index"].item() + target_current_end - kv_cache["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v_target

            target_x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size) : local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size) : local_end_index],
            )
            kv_cache["global_end_index"].fill_(target_current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        x = x + target_x

        if self.use_sp:
            x = all_to_all_4D(x, scatter_dim=1, gather_dim=2)  # [B, L/P, H, C/H]

        x = x.flatten(2)  # [B, L/P, C]

        x = self.o(x)
        return x


WAN_CROSSATTENTION_CLASSES = {"t2v_cross_attn": WanT2VCrossAttention}


def gradient_checkpointing(module: nn.Module, *args, enabled: bool, **kwargs):
    if enabled:
        return checkpoint(module, *args, use_reentrant=False, **kwargs)
    else:
        return module(*args, **kwargs)


@amp.autocast("cuda", enabled=False)
def rope_params(max_seq_len, dim, theta=10000, freqs_scaling=1.0):
    assert dim % 2 == 0
    pos = torch.arange(max_seq_len)
    freqs = 1.0 / torch.pow(theta, torch.arange(0, dim, 2).to(torch.float64).div(dim))
    freqs = freqs_scaling * freqs
    freqs = torch.outer(pos, freqs)
    freqs = torch.polar(torch.ones_like(freqs), freqs)
    return freqs

@amp.autocast("cuda", enabled=False)
def causal_rope_apply_1d(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2  ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"

    # loop over samples
    output = []
    for i, (l,) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))  # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[start_frame : start_frame + seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


def causal_rope_apply_3d(x, grid_sizes, freqs, start_frame=0):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []

    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][start_frame : start_frame + f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


@amp.autocast("cuda", enabled=False)
def causal_rope_apply(x, grid_sizes, freqs, start_frame=0):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return causal_rope_apply_3d(x, grid_sizes, freqs, start_frame=start_frame)
    else:
        return causal_rope_apply_1d(x, grid_sizes, freqs, start_frame=start_frame)


@amp.autocast("cuda", enabled=False)
def rope_apply_1d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2  ## b l h d
    c_rope = freqs.shape[1]  # number of complex dims to rotate
    assert c_rope <= c, "RoPE dimensions cannot exceed half of hidden size"

    # loop over samples
    output = []
    for i, (l,) in enumerate(grid_sizes.tolist()):
        seq_len = l
        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))  # [l n d//2]
        x_i_rope = x_i[:, :, :c_rope] * freqs[:seq_len, None, :]  # [L, N, c_rope]
        x_i_passthrough = x_i[:, :, c_rope:]  # untouched dims
        x_i = torch.cat([x_i_rope, x_i_passthrough], dim=2)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


@amp.autocast("cuda", enabled=False)
def rope_apply_3d(x, grid_sizes, freqs):
    n, c = x.size(2), x.size(3) // 2

    # split freqs
    freqs = freqs.split([c - 2 * (c // 3), c // 3, c // 3], dim=1)

    # loop over samples
    output = []
    for i, (f, h, w) in enumerate(grid_sizes.tolist()):
        seq_len = f * h * w

        # precompute multipliers
        x_i = torch.view_as_complex(x[i, :seq_len].to(torch.float64).reshape(seq_len, n, -1, 2))
        freqs_i = torch.cat(
            [
                freqs[0][:f].view(f, 1, 1, -1).expand(f, h, w, -1),
                freqs[1][:h].view(1, h, 1, -1).expand(f, h, w, -1),
                freqs[2][:w].view(1, 1, w, -1).expand(f, h, w, -1),
            ],
            dim=-1,
        ).reshape(seq_len, 1, -1)

        # apply rotary embedding
        x_i = torch.view_as_real(x_i * freqs_i).flatten(2)
        x_i = torch.cat([x_i, x[i, seq_len:]])

        # append to collection
        output.append(x_i)
    return torch.stack(output).type_as(x)


@amp.autocast("cuda", enabled=False)
def rope_apply(x, grid_sizes, freqs):
    x_ndim = grid_sizes.shape[-1]
    if x_ndim == 3:
        return rope_apply_3d(x, grid_sizes, freqs)
    else:
        return rope_apply_1d(x, grid_sizes, freqs)


class ChannelLastConv1d(nn.Conv1d):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.permute(0, 2, 1)
        x = super().forward(x)
        x = x.permute(0, 2, 1)
        return x


class ConvMLP(nn.Module):
    def __init__(self, dim: int, hidden_dim: int, multiple_of: int = 256, kernel_size: int = 3, padding: int = 1):
        """
        Initialize the FeedForward module.

        Args:
            dim (int): Input dimension.
            hidden_dim (int): Hidden dimension of the feedforward layer.
            multiple_of (int): Value to ensure hidden dimension is a multiple of this value.

        Attributes:
            w1 (ColumnParallelLinear): Linear transformation for the first layer.
            w2 (RowParallelLinear): Linear transformation for the second layer.
            w3 (ColumnParallelLinear): Linear transformation for the third layer.

        """
        super().__init__()
        hidden_dim = int(2 * hidden_dim / 3)
        hidden_dim = multiple_of * ((hidden_dim + multiple_of - 1) // multiple_of)

        self.w1 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w2 = ChannelLastConv1d(hidden_dim, dim, bias=False, kernel_size=kernel_size, padding=padding)
        self.w3 = ChannelLastConv1d(dim, hidden_dim, bias=False, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class ModulationAdd(nn.Module):
    def __init__(self, dim, num):
        super().__init__()
        self.modulation = nn.Parameter(torch.randn(1, num, dim) / dim**0.5)

    def forward(self, e):
        return self.modulation.unsqueeze(1) + e


class CausalWanSelfAttention(nn.Module):
    def __init__(self, dim, num_heads, local_attn_size=-1, sink_size=0, qk_norm=True, eps=1e-6):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.local_attn_size = local_attn_size
        self.sink_size = sink_size
        self.qk_norm = qk_norm
        self.eps = eps
        self.max_attention_size = 32760 if local_attn_size == -1 else local_attn_size * 1560

        # layers
        self.q = nn.Linear(dim, dim)
        self.k = nn.Linear(dim, dim)
        self.v = nn.Linear(dim, dim)
        self.o = nn.Linear(dim, dim)
        self.norm_q = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim, eps=eps) if qk_norm else nn.Identity()

    def forward(self, x, seq_lens, grid_sizes, freqs, block_mask, kv_cache=None, current_start=0, cache_start=None):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
            block_mask (BlockMask)
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim
        if cache_start is None:
            cache_start = current_start

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)

        if kv_cache is None:
            # if it is teacher forcing training?
            is_tf = s == seq_lens[0].item() * 2
            if is_tf:
                q_chunk = torch.chunk(q, 2, dim=1)
                k_chunk = torch.chunk(k, 2, dim=1)
                roped_query = []
                roped_key = []
                # rope should be same for clean and noisy parts
                for ii in range(2):
                    rq = rope_apply(q_chunk[ii], grid_sizes, freqs).type_as(v)
                    rk = rope_apply(k_chunk[ii], grid_sizes, freqs).type_as(v)
                    roped_query.append(rq)
                    roped_key.append(rk)

                roped_query = torch.cat(roped_query, dim=1)
                roped_key = torch.cat(roped_key, dim=1)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask,
                )[:, :, :-padded_length].transpose(2, 1)
            else:
                roped_query = rope_apply(q, grid_sizes, freqs).type_as(v)
                roped_key = rope_apply(k, grid_sizes, freqs).type_as(v)

                padded_length = math.ceil(q.shape[1] / 128) * 128 - q.shape[1]
                padded_roped_query = torch.cat(
                    [
                        roped_query,
                        torch.zeros(
                            [q.shape[0], padded_length, q.shape[2], q.shape[3]], device=q.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_roped_key = torch.cat(
                    [
                        roped_key,
                        torch.zeros(
                            [k.shape[0], padded_length, k.shape[2], k.shape[3]], device=k.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                padded_v = torch.cat(
                    [
                        v,
                        torch.zeros(
                            [v.shape[0], padded_length, v.shape[2], v.shape[3]], device=v.device, dtype=v.dtype
                        ),
                    ],
                    dim=1,
                )

                x = flex_attention(
                    query=padded_roped_query.transpose(2, 1),
                    key=padded_roped_key.transpose(2, 1),
                    value=padded_v.transpose(2, 1),
                    block_mask=block_mask,
                )[:, :, :-padded_length].transpose(2, 1)
        else:
            # video grid_sizes: [3, 16, 31]
            # audio grid_sizes: [15]
            frame_seqlen = torch.prod(grid_sizes[0][1:]).item()
            current_start_frame = current_start // frame_seqlen
            roped_query = causal_rope_apply(q, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            roped_key = causal_rope_apply(k, grid_sizes, freqs, start_frame=current_start_frame).type_as(v)
            current_end = current_start + roped_query.shape[1]
            sink_tokens = self.sink_size * frame_seqlen
            # If we are using local attention and the current KV cache size is larger than the local attention size, we need to truncate the KV cache
            kv_cache_size = kv_cache["k"].shape[1]
            num_new_tokens = roped_key.shape[1]
            if (
                self.local_attn_size != -1
                and (current_end > kv_cache["global_end_index"].item())
                and (num_new_tokens + kv_cache["local_end_index"].item() > kv_cache_size)
            ):
                # Calculate the number of new tokens added in this step
                # Shift existing cache content left to discard oldest tokens
                # Clone the source slice to avoid overlapping memory error
                num_evicted_tokens = num_new_tokens + kv_cache["local_end_index"].item() - kv_cache_size
                num_rolled_tokens = kv_cache["local_end_index"].item() - num_evicted_tokens - sink_tokens
                kv_cache["k"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["k"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                kv_cache["v"][:, sink_tokens : sink_tokens + num_rolled_tokens] = kv_cache["v"][
                    :, sink_tokens + num_evicted_tokens : sink_tokens + num_evicted_tokens + num_rolled_tokens
                ].clone()
                # Insert the new keys/values at the end
                local_end_index = (
                    kv_cache["local_end_index"].item()
                    + current_end
                    - kv_cache["global_end_index"].item()
                    - num_evicted_tokens
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            else:
                # Assign new keys/values directly up to current_end
                local_end_index = (
                    kv_cache["local_end_index"].item() + current_end - kv_cache["global_end_index"].item()
                )
                local_start_index = local_end_index - num_new_tokens
                kv_cache["k"][:, local_start_index:local_end_index] = roped_key
                kv_cache["v"][:, local_start_index:local_end_index] = v
            x = attention(
                roped_query,
                kv_cache["k"][:, max(0, local_end_index - self.max_attention_size) : local_end_index],
                kv_cache["v"][:, max(0, local_end_index - self.max_attention_size) : local_end_index],
            )
            kv_cache["global_end_index"].fill_(current_end)
            kv_cache["local_end_index"].fill_(local_end_index)

        # output
        x = x.flatten(2)
        x = self.o(x)
        return x


class CausalWanAttentionBlock(nn.Module):
    def __init__(
        self,
        cross_attn_type,
        dim,
        ffn_dim,
        num_heads,
        local_attn_size=-1,
        sink_size=0,
        qk_norm=True,
        cross_attn_norm=False,
        eps=1e-6,
    ):
        super().__init__()
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.num_heads = num_heads
        self.local_attn_size = local_attn_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps

        # layers
        self.norm1 = WanLayerNorm(dim, eps)
        self.self_attn = CausalWanSelfAttention(dim, num_heads, local_attn_size, sink_size, qk_norm, eps)
        self.norm3 = WanLayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()
        self.cross_attn = WAN_CROSSATTENTION_CLASSES[cross_attn_type](dim, num_heads, (-1, -1), qk_norm, eps)
        self.norm2 = WanLayerNorm(dim, eps)
        self.ffn = nn.Sequential(nn.Linear(dim, ffn_dim), nn.GELU(approximate="tanh"), nn.Linear(ffn_dim, dim))

        # modulation
        # self.modulation = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)  # [1, 6, 3072]
        self.modulation = ModulationAdd(dim, 6)  # [1, 6, 3072]

    def forward(
        self,
        x,
        e,
        seq_lens,
        grid_sizes,
        freqs,
        context,
        context_lens,
        block_mask,
        kv_cache=None,
        crossattn_cache=None,
        current_start=0,
        cache_start=None,
    ):
        r"""
        Args:
            x(Tensor): Shape [B, L, C] [4, 32760, 1536]
            e(Tensor): Shape [B, F, 6, C] [4, 21, 6, 1536]
            seq_lens(Tensor): Shape [B], length of each sequence in batch
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        e = (self.modulation.unsqueeze(1) + e).chunk(6, dim=2)
        # assert e[0].dtype == torch.float32

        # self-attention
        y = self.self_attn(
            (self.norm1(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0]).flatten(1, 2),
            seq_lens,
            grid_sizes,
            freqs,
            block_mask,
            kv_cache,
            current_start,
            cache_start,
        )

        # with amp.autocast(dtype=torch.float32):
        x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[2]).flatten(1, 2)

        # cross-attention & ffn function
        def cross_attn_ffn(x, context, context_lens, e, crossattn_cache=None):
            x = x + self.cross_attn(self.norm3(x), context, context_lens, crossattn_cache=crossattn_cache)
            y = self.ffn(
                (self.norm2(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[4]) + e[3]).flatten(1, 2)
            )
            # with amp.autocast(dtype=torch.float32):
            x = x + (y.unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * e[5]).flatten(1, 2)
            return x

        x = cross_attn_ffn(x, context, context_lens, e, crossattn_cache)
        return x


class CausalHead(nn.Module):
    def __init__(self, dim, out_dim, patch_size, eps=1e-6):
        super().__init__()
        self.dim = dim
        self.out_dim = out_dim
        self.patch_size = patch_size
        self.eps = eps

        # layers
        out_dim = math.prod(patch_size) * out_dim
        self.norm = WanLayerNorm(dim, eps)
        self.head = nn.Linear(dim, out_dim)

        # modulation
        self.modulation = nn.Parameter(torch.randn(1, 2, dim) / dim**0.5)

    def forward(self, x, e):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            e(Tensor): Shape [B, F, 1, C]
        """
        # assert e.dtype == torch.float32
        # with amp.autocast(dtype=torch.float32):
        num_frames, frame_seqlen = e.shape[1], x.shape[1] // e.shape[1]
        e = (self.modulation.unsqueeze(1) + e).chunk(2, dim=2)
        x = self.head(self.norm(x).unflatten(dim=1, sizes=(num_frames, frame_seqlen)) * (1 + e[1]) + e[0])
        return x


class CausalWanModel(ModelMixin, ConfigMixin):
    r"""
    Wan diffusion backbone supporting both text-to-video and image-to-video, text-to-audio.
    """

    ignore_for_config = ["patch_size", "cross_attn_norm", "qk_norm", "text_dim", "window_size"]
    _no_split_modules = ["WanAttentionBlock"]

    @register_to_config
    def __init__(
        self,
        model_type="t2v",
        patch_size=(1, 2, 2),
        text_len=512,
        in_dim=16,
        dim=2048,
        ffn_dim=8192,
        freq_dim=256,
        text_dim=4096,
        additional_emb_dim=None,
        additional_emb_length=None,
        out_dim=16,
        num_heads=16,
        num_layers=32,
        local_attn_size=-1,
        sink_size=0,
        window_size=(-1, -1),
        qk_norm=True,
        cross_attn_norm=True,
        gradient_checkpointing=False,
        temporal_rope_scaling_factor=1.0,
        eps=1e-6,
        **kwargs,
    ):
        r"""
        Initialize the diffusion model backbone.

        Args:
            model_type (`str`, *optional*, defaults to 't2v'):
                Model variant - 't2v' (text-to-video) or 'i2v' (image-to-video)
            patch_size (`tuple`, *optional*, defaults to (1, 2, 2)):
                3D patch dimensions for video embedding (t_patch, h_patch, w_patch)
            text_len (`int`, *optional*, defaults to 512):
                Fixed length for text embeddings
            in_dim (`int`, *optional*, defaults to 16):
                Input video channels (C_in)
            dim (`int`, *optional*, defaults to 2048):
                Hidden dimension of the transformer
            ffn_dim (`int`, *optional*, defaults to 8192):
                Intermediate dimension in feed-forward network
            freq_dim (`int`, *optional*, defaults to 256):
                Dimension for sinusoidal time embeddings
            text_dim (`int`, *optional*, defaults to 4096):
                Input dimension for text embeddings
            out_dim (`int`, *optional*, defaults to 16):
                Output video channels (C_out)
            num_heads (`int`, *optional*, defaults to 16):
                Number of attention heads
            num_layers (`int`, *optional*, defaults to 32):
                Number of transformer blocks
            window_size (`tuple`, *optional*, defaults to (-1, -1)):
                Window size for local attention (-1 indicates global attention)
            qk_norm (`bool`, *optional*, defaults to True):
                Enable query/key normalization
            cross_attn_norm (`bool`, *optional*, defaults to False):
                Enable cross-attention normalization
            eps (`float`, *optional*, defaults to 1e-6):
                Epsilon value for normalization layers
        """

        super().__init__()

        assert model_type in [
            "t2v",
            "i2v",
            "t2a",
            "tt2a",
            "ti2v",
        ]  ## tt2a means text transcript + text description to audio (to support both TTS and T2A
        self.model_type = model_type
        is_audio_type = "a" in self.model_type
        is_video_type = "v" in self.model_type
        assert is_audio_type ^ is_video_type, "Either audio or video model should be specified"
        if is_audio_type:
            ## audio model
            assert len(patch_size) == 1 and patch_size[0] == 1, (
                "Audio model should only accept 1 dimensional input, and we dont do patchify"
            )

        self.patch_size = patch_size
        self.text_len = text_len
        self.in_dim = in_dim
        self.dim = dim
        self.ffn_dim = ffn_dim
        self.freq_dim = freq_dim
        self.text_dim = text_dim
        self.out_dim = out_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.local_attn_size = local_attn_size
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.cross_attn_norm = cross_attn_norm
        self.eps = eps
        self.temporal_rope_scaling_factor = temporal_rope_scaling_factor
        self.is_audio_type = is_audio_type
        self.is_video_type = is_video_type
        # embeddings
        if is_audio_type:
            ## hardcoded to MMAudio
            self.patch_embedding = nn.Sequential(
                ChannelLastConv1d(in_dim, dim, kernel_size=7, padding=3),
                nn.SiLU(),
                ConvMLP(dim, dim * 4, kernel_size=7, padding=3),
            )
        else:
            self.patch_embedding = nn.Conv3d(in_dim, dim, kernel_size=patch_size, stride=patch_size)

        self.text_embedding = nn.Sequential(nn.Linear(text_dim, dim), nn.GELU(approximate="tanh"), nn.Linear(dim, dim))

        self.time_embedding = nn.Sequential(nn.Linear(freq_dim, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.time_projection = nn.Sequential(nn.SiLU(), nn.Linear(dim, dim * 6))
        self.use_sp = get_sequence_parallel_state()  # seq parallel
        if self.use_sp:
            self.sp_size = nccl_info.sp_size
            self.sp_rank = nccl_info.rank_within_group
            assert self.num_heads % self.sp_size == 0, (
                f"Num heads {self.num_heads} must be divisible by sp_size {self.sp_size}"
            )
        # blocks
        ## so i2v and tt2a share the same cross attention while t2v and t2a share the same cross attention
        cross_attn_type = "t2v_cross_attn" if model_type in ["t2v", "t2a", "ti2v"] else "i2v_cross_attn"

        if cross_attn_type == "t2v_cross_attn":
            assert additional_emb_dim is None and additional_emb_length is None, (
                "additional_emb_length should be None for t2v and t2a model"
            )
        else:
            assert additional_emb_dim is not None and additional_emb_length is not None, (
                "additional_emb_length should be specified for i2v and tt2a model"
            )

        self.blocks = nn.ModuleList(
            [
                CausalWanAttentionBlock(
                    cross_attn_type, dim, ffn_dim, num_heads, local_attn_size, sink_size, qk_norm, cross_attn_norm, eps
                )
                for _ in range(num_layers)
            ]
        )

        # head
        self.head = CausalHead(dim, out_dim, patch_size, eps)

        self.set_gradient_checkpointing(enable=gradient_checkpointing)
        self.set_rope_params()

        if model_type in ["i2v", "tt2a"]:
            self.img_emb = MLPProj(additional_emb_dim, dim)

        # initialize weights
        self.init_weights()

        self.gradient_checkpointing = False
        self.block_mask = None
        self.num_frame_per_block = 1
        self.independent_first_frame = False

    def set_rope_params(self):
        # buffers (don't use register_buffer otherwise dtype will be changed in to())
        dim = self.dim
        num_heads = self.num_heads
        assert (dim % num_heads) == 0 and (dim // num_heads) % 2 == 0
        d = dim // num_heads

        if self.is_audio_type:
            ## to be determined
            # self.freqs = rope_params(1024, d, freqs_scaling=temporal_rope_scaling_factor)
            self.freqs = rope_params(1024, d - 4 * (d // 6), freqs_scaling=self.temporal_rope_scaling_factor)
        else:
            self.freqs = torch.cat(
                [
                    rope_params(1024, d - 4 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                    rope_params(1024, 2 * (d // 6)),
                ],
                dim=1,
            )

    def set_gradient_checkpointing(self, enable: bool):
        self.gradient_checkpointing = enable

    # def forward(self, *args, **kwargs):
    #     if kwargs.get("kv_cache", None) is not None:
    #         return self._forward_inference(*args, **kwargs)
    #     else:
    #         return self._forward_train(*args, **kwargs)

    @staticmethod
    def _prepare_blockwise_causal_attn_mask(
        device: torch.device | str,
        num_frames: int = 21,
        frame_seqlen: int = 1560,
        num_frame_per_block=1,
        local_attn_size=-1,
    ) -> BlockMask:
        """
        we will divide the token sequence into the following format
        [1 latent frame] [1 latent frame] ... [1 latent frame]
        We use flexattention to construct the attention mask
        """
        total_length = num_frames * frame_seqlen

        # we do right padding to get to a multiple of 128
        padded_length = math.ceil(total_length / 128) * 128 - total_length

        ends = torch.zeros(total_length + padded_length, device=device, dtype=torch.long)

        # Block-wise causal mask will attend to all elements that are before the end of the current chunk
        frame_indices = torch.arange(start=0, end=total_length, step=frame_seqlen * num_frame_per_block, device=device)

        for tmp in frame_indices:
            ends[tmp : tmp + frame_seqlen * num_frame_per_block] = tmp + frame_seqlen * num_frame_per_block

        def attention_mask(b, h, q_idx, kv_idx):
            if local_attn_size == -1:
                return (kv_idx < ends[q_idx]) | (q_idx == kv_idx)
            else:
                return ((kv_idx < ends[q_idx]) & (kv_idx >= (ends[q_idx] - local_attn_size * frame_seqlen))) | (
                    q_idx == kv_idx
                )
            # return ((kv_idx < total_length) & (q_idx < total_length))  | (q_idx == kv_idx) # bidirectional mask

        block_mask = create_block_mask(
            attention_mask,
            B=None,
            H=None,
            Q_LEN=total_length + padded_length,
            KV_LEN=total_length + padded_length,
            _compile=False,
            device=device,
        )

        import torch.distributed as dist

        if not dist.is_initialized() or dist.get_rank() == 0:
            print(f" cache a block wise causal mask with block size of {num_frame_per_block} frames")
            print(block_mask)

        return block_mask

    def prepare_transformer_block_kwargs(
        self, x, t, context, seq_len, clip_fea=None, y=None, first_frame_is_clean=False, train_mode=False
    ):
        # params
        ## need to change!
        device = next(self.patch_embedding.parameters()).device
        # device = self.patch_embedding.weight.device
        if self.freqs.device != device:
            self.freqs = self.freqs.to(device)

        if train_mode:
            # Construct blockwise causal attn mask
            if self.is_audio_type:  # [4, 150, 20]
                frame_seqlen = 1
                num_frames = x.shape[1] // frame_seqlen
            else:  # [4, 48, 30, 32, 62]
                frame_seqlen = x.shape[-2] * x.shape[-1] // (self.patch_size[1] * self.patch_size[2])
                num_frames = x.shape[2]

            if self.block_mask is None:
                self.block_mask = self._prepare_blockwise_causal_attn_mask(
                    device,
                    num_frames=num_frames,
                    frame_seqlen=frame_seqlen,
                    num_frame_per_block=self.num_frame_per_block,
                    local_attn_size=self.local_attn_size,
                )

        if y is not None:
            x = [torch.cat([u, v], dim=0) for u, v in zip(x, y)]

        # embeddings
        x = [self.patch_embedding(u.unsqueeze(0)) for u in x]  ## x is list of [B L D] or [B C F H W]

        if self.is_audio_type:
            # [B, 1]
            grid_sizes = torch.stack([torch.tensor(u.shape[1:2], dtype=torch.long) for u in x])
        else:
            # [B, 3]
            grid_sizes = torch.stack([torch.tensor(u.shape[2:], dtype=torch.long) for u in x])
            x = [u.flatten(2).transpose(1, 2) for u in x]  # [B C F H W] -> [B (F H W) C] -> [B L C]

        seq_lens = torch.tensor([u.size(1) for u in x], dtype=torch.long)
        # assert seq_lens.max() <= seq_len, f"Sequence length {seq_lens.max()} exceeds maximum {seq_len}."

        if train_mode:
            x = torch.cat(
                [torch.cat([u, u.new_zeros(1, seq_lens[0] - u.size(1), u.size(2))], dim=1) for u in x]
            )  # [B, L, C] -> [B, L, C]
        else:
            x = torch.cat(x)

        # time embeddings
        # if t.dim() == 1:
        #     if first_frame_is_clean:
        #         t = torch.ones((t.size(0), seq_len), device=t.device, dtype=t.dtype) * t.unsqueeze(1)
        #         _first_images_seq_len = grid_sizes[:, 1:].prod(-1)
        #         for i in range(t.size(0)):
        #             t[i, : _first_images_seq_len[i]] = 0
        #         # print(f"zeroing out first {_first_images_seq_len} from t: {t.shape}, {t}")
        #     else:
        #         t = t.unsqueeze(1).expand(t.size(0), seq_len)
        # with amp.autocast("cuda", dtype=torch.bfloat16):
        #     bt = t.size(0)
        #     t = t.flatten()
        #     e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t).unflatten(0, (bt, seq_len)).bfloat16())
        #     e0 = self.time_projection(e).unflatten(2, (6, self.dim))  # [1, 26784, 6, 3072] - B, seq_len, 6, dim
        #     assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        # time embeddings
        with amp.autocast("cuda", dtype=torch.bfloat16):
            e = self.time_embedding(sinusoidal_embedding_1d(self.freq_dim, t.flatten()).type_as(x))
            e0 = self.time_projection(e).unflatten(1, (6, self.dim)).unflatten(dim=0, sizes=t.shape)
            assert e.dtype == torch.bfloat16 and e0.dtype == torch.bfloat16

        if self.use_sp:
            current_len = x.shape[1]
            # we will pad up to the next multiple of sp_size: eg. [157] -> [160]
            pad_size = (-current_len) % self.sp_size

            if pad_size > 0:
                padding = torch.zeros(x.shape[0], pad_size, x.shape[2], device=x.device, dtype=x.dtype)
                x = torch.cat([x, padding], dim=1)
                e_padding = torch.zeros(e.shape[0], pad_size, e.shape[2], device=e.device, dtype=e.dtype)
                e = torch.cat([e, e_padding], dim=1)
                e0_padding = torch.zeros(
                    e0.shape[0], pad_size, e0.shape[2], e0.shape[3], device=e0.device, dtype=e0.dtype
                )
                e0 = torch.cat([e0, e0_padding], dim=1)

            x = torch.chunk(x, self.sp_size, dim=1)[self.sp_rank]
            e = torch.chunk(e, self.sp_size, dim=1)[self.sp_rank]
            e0 = torch.chunk(e0, self.sp_size, dim=1)[self.sp_rank]

        # context
        context_lens = None
        context = self.text_embedding(
            torch.stack([torch.cat([u, u.new_zeros(self.text_len - u.size(0), u.size(1))]) for u in context])
        )

        if clip_fea is not None:
            context_clip = self.img_emb(clip_fea)  # bs x 257 x dim
            context = torch.concat([context_clip, context], dim=1)

        # arguments
        kwargs = dict(
            e=e0,
            seq_lens=seq_lens,
            grid_sizes=grid_sizes,
            freqs=self.freqs,
            context=context,
            context_lens=context_lens,
            block_mask=self.block_mask,
        )

        return x, e, kwargs

    def post_transformer_block_out(self, x, grid_sizes, e, t):
        # head
        x = self.head(x, e.unflatten(dim=0, sizes=t.shape).unsqueeze(2))

        if self.use_sp:
            x = all_gather(x, dim=1)
        # unpatchify
        if self.is_audio_type:
            ## grid_sizes is [B 1] where 1 is L,
            # converting grid_sizes from [B 1] -> [B]
            grid_sizes = [gs[0] for gs in grid_sizes]
            assert len(x) == len(grid_sizes)
            x = [u[:gs] for u, gs in zip(x, grid_sizes)]
        else:
            ## grid_sizes is [B 3] where 3 is F H w
            x = self.unpatchify(x, grid_sizes)

        return torch.stack(x)

    def forward(self, x, t, context, seq_len, clip_fea=None, y=None, first_frame_is_clean=False):
        r"""
        Forward pass through the diffusion model

        Args:
            x (List[Tensor]):
                List of input video tensors, each with shape [C_in, F, H, W]
                OR
                List of input audio tensors, each with shape [L, C_in]
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
                OR
                List of denoised audio tensors with original input shapes [L, C_in]
        """
        x, e, kwargs = self.prepare_transformer_block_kwargs(
            x=x,
            t=t,
            context=context,
            seq_len=seq_len,
            clip_fea=clip_fea,
            y=y,
            first_frame_is_clean=first_frame_is_clean,
        )

        for block in self.blocks:
            x = gradient_checkpointing(
                enabled=(torch.is_grad_enabled() and self.gradient_checkpointing), module=block, x=x, **kwargs
            )

        return self.post_transformer_block_out(x, kwargs["grid_sizes"], e, t)

    def unpatchify(self, x, grid_sizes):
        r"""
        Reconstruct video tensors from patch embeddings.

        Args:
            x (List[Tensor]):
                List of patchified features, each with shape [L, C_out * prod(patch_size)]
            grid_sizes (Tensor):
                Original spatial-temporal grid dimensions before patching,
                    shape [B, 3] (3 dimensions correspond to F_patches, H_patches, W_patches)

        Returns:
            List[Tensor]:
                Reconstructed video tensors with shape [C_out, F, H / 8, W / 8]
        """

        c = self.out_dim
        out = []
        for u, v in zip(x, grid_sizes.tolist()):
            # v is [F H w] F * H * 80, 100, it was right padded by 20.
            u = u[: math.prod(v)].view(*v, *self.patch_size, c)
            u = torch.einsum("fhwpqrc->cfphqwr", u)
            u = u.reshape(c, *[i * j for i, j in zip(v, self.patch_size)])
            out.append(u)
        # out is list of [C F H W]
        return out

    def init_weights(self):
        r"""
        Initialize model parameters using Xavier initialization.
        """

        # basic init
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # init embeddings
        if self.is_video_type:
            assert isinstance(self.patch_embedding, nn.Conv3d), (
                f"Patch embedding for video should be a Conv3d layer, got {type(self.patch_embedding)}"
            )
            nn.init.xavier_uniform_(self.patch_embedding.weight.flatten(1))
        for m in self.text_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)
        for m in self.time_embedding.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.02)

        # init output layer
        nn.init.zeros_(self.head.head.weight)
