#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Literal
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import threading
import queue
from concurrent.futures import Future
from tests.models.weight_utils import (hf_model_weights_iterator,
                                       convert_pyslice_to_tensor,
                                       load_tensor_parallel_weights,
                                       matrix_nd2nz, logger)


debug = False
world_size = 1
rank = 0

forward_backend = os.getenv("FORWARD_BACKEND", "torch_npu")
if forward_backend == "xlite":
    block_size = 128
    from xlite._C import (Runtime, ModelConfig, AttnMeta, AttnMetaV2,
                          AttnMHA, Model, RopeNeox, RopeGptj)
    import numpy as np
    from tests.models.xlite_utils import prepare_xlite_attnmeta_v2


@dataclass
class ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096
    max_num_batched_tokens: int = 4096
    dim: int = 4096
    head_dim: int = None
    inter_dim: int = 11008
    vocab_size: int = 32000
    n_layers: int = 32
    n_heads: int = 32
    n_kv_heads: int = 32
    norm_eps: float = 1e-5
    rope_theta: float = 10000.0
    rope_type: str = "default"
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    dtype: Literal["bfloat16", "float16"] = "float16"
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    qk_norm: bool = False
    model_type: str = "llama"
    # the raw model config path
    config_path: Optional[Path] = None

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.dim = dim
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        input_dtype = x.dtype
        x = x.to(torch.float32)
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(input_dtype)


class ParallelEmbedding(nn.Module):
    """
    Embedding layer with parallelism support across distributed processes.

    Args:
        vocab_size (int): Vocabulary size.
        dim (int): Embedding dimension.
    """
    def __init__(self, vocab_size: int, dim: int):
        super().__init__()
        self.vocab_size = vocab_size
        self.dim = dim
        assert vocab_size % world_size == 0, f"Vocabulary size must be divisible by world size (world_size={world_size})"
        self.part_vocab_size = (vocab_size // world_size)
        self.vocab_start_idx = rank * self.part_vocab_size
        self.vocab_end_idx = self.vocab_start_idx + self.part_vocab_size
        self.weight = nn.Parameter(torch.empty(self.part_vocab_size, self.dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for parallel embedding layer.

        Args:
            x (torch.Tensor): Input tensor containing token indices.

        Returns:
            torch.Tensor: Embedded representations.

        Raises:
            ValueError: If `world_size` is not defined.
        """
        if world_size > 1:
            mask = (x < self.vocab_start_idx) | (x >= self.vocab_end_idx)
            x = x - self.vocab_start_idx
            x[mask] = 0
        y = F.embedding(x, self.weight)
        if world_size > 1:
            y[mask] = 0
            dist.all_reduce(y)
        return y


def linear(x: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    return F.linear(x, weight, bias)


class Linear(nn.Module):
    dtype = torch.bfloat16

    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features, dtype=dtype or Linear.dtype), requires_grad=False)
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return linear(x, self.weight, self.bias)


class ColumnParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert out_features % world_size == 0, f"Output features must be divisible by world size (world_size={world_size})"
        self.part_out_features = out_features // world_size
        super().__init__(in_features, self.part_out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight, self.bias)
        return y


class RowParallelLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = False, dtype = None):
        assert in_features % world_size == 0, f"Input features must be divisible by world size (world_size={world_size})"
        self.part_in_features = in_features // world_size
        super().__init__(self.part_in_features, out_features, bias, dtype)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = linear(x, self.weight)
        if world_size > 1:
            dist.all_reduce(y)
        if self.bias is not None:
            y += self.bias
        return y


def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_cache = freqs.cos().to(torch.get_default_dtype())
    sin_cache = freqs.sin().to(torch.get_default_dtype())
    freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
    return freq_cis.to("npu")


def _mrope_axes(section: list[int], interleaved: bool, half_dim: int,
                device: torch.device) -> torch.Tensor:
    if len(section) != 3 or sum(section) != half_dim:
        raise ValueError(f"mrope_section must contain 3 entries summing to {half_dim}")
    if interleaved:
        axes = [0] * half_dim
        for index in range(1, min(3 * section[1], half_dim), 3):
            axes[index] = 1
        for index in range(2, min(3 * section[2], half_dim), 3):
            axes[index] = 2
    else:
        axes = [0] * section[0] + [1] * section[1] + [2] * section[2]
    return torch.tensor(axes, dtype=torch.long, device=device)


def apply_rotary_emb(x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                     positions: Optional[torch.Tensor] = None,
                     mrope_section: Optional[list[int]] = None,
                     mrope_interleaved: bool = False) -> torch.Tensor:
    seqlen = x.size(2) # [bsz, n_local_heads, seqlen, head_dim]
    cos_table, sin_table = freqs_cis.chunk(2, dim=-1)
    if positions is not None and positions.ndim == 2:
        if positions.shape != (3, seqlen):
            raise ValueError(f"mRoPE positions must have shape [3, {seqlen}]")
        axes = _mrope_axes(mrope_section or [], mrope_interleaved,
                           x.shape[-1] // 2, positions.device)
        pair_positions = positions.long().transpose(0, 1)[:, axes]
        pair_indices = torch.arange(x.shape[-1] // 2, device=positions.device) \
            .expand(seqlen, -1)
        cos = cos_table[pair_positions, pair_indices]
        sin = sin_table[pair_positions, pair_indices]
    else:
        if positions is not None:
            linear_positions = positions.reshape(-1).long()
            if linear_positions.numel() != seqlen:
                raise ValueError(f"positions must contain {seqlen} values")
            cos = cos_table[linear_positions]
            sin = sin_table[linear_positions]
        else:
            cos = cos_table[start_pos:start_pos + seqlen, :]
            sin = sin_table[start_pos:start_pos + seqlen, :]
    cos = cos.repeat(1, 2) # [seqlen, head_dim]
    sin = sin.repeat(1, 2)
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    x_rot = torch.cat((-x2, x1), dim=-1)
    return (x * cos) + (x_rot * sin)


class MHA(nn.Module):
    def __init__(self, args: ModelArgs):
        super().__init__()
        self.n_heads = args.n_heads
        self.n_kv_heads = args.n_kv_heads
        self.dim = args.dim
        self.head_dim = args.head_dim
        self.qk_norm = args.qk_norm
        self.mrope_section = args.mrope_section
        self.mrope_interleaved = args.mrope_interleaved
        self.n_local_heads = args.n_heads // world_size
        self.n_local_kv_heads = max(1, args.n_kv_heads // world_size)
        self.qkv_proj = ColumnParallelLinear(args.dim, (self.n_heads + 2 * self.n_local_kv_heads * world_size) * self.head_dim, bias=args.qkv_bias)
        self.o_proj = RowParallelLinear(self.n_heads * self.head_dim, args.dim, bias=False)
        if args.qk_norm:
            self.q_norm = RMSNorm(args.head_dim, args.norm_eps)
            self.k_norm = RMSNorm(args.head_dim, args.norm_eps)
        # Keep reference caches even for the xlite backend.  The 310P POC uses
        # them to compare the same weights against the torch_npu implementation.
        self.register_buffer("k_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), persistent=False)
        self.register_buffer("v_cache", torch.zeros(args.max_batch_size, args.max_seq_len, self.n_local_kv_heads, self.head_dim), persistent=False)

    def forward(self, x, start_pos: int, freqs_cis: torch.Tensor, mask: Optional[torch.Tensor],
                positions: Optional[torch.Tensor] = None):
        bsz, seqlen, _ = x.shape

        qkv = self.qkv_proj(x)
        q, k, v = qkv.split([
            self.n_local_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim,
            self.n_local_kv_heads * self.head_dim
        ], dim=2)

        q = q.view(bsz, seqlen, self.n_local_heads, self.head_dim)
        k = k.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)
        v = v.view(bsz, seqlen, self.n_local_kv_heads, self.head_dim)

        if self.qk_norm:
            q = self.q_norm(q)
            k = self.k_norm(k)

        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        q = apply_rotary_emb(q, start_pos, freqs_cis=freqs_cis, positions=positions,
                             mrope_section=self.mrope_section,
                             mrope_interleaved=self.mrope_interleaved)
        k = apply_rotary_emb(k, start_pos, freqs_cis=freqs_cis, positions=positions,
                             mrope_section=self.mrope_section,
                             mrope_interleaved=self.mrope_interleaved)
        q = q.transpose(1, 2).contiguous()
        k = k.transpose(1, 2).contiguous()

        q = q * self.head_dim ** -0.5

        self.k_cache[:bsz, start_pos:start_pos + seqlen] = k
        self.v_cache[:bsz, start_pos:start_pos + seqlen] = v

        keys = self.k_cache[:bsz, :start_pos + seqlen]
        values = self.v_cache[:bsz, :start_pos + seqlen]

        keys = keys.repeat_interleave(self.n_local_heads // self.n_local_kv_heads, dim=2)
        values = values.repeat_interleave(self.n_local_heads // self.n_local_kv_heads, dim=2)

        scores = torch.matmul(
            q.transpose(1, 2),  # [bsz, n_local_heads, seqlen, head_dim]
            keys.permute(0, 2, 3, 1)  # [bsz, n_local_heads, head_dim, seqlen+start_pos]
        ) # [bsz, n_local_heads, seqlen, seqlen+start_pos]

        if mask is not None:
            scores = scores.float() + mask.float()

        scores = torch.softmax(scores, dim=-1, dtype=torch.float).to(x.dtype)

        output = torch.matmul(
            scores,  # [bsz, n_local_heads, seqlen, seqlen+start_pos]
            values.transpose(1, 2)  # [bsz, n_local_heads, seqlen+start_pos, head_dim]
        ).transpose(1, 2).contiguous()  # [bsz, seqlen, n_local_heads, head_dim]

        output = output.reshape(bsz, seqlen, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    def __init__(self, dim: int, inter_dim: int):
        super().__init__()
        self.gate_up_proj = ColumnParallelLinear(dim, inter_dim * 2)
        self.down_proj = RowParallelLinear(inter_dim, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.gate_up_proj(x)
        y1, y3 = torch.split(y, y.shape[-1] // 2, dim=-1)
        return self.down_proj(F.silu(y1) * y3)


class Block(nn.Module):
    def __init__(self, args: ModelArgs, layer_id: int):
        super().__init__()
        self.self_attn = MHA(args)
        self.mlp = MLP(dim=args.dim, inter_dim=args.inter_dim)
        self.input_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.post_attention_layernorm = RMSNorm(args.dim, args.norm_eps)
        self.layer_id = layer_id

    def forward(self, x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                positions: Optional[torch.Tensor] = None) -> torch.Tensor:
        attn_norm_out = self.input_layernorm(x)
        if debug and rank == 0 and self.layer_id == 0:
            print(f"layer{self.layer_id} in: {attn_norm_out}")
        attn_out = self.self_attn(attn_norm_out, start_pos, freqs_cis, mask, positions)
        if debug and rank == 0 and self.layer_id == 0:
            print(f"layer{self.layer_id} after attn: {attn_out}")
        x = x + attn_out
        ffn_norm_out = self.post_attention_layernorm(x)
        ffn_out = self.mlp(ffn_norm_out)
        if debug and rank == 0 and self.layer_id == 0:
            print(f"layer{self.layer_id} after ffn: {ffn_out}")
        x = x + ffn_out
        return x


class Llama(nn.Module):
    def __init__(self, args: ModelArgs):
        global world_size, rank
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        # DP deployment: world_size = tp_size * dp_size. xlite handles DP natively
        # in C++ (self.global_rank/tp_size/dp_size); torch_npu does NOT support DP
        # (asserted in generate.py). Dense runs pure-TP: rebind to TP-local view.
        self.global_rank = rank
        self.dp_size = int(os.getenv("XLITE_DP_SIZE", "1"))
        assert world_size % self.dp_size == 0, (
            f"WORLD_SIZE ({world_size}) must be divisible by XLITE_DP_SIZE ({self.dp_size})"
        )
        self.tp_size = world_size // self.dp_size
        self.tp_rank = rank % self.tp_size
        world_size = self.tp_size
        rank = self.tp_rank
        Linear.dtype = torch.float16 if args.dtype == "float16" else torch.bfloat16
        super().__init__()
        self.args = args
        self.max_seq_len = args.max_seq_len
        self.vocab_size = args.vocab_size
        self.n_layers = args.n_layers
        self.dim = args.dim
        self.embed_tokens = ParallelEmbedding(args.vocab_size, args.dim)
        self.layers = nn.ModuleList()
        for layer_id in range(args.n_layers):
            self.layers.append(Block(args, layer_id))
        self.norm = RMSNorm(args.dim, args.norm_eps)
        self.lm_head = ColumnParallelLinear(args.dim, args.vocab_size, bias=False)
        self.freqs_cis = precompute_freqs_cis(args.head_dim, args.max_seq_len, args.rope_theta)
        self.multi_task_parallel = (os.getenv("MULTI_TASK_PARALLEL", "0") == "1")


    @torch.inference_mode()
    def forward_naive(self, tokens: torch.Tensor, start_pos: int = 0):
        _bsz, seqlen = tokens.shape
        h = self.embed_tokens(tokens)

        return self.forward_naive_with_inputs_embeds(h, start_pos)


    @torch.inference_mode()
    def forward_naive_with_inputs_embeds(self, inputs_embeds: torch.Tensor, start_pos: int = 0,
                                         return_hidden: bool = False,
                                         positions: Optional[torch.Tensor] = None):
        _bsz, seqlen, _hidden = inputs_embeds.shape
        h = inputs_embeds

        mask = None
        if seqlen > 1:
            mask = torch.full(
                (seqlen, seqlen), float("-inf"), device=inputs_embeds.device
            ).triu_(1)

        for layer in self.layers:
            h = layer(h, start_pos, self.freqs_cis, mask, positions)
        h = self.norm(h)[:, -1]
        logits = self.lm_head(h)
        if world_size > 1:
            all_logits = [torch.empty_like(logits) for _ in range(world_size)]
            dist.all_gather(all_logits, logits)
            logits = torch.cat(all_logits, dim=-1)
        return (logits, h) if return_hidden else logits


    @torch.inference_mode()
    def forward_xlite(self, tokens: torch.Tensor, start_pos: int = 0):
        """Forward using V2 device-tensor attention metadata (zero-copy path)."""
        logits = torch.empty(world_size, tokens.size(0), self.args.vocab_size // world_size, device=tokens.device)
        tokens = tokens.contiguous().view(tokens.size(0), tokens.size(1))
        batch = tokens.size(0)
        seqlen = tokens.size(1)
        logits_indices = torch.arange(batch, dtype=torch.int32, device=tokens.device) * seqlen + (seqlen - 1)
        attn_meta = self.prepare_xlite_attnmeta_v2(tokens, start_pos)
        stream = torch.npu.current_stream().npu_stream
        h = torch.empty(tokens.numel(), self.args.dim, device=tokens.device)
        self.xlite_model.forward_v2(self.xlite_rt, tokens.flatten(), attn_meta, self.xlite_kv_cache, [self.freqs_cis], h, stream)
        self.xlite_model.forward_get_logits(self.xlite_rt, h, logits_indices, logits)
        logits = logits.permute(1, 0, 2).reshape(tokens.size(0), self.args.vocab_size)
        return logits


    @torch.inference_mode()
    def forward_xlite_with_inputs_embeds(self, inputs_embeds: torch.Tensor, start_pos: int = 0,
                                         return_hidden: bool = False,
                                         positions: Optional[torch.Tensor] = None):
        if inputs_embeds.ndim != 3 or inputs_embeds.size(0) != 1:
            raise ValueError("310P LLM POC inputs_embeds must have shape [1, num_tokens, hidden_size]")
        if inputs_embeds.size(2) != self.args.dim:
            raise ValueError(
                f"inputs_embeds hidden size {inputs_embeds.size(2)} != model hidden size {self.args.dim}"
            )
        if inputs_embeds.dtype != torch.float16:
            raise ValueError("310P LLM POC inputs_embeds must use torch.float16")

        batch, seqlen, _hidden = inputs_embeds.shape
        meta_shape = torch.empty((batch, seqlen), dtype=torch.int32, device=inputs_embeds.device)
        attn_meta = self.prepare_xlite_attnmeta(meta_shape, start_pos, positions=positions)
        stream = torch.npu.current_stream().npu_stream
        hidden = torch.empty(batch * seqlen, self.args.dim, dtype=torch.float16,
                             device=inputs_embeds.device)
        self.xlite_model.forward_with_inputs_embeds(
            self.xlite_rt, inputs_embeds.contiguous().view(-1, self.args.dim), attn_meta,
            self.xlite_kv_cache, self.freqs_cis, hidden, stream
        )
        logits_indices = torch.tensor([seqlen - 1], dtype=torch.int32, device=inputs_embeds.device)
        logits = torch.empty(world_size, batch, self.args.vocab_size // world_size,
                             dtype=torch.float16, device=inputs_embeds.device)
        self.xlite_model.forward_get_logits(self.xlite_rt, hidden, logits_indices, logits)
        logits = logits.permute(1, 0, 2).reshape(batch, self.args.vocab_size)
        selected_hidden = hidden[logits_indices.long()]
        return (logits, selected_hidden) if return_hidden else logits


    @torch.inference_mode()
    def forward_with_inputs_embeds(self, inputs_embeds: torch.Tensor, start_pos: int = 0):
        if forward_backend == "xlite":
            return self.forward_xlite_with_inputs_embeds(inputs_embeds, start_pos)
        return self.forward_naive_with_inputs_embeds(inputs_embeds, start_pos)

    class PersistentThread(threading.Thread):
        def __init__(self, model, task_id):
            super().__init__(daemon=True)
            self.xlite_model = model.xlite_model
            self.task_queue = queue.Queue()
            self.xlite_rt = model.xlite_rt_list[task_id]
            self.xlite_rt.task_id = task_id

        def add_task(self, args):
            event = threading.Event()
            self.task_queue.put((args, event))
            return event

        @torch.inference_mode()
        def run(self) -> None:
            self.xlite_rt.set_current_context()
            while True:
                task = self.task_queue.get()
                args, event = task
                tokens, attn_meta, xlite_kv_cache, freqs_cis, logits_indices, logits, stream = args
                self.xlite_rt.multi_task_parallel = True
                self.xlite_model.forward_and_get_logits(self.xlite_rt, tokens, attn_meta, xlite_kv_cache, freqs_cis, logits_indices, logits, stream)
                event.set()

    @torch.inference_mode()
    def forward_xlite_parallel(self, tokens: torch.Tensor, start_pos: int = 0):
        mid = tokens.size(1) // 2
        tokens0 = tokens[:, :mid]
        tokens1 = tokens[:, mid:]
        tokens0 = tokens0.contiguous().view(tokens0.size(0), tokens0.size(1))
        tokens1 = tokens1.contiguous().view(tokens1.size(0), tokens1.size(1))
        attn_meta0 = self.prepare_xlite_attnmeta(tokens0, start_pos)
        attn_meta1 = self.prepare_xlite_attnmeta(tokens1, start_pos + mid)
        stream = torch.npu.current_stream().npu_stream
        logits0 = torch.empty(world_size, tokens0.size(0), self.args.vocab_size // world_size, device=tokens.device)
        logits1 = torch.empty(world_size, tokens1.size(0), self.args.vocab_size // world_size, device=tokens.device)
        # logits_indices: pick the last-token hidden of each sequence for the lm_head,
        # matching forward_xlite (arange(batch)*seqlen + seqlen-1).
        b0, s0 = tokens0.size(0), tokens0.size(1)
        b1, s1 = tokens1.size(0), tokens1.size(1)
        logits_indices0 = torch.arange(b0, dtype=torch.int32, device=tokens.device) * s0 + (s0 - 1)
        logits_indices1 = torch.arange(b1, dtype=torch.int32, device=tokens.device) * s1 + (s1 - 1)
        event0 = self.thread_pool[0].add_task(args = (tokens0.flatten(), attn_meta0, self.xlite_kv_cache, self.freqs_cis, logits_indices0, logits0, stream))
        event1 = self.thread_pool[1].add_task(args = (tokens1.flatten(), attn_meta1, self.xlite_kv_cache, self.freqs_cis, logits_indices1, logits1, stream))
        event0.wait()
        event1.wait()
        logits0 = logits0.permute(1, 0, 2).reshape(tokens0.size(0), self.args.vocab_size)
        logits1 = logits1.permute(1, 0, 2).reshape(tokens1.size(0), self.args.vocab_size)
        return logits1

    @torch.inference_mode()
    def forward(self, tokens: torch.Tensor, start_pos: int = 0):
        if forward_backend == "xlite":
            if not self.multi_task_parallel or tokens.size(0) * tokens.size(1) < 1024:
                self.xlite_rt.multi_task_parallel = False
                return self.forward_xlite(tokens, start_pos)

            return self.forward_xlite_parallel(tokens, start_pos)
        else:
            return self.forward_naive(tokens, start_pos)


    def load_weights(
        self,
        model_path : str,
    ) -> None:
        """ load llama weight """
        assert self.args.dim % world_size == 0, f"dim must be divisible by world_size (world_size={world_size})"
        assert self.args.n_heads % world_size == 0, f"n_heads must be divisible by world_size (world_size={world_size})"
        assert self.args.inter_dim % world_size == 0, f"inter_dim must be divisible by world_size (world_size={world_size})"
        assert self.args.vocab_size % world_size == 0, f"vocab_size must be divisible by world_size (world_size={world_size})"

        use_weight_nz = os.getenv("XLITE_WEIGHT_NZ", "1").strip().lower() not in {
            "0", "false", "no", "off"
        }
        self.xlite_weight_nz = (
            forward_backend == "xlite" and not self.args.tie_word_embeddings and use_weight_nz
        )

        q_proj_shard_size = (self.args.head_dim * self.args.n_heads // world_size)
        n_kv_heads_replicas = max(1, world_size // self.args.n_kv_heads)
        n_local_kv_heads = max(1, self.args.n_kv_heads // world_size)
        kv_proj_shard_size = (self.args.head_dim * n_local_kv_heads)
        attention_weight_specs = [
            ("q_proj", q_proj_shard_size, 0),
            ("k_proj", kv_proj_shard_size, q_proj_shard_size),
            ("v_proj", kv_proj_shard_size, q_proj_shard_size + kv_proj_shard_size),
        ]
        intermediate_size = self.args.inter_dim

        param_dict = {name if "lm_head" in name else "model." + name: param for name, param in self.named_parameters()}
        for _, param in self.named_parameters():
            param.requires_grad = False
        for name, loaded_weight in hf_model_weights_iterator(model_path):
            # Qwen3-ASR checkpoints nest the decoder below thinker.language_model.
            # Ignore the audio tower and present the decoder with ordinary Qwen3 names.
            language_model_marker = "language_model."
            if language_model_marker in name:
                name = name.split(language_model_marker, 1)[1]
            elif name.startswith("thinker.") or "audio_tower" in name or "audio_encoder" in name:
                continue
            if "rotary_emb.inv_freq" in name or "g_idx" in name:
                continue

            is_attention_weight = False
            for weight_name, shard_size, offset in attention_weight_specs:
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "qkv_proj")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.', param_name)
                    continue

                param = param_dict[param_name]
                if weight_name in ["k_proj", "v_proj"]:
                    shard_id = rank // n_kv_heads_replicas
                else:
                    shard_id = rank

                loaded_weight = loaded_weight[shard_size * shard_id: shard_size * (shard_id + 1)]
                param_slice = param.data[offset:offset + shard_size]

                if param_slice.shape != loaded_weight.shape:
                    raise ValueError(f"{param_name} model shape({param_slice.shape}) mismatch"
                                     f" checkpoint shape({loaded_weight.shape})")

                param_slice.copy_(loaded_weight)
                is_attention_weight = True
                break

            if is_attention_weight:
                continue

            is_gate_up_weight = False
            for stride_id, weight_name in enumerate(["gate_proj", "up_proj"]):
                if weight_name not in name:
                    continue

                param_name = name.replace(weight_name, "gate_up_proj")
                if param_name not in param_dict:
                    logger.warning('Loading model has no param named %s in checkpoints, bypass.', param_name)
                    continue
                param = param_dict[param_name]
                shard_size = param.shape[0] // 2

                loaded_weight = loaded_weight[shard_size * rank:shard_size * (rank + 1)]
                param_slice = param.data[shard_size * stride_id:shard_size * (stride_id + 1)]

                param_slice.data[:loaded_weight.shape[0]].copy_(loaded_weight)
                is_gate_up_weight = True
                break
            if is_gate_up_weight:
                continue

            if name not in param_dict:
                logger.warning('Loading model has no param named %s in checkpoints, bypass.', name)
                continue
            param = param_dict[name]

            if "embed_tokens" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.vocab_size,
                                             self.args.dim,
                                             name, True, False, rank, world_size)
                continue

            if "lm_head" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.dim,
                                             self.args.vocab_size,
                                             name, False, True, rank, world_size)
                continue

            if "o_proj" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             self.args.head_dim * self.args.n_heads, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            if "down_proj" in name:
                load_tensor_parallel_weights(param, loaded_weight,
                                             intermediate_size, self.args.dim,
                                             name, True, True, rank, world_size)
                continue

            loaded_weight = convert_pyslice_to_tensor(loaded_weight)
            param.copy_(loaded_weight)
            torch.npu.empty_cache()

        if self.args.tie_word_embeddings:
            self.lm_head.weight.data = self.embed_tokens.weight.data

        if self.xlite_weight_nz:
            self.lm_head.weight.data = matrix_nd2nz(self.lm_head.weight)
            for layer in self.layers:
                layer.self_attn.qkv_proj.weight.data = matrix_nd2nz(layer.self_attn.qkv_proj.weight)
                layer.self_attn.o_proj.weight.data = matrix_nd2nz(layer.self_attn.o_proj.weight)
                layer.mlp.gate_up_proj.weight.data = matrix_nd2nz(layer.mlp.gate_up_proj.weight)
                layer.mlp.down_proj.weight.data = matrix_nd2nz(layer.mlp.down_proj.weight)
            torch.npu.empty_cache()

        if forward_backend == "xlite":
            local_rank = int(os.getenv("LOCAL_RANK", "0"))
            # Runtime rank = GLOBAL rank; tp_size/dp_size explicit.
            self.xlite_rt = Runtime(local_rank, 0, self.global_rank, self.tp_size, self.dp_size)
            self.init_xlite_model(self.args)
            kv_size = self.init_xlite_kvcache(self.args)
            pool_size = self.xlite_model.get_tensor_pool_size()
            self.xlite_rt.init_tensor_pool(pool_size)

            total_model_memory = 0
            for _, param in self.named_parameters():
                memory_usage = param.element_size() * param.numel()
                total_model_memory += memory_usage
            if self.global_rank == 0:
                print(f"Memory usage: Model: {total_model_memory // 1024 // 1024} MB" +
                      f" KV Cache: {kv_size // 1024 // 1024} MB" +
                      f" Tensor pool: {pool_size} MB")

            if self.multi_task_parallel:
                self.xlite_rt_list = [self.xlite_rt, Runtime(local_rank, 0, self.global_rank, self.tp_size, self.dp_size)]
                self.xlite_rt_list[1].init_tensor_pool(pool_size)
                self.xlite_rt_list[0].peer_notify = self.xlite_rt_list[1].notify
                self.xlite_rt_list[1].peer_notify = self.xlite_rt_list[0].notify
                self.thread_pool = [self.PersistentThread(model = self, task_id = i) for i in range(2)]
                for thread in self.thread_pool:
                    thread.start()

    def init_xlite_model(self, args: ModelArgs):
        config = ModelConfig()
        config.vocab_size = args.vocab_size
        config.hidden_size = args.dim
        config.n_layers = args.n_layers
        config.n_heads = args.n_heads
        config.n_kv_heads = args.n_kv_heads
        config.head_dim = args.head_dim
        config.rope_head_dim = args.head_dim
        config.norm_eps = args.norm_eps
        config.rope_theta = args.rope_theta
        config.rope_type = RopeGptj if args.rope_type.lower() == "gptj" else RopeNeox
        config.mrope_section = args.mrope_section or []
        config.mrope_interleaved = args.mrope_interleaved
        config.softmax_scale = args.head_dim ** -0.5
        config.n_dense_layers = args.n_layers
        config.intermediate_size = args.inter_dim
        config.def_tp_size = self.tp_size
        config.def_dp_size = self.dp_size
        config.moe_ep_size = 1
        config.moe_tp_size = 1
        config.block_sizes = [block_size]
        config.max_seq_len = args.max_seq_len
        config.max_batch_size = args.max_batch_size
        config.max_num_batched_tokens = args.max_num_batched_tokens
        config.attn_type = AttnMHA
        config.weight_nz = self.xlite_weight_nz
        config.qkv_bias = args.qkv_bias
        config.qk_norm = args.qk_norm

        self.xlite_model = Model()
        self.xlite_model.embed = self.embed_tokens.weight
        self.xlite_model.norm = self.norm.weight
        self.xlite_model.head = self.lm_head.weight
        self.xlite_model.attn_norm = [layer.input_layernorm.weight for layer in self.layers]
        self.xlite_model.attn_out = [layer.self_attn.o_proj.weight for layer in self.layers]
        self.xlite_model.mha_qkv = [layer.self_attn.qkv_proj.weight for layer in self.layers]
        self.xlite_model.mlp_norm = [layer.post_attention_layernorm.weight for layer in self.layers]
        self.xlite_model.mlp_up_gate = [self.layers[i].mlp.gate_up_proj.weight for i in range(args.n_layers)]
        self.xlite_model.mlp_down = [self.layers[i].mlp.down_proj.weight for i in range(args.n_layers)]
        if args.qk_norm:
            self.xlite_model.mha_q_norm = [self.layers[i].self_attn.q_norm.weight for i in range(args.n_layers)]
            self.xlite_model.mha_k_norm = [self.layers[i].self_attn.k_norm.weight for i in range(args.n_layers)]
        if args.qkv_bias:
            self.xlite_model.mha_qkv_bias = [layer.self_attn.qkv_proj.bias for layer in self.layers]
        # init() takes the GLOBAL rank (C++ derives tp_rank/dp_rank from it).
        self.xlite_model.init(config, self.global_rank)

    def init_xlite_kvcache(self, args: ModelArgs):
        block_num = (args.max_seq_len + block_size - 1) // block_size * args.max_batch_size
        head_num = max(args.n_kv_heads // world_size, 1)
        self.xlite_kv_cache = [(torch.zeros(block_num, block_size, head_num, args.head_dim, dtype=torch.get_default_dtype(), device='npu'),
                                torch.zeros(block_num, block_size, head_num, args.head_dim, dtype=torch.get_default_dtype(), device='npu'))
                               for _ in range(args.n_layers)]
        kv_size = (block_num * head_num * block_size * (args.head_dim + args.head_dim) *
                   self.xlite_kv_cache[0][0].element_size() * args.n_layers)
        return kv_size

    def prepare_xlite_attnmeta(self, tokens: torch.Tensor, start_pos: int,
                               positions: Optional[torch.Tensor] = None):
        batch = tokens.size(0)
        seqlen = tokens.size(1)
        step = (self.args.max_seq_len + block_size - 1) // block_size
        block_num = (seqlen + start_pos + block_size - 1) // block_size
        attn_meta = AttnMeta()
        attn_meta.lens = [seqlen] * batch
        attn_meta.cached_lens = [start_pos] * batch
        batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
        block_indices = np.arange(block_num, dtype=np.uint32)
        attn_meta.block_tables_cpu = batch_indices * step + block_indices
        if positions is None:
            positions = torch.arange(start_pos, start_pos + seqlen, dtype=torch.int64,
                                     device=tokens.device).repeat(batch)
        else:
            positions = positions.to(device=tokens.device, dtype=torch.int64).contiguous()
            valid_shapes = ((batch * seqlen,), (3, batch * seqlen))
            if tuple(positions.shape) not in valid_shapes:
                raise ValueError(
                    f"positions must have shape [{batch * seqlen}] or [3, {batch * seqlen}]"
                )
        attn_meta.positions = positions
        return attn_meta

    def prepare_xlite_attnmeta_v2(self, tokens: torch.Tensor, start_pos: int):
        return prepare_xlite_attnmeta_v2(
            AttnMetaV2, tokens, start_pos, self.args.max_seq_len, [block_size]
        )
