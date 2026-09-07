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
from typing import Literal


@dataclass
class Qwen3ModelArgs:
    max_batch_size: int = 8
    max_seq_len: int = 4096
    dim: int = 5120
    head_dim: int = 128
    inter_dim: int = 25600
    vocab_size: int = 151936
    n_layers: int = 64
    n_heads: int = 64
    n_kv_heads: int = 8
    norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    rope_type: str = "default"
    mrope_section: list[int] | None = None
    mrope_interleaved: bool = False
    dtype: Literal["bfloat16", "float16"] = "bfloat16"
    tie_word_embeddings: bool = False
    qkv_bias: bool = False
    qk_norm: bool = True
    model_type: str = "qwen3"

    def __post_init__(self):
        self.max_num_batched_tokens = self.max_seq_len * self.max_batch_size
        if self.head_dim is None:
            self.head_dim = self.dim // self.n_heads
