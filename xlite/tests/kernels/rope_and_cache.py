#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import
import logging
import os
import torch
from xlite._C import Runtime, rope_and_cache

logging.getLogger().setLevel(logging.INFO)

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0):
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32, device="cpu")[: (dim // 2)] / dim))
    t = torch.arange(end, device=freqs.device)  # type: ignore
    freqs = torch.outer(t, freqs).float()  # type: ignore
    cos_cache = freqs.cos().to(torch.get_default_dtype())
    sin_cache = freqs.sin().to(torch.get_default_dtype())
    freq_cis = torch.cat((cos_cache, sin_cache), dim=-1)
    return freq_cis.to("npu")


def apply_rotary_emb(x: torch.Tensor, start_pos: int, freqs_cis: torch.Tensor) -> torch.Tensor:
    seqlen = x.size(2) # [bsz, n_local_heads, seqlen, head_dim]
    cos, sin = freqs_cis[start_pos:start_pos + seqlen, :].chunk(2, dim=-1)
    cos = cos.repeat(1, 2) # [seqlen, head_dim]
    sin = sin.repeat(1, 2)

    rotary_dim = cos.shape[-1]
    x_rot, x_pass = x[..., :rotary_dim], x[..., rotary_dim:]

    x1 = x_rot[..., :x_rot.shape[-1] // 2]
    x2 = x_rot[..., x_rot.shape[-1] // 2:]
    x_rot_half = torch.cat((-x2, x1), dim=-1)

    x_rot_embedded = (x_rot * cos) + (x_rot_half * sin)
    return torch.cat([x_rot_embedded, x_pass], dim=-1)


rt = Runtime(0, 500)
torch.npu.set_device(0)

ROPE_THETA = 10000.0
poc_fp16 = os.getenv("XLITE_TEST_FP16_ONLY") == "1"
N_HEADS = 16 if poc_fp16 else 32
N_KV_HEADS = 8 if poc_fp16 else 32

MAX_SEQ_LEN = 1024
MAX_BATCH_SIZE = 8
BATCH_SIZE = 1 if poc_fp16 else 8
SEQ_LEN = 129 if poc_fp16 else 10

START_POS = 0
BLOCK_SIZE = 128
BLOCK_NUM = (SEQ_LEN + BLOCK_SIZE - 1) // BLOCK_SIZE

test_cases = {(torch.float16, 64, 64), (torch.float16, 128, 128), (torch.float16, 128, 64),
              (torch.bfloat16, 64, 64), (torch.bfloat16, 128, 128), (torch.bfloat16, 128, 64)}
if poc_fp16:
    test_cases = {(torch.float16, 128, 128)}

for test_dtype, head_dim, rot_dim in test_cases:
    out_features = (N_HEADS + 2 * N_KV_HEADS) * head_dim
    torch.set_default_dtype(test_dtype)
    with torch.device("npu"):
        qkv_standard = torch.randn(BATCH_SIZE, SEQ_LEN, out_features)
        freqs_cis_standard = precompute_freqs_cis(rot_dim, MAX_SEQ_LEN, ROPE_THETA)

        k_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, head_dim)
        v_cache = torch.zeros(MAX_BATCH_SIZE, MAX_SEQ_LEN, N_KV_HEADS, head_dim)

        qkv_xlite = qkv_standard.clone().view(BATCH_SIZE * SEQ_LEN, out_features)
        freqs_cis_xlite = freqs_cis_standard.clone()

        k_cache_xlite = torch.full(
            (BLOCK_NUM, BLOCK_SIZE, N_KV_HEADS, head_dim), torch.nan)
        v_cache_xlite = torch.full(
            (BLOCK_NUM, BLOCK_SIZE, N_KV_HEADS, head_dim), torch.nan)

        len = torch.arange(SEQ_LEN, dtype=torch.int64)
        position = len.unsqueeze(0).repeat(BATCH_SIZE, 1)
        len = torch.arange(SEQ_LEN, dtype=torch.int32)
        slot_mapping = len.unsqueeze(0).repeat(BATCH_SIZE, 1)

    # standard
    q, k, v = qkv_standard.split([N_HEADS * head_dim, N_KV_HEADS * head_dim, N_KV_HEADS * head_dim], dim=2)

    q = q.view(BATCH_SIZE, SEQ_LEN, N_HEADS, head_dim)
    k = k.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, head_dim)
    v = v.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, head_dim)

    q = q.transpose(1, 2)
    k = k.transpose(1, 2)
    q = apply_rotary_emb(q, START_POS, freqs_cis=freqs_cis_standard)
    k = apply_rotary_emb(k, START_POS, freqs_cis=freqs_cis_standard)
    q = q.transpose(1, 2).contiguous()
    k = k.transpose(1, 2).contiguous()

    q = q * head_dim ** -0.5

    q = q.view(BATCH_SIZE * SEQ_LEN, N_HEADS * head_dim)
    k = k.view(BATCH_SIZE * SEQ_LEN, N_KV_HEADS * head_dim)
    v = v.view(BATCH_SIZE * SEQ_LEN, N_KV_HEADS * head_dim)
    qkv_standard_out = torch.cat([q, k, v], dim=1)

    # xlite
    torch.npu.synchronize()
    rope_and_cache(rt, qkv_xlite, k_cache_xlite, v_cache_xlite, position, freqs_cis_xlite, slot_mapping,
                N_HEADS, N_KV_HEADS, head_dim, rot_dim, BLOCK_SIZE, True)
    torch.npu.synchronize()

    logging.info(f'rope and cache (head_dim={head_dim}, rot_dim={rot_dim}, {test_dtype}) executed!')

    try:
        torch.testing.assert_close(qkv_standard_out, qkv_xlite, atol=1e-5, rtol=1e-3)
        # The test deliberately aliases every batch onto the same block; the last
        # batch must therefore be the final cache writer.
        expected_k_cache = k.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, head_dim)[-1]
        expected_v_cache = v.view(BATCH_SIZE, SEQ_LEN, N_KV_HEADS, head_dim)[-1]
        torch.testing.assert_close(
            expected_k_cache,
            k_cache_xlite.view(-1, N_KV_HEADS, head_dim)[:SEQ_LEN],
            atol=1e-5, rtol=1e-3)
        torch.testing.assert_close(
            expected_v_cache,
            v_cache_xlite.view(-1, N_KV_HEADS, head_dim)[:SEQ_LEN],
            atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
            raise
        logging.error(f'{e}')
        logging.error(f'torch_npu: {qkv_standard_out}')
        logging.error(f'xlite: {qkv_xlite}')
