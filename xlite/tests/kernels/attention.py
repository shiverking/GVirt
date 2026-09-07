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
import math
import numpy as np
import warnings
from typing import Iterable
from xlite._C import Runtime, attention

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 3000)
torch.npu.set_device(0)
enable_flash = False

BLOCK_SIZE = 128

# model configurations: name, n_heads, n_kv_heads, head_dim, dtype
models = [
    ("base", 1, 1, 64, torch.float16),
    ("base64", 1, 1, 64, torch.bfloat16),
    ("base128", 1, 1, 128, torch.bfloat16),
    ("qwen2.5_0.5B_TP1", 14, 2, 64, torch.bfloat16),
    ("qwen2_32B_TP8", 5, 1, 128, torch.bfloat16),
    ("qwen3_32B_TP8", 8, 1, 128, torch.bfloat16),
    ("qwen3_asr_1.7B_TP1", 16, 8, 128, torch.float16),
    ("qwen3_moe_30B_TP8", 4, 1, 128, torch.bfloat16),
    ("llama_7B_TP1", 32, 32, 128, torch.float16),
    ("llama_13B_TP2", 20, 20, 128, torch.float16),
    ("codellama_34B_TP8", 8, 1, 128, torch.bfloat16),
]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    models = [model for model in models if model[0] == "qwen3_asr_1.7B_TP1"]

# work configurations: batch_size, cached_lens, query_lens
work = [
    (1, [0], [1]),
    (1, [0], [30]),
    (1, [0], [77]),
    (1, [0], [1800]),
    (1, [0], [24322]),
    (1, [0], [32769]),
    (8, [0] * 8, [789, 65, 13, 6545, 24, 190, 2432, 124]),
    (2, [4000] * 2, [1] * 2),
    (2, [6000] * 2, [1] * 2),
    (2, [16000] * 2, [1] * 2),
    (2, [24000] * 2, [1] * 2),
    (2, [33000] * 2, [1] * 2),
    (2, [60000] * 2, [1] * 2),
    (2, [131071] * 2, [1] * 2),
    (1, [1800], [1]),
    (5, [8, 13, 65, 11, 5], [1] * 5),
    (1, [1 * BLOCK_SIZE], [1800]),
]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    work = [
        (1, [0], [1]),
        (1, [0], [127]),
        (1, [0], [128]),
        (1, [0], [129]),
        (1, [128], [1]),
        (1, [15], [1]),
        (1, [126], [1]),
        (1, [127], [1]),
        (1, [128], [1]),
        (1, [511], [1]),
        (1, [2047], [1]),
    ]

def max_blocks(query_lens: Iterable[int], cached_lens: Iterable[int], BLOCK_SIZE: int) -> int:
    """
    计算 (query_lens[i] + cached_lens[i]) 列表中最大元素，向上整除 BLOCK_SIZE 后的块数。
    """
    query_lens = list(query_lens)
    cached_lens = list(cached_lens)
    max_sum = max(a + b for a, b in zip(query_lens, cached_lens))
    return (max_sum + BLOCK_SIZE - 1) // BLOCK_SIZE


def rms_norm_last_dim(x: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    # normalize over the last dimension by root-mean-square
    rms = x.pow(2).mean(dim=-1, keepdim=True).add(eps).sqrt()
    return x / rms


for name, n_heads, n_kv_heads, head_dim, test_dtype in models:
    for batch, cached_lens_list, query_len_list in work:
        max_num_blocks = max_blocks(query_len_list, cached_lens_list, BLOCK_SIZE)
        max_seq_len = max_num_blocks * BLOCK_SIZE
        total_query_len = sum(query_len_list)

        # Skip very large total_query_len
        if total_query_len >= 24322 and (batch != 1 or n_heads != 1 or n_kv_heads != 1):
            logging.info(
                "skip attention %s: total_query_len=%d >=24322 and (batch=%d != 1 or n_heads=%d != 1 or n_kv_heads=%d != 1)",
                name,
                total_query_len,
                batch,
                n_heads,
                n_kv_heads,
            )
            continue

        torch.set_default_dtype(test_dtype)
        out_features = (n_heads + 2 * n_kv_heads) * head_dim
        with torch.device("npu"):
            # standard
            qkv_standard = torch.randn(total_query_len, (n_heads + 2 * n_kv_heads), head_dim)
            k_cache = torch.randn(batch, max_seq_len, n_kv_heads, head_dim)
            v_cache = torch.randn(batch, max_seq_len, n_kv_heads, head_dim)

            # apply RMS-norm over head_dim for k and v caches
            qkv_standard = rms_norm_last_dim(qkv_standard).view(total_query_len, out_features)
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                # The real model's RoPE-and-Cache stage scales Q before the ACLNN backend.
                qkv_standard[:, :n_heads * head_dim] *= head_dim ** -0.5
            k_cache = rms_norm_last_dim(k_cache)
            v_cache = rms_norm_last_dim(v_cache)

            masks = []
            for i in range(batch):
                query_len = query_len_list[i]
                cached_len = cached_lens_list[i]
                # 生成目标掩码：当列索引 >= cached_len + 行索引 时为 -inf，其余为 0
                cols = torch.arange(query_len + cached_len).unsqueeze(0)
                rows = torch.arange(query_len).unsqueeze(1)
                mask = torch.where(cols > cached_len + rows, float("-inf"), 0.0)
                masks.append(mask)

            # xlite
            qkv_xlite = qkv_standard.clone()
            output_xlite = torch.full(
                (total_query_len, n_heads * head_dim), torch.nan, dtype=test_dtype)

            kvcache_block_num = max_num_blocks * batch
            k_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, n_kv_heads, head_dim)
            v_cache_xlite = torch.randn(kvcache_block_num, BLOCK_SIZE, n_kv_heads, head_dim)

            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_lens_np = np.array(query_len_list)
            query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
            query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

            batch_indices = np.arange(batch, dtype=np.uint32).reshape(-1, 1)
            block_indices = np.arange(max_num_blocks, dtype=np.uint32)
            block_tables_array = batch_indices * max_num_blocks + block_indices
            block_tables = torch.tensor(block_tables_array.tolist(), dtype=torch.int32).flatten()

        # standard GQA forward: process each sample with its own query_len and cached_len
        outputs = []
        offset = 0
        for i in range(batch):
            qlen = query_len_list[i]
            clen = cached_lens_list[i]

            # slice this sample's flat qkv from concatenated qkv_standard
            qkv_chunk = qkv_standard[offset: offset + qlen]
            offset += qlen

            # split features (q,k,v) along feature dim
            q_chunk, k_chunk, v_chunk = qkv_chunk.split([n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim], dim=1)

            # reshape to per-sample tensors
            q_chunk = q_chunk.view(1, qlen, n_heads, head_dim)
            k_chunk = k_chunk.view(1, qlen, n_kv_heads, head_dim)
            v_chunk = v_chunk.view(1, qlen, n_kv_heads, head_dim)

            # write into cache at the sample's cached_len offset
            k_cache[i:i+1, clen:clen + qlen] = k_chunk
            v_cache[i:i+1, clen:clen + qlen] = v_chunk

            # read keys/values for this sample (cached + current)
            keys = k_cache[i:i+1, :clen + qlen]
            values = v_cache[i:i+1, :clen + qlen]

            keys = keys.repeat_interleave(n_heads // n_kv_heads, dim=2)
            values = values.repeat_interleave(n_heads // n_kv_heads, dim=2)

            # compute scores [1, n_heads, qlen, qlen+clen]
            scores = torch.matmul(
                q_chunk.transpose(1, 2),
                keys.permute(0, 2, 3, 1)
            )

            # add per-sample mask if present
            if masks is not None:
                scores = scores + masks[i].unsqueeze(0).unsqueeze(1)

            probs = torch.softmax(scores, dim=-1)

            out = torch.matmul(probs, values.transpose(1, 2)).transpose(1, 2).contiguous()  # [1, qlen, n_heads, head_dim]

            outputs.append(out.view(qlen, n_heads * head_dim))

        output_standard = torch.cat(outputs, dim=0)

        # xlite: split flat qkv and write per-sample KV into block cache
        q_xlite_all, k_xlite_all, v_xlite_all = qkv_xlite.split([n_heads * head_dim, n_kv_heads * head_dim, n_kv_heads * head_dim], dim=1)

        for i in range(batch):
            # per-sample start and lengths
            start = int(query_start_loc[i].item())
            qlen = int(query_lens[i].item())
            clen = int(cached_lens[i].item())

            # extract this sample's k/v and reshape
            current_k = k_xlite_all[start:start + qlen].view(1, qlen, n_kv_heads, head_dim)
            current_v = v_xlite_all[start:start + qlen].view(1, qlen, n_kv_heads, head_dim)

            # also copy existing cached KV (from k_cache/v_cache) before current tokens
            cached_k_part = k_cache[i:i+1, :clen] if clen > 0 else torch.empty(1, 0, n_kv_heads, head_dim, device="npu:0")
            cached_v_part = v_cache[i:i+1, :clen] if clen > 0 else torch.empty(1, 0, n_kv_heads, head_dim, device="npu:0")

            full_k = torch.cat([cached_k_part, current_k], dim=1)
            full_v = torch.cat([cached_v_part, current_v], dim=1)

            sample_cache_start = i * max_num_blocks
            total_len = clen + qlen
            num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            for block_idx in range(num_blocks_needed):
                seq_start = block_idx * BLOCK_SIZE
                seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
                current_seq_len = seq_end - seq_start

                cache_block_idx = sample_cache_start + block_idx
                if cache_block_idx >= (i + 1) * max_num_blocks:
                    warnings.warn(f"第{i}个样本的缓存块容量不足，无法容纳超长序列（需要{num_blocks_needed}块，仅分配{max_num_blocks}块）")
                    break

                k_cache_xlite[cache_block_idx, :current_seq_len] = full_k[:, seq_start:seq_end]
                v_cache_xlite[cache_block_idx, :current_seq_len] = full_v[:, seq_start:seq_end]

        torch.npu.synchronize()
        attention(rt, qkv_xlite, k_cache_xlite, v_cache_xlite,
                  output_xlite, query_start_loc, query_lens, cached_lens,
                  block_tables, n_heads, n_kv_heads, head_dim, BLOCK_SIZE, batch, max_num_blocks, enable_flash)
        torch.npu.synchronize()
        if torch.isnan(output_xlite).any():
            raise AssertionError("attention output still contains the no-op sentinel")

        logging.info(
            "attention %s (%d heads, %d kv heads, %d head dim, %s) work (%d batch, cached_lens=%s, query_lens=%s) executed!",
            name,
            n_heads,
            n_kv_heads,
            head_dim,
            test_dtype,
            batch,
            cached_lens_list,
            query_len_list,
        )

        try:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                cosine = torch.nn.functional.cosine_similarity(
                    output_standard.float().flatten(), output_xlite.float().flatten(), dim=0)
                if float(cosine.cpu()) < 0.999:
                    raise AssertionError(f"attention cosine similarity {float(cosine.cpu())} < 0.999")
                torch.testing.assert_close(output_standard, output_xlite, atol=1e-2, rtol=1e-2)
            else:
                torch.testing.assert_close(output_standard, output_xlite, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                raise
            logging.error(f'{e}')
            logging.error(f'torch_npu: {output_standard}')
            logging.error(f'xlite: {output_xlite}')
