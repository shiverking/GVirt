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
import argparse
import logging
import json
import os
from pathlib import Path
import subprocess
import sys
import torch
import math
import numpy as np
import warnings
from typing import Iterable
from xlite._C import Runtime, attention

logging.getLogger().setLevel(logging.INFO)

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
        (1, [15], [1]),
        (1, [126], [1]),
        (1, [127], [1]),
        (1, [128], [1]),
        (1, [511], [1]),
        (1, [2047], [1]),
        (2, [0, 127], [129, 1]),
        (2, [128, 64], [65, 129]),
        (8, [15, 31, 63, 127, 255, 511, 1023, 2047], [1] * 8),
        (20, [16 + index * 7 for index in range(20)], [1] * 20),
    ]

def case_name(batch: int, cached_lens: list[int], query_lens: list[int]) -> str:
    if batch == 1:
        return f"cache{cached_lens[0]}-query{query_lens[0]}"
    return (f"batch{batch}-cache{'_'.join(map(str, cached_lens))}"
            f"-query{'_'.join(map(str, query_lens))}")

# Keep every 310P attention shape diagnosable even when one ACLNN invocation
# fails or poisons its process.  The parent runs each shape in isolation and
# prints all failures at the end; a hidden environment variable selects the
# single child case.
parser = argparse.ArgumentParser(description="Xlite attention correctness test")
parser.add_argument("--decode-attention-backend", choices=("legacy", "paged_310p"), default="legacy")
parser.add_argument("--rerun-failed", action="store_true",
                    help="310P FP16: rerun failures from attention_310p_report/summary.json")
parser.add_argument("--async-stress-iters", type=int, default=0,
                    help="310P FP16: queue this many calls before one final synchronization")
test_args = parser.parse_args()
if test_args.decode_attention_backend == "paged_310p":
    os.environ["XLITE_TEST_FP16_ONLY"] = "1"
    models = [model for model in models if model[0] == "qwen3_asr_1.7B_TP1"]
    work = [(1, [length - 1], [1]) for length in (1, 16, 127, 128, 129, 512, 2048)]
    work += [
        (2, [2047, 0], [1, 1]),
        (8, [2047, 0, 511, 15, 128, 126, 127, 63], [1] * 8),
        (20, [2047, 0, 511, 15, 128] * 4, [1] * 20),
        (20, [0, 15, 126, 127, 128] * 4, [1] * 20),
        (4, [2047, 128, 15, 64], [1, 65, 1, 129]),
        (4, [64, 15, 128, 2047], [129, 1, 65, 1]),
    ]
    # Every new-path case checks repeated scratch reuse, not just the last output.
    if not test_args.async_stress_iters:
        test_args.async_stress_iters = 4
poc_case_index = os.getenv("XLITE_ATTENTION_CASE_INDEX")
if os.getenv("XLITE_TEST_FP16_ONLY") == "1" and poc_case_index is None:
    report_dir = Path("attention_paged_310p_report" if test_args.decode_attention_backend == "paged_310p"
                      else "attention_310p_report")
    report_dir.mkdir(parents=True, exist_ok=True)
    selected_indices = list(range(len(work)))
    if test_args.async_stress_iters:
        if test_args.async_stress_iters <= 0:
            parser.error("--async-stress-iters must be positive")
        # One ordinary decode and one batch-20 decode exercise workspace reuse
        # without turning the stress check into the full correctness sweep.
        if test_args.decode_attention_backend == "legacy":
            selected_indices = [6, 13]
    if test_args.rerun_failed:
        summary_path = report_dir / "summary.json"
        if not summary_path.is_file():
            parser.error(f"cannot rerun failures: {summary_path} does not exist")
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            failed_names = {
                item["name"] for item in previous if int(item["exit_code"]) != 0
            }
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"cannot read failures from {summary_path}: {error}")
        selected_indices = [
            index for index, (_batch, cached, query) in enumerate(work)
            if case_name(_batch, cached, query) in failed_names
        ]
        if not selected_indices:
            print(f"No failed cases recorded in {summary_path}")
            raise SystemExit(0)
    results = []
    for index in selected_indices:
        batch, cached_lens_list, query_len_list = work[index]
        current_case_name = case_name(batch, cached_lens_list, query_len_list)
        print(f"[ RUN      ] {current_case_name}", flush=True)
        log_path = report_dir / f"{current_case_name}.log"
        env = dict(os.environ, XLITE_ATTENTION_CASE_INDEX=str(index))
        if test_args.async_stress_iters:
            env["XLITE_ATTENTION_ASYNC_STRESS_ITERS"] = str(test_args.async_stress_iters)
            env["XLITE_310P_STRESS_WORKSPACE_REUSE"] = "1"
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--decode-attention-backend",
                     test_args.decode_attention_backend], env=env,
                    stdout=log, stderr=subprocess.STDOUT, timeout=1200, check=False)
                status = result.returncode
            except subprocess.TimeoutExpired:
                log.write("\nTimeout after 1200 seconds\n")
                status = 124
        results.append({"name": current_case_name, "exit_code": status,
                        "log": str(log_path.resolve())})
        print(f"[ {'      OK' if status == 0 else ' FAILED '} ] {current_case_name}", flush=True)
        (report_dir / "summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    failures = [item for item in results if item["exit_code"] != 0]
    print(f"\nAttention summary: {len(results) - len(failures)} passed, "
          f"{len(failures)} failed")
    if failures:
        print("\nAGGREGATED FAILURES")
        for item in failures:
            print(f"\n[{item['name']}] exit={item['exit_code']} log={item['log']}")
            print(Path(item["log"]).read_text(encoding="utf-8", errors="replace"))
    print(f"Reports: {report_dir.resolve()}")
    raise SystemExit(1 if failures else 0)

if poc_case_index is not None:
    try:
        selected_index = int(poc_case_index)
        work = [work[selected_index]]
    except (ValueError, IndexError):
        raise SystemExit(f"invalid XLITE_ATTENTION_CASE_INDEX={poc_case_index!r}")

torch.npu.set_device(0)
rt = Runtime(0, 768 if test_args.decode_attention_backend == "paged_310p" else 3000)
rt.set_decode_attention_backend(test_args.decode_attention_backend)

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
            k_cache_xlite = torch.full((kvcache_block_num, BLOCK_SIZE, n_kv_heads, head_dim), 123.0)
            v_cache_xlite = torch.full_like(k_cache_xlite, -123.0)

            query_lens = torch.tensor(query_len_list, dtype=torch.int32).flatten()
            cached_lens = torch.tensor(cached_lens_list, dtype=torch.int32).flatten()
            query_lens_np = np.array(query_len_list)
            query_start_loc_np = np.cumsum(query_lens_np) - query_lens_np
            query_start_loc = torch.tensor(query_start_loc_np.tolist(), dtype=torch.int32).flatten()

            # Reverse the physical allocation to ensure the backend follows each
            # request's block table instead of assuming contiguous cache storage.
            physical_blocks = np.arange(kvcache_block_num, dtype=np.uint32)[::-1]
            block_tables_array = physical_blocks.reshape(batch, max_num_blocks)
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

            total_len = clen + qlen
            num_blocks_needed = (total_len + BLOCK_SIZE - 1) // BLOCK_SIZE
            for block_idx in range(num_blocks_needed):
                seq_start = block_idx * BLOCK_SIZE
                seq_end = min((block_idx + 1) * BLOCK_SIZE, total_len)
                current_seq_len = seq_end - seq_start

                cache_block_idx = int(block_tables_array[i, block_idx])

                k_cache_xlite[cache_block_idx, :current_seq_len] = full_k[:, seq_start:seq_end]
                v_cache_xlite[cache_block_idx, :current_seq_len] = full_v[:, seq_start:seq_end]

        cache_before = (k_cache_xlite.clone(), v_cache_xlite.clone())
        torch.npu.synchronize()
        if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
            rt.set_host_attention_metadata(
                query_len_list, cached_lens_list,
                block_tables_array.reshape(-1).tolist(), max_num_blocks)
        stress_iterations = int(os.getenv("XLITE_ATTENTION_ASYNC_STRESS_ITERS", "1"))
        retained = []
        for iteration in range(stress_iterations):
            order = list(range(batch)) if iteration % 2 == 0 else list(reversed(range(batch)))
            rows = []
            for request in order:
                start = sum(query_len_list[:request])
                rows.extend(range(start, start + query_len_list[request]))
            frame_q = qkv_xlite[rows].contiguous()
            frame_ref = output_standard[rows].contiguous()
            frame_lens = [query_len_list[i] for i in order]
            frame_cached = [cached_lens_list[i] for i in order]
            frame_table = block_tables_array[order].copy()
            frame_starts = np.cumsum(frame_lens) - frame_lens
            guard = torch.full((total_query_len + 2, n_heads * head_dim), 123.0,
                               dtype=test_dtype, device="npu")
            guard[1:-1].fill_(torch.nan)
            device_meta = [torch.tensor(value, dtype=torch.int32, device="npu") for value in
                           (frame_starts.tolist(), frame_lens, frame_cached, frame_table.reshape(-1).tolist())]
            retained.append((frame_q, frame_ref, frame_lens, frame_cached, frame_table, guard, device_meta))
        torch.npu.synchronize()
        rt.reset_stats()
        for frame_q, frame_ref, frame_lens, frame_cached, frame_table, guard, device_meta in retained:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                rt.set_host_attention_metadata(frame_lens, frame_cached, frame_table.reshape(-1).tolist(), max_num_blocks)
            attention(rt, frame_q, k_cache_xlite, v_cache_xlite,
                      guard[1:-1], *device_meta, n_heads, n_kv_heads, head_dim, BLOCK_SIZE,
                      batch, max_num_blocks, enable_flash,
                      synchronize=stress_iterations == 1)
        torch.npu.synchronize()
        for frame in retained:
            reference, guard = frame[1], frame[5]
            actual = guard[1:-1]
            assert torch.isfinite(actual).all(), "nonfinite/unwritten attention output"
            assert torch.all(guard[0] == 123) and torch.all(guard[-1] == 123), "output guard overwritten"
            cosine = torch.nn.functional.cosine_similarity(reference.float().flatten(), actual.float().flatten(), dim=0)
            max_error = (reference.float() - actual.float()).abs().max()
            print(f"retained output cosine={float(cosine.cpu()):.9f} max_abs_error={float(max_error.cpu()):.6g}")
            assert float(cosine.cpu()) >= 0.999, "retained async output differs"
            offset = 0
            for length in frame[2]:
                per_request = torch.nn.functional.cosine_similarity(
                    reference[offset:offset + length].float().flatten(),
                    actual[offset:offset + length].float().flatten(), dim=0)
                assert float(per_request.cpu()) >= 0.999, f"request at row {offset} differs"
                offset += length
        output_xlite = retained[0][5][1:-1]
        assert torch.equal(k_cache_xlite, cache_before[0]), "attention modified K cache"
        assert torch.equal(v_cache_xlite, cache_before[1]), "attention modified V cache"
        if test_args.decode_attention_backend == "paged_310p":
            stats = rt.get_stats()
            decode_count = sum(length == 1 for length in query_len_list)
            assert stats["paged_decode_requests"] == stress_iterations * decode_count, stats
            assert stats["paged_decode_kernel_launches"] == stress_iterations, stats
            expected_merges = stress_iterations if max(cached_lens_list[i] + 1 for i in range(batch)
                                                       if query_len_list[i] == 1) > 512 else 0
            assert stats["paged_decode_merge_launches"] == expected_merges, stats
            assert stats["legacy_decode_requests"] == stats["decode_kv_gather_bytes"] == 0, stats
            assert stats["attention_metadata_d2h_bytes"] == 0, stats
            assert stats["stream_synchronizations"] == 0, stats
            print(f"paged_decode_stats={stats}")
            # Reuse the SAME cache allocations with a different physical ownership
            # map, after prior consumers have finished. Unused slots stay poisoned.
            k_cache_xlite.copy_(cache_before[0].flip(0))
            v_cache_xlite.copy_(cache_before[1].flip(0))
            reused_table = kvcache_block_num - 1 - block_tables_array
            reused_device_table = torch.tensor(reused_table.reshape(-1).tolist(), dtype=torch.int32, device="npu")
            reused_output = torch.full_like(output_xlite, torch.nan)
            expected_k, expected_v = k_cache_xlite.clone(), v_cache_xlite.clone()
            torch.npu.synchronize()
            rt.set_host_attention_metadata(query_len_list, cached_lens_list,
                                           reused_table.reshape(-1).tolist(), max_num_blocks)
            attention(rt, qkv_xlite, k_cache_xlite, v_cache_xlite, reused_output,
                      query_start_loc, query_lens, cached_lens, reused_device_table,
                      n_heads, n_kv_heads, head_dim, BLOCK_SIZE, batch, max_num_blocks,
                      enable_flash)
            torch.npu.synchronize()
            assert torch.isfinite(reused_output).all(), "block reuse produced nonfinite output"
            offset = 0
            for length in query_len_list:
                reuse_cosine = torch.nn.functional.cosine_similarity(
                    output_standard[offset:offset + length].float().flatten(),
                    reused_output[offset:offset + length].float().flatten(), dim=0)
                assert float(reuse_cosine.cpu()) >= 0.999, "reassigned physical block output differs"
                offset += length
            assert torch.equal(k_cache_xlite, expected_k) and torch.equal(v_cache_xlite, expected_v)
        elif stress_iterations > 1:
            stats = rt.get_stats()
            if stats["stream_synchronizations"] != 0:
                raise AssertionError(f"async attention synchronized its runtime stream: {stats}")
            if stats["aclnn_launches"] != stress_iterations * batch:
                raise AssertionError(f"unexpected async attention launch count: {stats}")
            if stats["attention_metadata_d2h_bytes"] != 0:
                raise AssertionError(f"async attention copied metadata D2H: {stats}")
            if stats["workspace_reuses"] == 0:
                raise AssertionError(f"async attention did not reuse ACLNN workspace: {stats}")
            print(f"async_attention_stats={stats}", flush=True)
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
                    per_request_cosine = []
                    request_offset = 0
                    for request_index, request_len in enumerate(query_len_list):
                        request_slice = slice(request_offset, request_offset + request_len)
                        request_cosine = torch.nn.functional.cosine_similarity(
                            output_standard[request_slice].float().flatten(),
                            output_xlite[request_slice].float().flatten(),
                            dim=0,
                        )
                        per_request_cosine.append(
                            f"request{request_index}={float(request_cosine.cpu()):.6f}"
                        )
                        request_offset += request_len
                    raise AssertionError(
                        f"attention cosine similarity {float(cosine.cpu())} < 0.999; "
                        + ", ".join(per_request_cosine)
                    )
                if test_args.decode_attention_backend == "legacy":
                    torch.testing.assert_close(output_standard, output_xlite, atol=1e-2, rtol=1e-2)
            else:
                torch.testing.assert_close(output_standard, output_xlite, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                raise
            logging.error(f'{e}')
            logging.error(f'torch_npu: {output_standard}')
            logging.error(f'xlite: {output_xlite}')

# Destroy the native runtime while Python/NPU modules are still alive.  This
# avoids misleading NoneType callback noise during interpreter finalization.
torch.npu.synchronize()
del rt
