#!/usr/bin/env python3
"""Probe the native 310P paged-attention contract used by vLLM-Ascend.

This is intentionally independent of vLLM.  It validates the two torch_npu
operators that a no-gather Xlite decode backend must reproduce or invoke:
``_npu_reshape_and_cache`` and ``_npu_paged_attention``.  The shapes are fixed
to the Qwen3-ASR-1.7B decoder supported by the 310P POC.
"""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
import tempfile
import time
import traceback
from dataclasses import dataclass

import torch
import torch_npu


BLOCK_SIZE = 128
NUM_HEADS = 16
NUM_KV_HEADS = 8
HEAD_DIM = 128
NZ_FORMAT = 29


@dataclass(frozen=True)
class ProbeCase:
    name: str
    lengths: tuple[int, ...]


def _make_cases(selection: str) -> list[ProbeCase]:
    cases = {
        "batch1": ProbeCase("batch1-kv129", (129,)),
        "batch20": ProbeCase(
            "batch20-variable-kv",
            tuple(16 + 7 * index for index in range(20)),
        ),
    }
    if selection == "all":
        return list(cases.values())
    return [cases[selection]]


def _reference(
    query: torch.Tensor,
    keys: list[torch.Tensor],
    values: list[torch.Tensor],
) -> torch.Tensor:
    outputs = []
    query_cpu = query.float().cpu().view(-1, NUM_HEADS, HEAD_DIM)
    for request, (key, value) in enumerate(zip(keys, values, strict=True)):
        # Q16/KV8 is fixed GQA 2:1 for Qwen3-ASR-1.7B.
        key = key.float().repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        value = value.float().repeat_interleave(NUM_HEADS // NUM_KV_HEADS, dim=1)
        # Xlite RoPE-and-Cache has already applied 1/sqrt(head_dim) to Q.
        scores = torch.einsum("hd,thd->ht", query_cpu[request], key)
        probs = torch.softmax(scores, dim=-1)
        outputs.append(torch.einsum("ht,thd->hd", probs, value))
    return torch.stack(outputs).reshape(len(outputs), NUM_HEADS * HEAD_DIM)


def _run_case(case: ProbeCase, warmup: int, iterations: int) -> dict[str, object]:
    generator = torch.Generator(device="cpu").manual_seed(20260909 + len(case.lengths))
    blocks_per_request = [(length + BLOCK_SIZE - 1) // BLOCK_SIZE for length in case.lengths]
    total_blocks = sum(blocks_per_request) + len(case.lengths)
    max_blocks = max(blocks_per_request)

    # Reverse the physical allocation order and leave one unused block between
    # requests.  This catches accidental assumptions that block tables are
    # contiguous or identity-mapped.
    available = list(reversed(range(total_blocks)))
    rows: list[list[int]] = []
    cursor = 0
    for needed in blocks_per_request:
        rows.append(available[cursor : cursor + needed])
        cursor += needed + 1

    block_table_cpu = torch.zeros((len(case.lengths), max_blocks), dtype=torch.int32)
    keys_cpu: list[torch.Tensor] = []
    values_cpu: list[torch.Tensor] = []
    slots: list[int] = []
    for request, (length, row) in enumerate(zip(case.lengths, rows, strict=True)):
        block_table_cpu[request, : len(row)] = torch.tensor(row, dtype=torch.int32)
        keys_cpu.append(
            torch.randn(
                (length, NUM_KV_HEADS, HEAD_DIM),
                generator=generator,
                dtype=torch.float16,
            )
        )
        values_cpu.append(
            torch.randn(
                (length, NUM_KV_HEADS, HEAD_DIM),
                generator=generator,
                dtype=torch.float16,
            )
        )
        slots.extend(
            row[token // BLOCK_SIZE] * BLOCK_SIZE + token % BLOCK_SIZE
            for token in range(length)
        )

    key = torch.cat(keys_cpu).npu()
    value = torch.cat(values_cpu).npu()
    slot_mapping = torch.tensor(slots, dtype=torch.int32, device="npu")
    cache_shape = (total_blocks, NUM_KV_HEADS * HEAD_DIM // 16, BLOCK_SIZE, 16)
    key_cache = torch_npu.empty_with_format(
        size=cache_shape, dtype=torch.float16, device="npu", acl_format=NZ_FORMAT
    )
    value_cache = torch_npu.empty_with_format(
        size=cache_shape, dtype=torch.float16, device="npu", acl_format=NZ_FORMAT
    )
    torch_npu._npu_reshape_and_cache(
        key=key,
        value=value,
        key_cache=key_cache,
        value_cache=value_cache,
        slot_indices=slot_mapping,
    )

    query_cpu = torch.randn(
        (len(case.lengths), NUM_HEADS, HEAD_DIM),
        generator=generator,
        dtype=torch.float16,
    ) * (1.0 / math.sqrt(HEAD_DIM))
    query = query_cpu.npu()
    output = torch.full_like(query, torch.nan)
    block_table = block_table_cpu.npu()
    context_lens = torch.tensor(case.lengths, dtype=torch.int32, device="npu")

    def launch() -> None:
        torch_npu._npu_paged_attention(
            query=query,
            key_cache=key_cache,
            value_cache=value_cache,
            num_kv_heads=NUM_KV_HEADS,
            num_heads=NUM_HEADS,
            # Q is pre-scaled by Xlite RoPE-and-Cache.
            scale_value=1.0,
            block_table=block_table,
            context_lens=context_lens,
            out=output,
        )

    torch.npu.synchronize()
    for _ in range(warmup):
        launch()
    torch.npu.synchronize()
    started = time.perf_counter()
    for _ in range(iterations):
        launch()
    torch.npu.synchronize()
    average_ms = (time.perf_counter() - started) * 1000.0 / iterations

    actual = output.float().cpu().reshape(len(case.lengths), NUM_HEADS * HEAD_DIM)
    expected = _reference(query_cpu, keys_cpu, values_cpu)
    cosine = float(
        torch.nn.functional.cosine_similarity(
            actual.flatten(), expected.flatten(), dim=0
        )
    )
    max_abs_error = float((actual - expected).abs().max())
    if not torch.isfinite(actual).all():
        raise AssertionError("native paged attention produced non-finite output")
    if cosine < 0.999:
        raise AssertionError(f"native paged attention cosine {cosine:.8f} < 0.999")

    return {
        "name": case.name,
        "batch": len(case.lengths),
        "lengths": case.lengths,
        "cache_shape": cache_shape,
        "cache_format": "FRACTAL_NZ",
        "non_contiguous_block_table": True,
        "average_ms": average_ms,
        "cosine": cosine,
        "max_abs_error": max_abs_error,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", choices=("all", "batch1", "batch20"), default="all")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--report", default="native_paged_attention_310p_report.json")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()

    if not args.worker:
        # A failed asynchronous NPU launch poisons the process context.  Run
        # every case in a fresh process so later results remain meaningful.
        passed: list[dict[str, object]] = []
        failed: list[dict[str, str]] = []
        with tempfile.TemporaryDirectory(prefix="xlite_native_pa_") as temp_dir:
            for case in _make_cases(args.case):
                child_report = f"{temp_dir}/{case.name}.json"
                command = [
                    sys.executable,
                    __file__,
                    "--case",
                    "batch1" if len(case.lengths) == 1 else "batch20",
                    "--warmup",
                    str(args.warmup),
                    "--iterations",
                    str(args.iterations),
                    "--report",
                    child_report,
                    "--worker",
                ]
                completed = subprocess.run(
                    command,
                    check=False,
                    text=True,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                )
                print(completed.stdout, end="")
                try:
                    with open(child_report, encoding="utf-8") as report:
                        child_payload = json.load(report)
                    passed.extend(child_payload["passed"])
                    failed.extend(child_payload["failed"])
                except (OSError, json.JSONDecodeError, KeyError):
                    failed.append(
                        {
                            "name": case.name,
                            "traceback": (
                                f"worker exited {completed.returncode} without a valid report\n"
                                f"{completed.stdout}"
                            ),
                        }
                    )

        payload = _payload(passed, failed)
        with open(args.report, "w", encoding="utf-8") as report:
            json.dump(payload, report, indent=2)
        print(
            f"\nIsolated native paged attention summary: "
            f"{len(passed)} passed, {len(failed)} failed"
        )
        print(f"Report: {args.report}")
        return 1 if failed else 0

    torch.npu.set_device(0)
    passed: list[dict[str, object]] = []
    failed: list[dict[str, str]] = []
    for case in _make_cases(args.case):
        print(f"[ RUN      ] {case.name}", flush=True)
        try:
            result = _run_case(case, args.warmup, args.iterations)
            passed.append(result)
            print(
                f"[       OK ] {case.name}: {result['average_ms']:.6f} ms, "
                f"cosine={result['cosine']:.8f}, max_abs={result['max_abs_error']:.6f}",
                flush=True,
            )
        except Exception:
            failed.append({"name": case.name, "traceback": traceback.format_exc()})
            print(f"[  FAILED  ] {case.name} (recorded; continuing)", flush=True)

    payload = _payload(passed, failed)
    with open(args.report, "w", encoding="utf-8") as report:
        json.dump(payload, report, indent=2)

    print(f"\nNative paged attention summary: {len(passed)} passed, {len(failed)} failed")
    if failed:
        print("\n" + "=" * 80 + "\nAGGREGATED FAILURES")
        for failure in failed:
            print(f"\n## {failure['name']}\n\n{failure['traceback']}")
        return 1
    return 0


def _payload(
    passed: list[dict[str, object]], failed: list[dict[str, str]]
) -> dict[str, object]:
    return {
        "soc": "Ascend310P3",
        "operator": "torch_npu._npu_paged_attention",
        "cache_writer": "torch_npu._npu_reshape_and_cache",
        "query_contract": "Xlite pre-scaled Q; paged-attention scale_value=1.0",
        "shape": {
            "q_heads": NUM_HEADS,
            "kv_heads": NUM_KV_HEADS,
            "head_dim": HEAD_DIM,
            "block_size": BLOCK_SIZE,
        },
        "passed": passed,
        "failed": failed,
    }


if __name__ == "__main__":
    raise SystemExit(main())
