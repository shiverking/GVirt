#!/usr/bin/env python3
"""Gate-1/2 smoke tests for the Qwen3-ASR 310P vector path."""

from __future__ import annotations

import gc
import traceback
from collections.abc import Callable

import torch
import torch.nn.functional as F

from xlite._C import Runtime, add, embed, qk_rmsnorm_310p, rmsnorm, silu_and_mul


def _assert_written(value: torch.Tensor, name: str) -> None:
    if torch.isnan(value).any():
        raise AssertionError(f"{name} left the non-zero/no-op sentinel unchanged")


def _rms(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    variance = x.float().square().mean(dim=-1, keepdim=True)
    return (x.float() * torch.rsqrt(variance + eps) * weight.float()).half()


def _test_add(runtime: Runtime, device: str) -> None:
    left = torch.randn(8, 2048, dtype=torch.float16, device=device)
    right = torch.randn_like(left)
    output = torch.full_like(left, torch.nan)
    add(runtime, left, right, output)
    torch.npu.synchronize()
    _assert_written(output, "add")
    torch.testing.assert_close(output, left + right, rtol=1e-2, atol=1e-2)


def _test_embedding(runtime: Runtime, device: str) -> None:
    table = torch.randn(256, 2048, dtype=torch.float16, device=device)
    ids = torch.tensor([0, 1, 127, 255], dtype=torch.int32, device=device)
    embedded = torch.full((4, 2048), torch.nan, dtype=torch.float16, device=device)
    embed(runtime, table, ids, embedded, 0, 256)
    torch.npu.synchronize()
    _assert_written(embedded, "embedding")
    torch.testing.assert_close(embedded, F.embedding(ids.long(), table), rtol=1e-2, atol=1e-2)


def _test_rmsnorm(runtime: Runtime, device: str) -> None:
    norm_input = torch.randn(8, 2048, dtype=torch.float16, device=device)
    norm_weight = torch.randn(2048, dtype=torch.float16, device=device)
    norm_output = torch.full_like(norm_input, torch.nan)
    rmsnorm(runtime, norm_input, norm_weight, norm_output, 1e-6)
    torch.npu.synchronize()
    _assert_written(norm_output, "rmsnorm")
    torch.testing.assert_close(
        norm_output, _rms(norm_input, norm_weight), rtol=1e-2, atol=1e-2)


def _test_qk_rmsnorm(runtime: Runtime, device: str) -> None:
    n_heads, n_kv_heads, head_dim = 16, 8, 128
    qkv = torch.randn(
        8, (n_heads + 2 * n_kv_heads) * head_dim,
        dtype=torch.float16, device=device)
    q_weight = torch.randn(head_dim, dtype=torch.float16, device=device)
    k_weight = torch.randn(head_dim, dtype=torch.float16, device=device)
    qkv_output = qkv.clone()
    qkv_output[:, : (n_heads + n_kv_heads) * head_dim] = torch.nan
    qk_rmsnorm_310p(runtime, qkv, q_weight, k_weight, qkv_output, 1e-6)
    torch.npu.synchronize()
    _assert_written(qkv_output, "qk-rmsnorm")
    q, k, v = qkv.split((n_heads * head_dim, n_kv_heads * head_dim,
                         n_kv_heads * head_dim), dim=-1)
    expected = torch.cat((
        _rms(q.view(8, n_heads, head_dim), q_weight).flatten(1),
        _rms(k.view(8, n_kv_heads, head_dim), k_weight).flatten(1),
        v,
    ), dim=-1)
    torch.testing.assert_close(qkv_output, expected, rtol=1e-2, atol=1e-2)


def _test_silu_and_mul(runtime: Runtime, device: str) -> None:
    silu_input = torch.randn(8, 2 * 6144, dtype=torch.float16, device=device)
    silu_output = torch.full(
        (8, 6144), torch.nan, dtype=torch.float16, device=device)
    silu_and_mul(runtime, silu_input, silu_output)
    torch.npu.synchronize()
    _assert_written(silu_output, "silu-and-mul")
    torch.testing.assert_close(
        silu_output, F.silu(silu_input[:, :6144]) * silu_input[:, 6144:],
        rtol=1e-2, atol=1e-2)


def _run_case(
    name: str,
    test: Callable[[Runtime, str], None],
    runtime: Runtime,
    device: str,
    failures: list[tuple[str, str]],
) -> None:
    print(f"[ RUN      ] {name}", flush=True)
    try:
        test(runtime, device)
    except Exception:  # Keep running independent kernels and aggregate diagnostics.
        failures.append((name, traceback.format_exc()))
        print(f"[  FAILED  ] {name} (recorded; continuing)", flush=True)
    else:
        print(f"[       OK ] {name}", flush=True)
    finally:
        gc.collect()


def main() -> int:
    torch.npu.set_device(0)
    runtime = Runtime(0, 1024)
    device = "npu:0"
    failures: list[tuple[str, str]] = []

    cases = (
        ("add", _test_add),
        ("embedding", _test_embedding),
        ("rmsnorm", _test_rmsnorm),
        ("qk-rmsnorm", _test_qk_rmsnorm),
        ("silu-and-mul", _test_silu_and_mul),
    )
    for name, test in cases:
        _run_case(name, test, runtime, device, failures)

    print(f"\nVector kernel summary: {len(cases) - len(failures)} passed, "
          f"{len(failures)} failed")
    if failures:
        print("\n" + "=" * 80)
        print("AGGREGATED FAILURES")
        for index, (name, details) in enumerate(failures, 1):
            print(f"\n[{index}/{len(failures)}] {name}\n{'-' * 80}\n{details.rstrip()}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
