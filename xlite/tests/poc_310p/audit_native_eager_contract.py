#!/usr/bin/env python3
"""Audit the real Ascend310P Native eager Linear storage/execution contract.

The audit never selects or resets a device.  Logical device mapping is owned
by the caller.  Every case runs in a fresh process and compares ND and
FRACTAL_NZ weights in the same process, on the same current NPU stream.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


M_VALUES = (1, 2, 4, 6, 8, 12, 16, 20)
SHAPES = (
    ("qkv", 4096, 2048),
    ("o", 2048, 2048),
    ("gate-up", 12288, 2048),
    ("down", 2048, 6144),
    ("lm-head", 151936, 2048),
)


def _format(torch_npu, tensor) -> int:
    getter = getattr(torch_npu, "get_npu_format", None)
    if getter is None:
        raise RuntimeError("torch_npu.get_npu_format is unavailable")
    return int(getter(tensor))


def _time_window(torch, call, minimum_ms: float) -> dict:
    for _ in range(3):
        call()
    torch.npu.synchronize()
    iterations = 1
    while True:
        begin = torch.npu.Event(enable_timing=True)
        end = torch.npu.Event(enable_timing=True)
        wall_begin = time.perf_counter_ns()
        begin.record()
        for _ in range(iterations):
            call()
        end.record()
        end.synchronize()
        wall_ms = (time.perf_counter_ns() - wall_begin) / 1_000_000
        device_ms = float(begin.elapsed_time(end))
        if wall_ms >= minimum_ms:
            return {
                "iterations": iterations,
                "device_ms": device_ms / iterations,
                "wall_ms": wall_ms / iterations,
                "host_gap_ms": max(0.0, wall_ms - device_ms) / iterations,
            }
        iterations = min(iterations * 2, 1 << 20)


def _run_case(m: int, n: int, k: int, minimum_ms: float) -> None:
    import torch
    import torch.nn.functional as F
    import torch_npu

    torch.manual_seed(310)
    x = torch.randn((m, k), dtype=torch.float16, device="npu")
    weight_nd = torch.randn((n, k), dtype=torch.float16, device="npu")
    weight_nz = torch_npu.npu_format_cast(weight_nd, 29)
    output_nd = F.linear(x, weight_nd)
    output_nz = F.linear(x, weight_nz)
    torch.npu.synchronize()

    formats = {
        "input": _format(torch_npu, x),
        "weight_nd": _format(torch_npu, weight_nd),
        "weight_nz": _format(torch_npu, weight_nz),
        "output_nd": _format(torch_npu, output_nd),
        "output_nz": _format(torch_npu, output_nz),
    }
    expected_formats = {
        "input": 2, "weight_nd": 2, "weight_nz": 29,
        "output_nd": 2, "output_nz": 2,
    }
    if formats != expected_formats:
        raise AssertionError(
            f"Native eager format contract changed: {formats} != {expected_formats}")
    torch.testing.assert_close(output_nz.cpu().float(), output_nd.cpu().float(),
                               rtol=1e-2, atol=1e-2)

    addresses_before = {
        "input": int(x.data_ptr()),
        "weight_nd": int(weight_nd.data_ptr()),
        "weight_nz": int(weight_nz.data_ptr()),
    }
    nd = _time_window(torch, lambda: F.linear(x, weight_nd), minimum_ms)
    nz = _time_window(torch, lambda: F.linear(x, weight_nz), minimum_ms)
    addresses_after = {
        "input": int(x.data_ptr()),
        "weight_nd": int(weight_nd.data_ptr()),
        "weight_nz": int(weight_nz.data_ptr()),
    }
    if addresses_before != addresses_after:
        raise AssertionError("persistent Native audit inputs changed address")

    task_queue = os.environ.get("TASK_QUEUE_ENABLE",
                                os.environ.get("ASCEND_LAUNCH_BLOCKING", "runtime-default"))
    print(json.dumps({
        "m": m, "n": n, "k": k,
        "formats": formats,
        "addresses_stable": True,
        "native_nd": nd,
        "native_nz": nz,
        "nz_speedup_over_nd": nd["device_ms"] / nz["device_ms"],
        "task_queue_contract": task_queue,
        "minimum_synchronized_window_ms": minimum_ms,
    }), flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path,
                        default=Path("native_eager_contract_report"))
    parser.add_argument("--minimum-ms", type=float, default=200.0)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--shape", type=int, nargs=3, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shape:
        _run_case(*args.shape, args.minimum_ms)
        return 0
    if args.minimum_ms < 200.0:
        parser.error("--minimum-ms must be at least 200 for a promotion audit")

    cases = [(f"{name}-m{m}", m, n, k)
             for name, n, k in SHAPES for m in M_VALUES]
    if args.case:
        selected = set(args.case)
        known = {case[0] for case in cases}
        unknown = sorted(selected - known)
        if unknown:
            parser.error("unknown --case: " + ", ".join(unknown))
        cases = [case for case in cases if case[0] in selected]

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for name, m, n, k in cases:
        print(f"[ RUN      ] {name} M={m} N={n} K={k}", flush=True)
        log = args.report_dir / f"{name}.log"
        with log.open("w", encoding="utf-8") as output:
            try:
                completed = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()),
                     "--shape", str(m), str(n), str(k),
                     "--minimum-ms", str(args.minimum_ms)],
                    stdout=output, stderr=subprocess.STDOUT, check=False,
                    timeout=args.timeout, env=os.environ.copy())
                status = completed.returncode
            except subprocess.TimeoutExpired:
                output.write(f"timeout after {args.timeout} seconds\n")
                status = 124
        results.append({"name": name, "m": m, "n": n, "k": k,
                        "exit_code": status, "log": str(log.resolve())})
        print(f"[ {'      OK' if status == 0 else ' FAILED '} ] {name}", flush=True)
    summary = args.report_dir / "summary.json"
    summary.write_text(json.dumps(results, indent=2), encoding="utf-8")
    failed = [item for item in results if item["exit_code"]]
    print(f"\nNative eager contract summary: {len(results)-len(failed)} passed, "
          f"{len(failed)} failed")
    print(f"Report: {summary.resolve()}")
    if failed:
        print("\nAGGREGATED FAILURES")
        for item in failed:
            print(f"\n## {item['name']}")
            print(Path(item["log"]).read_text(encoding="utf-8", errors="replace"))
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
