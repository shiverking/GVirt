#!/usr/bin/env python3
"""310P FP16/ND MatMul gate; isolate each shape and report every failure."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time


PROJECTIONS = (
    ("qkv", 4096, 2048),
    ("o", 2048, 2048),
    ("gate-up", 12288, 2048),
    ("down", 2048, 6144),
    ("lm-head", 151936, 2048),
)
M_VALUES = (1, 8, 20, 127, 128, 129)


def run_shape(m: int, n: int, k: int) -> None:
    import torch
    import torch_npu  # noqa: F401: register the NPU backend
    import torch.nn.functional as F
    from xlite._C import Runtime, get_310p_matmul_stats, matmul

    torch.npu.set_device(0)
    # The native ACL descriptors assume ND storage, not Torch internal NZ.
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(310)
    runtime = Runtime(0, 768)  # ACLNN workspace budget is at most 512 MiB.
    x = torch.randn(m, k, dtype=torch.float16, device="npu:0")
    weight = torch.randn(n, k, dtype=torch.float16, device="npu:0")
    output = torch.full((m, n), torch.nan, dtype=torch.float16, device="npu:0")
    reference = F.linear(x, weight)

    # Xlite owns a separate ACL stream: complete producers, including the
    # sentinel fill, before calling the native binding.
    torch.npu.synchronize()
    started = time.perf_counter()
    matmul(runtime, x, weight, output, False, False)
    torch.npu.synchronize()
    elapsed_ms = (time.perf_counter() - started) * 1000
    backend_stats = dict(get_310p_matmul_stats(runtime))

    actual = output.cpu().float()
    expected = reference.cpu().float()
    if not torch.isfinite(actual).all():
        raise AssertionError(
            f"MatMul non-finite output: {int((~torch.isfinite(actual)).sum())}/{actual.numel()}")
    if not torch.isfinite(expected).all():
        raise AssertionError("torch_npu reference is non-finite")
    cosine = F.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
    print(json.dumps({
        "m": m, "n": n, "k": k,
        "max_abs_error": (actual - expected).abs().max().item(),
        "cosine": cosine,
        "cold_call_ms": elapsed_ms,
        "torch_peak_allocated_bytes": torch.npu.max_memory_allocated(),
        "memory_note": "Torch allocator only; excludes native Xlite TensorPool",
        "backend_stats": backend_stats,
    }), flush=True)
    # Compare on CPU to avoid the device-side isclose double-tolerance warning.
    torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    use_m200 = m <= 20 and not (n == 151936 and m > 8)
    if use_m200:
        expected_launches = 13 if n == 151936 else 1
        if (backend_stats["m200_requests"] != 1 or
                backend_stats["m200_kernel_launches"] != expected_launches or
                backend_stats["aclnn_requests"] != 0):
            raise AssertionError(
                f"supported ASR shape did not exclusively use M200 Cube: {backend_stats}")
    elif (backend_stats["m200_requests"] != 0 or
          backend_stats["aclnn_requests"] != 1):
        raise AssertionError(
            f"shape outside the verified M200 range did not use ACLNN fallback: {backend_stats}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("matmul_310p_report"))
    parser.add_argument("--timeout", type=int, default=600, help="Seconds per shape")
    parser.add_argument("--list", action="store_true", help="List shapes without loading NPU")
    parser.add_argument(
        "--case", action="append", default=[], metavar="NAME",
        help="Run only a named case; repeat to select multiple cases (see --list)")
    parser.add_argument(
        "--rerun-failed", action="store_true",
        help="Run only failures recorded in REPORT_DIR/summary.json")
    parser.add_argument("--shape", type=int, nargs=3, metavar=("M", "N", "K"),
                        help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.shape:
        run_shape(*args.shape)
        return 0
    cases = [(f"{name}-m{m}", m, n, k)
             for name, n, k in PROJECTIONS for m in M_VALUES]
    if args.list:
        for name, m, n, k in cases:
            print(f"{name}: [{m},{k}] @ [{n},{k}].T")
        return 0
    if args.rerun_failed:
        summary_path = args.report_dir / "summary.json"
        if not summary_path.is_file():
            parser.error(f"cannot rerun failures: {summary_path} does not exist")
        try:
            previous = json.loads(summary_path.read_text(encoding="utf-8"))
            args.case.extend(
                item["name"] for item in previous if int(item["exit_code"]) != 0)
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
            parser.error(f"cannot read failures from {summary_path}: {error}")
        if not args.case:
            print(f"No failed cases recorded in {summary_path}")
            return 0
    if args.case:
        known = {name for name, _m, _n, _k in cases}
        unknown = [name for name in args.case if name not in known]
        if unknown:
            parser.error("unknown --case: " + ", ".join(unknown) + "; use --list")
        selected = set(args.case)
        cases = [case for case in cases if case[0] in selected]
    if args.timeout <= 0:
        parser.error("--timeout must be positive")
    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    env = dict(os.environ, XLITE_TEST_FP16_ONLY="1")
    for name, m, n, k in cases:
        print(f"[ RUN      ] {name} M={m} N={n} K={k}", flush=True)
        log_path = args.report_dir / f"{name}.log"
        # A failed ACLNN call can leave device state or pool allocations behind.
        # A fresh process keeps later independent shapes diagnosable.
        with log_path.open("w", encoding="utf-8") as log:
            try:
                result = subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--shape",
                     str(m), str(n), str(k)], env=env, stdout=log,
                    stderr=subprocess.STDOUT, timeout=args.timeout, check=False)
                status = result.returncode
            except subprocess.TimeoutExpired:
                log.write(f"\nTimeout after {args.timeout} seconds\n")
                status = 124
        results.append({"name": name, "m": m, "n": n, "k": k,
                        "exit_code": status, "log": str(log_path.resolve())})
        print(f"[ {'      OK' if status == 0 else ' FAILED '} ] {name}", flush=True)
        (args.report_dir / "summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")
    failures = [item for item in results if item["exit_code"] != 0]
    print(f"\nMatMul summary: {len(results) - len(failures)} passed, {len(failures)} failed")
    if failures:
        print("\nAGGREGATED FAILURES")
        for item in failures:
            print(f"\n[{item['name']}] exit={item['exit_code']} log={item['log']}")
            print(Path(item["log"]).read_text(encoding="utf-8", errors="replace"))
    print(f"Reports: {args.report_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
