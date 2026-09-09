#!/usr/bin/env python3
"""Compare validated M200 Cube timings with the current 310P ACLNN MatMul path."""

from __future__ import annotations

import argparse
import csv
import json
import os
from pathlib import Path
import re
import subprocess
import sys


PROJECTIONS = {
    "qkv": (4096, 2048),
    "o": (2048, 2048),
    "gate-up": (12288, 2048),
    "down": (2048, 6144),
    "lm-head": (151936, 2048),
}
M_VALUES = (1, 8, 20)
LM_HEAD_CHUNK_N = 12288
CUBE_RESULT = re.compile(
    r"name=(?P<name>[^,]+), M=(?P<m>\d+), N=(?P<n>\d+), K=(?P<k>\d+),.*?"
    r"average_ms=(?P<average_ms>[0-9.]+)")


def _bench_chunk(runtime, x, n: int, k: int, warmup: int, iterations: int) -> dict:
    import torch
    import torch.nn.functional as F
    from xlite._C import matmul_bench

    weight = torch.randn(n, k, dtype=torch.float16, device="npu:0")
    output = torch.full((x.shape[0], n), torch.nan,
                        dtype=torch.float16, device="npu:0")
    reference = F.linear(x, weight)
    torch.npu.synchronize()
    average_ns = matmul_bench(
        runtime, x, weight, output, x, weight, output,
        iterations, warmup, False, False)
    torch.npu.synchronize()
    actual_cpu = output.cpu().float()
    reference_cpu = reference.cpu().float()
    if not torch.isfinite(actual_cpu).all():
        raise AssertionError("ACLNN output contains non-finite values")
    torch.testing.assert_close(actual_cpu, reference_cpu, rtol=1e-2, atol=1e-2)
    cosine = F.cosine_similarity(
        actual_cpu.flatten(), reference_cpu.flatten(), dim=0).item()
    return {
        "average_ms": float(average_ns) / 1_000_000.0,
        "cosine": cosine,
        "max_abs_error": (actual_cpu - reference_cpu).abs().max().item(),
    }


def run_worker(name: str, m: int, n: int, k: int,
               warmup: int, iterations: int) -> None:
    import torch
    import torch_npu  # noqa: F401
    from xlite._C import Runtime

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(310)
    runtime = Runtime(0, 768)
    x = torch.randn(m, k, dtype=torch.float16, device="npu:0")

    if name != "lm-head":
        result = _bench_chunk(runtime, x, n, k, warmup, iterations)
        launches = 1
    else:
        full_chunks, tail_n = divmod(n, LM_HEAD_CHUNK_N)
        full = _bench_chunk(runtime, x, LM_HEAD_CHUNK_N, k, warmup, iterations)
        tail = (_bench_chunk(runtime, x, tail_n, k, warmup, iterations)
                if tail_n else {"average_ms": 0.0, "cosine": 1.0,
                                "max_abs_error": 0.0})
        result = {
            "average_ms": full_chunks * full["average_ms"] + tail["average_ms"],
            "cosine": min(full["cosine"], tail["cosine"]),
            "max_abs_error": max(full["max_abs_error"], tail["max_abs_error"]),
        }
        launches = full_chunks + (1 if tail_n else 0)

    print(json.dumps({
        "name": name,
        "m": m,
        "n": n,
        "k": k,
        "backend": "aclnn_310p_current",
        "warmup": warmup,
        "iterations": iterations,
        "launches_per_iteration": launches,
        **result,
    }), flush=True)


def load_cube_results(paths: list[Path]) -> dict[tuple[str, int], dict]:
    results: dict[tuple[str, int], dict] = {}
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(f"Cube report does not exist: {path}")
        for match in CUBE_RESULT.finditer(path.read_text(
                encoding="utf-8", errors="replace")):
            item = {
                "name": match.group("name"),
                "m": int(match.group("m")),
                "n": int(match.group("n")),
                "k": int(match.group("k")),
                "cube_ms": float(match.group("average_ms")),
            }
            results[(item["name"], item["m"])] = item
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cube-log", type=Path, action="append", required=True,
                        help="Existing Cube report; repeat for multiple reports")
    parser.add_argument("--report-dir", type=Path,
                        default=Path("m200_vs_aclnn_report"))
    parser.add_argument("--projection", action="append", choices=PROJECTIONS,
                        help="Limit comparison to selected projections")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--worker", nargs=4,
                        metavar=("NAME", "M", "N", "K"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args.worker:
        name, m, n, k = args.worker
        run_worker(name, int(m), int(n), int(k), args.warmup, args.iterations)
        return 0
    if args.warmup < 0 or args.iterations <= 0 or args.timeout <= 0:
        parser.error("warmup must be non-negative; iterations and timeout must be positive")

    cube = load_cube_results(args.cube_log)
    selected = set(args.projection or PROJECTIONS)
    cases = [(name, m, *PROJECTIONS[name]) for name in PROJECTIONS
             for m in M_VALUES if name in selected]
    missing = [f"{name}-m{m}" for name, m, _n, _k in cases
               if (name, m) not in cube]
    if missing:
        parser.error("Cube reports are missing: " + ", ".join(missing))

    args.report_dir.mkdir(parents=True, exist_ok=True)
    results = []
    env = dict(os.environ, XLITE_TEST_FP16_ONLY="1")
    for name, m, n, k in cases:
        case = f"{name}-m{m}"
        print(f"[ RUN      ] {case} ACLNN comparison", flush=True)
        log_path = args.report_dir / f"{case}.log"
        command = [
            sys.executable, str(Path(__file__).resolve()),
            "--cube-log", str(args.cube_log[0]),
            "--worker", name, str(m), str(n), str(k),
            "--warmup", str(args.warmup),
            "--iterations", str(args.iterations),
        ]
        try:
            completed = subprocess.run(
                command, env=env, capture_output=True, text=True,
                timeout=args.timeout, check=False)
            output = completed.stdout + completed.stderr
            log_path.write_text(output, encoding="utf-8")
            status = completed.returncode
        except subprocess.TimeoutExpired as error:
            output = (error.stdout or "") + (error.stderr or "")
            log_path.write_text(output + f"\nTimeout after {args.timeout}s\n",
                                encoding="utf-8")
            status = 124

        row = {**cube[(name, m)], "exit_code": status,
               "log": str(log_path.resolve())}
        if status == 0:
            payload = next(json.loads(line) for line in reversed(output.splitlines())
                           if line.startswith("{"))
            row.update({"aclnn_ms": payload["average_ms"],
                        "speedup": payload["average_ms"] / row["cube_ms"],
                        "cosine": payload["cosine"],
                        "max_abs_error": payload["max_abs_error"]})
            print(f"[       OK ] {case}: Cube={row['cube_ms']:.6f} ms, "
                  f"ACLNN={row['aclnn_ms']:.6f} ms, speedup={row['speedup']:.3f}x",
                  flush=True)
        else:
            print(f"[  FAILED  ] {case} (recorded; continuing)", flush=True)
        results.append(row)
        (args.report_dir / "summary.json").write_text(
            json.dumps(results, indent=2), encoding="utf-8")

    fields = ["name", "m", "n", "k", "cube_ms", "aclnn_ms", "speedup",
              "cosine", "max_abs_error", "exit_code", "log"]
    with (args.report_dir / "summary.csv").open(
            "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    failures = [row for row in results if row["exit_code"] != 0]
    print(f"\nM200 vs ACLNN summary: {len(results) - len(failures)} passed, "
          f"{len(failures)} failed")
    if failures:
        print("\nAGGREGATED FAILURES")
        for row in failures:
            print(f"\n[{row['name']}-m{row['m']}] log={row['log']}")
            print(Path(row["log"]).read_text(encoding="utf-8", errors="replace"))
    print(f"Reports: {args.report_dir.resolve()}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
