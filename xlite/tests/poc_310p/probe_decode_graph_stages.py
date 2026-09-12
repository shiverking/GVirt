#!/usr/bin/env python3
"""Isolate 310P Model-RI compatibility for the ASR decoder stages."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


CASES = ("m200_matmul", "layer_pre", "layer_post")


def _rope_table(torch, length: int = 2048):
    indices = torch.arange(0, 128, 2, dtype=torch.float32, device="npu:0")
    inv_freq = 1.0 / (10000.0 ** (indices / 128.0))
    positions = torch.arange(length, dtype=torch.float32, device="npu:0")
    angles = torch.outer(positions, inv_freq)
    return torch.cat((angles.cos(), angles.sin()), dim=-1).to(torch.float16)


def _worker(case: str, iterations: int) -> int:
    import torch
    import torch_npu  # noqa: F401
    from xlite._C import Runtime, get_build_info, probe_decode_graph_310p

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(310)
    runtime = Runtime(0, 768)
    runtime.set_matmul_backend_310p("m200_asr")

    def rand(*shape):
        return torch.randn(*shape, dtype=torch.float16, device="npu:0") * 0.02

    if case == "m200_matmul":
        tensors = [rand(1, 2048), rand(4096, 2048),
                   torch.full((1, 4096), torch.nan, dtype=torch.float16,
                              device="npu:0")]
        output = tensors[-1]
    elif case == "layer_pre":
        qkv = torch.full((1, 4096), torch.nan, dtype=torch.float16,
                         device="npu:0")
        tensors = [
            rand(1, 2048), rand(4096, 2048), torch.ones(2048, dtype=torch.float16,
                                                       device="npu:0"),
            torch.ones(1024, dtype=torch.float16, device="npu:0"), qkv,
            torch.zeros(2, 128, 8, 128, dtype=torch.float16, device="npu:0"),
            torch.zeros(2, 128, 8, 128, dtype=torch.float16, device="npu:0"),
            torch.zeros((1, 1), dtype=torch.int64, device="npu:0"),
            _rope_table(torch),
            torch.zeros((1, 1), dtype=torch.int32, device="npu:0"),
        ]
        output = qkv
    else:
        output = torch.full((1, 2048), torch.nan, dtype=torch.float16,
                            device="npu:0")
        tensors = [
            rand(1, 2048), rand(2048, 2048), rand(1, 2048),
            torch.empty((1, 2048), dtype=torch.float16, device="npu:0"),
            torch.ones(2048, dtype=torch.float16, device="npu:0"),
            torch.empty((1, 2048), dtype=torch.float16, device="npu:0"),
            rand(12288, 2048),
            torch.empty((1, 12288), dtype=torch.float16, device="npu:0"),
            torch.empty((1, 6144), dtype=torch.float16, device="npu:0"),
            rand(2048, 6144),
            torch.empty((1, 2048), dtype=torch.float16, device="npu:0"),
            torch.ones(2048, dtype=torch.float16, device="npu:0"), output,
        ]

    torch.npu.synchronize()
    result = dict(probe_decode_graph_310p(runtime, case, tensors, iterations))
    actual = output.float().cpu()
    if not bool(actual.isfinite().all()):
        raise AssertionError(f"{case} produced non-finite output")
    if float(actual.abs().max()) == 0.0:
        raise AssertionError(f"{case} produced an all-zero output")
    result.update({"passed": True, "device": str(torch.npu.get_device_name(0)),
                   "build_info": dict(get_build_info()),
                   "max_abs_output": float(actual.abs().max())})
    print(json.dumps(result), flush=True)
    del runtime
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", choices=("all",) + CASES, default="all")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--report-dir", type=Path,
                        default=Path("decode_graph_310p_probe_report"))
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.iterations <= 0:
        parser.error("--iterations must be positive")
    if args.worker:
        return _worker(args.case, args.iterations)

    args.report_dir.mkdir(parents=True, exist_ok=True)
    selected = CASES if args.case == "all" else (args.case,)
    results = []
    failures = []
    for case in selected:
        print(f"[ RUN      ] {case}", flush=True)
        command = [sys.executable, str(Path(__file__).resolve()), "--worker",
                   "--case", case, "--iterations", str(args.iterations)]
        completed = subprocess.run(command, text=True, capture_output=True)
        log_path = args.report_dir / f"{case}.log"
        log_path.write_text(completed.stdout + completed.stderr, encoding="utf-8")
        payload = {"case": case, "exit_code": completed.returncode,
                   "log": str(log_path)}
        if completed.returncode == 0:
            json_lines = [line for line in completed.stdout.splitlines()
                          if line.startswith("{")]
            if json_lines:
                payload.update(json.loads(json_lines[-1]))
            print(f"[       OK ] {case}", flush=True)
        else:
            payload["passed"] = False
            failures.append(payload)
            print(f"[  FAILED  ] {case} (recorded; continuing)", flush=True)
        results.append(payload)

    report = {"iterations": args.iterations, "results": results,
              "passed": not failures}
    report_path = args.report_dir / "summary.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(f"\n310P Decode Graph probe summary: "
          f"{len(results) - len(failures)} passed, {len(failures)} failed")
    print(f"Report: {report_path.resolve()}")
    if failures:
        print("\nAGGREGATED FAILURES")
        for failure in failures:
            print(f"\n## {failure['case']}\n")
            print(Path(failure["log"]).read_text(encoding="utf-8"))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
