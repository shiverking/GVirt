#!/usr/bin/env python3
"""Isolated P3 candidates, retained-output correctness, and offline policy selection."""
import argparse
import json
from pathlib import Path
import statistics
import subprocess
import sys
import time
import traceback

SHAPES = {"qkv": (4096, 2048), "o": (2048, 2048), "gate_up": (12288, 2048),
          "down": (2048, 6144), "lm_head": (151936, 2048)}


def worker(spec, diagnostics=False, profile_dir=None):
    import torch
    import torch_npu  # noqa: F401
    from xlite._C import Runtime, matmul
    from xlite.p3 import fingerprint

    torch.npu.set_device(0)
    torch.npu.config.allow_internal_format = False
    torch.manual_seed(310)
    m, n, k = spec["m"], spec["n"], spec["k"]
    rt = Runtime(0, 768)
    identity = fingerprint(rt)
    rt.set_matmul_optimization(spec["mode"])
    if spec["mode"] == "p3_aclnn":
        rt.set_matmul_plan(m, n, k, spec["chunk"], spec["direct"], True)
    # Small inputs avoid near-zero FP16 tolerance being dominated by huge sums.
    x = torch.randn(m, k, dtype=torch.float16, device="npu") * 0.1
    w = torch.randn(n, k, dtype=torch.float16, device="npu") * 0.1
    expected = torch.nn.functional.linear(x, w).cpu().float()
    guards = [torch.full((m + 2, n), 123.0, dtype=torch.float16, device="npu") for _ in range(10)]
    for guard in guards:
        guard[1:-1].fill_(torch.nan)
    torch.npu.synchronize()
    rt.reset_stats()
    # One cold warmup; 3 complete submissions, including output copies, timed in
    # one interval. Keep EVERY output alive until device completion.
    matmul(rt, x, w, guards[0][1:-1], False, False)
    torch.npu.synchronize()
    samples = []
    for sample in range(3):
        started = time.perf_counter()
        for guard in guards[1 + sample * 3:4 + sample * 3]:
            matmul(rt, x, w, guard[1:-1], False, False)
        torch.npu.synchronize()
        samples.append((time.perf_counter() - started) * 1000 / 3)
    worst_error, worst_cosine = 0.0, 1.0
    for guard in guards:
        cpu = guard.cpu().float()
        assert torch.all(cpu[0] == 123) and torch.all(cpu[-1] == 123), "output guard overwritten"
        actual = cpu[1:-1]
        assert torch.isfinite(actual).all(), "nonfinite/unwritten output"
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(), expected.flatten(), dim=0).item()
        worst_cosine = min(worst_cosine, cosine)
        worst_error = max(worst_error, (actual - expected).abs().max().item())
        assert cosine >= 0.999, f"cosine={cosine}"
        torch.testing.assert_close(actual, expected, rtol=1e-2, atol=1e-2)
    normal_stats = dict(rt.get_stats())
    if diagnostics:
        rt.reset_stats()
        rt.matmul_diagnostics = True
        matmul(rt, x, w, guards[0][1:-1], False, False)
        torch.npu.synchronize()
    result = {"spec": spec, "fingerprint": identity, "status": "passed",
              "median_ms": statistics.median(samples), "samples_ms": samples,
              "cosine": worst_cosine, "max_abs_error": worst_error,
              "runtime_stats": normal_stats,
              "diagnostic_stats": dict(rt.get_stats()) if diagnostics else None}
    if profile_dir:
        rt.matmul_diagnostics = False
        with torch_npu.profiler.profile(
            activities=[torch_npu.profiler.ProfilerActivity.CPU, torch_npu.profiler.ProfilerActivity.NPU],
            on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(str(profile_dir)),
            record_shapes=True,
        ) as profiler:
            for _ in range(3):
                matmul(rt, x, w, guards[0][1:-1], False, False)
                profiler.step()
            torch.npu.synchronize()
    del rt
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report-dir", type=Path, default=Path("p3_matmul_report"))
    parser.add_argument("--rerun-failed", action="store_true")
    parser.add_argument("--diagnostics", action="store_true", help="extra timing pass, excluded from selection")
    parser.add_argument("--profile", action="store_true", help="short trace for anchor LM Head candidates")
    parser.add_argument("--profile-selected", action="store_true",
                        help="profile only legacy and selected LM Head at M1/20; keep existing policy")
    parser.add_argument("--worker", help=argparse.SUPPRESS)
    parser.add_argument("--output", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        spec = json.loads(args.worker)
        try:
            result = worker(spec, args.diagnostics,
                            args.output.with_suffix(".trace") if args.profile else None)
        except Exception:
            result = {"spec": spec, "status": "failed", "error": traceback.format_exc()}
            traceback.print_exc()
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return 0 if result["status"] == "passed" else 1

    root = args.report_dir
    root.mkdir(parents=True, exist_ok=True)
    if args.profile_selected:
        policy = json.loads((root / "policy.json").read_text())
        diagnostic_root = root / "selected_profile"
        diagnostic_root.mkdir(exist_ok=True)
        failed = 0
        for m in (1, 20):
            entry = next(e for e in policy["entries"] if e["m"] == m and e["n"] == 151936)
            specs = [{"m": m, "n": 151936, "k": 2048, "chunk": 12288, "direct": False, "mode": "legacy"}]
            if entry["enabled"]:
                specs.append({**{key: entry[key] for key in ("m", "n", "k", "chunk", "direct")},
                              "mode": "p3_aclnn"})
            for spec in specs:
                output = diagnostic_root / f"lm_head-m{m}-{spec['mode']}.json"
                print(f"[PROFILE] {output.stem}", flush=True)
                with output.with_suffix(".log").open("w") as log:
                    command = [sys.executable, __file__, "--worker", json.dumps(spec), "--output", str(output),
                               "--diagnostics", "--profile"]
                    try:
                        completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                        failed += completed.returncode != 0
                    except subprocess.TimeoutExpired:
                        failed += 1
        return 1 if failed else 0
    (root / "policy.json").unlink(missing_ok=True)
    results = {}

    def run(m, shape, chunk=12288, direct=False, mode="p3_aclnn", phase="validation"):
        n, k = SHAPES[shape]
        spec = {"m": m, "n": n, "k": k, "chunk": chunk, "direct": direct, "mode": mode}
        name = f"{shape}-m{m}-{mode}-n{chunk}-{'direct' if direct else 'packed'}"
        if name in results:
            return results[name]
        output = root / f"{name}.json"
        previous = json.loads(output.read_text()) if output.exists() and args.rerun_failed else None
        if previous and previous.get("status") == "passed":
            result = previous
            print(f"[REUSE] {name}", flush=True)
        else:
            print(f"[{phase}] {name}", flush=True)
            output.unlink(missing_ok=True)
            command = [sys.executable, __file__, "--worker", json.dumps(spec), "--output", str(output)]
            if args.diagnostics:
                command.append("--diagnostics")
            if args.profile and phase == "screen":
                command.append("--profile")
            with (root / f"{name}.log").open("w", encoding="utf-8") as log:
                try:
                    completed = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, timeout=900)
                    result = json.loads(output.read_text()) if completed.returncode in (0, 1) and output.exists() else {
                        "spec": spec, "status": "failed", "error": f"worker exit={completed.returncode}"}
                except subprocess.TimeoutExpired:
                    result = {"spec": spec, "status": "failed", "error": "timeout900s"}
            output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        results[name] = result
        print(f"[{result['status']}] {name}: {result.get('median_ms', result.get('error', ''))}", flush=True)
        (root / "summary.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
        return result

    anchors = {}
    for m in (1, 8, 20):
        candidates = [run(m, "lm_head", mode="legacy", phase="screen")]
        for chunk in (12288, 24576, 49152, 151936):
            for direct in ((True,) if m == 1 or chunk == 151936 else (False, True)):
                candidates.append(run(m, "lm_head", chunk, direct, phase="screen"))
        valid = [r for r in candidates if r["status"] == "passed"]
        if not valid:
            anchors[m] = None
        else:
            anchors[m] = min(valid, key=lambda r: r["median_ms"])["spec"]
    entries, baseline_failures = [], []
    for shape in SHAPES:
        for m in (range(1, 21) if shape == "lm_head" else (1, 8, 20)):
            baseline = run(m, shape, mode="legacy")
            if baseline["status"] != "passed":
                baseline_failures.append(baseline)
                continue
            chosen = anchors[min(anchors, key=lambda anchor: abs(anchor - m))] if shape == "lm_head" else {
                "chunk": 12288, "direct": m == 1, "mode": "p3_aclnn"}
            candidate = run(m, shape, chosen["chunk"], chosen["direct"], chosen["mode"]) if chosen else baseline
            optimized = candidate["status"] == "passed" and candidate["median_ms"] < baseline["median_ms"]
            selected = candidate if optimized else baseline
            entries.append({**{key: selected["spec"][key] for key in ("m", "n", "k", "chunk", "direct")},
                            "enabled": optimized and selected["spec"]["mode"] == "p3_aclnn"})
    identities = {json.dumps(r["fingerprint"], sort_keys=True) for r in results.values() if r["status"] == "passed"}
    rejected = {name: result.get("error") for name, result in results.items() if result["status"] != "passed"}
    print("\nAGGREGATED REJECTIONS (candidates fall back to measured legacy):")
    for name, error in rejected.items():
        print(f"{name}: {error}")
    if baseline_failures or len(identities) != 1:
        print("No policy exported: baseline failure or mixed build/environment results. Reports retained.")
        return 1
    policy = {"schema": 1, "fingerprint": json.loads(next(iter(identities))), "entries": entries}
    (root / "policy.json").write_text(json.dumps(policy, indent=2), encoding="utf-8")
    print(f"Policy: {root / 'policy.json'}; optimized shapes={sum(e['enabled'] for e in entries)}")
    print("Hardware profile is still required to identify hidden ACLNN conversions; zero external copies is not proof.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
