"""Summarize isolated QK/MRoPE/Cache baseline/grouped measurements."""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path


FULL_BATCHES = (1, 2, 4, 6, 8, 12, 16, 20)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("cases")
    parser.add_argument("repeats", type=int)
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    if args.cases == "all":
        batches = FULL_BATCHES
    elif args.cases == "quick":
        batches = (1, 8, 20)
    else:
        batches = (int(args.cases.removeprefix("m")),)

    rows = []
    for batch in batches:
        measurements = {}
        for variant in ("baseline", "grouped"):
            samples = []
            for run in range(1, args.repeats + 1):
                path = args.build_dir / f"m{batch}-{variant}-r{run}.log"
                match = re.search(
                    r"ASR QK-Norm-MRoPE-Cache PASS:.*average_ms=([0-9.]+)",
                    path.read_text(),
                )
                if not match:
                    raise RuntimeError(f"missing passing measurement: {path}")
                samples.append(float(match.group(1)))
            mean = statistics.mean(samples)
            stdev = statistics.stdev(samples) if len(samples) > 1 else None
            measurements[variant] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "stdev_ms": stdev,
                "cv": stdev / mean if stdev is not None else None,
            }
        before = measurements["baseline"]["median_ms"]
        after = measurements["grouped"]["median_ms"]
        noise_ms = 3 * max(measurements[v]["stdev_ms"] or 0.0
                           for v in ("baseline", "grouped"))
        regression_limit_ms = max(0.02 * before, noise_ms)
        row = {
            "tokens": batch,
            **measurements,
            "latency_reduction": 1 - after / before,
            "regression_ms": after - before,
            "regression_limit_ms": regression_limit_ms,
            "regression_within_limit": after - before <= regression_limit_ms,
        }
        rows.append(row)
        print(
            f"QK_MROPE_AB M={batch} baseline_ms={before:.6f} "
            f"grouped_ms={after:.6f} reduction={100 * row['latency_reduction']:.2f}% "
            f"regression_limit_ms={regression_limit_ms:.6f} "
            f"baseline_cv={measurements['baseline']['cv']} "
            f"grouped_cv={measurements['grouped']['cv']}"
        )

    measured_gate = (
        args.repeats >= 3
        and tuple(batches) == FULL_BATCHES
        and any(row["latency_reduction"] >= 0.05 for row in rows)
        and all(
            row["regression_within_limit"]
            and all(row[v]["cv"] <= 0.05 for v in ("baseline", "grouped"))
            for row in rows
        )
    )
    report = {
        "rows": rows,
        "microprobe_performance_gate": measured_gate,
        "runtime_promotion": False,
        "note": "Synthetic microprobe only; model-weight and whole-model gates remain.",
    }
    path = args.build_dir / "qk_mrope_cache_ab_summary.json"
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(
        f"Microprobe performance gate: {'PASS' if measured_gate else 'NOT_MET'}; "
        f"report={path}"
    )
    if args.require_gate and not measured_gate:
        print("Required performance gate failed", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
