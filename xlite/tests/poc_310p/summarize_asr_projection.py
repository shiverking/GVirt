"""Summarize same-build projection A/B logs; never infer device speedups."""
import argparse
import json
import re
import statistics
import sys
from pathlib import Path


SHAPES = [
    (name, m)
    for name in ("qkv", "o", "gate-up", "down")
    for m in (1, 2, 4, 6, 8, 12, 16, 20)
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("build_dir", type=Path)
    parser.add_argument("cases")
    parser.add_argument("repeats", type=int)
    parser.add_argument("--require-gate", action="store_true")
    args = parser.parse_args()
    shapes = SHAPES
    if args.cases == "quick":
        shapes = [(name, m) for name, m in SHAPES if m in (1, 8, 20)]
    elif args.cases != "all":
        prefix, batch = args.cases.rsplit("-m", 1)
        shapes = [(prefix, int(batch))]

    rows = []
    for name, batch in shapes:
        case = f"{name}-m{batch}"
        values = {}
        for variant in ("baseline", "cached"):
            samples = []
            for run in range(1, args.repeats + 1):
                path = args.build_dir / f"{case}-{variant}-r{run}.log"
                match = re.search(
                    r"ASR low-level projection PASS:.*average_ms=([0-9.]+)",
                    path.read_text(),
                )
                if not match:
                    raise RuntimeError(f"missing passing measurement: {path}")
                samples.append(float(match.group(1)))
            mean = statistics.mean(samples)
            stdev = statistics.stdev(samples) if len(samples) > 1 else None
            values[variant] = {
                "samples_ms": samples,
                "median_ms": statistics.median(samples),
                "stdev_ms": stdev,
                "cv": stdev / mean if stdev is not None else None,
            }
        before = values["baseline"]["median_ms"]
        after = values["cached"]["median_ms"]
        noise_ms = 3 * max(values[v]["stdev_ms"] or 0.0
                           for v in ("baseline", "cached"))
        regression_limit_ms = max(0.02 * before, noise_ms)
        row = {
            "shape": name,
            "m": batch,
            **values,
            "latency_reduction": 1 - after / before,
            "regression_ms": after - before,
            "regression_limit_ms": regression_limit_ms,
            "regression_within_limit": after - before <= regression_limit_ms,
        }
        rows.append(row)
        print(
            f"PROJECTION_AB shape={name} M={batch} baseline_ms={before:.6f} "
            f"cached_ms={after:.6f} reduction={100 * row['latency_reduction']:.2f}% "
            f"regression_limit_ms={regression_limit_ms:.6f} "
            f"baseline_cv={values['baseline']['cv']} cached_cv={values['cached']['cv']}"
        )

    measured_gate = (
        args.repeats >= 3
        and len(rows) == len(SHAPES)
        and any(row["latency_reduction"] >= 0.05 for row in rows)
        and all(
            row["regression_within_limit"]
            and all(row[v]["cv"] <= 0.05 for v in ("baseline", "cached"))
            for row in rows
        )
    )
    report = {
        "rows": rows,
        "microprobe_performance_gate": measured_gate,
        "runtime_promotion": False,
        "note": "Structured-input microprobe only; model-weight and full-model gates remain.",
    }
    path = args.build_dir / "projection_ab_summary.json"
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
