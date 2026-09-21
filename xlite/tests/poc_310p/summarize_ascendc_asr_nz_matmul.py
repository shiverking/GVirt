#!/usr/bin/env python3
"""Rank AscendC NZ MatMul shapes against the Native NZ production oracle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("report_dir", type=Path)
    parser.add_argument("--report", type=Path)
    parser.add_argument("--require-faster", action="store_true")
    args = parser.parse_args()

    rows = []
    errors = []
    for log in sorted(args.report_dir.glob("*.log")):
        records = []
        for line in log.read_text(encoding="utf-8", errors="replace").splitlines():
            if line.startswith("{"):
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    pass
        if len(records) != 1:
            errors.append(f"{log}: expected one JSON record, found {len(records)}")
            continue
        item = records[0]
        native_ms = float(item["native_nz"]["device_ms"])
        candidate_ms = float(item["xlite_candidate"]["device_and_submit_ms"])
        delta = (candidate_ms / native_ms - 1.0) * 100.0
        rows.append({
            "case": log.stem,
            "m": int(item["m"]),
            "n": int(item["n"]),
            "k": int(item["k"]),
            "native_nz_device_ms": native_ms,
            "ascendc_nz_device_and_submit_ms": candidate_ms,
            "candidate_delta_percent": delta,
            "speedup": native_ms / candidate_ms,
            "candidate_window_ms": float(
                item["xlite_candidate"]["synchronized_window_ms"]
            ),
        })
    rows.sort(key=lambda row: row["candidate_delta_percent"], reverse=True)
    if not rows:
        errors.append(f"no MatMul JSON records found in {args.report_dir}")
    for row in rows:
        if row["candidate_window_ms"] < 200.0:
            errors.append(
                f"{row['case']}: synchronized window below 200 ms "
                f"({row['candidate_window_ms']:.3f} ms)"
            )
    regressions = [row for row in rows if row["candidate_delta_percent"] > 0.0]
    if args.require_faster and regressions:
        errors.append(f"{len(regressions)} shape(s) are slower than Native NZ")
    payload = {
        "oracle": "F.linear_FP16_FRACTAL_NZ_format29",
        "candidate": "ascendc_asr_nz",
        "shape_count": len(rows),
        "regression_count": len(regressions),
        "top_three_bottlenecks": rows[:3],
        "rows": rows,
        "errors": errors,
    }
    rendered = json.dumps(payload, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
