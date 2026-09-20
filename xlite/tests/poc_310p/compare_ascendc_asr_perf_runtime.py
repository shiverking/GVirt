#!/usr/bin/env python3
"""Compare baseline and performance-candidate whole-model reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    args = parser.parse_args()

    baseline = load(args.baseline)
    candidate = load(args.candidate)
    errors: list[str] = []
    if not baseline.get("passed"):
        errors.append("baseline report did not pass")
    if not candidate.get("passed"):
        errors.append("candidate report did not pass")
    if candidate.get("generated_token_ids") != baseline.get("generated_token_ids"):
        errors.append("candidate tokens differ from ascendc_asr baseline")
    if candidate.get("generated_token_ids") != candidate.get("reference_token_ids"):
        errors.append("candidate tokens differ from the reference sequence")

    stats = candidate.get("xlite_runtime_stats", {})
    if stats.get("matmul_backend") != "ascendc_asr_perf":
        errors.append(f"unexpected candidate backend: {stats.get('matmul_backend')!r}")
    for key in (
        "ascendc_asr_perf_projection_requests",
        "ascendc_asr_perf_lm_head_requests",
        "ascendc_asr_perf_silu_mul_requests",
    ):
        if int(stats.get(key, 0)) <= 0:
            errors.append(f"candidate did not hit {key}")

    baseline_ms = float(baseline["mean_decode_ms"])
    candidate_ms = float(candidate["mean_decode_ms"])
    delta = (candidate_ms / baseline_ms - 1.0) * 100.0
    print(json.dumps({
        "baseline_mean_decode_ms": baseline_ms,
        "candidate_mean_decode_ms": candidate_ms,
        "candidate_delta_percent": delta,
        "tokens_match": not any("tokens differ" in error for error in errors),
        "candidate_stats": {
            key: stats.get(key) for key in (
                "ascendc_asr_perf_projection_requests",
                "ascendc_asr_perf_lm_head_requests",
                "ascendc_asr_perf_silu_mul_requests",
                "ascendc_asr_bypass_requests",
                "aclnn_requests",
            )
        },
        "errors": errors,
    }, indent=2))
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
