#!/usr/bin/env python3
"""Compare the same-NZ-weight ACLNN and AscendC whole-model runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _load(path: Path) -> dict:
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--report", type=Path)
    args = parser.parse_args()

    baseline = _load(args.baseline)
    candidate = _load(args.candidate)
    errors: list[str] = []
    if not baseline.get("passed"):
        errors.append("same-NZ-weight ACLNN baseline did not pass")
    if not candidate.get("passed"):
        errors.append("AscendC NZ candidate did not pass")
    if candidate.get("generated_token_ids") != baseline.get("generated_token_ids"):
        errors.append("candidate tokens differ from same-NZ-weight ACLNN baseline")
    if candidate.get("generated_token_ids") != candidate.get("reference_token_ids"):
        errors.append("candidate tokens differ from the reference sequence")

    stats = candidate.get("xlite_runtime_stats", {})
    if stats.get("matmul_backend") != "ascendc_asr_nz":
        errors.append(f"unexpected candidate backend: {stats.get('matmul_backend')!r}")
    required_hits = (
        "ascendc_asr_nz_projection_requests",
        "ascendc_asr_nz_lm_head_requests",
    )
    for key in required_hits:
        if int(stats.get(key, 0)) <= 0:
            errors.append(f"candidate did not hit {key}")
    if int(stats.get("ascendc_asr_bypass_requests", 0)) != 0:
        errors.append(
            "candidate crossed the strict NZ MatMul boundary: "
            f"bypass={stats.get('ascendc_asr_bypass_requests')}"
        )

    baseline_ms = float(baseline["mean_decode_ms"])
    candidate_ms = float(candidate["mean_decode_ms"])
    delta = (candidate_ms / baseline_ms - 1.0) * 100.0
    result = {
        "baseline": "xlite_aclnn_same_nz_weights",
        "candidate": "ascendc_asr_nz",
        "baseline_mean_decode_ms": baseline_ms,
        "candidate_mean_decode_ms": candidate_ms,
        "candidate_delta_percent": delta,
        "candidate_faster_in_this_sample": candidate_ms < baseline_ms,
        "tokens_match": not any("tokens differ" in item for item in errors),
        "candidate_stats": {
            key: stats.get(key) for key in (
                *required_hits,
                "ascendc_asr_requests",
                "ascendc_asr_kernel_launches",
                "ascendc_asr_bypass_requests",
                "aclnn_requests",
                "stream_synchronizations",
            )
        },
        "errors": errors,
    }
    rendered = json.dumps(result, indent=2)
    print(rendered)
    if args.report is not None:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(rendered + "\n", encoding="utf-8")
    if errors:
        raise RuntimeError("; ".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
