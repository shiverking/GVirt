#!/usr/bin/env python3
"""Extract real 310P ASR prefill batches from Xlite runtime statistics."""

import argparse
import ast
import json
import re
from pathlib import Path


PREFIX = "Xlite optimization runtime stats:"
BATCH_RE = re.compile(r"^b(?P<batch>\d+)-q(?P<query>[\d_]+)-c(?P<cached>[\d_]+)$")


def _load_stats(path: Path) -> dict:
    text = path.read_text(encoding="utf-8", errors="replace")
    if path.suffix == ".json":
        payload = json.loads(text)
        return payload.get("runtime", payload)

    candidates = []
    for line in text.splitlines():
        marker = line.find(PREFIX)
        if marker >= 0:
            candidates.append(line[marker + len(PREFIX):].strip())
    if not candidates:
        raise ValueError(f"no {PREFIX!r} entry found in {path}")
    payload = ast.literal_eval(candidates[-1])
    return payload.get("runtime", payload)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Generate attention regression cases from real ASR server telemetry")
    parser.add_argument("--input", required=True, type=Path,
                        help="server log containing runtime stats, or a stats JSON file")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top", type=int, default=32,
                        help="maximum number of distinct batch shapes to retain")
    args = parser.parse_args()

    stats = _load_stats(args.input)
    histogram = stats.get("prefill_batch_shape_histogram", {})
    cases = []
    rejected = []
    for signature, count in sorted(histogram.items(), key=lambda item: (-item[1], item[0])):
        match = BATCH_RE.fullmatch(signature)
        if match is None:
            rejected.append(signature)
            continue
        query = [int(value) for value in match.group("query").split("_")]
        cached = [int(value) for value in match.group("cached").split("_")]
        batch = int(match.group("batch"))
        if len(query) != batch or len(cached) != batch:
            rejected.append(signature)
            continue
        # Preserve the complete scheduler batch, including interleaved decode
        # requests, because output-offset bugs are invisible in isolated rows.
        cases.append({
            "name": f"real-prefill-{len(cases) + 1}",
            "batch": batch,
            "query_lens": query,
            "cached_lens": cached,
            "observations": int(count),
            "signature": signature,
        })
        if len(cases) == args.top:
            break

    payload = {
        "schema": 1,
        "source": str(args.input.resolve()),
        "telemetry": {
            key: stats.get(key, 0)
            for key in (
                "prefill_shape_forward_calls",
                "prefill_shape_requests",
                "prefill_exact_groupable_requests",
                "prefill_actual_query_tokens",
                "prefill_padded_query_tokens",
            )
        },
        "cases": cases,
        "rejected_signatures": rejected,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    groupable = int(payload["telemetry"]["prefill_exact_groupable_requests"])
    requests = int(payload["telemetry"]["prefill_shape_requests"])
    actual = int(payload["telemetry"]["prefill_actual_query_tokens"])
    padded = int(payload["telemetry"]["prefill_padded_query_tokens"])
    print(f"Wrote {len(cases)} real prefill batch shapes to {args.output}")
    print(f"Exact-shape groupability: {groupable}/{requests} "
          f"({100.0 * groupable / max(requests, 1):.1f}%)")
    print(f"Query padding ratio: {padded / max(actual, 1):.3f}x")
    return 0 if cases else 1


if __name__ == "__main__":
    raise SystemExit(main())
