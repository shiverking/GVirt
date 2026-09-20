#!/usr/bin/env python3
"""Print the compact teacher-forced Decode diagnostic table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    diagnostics = payload.get("decode_diagnostics")
    if not diagnostics:
        raise SystemExit(f"decode_diagnostics missing from {args.report}")

    print("\nTeacher-forced Decode diagnostic summary")
    print("first divergence (zero-based generated index):",
          diagnostics["first_argmax_divergence_generated_index_zero_based"])
    print("first divergence (one-based generated ordinal):",
          diagnostics["first_argmax_divergence_generated_ordinal_one_based"])
    print("AscendC repeat bitwise equal:",
          diagnostics["ascendc_repeat_bitwise_equal"])
    print("idx\tkv\tlegacy\tascendc\tmatch\tmargin\tlogit_cos\tlogit_max\t"
          "hidden_cos\thidden_max\trepeat")
    for step in diagnostics["steps"]:
        print(
            f"{step['generated_token_index_zero_based']}\t"
            f"{step['kv_length']}\t"
            f"{step['legacy_top2_ids'][0]}\t"
            f"{step['ascendc_top2_ids'][0]}\t"
            f"{step['argmax_equal']}\t"
            f"{step['legacy_top1_margin']:.8g}\t"
            f"{step['logits']['cosine']:.9f}\t"
            f"{step['logits']['max_abs']:.8g}\t"
            f"{step['hidden']['cosine']:.9f}\t"
            f"{step['hidden']['max_abs']:.8g}\t"
            f"{step['ascendc_repeat_bitwise_equal']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
