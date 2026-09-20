#!/usr/bin/env python3
"""Print the compact three-tier teacher-forced Decode diagnostic table."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("report", type=Path)
    args = parser.parse_args()
    payload = json.loads(args.report.read_text(encoding="utf-8"))
    diagnostics = payload.get("decode_diagnostics") or payload

    print("tiers:", diagnostics.get("tiers"))
    print("ACLNN + legacy oracle valid:",
          diagnostics["legacy_oracle_valid"])
    if not diagnostics["legacy_oracle_valid"]:
        print("oracle mismatches:",
              diagnostics["legacy_oracle_mismatches"])
        return 1

    print("first common-path divergence (A vs B):",
          diagnostics[
              "first_common_path_divergence_generated_index_zero_based"])
    print("first attention divergence (B vs C):",
          diagnostics[
              "first_attention_divergence_generated_index_zero_based"])
    print("C repeat bitwise equal:",
          diagnostics["ascendc_repeat_bitwise_equal"])
    print("idx\tkv\toracle\tcommon\tattention\tA=B\tB=C\t"
          "AB_logit_cos\tAB_hidden_cos\tBC_logit_cos\tBC_hidden_cos")
    for step in diagnostics["steps"]:
        print(
            f"{step['generated_token_index_zero_based']}\t"
            f"{step['kv_length']}\t"
            f"{step['oracle_top2_ids'][0]}\t"
            f"{step['common_candidate_top2_ids'][0]}\t"
            f"{step['attention_candidate_top2_ids'][0]}\t"
            f"{step['common_argmax_equal']}\t"
            f"{step['attention_argmax_equal']}\t"
            f"{step['common_vs_oracle_logits']['cosine']:.9f}\t"
            f"{step['common_vs_oracle_hidden']['cosine']:.9f}\t"
            f"{step['attention_vs_common_logits']['cosine']:.9f}\t"
            f"{step['attention_vs_common_hidden']['cosine']:.9f}"
        )
    return 0 if diagnostics["ascendc_repeat_bitwise_equal"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
