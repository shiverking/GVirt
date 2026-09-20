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
    print("Naive oracle agrees across fresh processes:",
          diagnostics.get("naive_processes_agree"))
    print("Legacy Xlite matches local Naive oracle:",
          diagnostics.get("legacy_xlite_matches_local_naive"))
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
    print("Attention layer diagnostic complete:",
          diagnostics.get("attention_layer_diagnostic_complete"))
    print("first non-bitwise Attention layer:",
          diagnostics.get("first_attention_layer_non_bitwise"))
    print("worst Attention layer by max abs:",
          diagnostics.get("worst_attention_layer_by_max_abs"))
    layer_diagnostics = diagnostics.get("attention_layer_diagnostics", [])
    if layer_diagnostics:
        print("layer\tkv\tcosine\tmean_abs\tmax_abs\tworst_head\t"
              "head_cosine\thead_max_abs")
        for item in layer_diagnostics:
            print(
                f"{item['layer']}\t{item['kv_length']}\t"
                f"{item['cosine']:.9f}\t{item['mean_abs']:.9f}\t"
                f"{item['max_abs']:.9f}\t{item['worst_head']}\t"
                f"{item['worst_head_cosine']:.9f}\t"
                f"{item['worst_head_max_abs']:.9f}"
            )
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
    return 0 if diagnostics.get("passed") else 1


if __name__ == "__main__":
    raise SystemExit(main())
