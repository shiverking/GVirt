#!/usr/bin/env python3
"""Run isolated A/B/C Decode diagnostics in fresh processes.

The tiers change one backend boundary at a time:
A = ACLNN MatMul + legacy attention, B = AscendC MatMul + legacy
attention, and C = AscendC MatMul + AscendC attention. A repeated C
process checks determinism.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn.functional as F


def _top2(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    values, indices = torch.topk(logits.float().flatten(), k=2)
    return [int(v) for v in indices.tolist()], [float(v) for v in values.tolist()]


def _metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_fp32 = reference.float().flatten()
    candidate_fp32 = candidate.float().flatten()
    difference = (reference_fp32 - candidate_fp32).abs()
    max_abs, max_index = difference.max(dim=0)
    return {
        "cosine": float(F.cosine_similarity(
            reference_fp32, candidate_fp32, dim=0)),
        "max_abs": float(max_abs),
        "mean_abs": float(difference.mean()),
        "max_abs_index": int(max_index),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }


def _run_worker(args: argparse.Namespace, name: str, matmul_backend: str,
                attention_backend: str, work_dir: Path,
                capture_attention: bool = False) -> dict:
    worker = Path(__file__).with_name("decode_diagnostic_worker.py")
    output = work_dir / f"{name}.pt"
    log = work_dir / f"{name}.log"
    command = [
        sys.executable, str(worker),
        "--checkpoint", args.checkpoint,
        "--bundle", str(args.bundle),
        "--backend", attention_backend,
        "--matmul-backend", matmul_backend,
        "--max-seq-len", str(args.max_seq_len),
        "--output", str(output),
    ]
    if args.num_layers is not None:
        command.extend(("--num-layers", str(args.num_layers)))
    if capture_attention and args.attention_diagnostic_kv:
        command.extend((
            "--attention-diagnostic-kv",
            str(args.attention_diagnostic_kv),
        ))
    print(
        f"[ RUN      ] {name} (matmul={matmul_backend}, "
        f"attention={attention_backend}, fresh process)", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit={completed.returncode}; log={log}")
    print(f"[       OK ] {name}", flush=True)
    return torch.load(output, map_location="cpu", weights_only=True)


def _predictions(result: dict, prefix: str = "") -> list[int]:
    return [
        int(result[f"{prefix}prefill_logits"].argmax()),
        *[int(step.argmax()) for step in result[f"{prefix}decode_logits"][:-1]],
    ]


def _first_mismatch(left: list[int], right: list[int]) -> int | None:
    return next((index for index, values in enumerate(zip(left, right))
                 if values[0] != values[1]), None)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument(
        "--matmul-backend", default="ascendc_asr",
        help="candidate MatMul/vector backend used by tiers B and C")
    parser.add_argument("--candidate-backend", default="ascendc_asr",
                        choices=("ascendc_asr",))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--attention-diagnostic-kv", type=int, default=142)
    args = parser.parse_args()

    if not args.bundle.is_file():
        parser.error(
            f"diagnostic bundle does not exist: {args.bundle}; generate a "
            "fresh bundle with run_qwen3_asr_llm.py "
            "--save-decode-diagnostic-bundle before running diagnostics")

    work_dir = args.work_dir or args.report.with_suffix("").with_name(
        args.report.stem + "_workers")
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 2:
        raise RuntimeError(
            "unsupported Decode diagnostic bundle; regenerate a version 2 "
            "bundle with independently cleared Naive/Xlite caches")
    reference_tokens = [int(token) for token in bundle["reference_token_ids"]]
    prompt_tokens = int(bundle["prompt_tokens"])

    oracle = _run_worker(
        args, "a-aclnn-legacy", "aclnn", "legacy", work_dir)
    common_candidate = _run_worker(
        args, "b-ascendc-legacy", args.matmul_backend, "legacy", work_dir)
    attention_candidate = _run_worker(
        args, "c-ascendc-ascendc", args.matmul_backend,
        args.candidate_backend, work_dir, capture_attention=True)
    attention_repeat = _run_worker(
        args, "c-repeat-ascendc-ascendc", args.matmul_backend,
        args.candidate_backend, work_dir)

    oracle_predictions = _predictions(oracle, "naive_")
    legacy_predictions = _predictions(oracle)
    common_predictions = _predictions(common_candidate)
    attention_predictions = _predictions(attention_candidate)
    oracle_mismatches = [
        {
            "generated_token_index_zero_based": index,
            "expected": expected,
            "oracle": actual,
        }
        for index, (expected, actual) in enumerate(
            zip(reference_tokens, oracle_predictions))
        if expected != actual
    ]
    local_naive_predictions = [
        _predictions(result, "naive_") for result in (
            oracle, common_candidate, attention_candidate, attention_repeat)
    ]
    naive_processes_agree = all(
        prediction == oracle_predictions
        for prediction in local_naive_predictions[1:]
    )
    legacy_matches_naive = legacy_predictions == oracle_predictions
    oracle_valid = bool(
        not oracle_mismatches and naive_processes_agree and
        legacy_matches_naive)

    layer_diagnostics = []
    for record in attention_candidate.get("attention_diagnostics", []):
        legacy = record["legacy"]
        ascendc = record["ascendc"]
        heads = []
        for head in range(legacy.shape[0]):
            heads.append({
                "head": head,
                **_metrics(legacy[head], ascendc[head]),
            })
        worst_head = max(heads, key=lambda item: item["max_abs"])
        layer_diagnostics.append({
            "layer": int(record["layer"]),
            "kv_length": int(record["kv_length"]),
            **_metrics(legacy, ascendc),
            "worst_head": int(worst_head["head"]),
            "worst_head_cosine": float(worst_head["cosine"]),
            "worst_head_max_abs": float(worst_head["max_abs"]),
            "heads": heads,
        })
    layer_diagnostics.sort(key=lambda item: item["layer"])
    diagnostic_layers = [item["layer"] for item in layer_diagnostics]
    expected_layers = list(range(args.num_layers or 28))
    layer_diagnostic_complete = (
        not args.attention_diagnostic_kv or diagnostic_layers == expected_layers)

    report = {
        "mode": "isolated_process_three_tier_teacher_forced_reference_tokens",
        "fresh_process_per_run": True,
        "bundle": str(args.bundle.resolve()),
        "worker_artifacts": str(work_dir.resolve()),
        "tiers": {
            "a_oracle": {"matmul": "aclnn", "attention": "legacy"},
            "b_common_candidate": {
                "matmul": args.matmul_backend, "attention": "legacy"},
            "c_attention_candidate": {
                "matmul": args.matmul_backend,
                "attention": args.candidate_backend,
            },
        },
        "legacy_oracle_valid": oracle_valid,
        "legacy_oracle_mismatches": oracle_mismatches,
        "naive_processes_agree": naive_processes_agree,
        "legacy_xlite_matches_local_naive": legacy_matches_naive,
        "first_common_path_divergence_generated_index_zero_based": None,
        "first_attention_divergence_generated_index_zero_based": None,
        "ascendc_repeat_bitwise_equal": None,
        "attention_diagnostic_kv": args.attention_diagnostic_kv,
        "attention_layer_diagnostic_complete": layer_diagnostic_complete,
        "attention_layer_diagnostics": layer_diagnostics,
        "first_attention_layer_non_bitwise": next(
            (item["layer"] for item in layer_diagnostics
             if not item["bitwise_equal"]), None),
        "worst_attention_layer_by_max_abs": (
            max(layer_diagnostics, key=lambda item: item["max_abs"])["layer"]
            if layer_diagnostics else None),
        "prefill": None,
        "steps": [],
    }

    if oracle_valid:
        first_common = _first_mismatch(oracle_predictions, common_predictions)
        first_attention = _first_mismatch(
            common_predictions, attention_predictions)
        repeat_equal = bool(
            torch.equal(attention_candidate["prefill_logits"],
                        attention_repeat["prefill_logits"]) and
            torch.equal(attention_candidate["prefill_hidden"],
                        attention_repeat["prefill_hidden"]) and
            torch.equal(attention_candidate["decode_logits"],
                        attention_repeat["decode_logits"]) and
            torch.equal(attention_candidate["decode_hidden"],
                        attention_repeat["decode_hidden"]))
        report[
            "first_common_path_divergence_generated_index_zero_based"
        ] = first_common
        report[
            "first_attention_divergence_generated_index_zero_based"
        ] = first_attention
        report["ascendc_repeat_bitwise_equal"] = repeat_equal
        report["prefill"] = {
            "oracle_top1": oracle_predictions[0],
            "legacy_xlite_top1": legacy_predictions[0],
            "common_candidate_top1": common_predictions[0],
            "attention_candidate_top1": attention_predictions[0],
            "common_vs_oracle_logits": _metrics(
                oracle["naive_prefill_logits"],
                common_candidate["prefill_logits"]),
            "common_vs_oracle_hidden": _metrics(
                oracle["naive_prefill_hidden"],
                common_candidate["prefill_hidden"]),
            "attention_vs_common_logits": _metrics(
                common_candidate["prefill_logits"],
                attention_candidate["prefill_logits"]),
            "attention_vs_common_hidden": _metrics(
                common_candidate["prefill_hidden"],
                attention_candidate["prefill_hidden"]),
        }

        for index in range(len(reference_tokens) - 1):
            oracle_logits = oracle["naive_decode_logits"][index]
            common_logits = common_candidate["decode_logits"][index]
            attention_logits = attention_candidate["decode_logits"][index]
            oracle_hidden = oracle["naive_decode_hidden"][index]
            common_hidden = common_candidate["decode_hidden"][index]
            attention_hidden = attention_candidate["decode_hidden"][index]
            oracle_ids, oracle_values = _top2(oracle_logits)
            common_ids, common_values = _top2(common_logits)
            attention_ids, attention_values = _top2(attention_logits)
            report["steps"].append({
                "generated_token_index_zero_based": index + 1,
                "generated_token_ordinal_one_based": index + 2,
                "kv_length": prompt_tokens + index + 1,
                "input_reference_token": reference_tokens[index],
                "expected_next_token": reference_tokens[index + 1],
                "oracle_top2_ids": oracle_ids,
                "oracle_top2_logits": oracle_values,
                "oracle_top1_margin": oracle_values[0] - oracle_values[1],
                "common_candidate_top2_ids": common_ids,
                "common_candidate_top2_logits": common_values,
                "attention_candidate_top2_ids": attention_ids,
                "attention_candidate_top2_logits": attention_values,
                "common_argmax_equal": oracle_ids[0] == common_ids[0],
                "attention_argmax_equal": common_ids[0] == attention_ids[0],
                "common_vs_oracle_logits": _metrics(
                    oracle_logits, common_logits),
                "common_vs_oracle_hidden": _metrics(
                    oracle_hidden, common_hidden),
                "attention_vs_common_logits": _metrics(
                    common_logits, attention_logits),
                "attention_vs_common_hidden": _metrics(
                    common_hidden, attention_hidden),
            })

    report["passed"] = bool(
        oracle_valid and report["ascendc_repeat_bitwise_equal"] and
        layer_diagnostic_complete and
        report["first_common_path_divergence_generated_index_zero_based"] is None and
        report["first_attention_divergence_generated_index_zero_based"] is None)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not oracle_valid:
        print(
            "ACLNN + legacy oracle self-check FAILED; candidate comparisons "
            "were not interpreted.", file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
