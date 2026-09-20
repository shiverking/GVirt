#!/usr/bin/env python3
"""Coordinate fresh-process legacy/AscendC teacher-forced diagnostics."""

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
        "max_abs_index": int(max_index),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }


def _run_worker(args: argparse.Namespace, name: str, backend: str,
                work_dir: Path) -> dict:
    worker = Path(__file__).with_name("decode_diagnostic_worker.py")
    output = work_dir / f"{name}.pt"
    log = work_dir / f"{name}.log"
    command = [
        sys.executable, str(worker),
        "--checkpoint", args.checkpoint,
        "--bundle", str(args.bundle),
        "--backend", backend,
        "--matmul-backend", args.matmul_backend,
        "--max-seq-len", str(args.max_seq_len),
        "--output", str(output),
    ]
    if args.num_layers is not None:
        command.extend(("--num-layers", str(args.num_layers)))
    print(f"[ RUN      ] {name} ({backend}, fresh process)", flush=True)
    with log.open("w", encoding="utf-8") as stream:
        completed = subprocess.run(
            command, stdout=stream, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            f"{name} failed with exit={completed.returncode}; log={log}")
    print(f"[       OK ] {name}", flush=True)
    return torch.load(output, map_location="cpu", weights_only=True)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--work-dir", type=Path)
    parser.add_argument("--matmul-backend", default="ascendc_asr")
    parser.add_argument("--candidate-backend", default="ascendc_asr",
                        choices=("ascendc_asr",))
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int)
    args = parser.parse_args()

    work_dir = args.work_dir or args.report.with_suffix("").with_name(
        args.report.stem + "_workers")
    work_dir.mkdir(parents=True, exist_ok=True)
    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise RuntimeError("unsupported Decode diagnostic bundle")
    reference_tokens = [int(token) for token in bundle["reference_token_ids"]]
    prompt_tokens = int(bundle["prompt_tokens"])

    legacy = _run_worker(args, "legacy", "legacy", work_dir)
    candidate = _run_worker(
        args, "ascendc-a", args.candidate_backend, work_dir)
    candidate_repeat = _run_worker(
        args, "ascendc-b", args.candidate_backend, work_dir)

    legacy_prefill_top1 = int(legacy["prefill_logits"].argmax())
    legacy_decode_top1 = [
        int(step.argmax()) for step in legacy["decode_logits"]
    ]
    oracle_predictions = [legacy_prefill_top1, *legacy_decode_top1[:-1]]
    oracle_mismatches = [
        {
            "generated_token_index_zero_based": index,
            "expected": expected,
            "legacy": actual,
        }
        for index, (expected, actual) in enumerate(
            zip(reference_tokens, oracle_predictions))
        if expected != actual
    ]
    oracle_valid = not oracle_mismatches

    report = {
        "mode": "isolated_process_teacher_forced_reference_tokens",
        "oracle_backend": "legacy",
        "candidate_backend": args.candidate_backend,
        "fresh_process_per_run": True,
        "worker_artifacts": str(work_dir.resolve()),
        "legacy_oracle_valid": oracle_valid,
        "legacy_oracle_mismatches": oracle_mismatches,
        "prefill": None,
        "ascendc_prefill_repeat_bitwise_equal": None,
        "ascendc_repeat_bitwise_equal": None,
        "first_argmax_divergence_generated_index_zero_based": None,
        "first_argmax_divergence_generated_ordinal_one_based": None,
        "steps": [],
    }

    if oracle_valid:
        first_divergence = None
        prefill_repeat_equal = bool(
            torch.equal(candidate["prefill_logits"],
                        candidate_repeat["prefill_logits"]) and
            torch.equal(candidate["prefill_hidden"],
                        candidate_repeat["prefill_hidden"]))
        repeat_all = prefill_repeat_equal
        report["prefill"] = {
            "legacy_top1": legacy_prefill_top1,
            "ascendc_top1": int(candidate["prefill_logits"].argmax()),
            "logits": _metrics(
                legacy["prefill_logits"], candidate["prefill_logits"]),
            "hidden": _metrics(
                legacy["prefill_hidden"], candidate["prefill_hidden"]),
        }
        report["ascendc_prefill_repeat_bitwise_equal"] = prefill_repeat_equal
        for index, (legacy_logits, legacy_hidden, candidate_logits,
                    candidate_hidden, repeat_logits, repeat_hidden) in enumerate(zip(
                        legacy["decode_logits"], legacy["decode_hidden"],
                        candidate["decode_logits"], candidate["decode_hidden"],
                        candidate_repeat["decode_logits"],
                        candidate_repeat["decode_hidden"])):
            legacy_ids, legacy_values = _top2(legacy_logits)
            candidate_ids, candidate_values = _top2(candidate_logits)
            repeat_equal = bool(
                torch.equal(candidate_logits, repeat_logits) and
                torch.equal(candidate_hidden, repeat_hidden))
            repeat_all = repeat_all and repeat_equal
            argmax_equal = legacy_ids[0] == candidate_ids[0]
            if not argmax_equal and first_divergence is None:
                first_divergence = index + 1
            report["steps"].append({
                "generated_token_index_zero_based": index + 1,
                "generated_token_ordinal_one_based": index + 2,
                "kv_length": prompt_tokens + index + 1,
                "input_reference_token": reference_tokens[index],
                "expected_next_token": (
                    reference_tokens[index + 1]
                    if index + 1 < len(reference_tokens) else None),
                "legacy_top2_ids": legacy_ids,
                "legacy_top2_logits": legacy_values,
                "legacy_top1_margin": legacy_values[0] - legacy_values[1],
                "ascendc_top2_ids": candidate_ids,
                "ascendc_top2_logits": candidate_values,
                "ascendc_logit_at_legacy_top1": float(
                    candidate_logits.float().flatten()[legacy_ids[0]]),
                "argmax_equal": argmax_equal,
                "logits": _metrics(legacy_logits, candidate_logits),
                "hidden": _metrics(legacy_hidden, candidate_hidden),
                "ascendc_repeat_bitwise_equal": repeat_equal,
            })
        report["ascendc_repeat_bitwise_equal"] = repeat_all
        report[
            "first_argmax_divergence_generated_index_zero_based"
        ] = first_divergence
        report["first_argmax_divergence_generated_ordinal_one_based"] = (
            first_divergence + 1 if first_divergence is not None else None)

    report["passed"] = bool(
        oracle_valid and report["ascendc_repeat_bitwise_equal"])
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    if not oracle_valid:
        print("Legacy oracle self-check FAILED; backend comparison was skipped.",
              file=sys.stderr)
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
