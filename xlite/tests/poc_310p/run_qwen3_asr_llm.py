#!/usr/bin/env python3
"""Ascend 310P single-card FP16 Qwen3-ASR decoder penetration test."""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

# When this file is executed by path, Python puts tests/poc_310p rather than
# the repository root first on sys.path.  Pin the checkout root so
# ``tests.models`` cannot be shadowed by an unrelated site-packages ``tests``.
REPO_ROOT = Path(__file__).resolve().parents[2]
repo_root_str = str(REPO_ROOT)
if repo_root_str in sys.path:
    sys.path.remove(repo_root_str)
sys.path.insert(0, repo_root_str)

os.environ.setdefault("FORWARD_BACKEND", "xlite")
os.environ.setdefault("XLITE_DP_SIZE", "1")
os.environ.setdefault("XLITE_TP_SIZE", "1")
os.environ.setdefault("XLITE_WEIGHT_NZ", "0")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tests.models.llama import Llama
from tests.models.qwen3 import Qwen3ModelArgs
from xlite import _C as xlite_c
from xlite.poc_310p import load_qwen3_asr_llm_args


def _sync_ms(callable_):
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    result = callable_()
    torch.npu.synchronize()
    return result, (time.perf_counter_ns() - started) / 1_000_000


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).cpu())


def _require_nonzero_finite(name: str, value: torch.Tensor) -> None:
    if not bool(torch.isfinite(value).all().item()):
        raise RuntimeError(f"{name} contains NaN or Inf")
    if float(value.float().abs().max().item()) == 0.0:
        raise RuntimeError(
            f"{name} is entirely zero; checkpoint loading or the execution path is invalid"
        )


def _clear_caches(model: Llama) -> None:
    for layer in model.layers:
        layer.self_attn.k_cache.zero_()
        layer.self_attn.v_cache.zero_()
    for key_cache, value_cache in model.xlite_kv_cache:
        key_cache.zero_()
        value_cache.zero_()


def _device_name() -> str:
    get_name = getattr(torch.npu, "get_device_name", None)
    return str(get_name(0)) if get_name is not None else "unknown"


def _error_metrics(reference: torch.Tensor, candidate: torch.Tensor) -> dict:
    reference_fp32 = reference.float().flatten()
    candidate_fp32 = candidate.float().flatten()
    difference = (reference_fp32 - candidate_fp32).abs()
    max_abs, max_index = difference.max(dim=0)
    return {
        "cosine": _cosine(reference_fp32, candidate_fp32),
        "max_abs": float(max_abs.cpu()),
        "max_abs_index": int(max_index.cpu()),
        "bitwise_equal": bool(torch.equal(reference, candidate)),
    }


def _top2(logits: torch.Tensor) -> tuple[list[int], list[float]]:
    values, indices = torch.topk(logits.float().flatten(), k=2)
    return ([int(value) for value in indices.cpu().tolist()],
            [float(value) for value in values.cpu().tolist()])


def _run_teacher_forced_xlite(
    model: Llama,
    backend: str,
    inputs_embeds: torch.Tensor,
    prompt_positions: torch.Tensor,
    all_positions: torch.Tensor,
    reference_tokens: list[int],
) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """Run Decode with identical reference tokens for backend comparison."""
    model.xlite_rt.set_decode_attention_backend(backend)
    _clear_caches(model)
    model.forward_xlite_with_inputs_embeds(
        inputs_embeds, 0, positions=prompt_positions)
    torch.npu.synchronize()
    snapshots = []
    position = inputs_embeds.size(1)
    for token in reference_tokens:
        token_tensor = torch.tensor([[token]], dtype=torch.int64, device="npu")
        token_embed = model.embed_tokens(token_tensor).to(torch.float16)
        logits, hidden = model.forward_xlite_with_inputs_embeds(
            token_embed, position, return_hidden=True,
            positions=all_positions[..., position:position + 1])
        torch.npu.synchronize()
        # Xlite TensorPool storage is reused by the next forward.  Clone every
        # diagnostic output so later comparisons observe the original step.
        snapshots.append((logits.detach().clone(), hidden.detach().clone()))
        position += 1
    return snapshots


def _build_decode_diagnostics(
    model: Llama,
    selected_backend: str,
    inputs_embeds: torch.Tensor,
    prompt_positions: torch.Tensor,
    all_positions: torch.Tensor,
    reference_tokens: list[int],
) -> dict:
    """Compare legacy and AscendC Attention under teacher forcing.

    Runtime counters used by normal acceptance must be captured before this
    function because the oracle run intentionally exercises legacy Attention.
    """
    legacy = _run_teacher_forced_xlite(
        model, "legacy", inputs_embeds, prompt_positions,
        all_positions, reference_tokens)
    ascendc = _run_teacher_forced_xlite(
        model, "ascendc_asr", inputs_embeds, prompt_positions,
        all_positions, reference_tokens)
    ascendc_repeat = _run_teacher_forced_xlite(
        model, "ascendc_asr", inputs_embeds, prompt_positions,
        all_positions, reference_tokens)

    steps = []
    first_argmax_divergence = None
    all_repeat_bitwise_equal = True
    prompt_tokens = int(inputs_embeds.size(1))
    for index, ((legacy_logits, legacy_hidden),
                (ascendc_logits, ascendc_hidden),
                (repeat_logits, repeat_hidden)) in enumerate(
                    zip(legacy, ascendc, ascendc_repeat)):
        legacy_ids, legacy_values = _top2(legacy_logits)
        ascendc_ids, ascendc_values = _top2(ascendc_logits)
        repeat_equal = bool(
            torch.equal(ascendc_logits, repeat_logits) and
            torch.equal(ascendc_hidden, repeat_hidden)
        )
        all_repeat_bitwise_equal = all_repeat_bitwise_equal and repeat_equal
        argmax_equal = legacy_ids[0] == ascendc_ids[0]
        if not argmax_equal and first_argmax_divergence is None:
            # index 0 is the forward consuming reference token 0 and predicts
            # generated token 1.  Token 0 itself comes from Prefill logits.
            first_argmax_divergence = index + 1
        legacy_top1_in_ascendc = float(
            ascendc_logits.float().flatten()[legacy_ids[0]].cpu())
        steps.append({
            "generated_token_index_zero_based": index + 1,
            "generated_token_ordinal_one_based": index + 2,
            "kv_length": prompt_tokens + index + 1,
            "input_reference_token": int(reference_tokens[index]),
            "expected_next_token": (
                int(reference_tokens[index + 1])
                if index + 1 < len(reference_tokens) else None
            ),
            "legacy_top2_ids": legacy_ids,
            "legacy_top2_logits": legacy_values,
            "legacy_top1_margin": legacy_values[0] - legacy_values[1],
            "ascendc_top2_ids": ascendc_ids,
            "ascendc_top2_logits": ascendc_values,
            "ascendc_logit_at_legacy_top1": legacy_top1_in_ascendc,
            "argmax_equal": argmax_equal,
            "logits": _error_metrics(legacy_logits, ascendc_logits),
            "hidden": _error_metrics(legacy_hidden, ascendc_hidden),
            "ascendc_repeat_bitwise_equal": repeat_equal,
        })

    model.xlite_rt.set_decode_attention_backend(selected_backend)
    _clear_caches(model)
    return {
        "mode": "teacher_forced_reference_tokens",
        "oracle_backend": "legacy",
        "candidate_backend": "ascendc_asr",
        "first_argmax_divergence_generated_index_zero_based": (
            first_argmax_divergence
        ),
        "first_argmax_divergence_generated_ordinal_one_based": (
            first_argmax_divergence + 1
            if first_argmax_divergence is not None else None
        ),
        "ascendc_repeat_bitwise_equal": all_repeat_bitwise_equal,
        "steps": steps,
    }


def _load_tensor_file(path: Path) -> torch.Tensor:
    if path.suffix == ".npy":
        import numpy as np
        return torch.from_numpy(np.load(path))
    value = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(value, torch.Tensor):
        raise ValueError(f"{path} must contain one tensor")
    return value


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt", default="Transcribe the supplied audio embedding.")
    parser.add_argument("--input-mode", choices=("tokens", "synthetic", "file"), default="tokens")
    parser.add_argument("--embeds-file", type=Path)
    parser.add_argument("--positions-file", type=Path)
    parser.add_argument("--prompt-tokens", type=int, default=32)
    parser.add_argument("--decode-tokens", type=int, default=16)
    parser.add_argument("--max-seq-len", type=int, default=512)
    parser.add_argument("--num-layers", type=int, help="POC gate override; use 1 for Gate 5")
    parser.add_argument("--stability-iters", type=int, default=50)
    parser.add_argument(
        "--aclnn-matmul-async", action="store_true",
        help="event-retire Decoder ACLNN MatMul resources; keep LM Head synchronized",
    )
    parser.add_argument(
        "--matmul-backend",
        choices=("ascendc_asr", "m200_asr_prefill", "m200_asr", "aclnn"),
        default="m200_asr",
        help=("310P MatMul backend; ascendc_asr currently accelerates fixed "
              "decode projections while prefill and LM Head remain explicit "
              "ACLNN boundaries"),
    )
    parser.add_argument(
        "--decode-attention-backend",
        choices=("ascendc_asr", "direct_atb", "native_atb", "batched_aclnn", "legacy"),
        default="legacy",
        help="310P Decode Attention backend; Prefill remains on its validated path",
    )
    parser.add_argument(
        "--decode-diagnostics", action="store_true",
        help=("teacher-force reference tokens through legacy and ascendc_asr "
              "Attention and report per-step hidden/logits drift"),
    )
    parser.add_argument("--allow-non-310p", action="store_true")
    parser.add_argument("--report", type=Path, default=Path("poc_310p_report.json"))
    args = parser.parse_args()

    if args.decode_tokens < 16:
        parser.error("--decode-tokens must be at least 16 for POC acceptance")
    if args.prompt_tokens + args.decode_tokens > args.max_seq_len:
        parser.error("prompt and decode lengths exceed --max-seq-len")
    if os.environ.get("WORLD_SIZE", "1") != "1":
        parser.error("310P POC requires WORLD_SIZE=1")

    device_name = _device_name()
    if "310P" not in device_name.upper() and not args.allow_non_310p:
        parser.error(f"expected an Ascend 310P device, detected {device_name!r}")

    torch.npu.set_device(0)
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(20260907)
    model_config = load_qwen3_asr_llm_args(
        args.checkpoint, max_seq_len=args.max_seq_len, max_batch_size=1)
    if args.num_layers is not None:
        if args.num_layers <= 0 or args.num_layers > model_config["n_layers"]:
            parser.error("--num-layers must be between 1 and the checkpoint layer count")
        model_config["n_layers"] = args.num_layers
    model_args = Qwen3ModelArgs(**model_config)
    with torch.device("npu"):
        model = Llama(model_args)
    model.load_weights(args.checkpoint)
    if not hasattr(model.xlite_rt, "set_matmul_backend_310p"):
        raise RuntimeError("installed Xlite does not expose selectable 310P MatMul backends")
    model.xlite_rt.set_matmul_backend_310p(args.matmul_backend)
    if not hasattr(model.xlite_rt, "set_decode_attention_backend"):
        raise RuntimeError("installed Xlite does not expose selectable Decode Attention")
    model.xlite_rt.set_decode_attention_backend(args.decode_attention_backend)
    if args.matmul_backend == "ascendc_asr":
        required_stats = {
            "ascendc_asr_requests",
            "ascendc_asr_kernel_launches",
            "ascendc_asr_bypass_requests",
            "ascendc_asr_rmsnorm_requests",
            "ascendc_asr_add_rmsnorm_requests",
            "ascendc_asr_qk_norm_mrope_cache_requests",
            "ascendc_asr_silu_mul_requests",
        }
        available_stats = set(dict(model.xlite_rt.get_stats()))
        missing_stats = sorted(required_stats - available_stats)
        if missing_stats:
            raise RuntimeError(
                "loaded Xlite extension is stale for ascendc_asr: "
                f"extension={Path(xlite_c.__file__).resolve()}, "
                f"missing runtime stats={missing_stats}; rebuild with "
                "'pip install -v -e . --no-build-isolation'"
            )
    if args.decode_attention_backend == "ascendc_asr":
        required_attention_stats = {
            "ascendc_asr_decode_attention_requests",
            "ascendc_asr_decode_attention_launches",
            "legacy_decode_attention_requests",
            "decode_kv_gather_bytes",
        }
        available_stats = set(dict(model.xlite_rt.get_stats()))
        missing_stats = sorted(required_attention_stats - available_stats)
        if missing_stats:
            raise RuntimeError(
                "loaded Xlite extension is stale for ascendc_asr attention: "
                f"extension={Path(xlite_c.__file__).resolve()}, "
                f"missing runtime stats={missing_stats}; rebuild with "
                "'pip install -v -e . --no-build-isolation'"
            )
    if args.aclnn_matmul_async:
        if not hasattr(model.xlite_rt, "set_aclnn_matmul_async_310p"):
            raise RuntimeError("installed Xlite does not expose ACLNN MatMul event leases")
        model.xlite_rt.set_aclnn_matmul_async_310p(True)

    non_fp16_parameters = [name for name, value in model.named_parameters()
                           if value.dtype != torch.float16]
    if non_fp16_parameters:
        raise RuntimeError(f"BF16/non-FP16 model parameters remain: {non_fp16_parameters[:8]}")
    if any(cache.dtype != torch.float16 for pair in model.xlite_kv_cache for cache in pair):
        raise RuntimeError("BF16/non-FP16 tensors remain in Xlite KV cache")

    if args.input_mode == "tokens":
        tokenizer = AutoTokenizer.from_pretrained(args.checkpoint, trust_remote_code=True)
        token_ids = tokenizer.encode(args.prompt)[:args.prompt_tokens]
        if not token_ids:
            raise ValueError("prompt produced no tokens")
        input_ids = torch.tensor([token_ids], dtype=torch.int64, device="npu")
        inputs_embeds = model.embed_tokens(input_ids).to(torch.float16)
    elif args.input_mode == "synthetic":
        inputs_embeds = torch.randn(
            1, args.prompt_tokens, model_args.dim, dtype=torch.float16, device="npu"
        ) * 0.02
    else:
        if args.embeds_file is None or args.positions_file is None:
            parser.error("--input-mode=file requires --embeds-file and --positions-file")
        inputs_embeds = _load_tensor_file(args.embeds_file)
        if inputs_embeds.ndim == 2:
            inputs_embeds = inputs_embeds.unsqueeze(0)
        if inputs_embeds.ndim != 3 or inputs_embeds.shape[0] != 1:
            parser.error("embedding file must have shape [tokens, hidden] or [1, tokens, hidden]")
        if inputs_embeds.shape[2] != model_args.dim:
            parser.error("embedding file hidden size does not match thinker_config.text_config")
        inputs_embeds = inputs_embeds.to(device="npu", dtype=torch.float16)

    total_position_count = inputs_embeds.size(1) + args.decode_tokens
    if args.positions_file is not None:
        all_positions = _load_tensor_file(args.positions_file).to(device="npu", dtype=torch.int64)
        if tuple(all_positions.shape) not in ((total_position_count,), (3, total_position_count)):
            parser.error(
                "positions file must cover prompt plus decode and have shape "
                f"[{total_position_count}] or [3, {total_position_count}]"
            )
    elif model_args.mrope_section:
        if args.input_mode == "file":
            parser.error("real embedding validation requires its processor-produced positions")
        linear = torch.arange(total_position_count, device="npu", dtype=torch.int64)
        all_positions = linear.repeat(3, 1)
    else:
        all_positions = torch.arange(total_position_count, device="npu", dtype=torch.int64)

    if int(all_positions.min().item()) < 0 or int(all_positions.max().item()) >= args.max_seq_len:
        parser.error("positions must be in [0, --max-seq-len)")

    prompt_positions = all_positions[..., :inputs_embeds.size(1)]

    _clear_caches(model)
    (reference_logits, reference_hidden), reference_prefill_ms = _sync_ms(
        lambda: model.forward_naive_with_inputs_embeds(
            inputs_embeds, 0, return_hidden=True, positions=prompt_positions)
    )
    (xlite_logits, xlite_hidden), xlite_prefill_ms = _sync_ms(
        lambda: model.forward_xlite_with_inputs_embeds(
            inputs_embeds, 0, return_hidden=True, positions=prompt_positions)
    )

    _require_nonzero_finite("reference hidden state", reference_hidden)
    _require_nonzero_finite("Xlite hidden state", xlite_hidden)
    _require_nonzero_finite("reference logits", reference_logits)
    _require_nonzero_finite("Xlite logits", xlite_logits)

    hidden_cosine = _cosine(reference_hidden, xlite_hidden)
    logits_cosine = _cosine(reference_logits, xlite_logits)
    reference_next_token = reference_logits.argmax(dim=-1)
    reference_generated = []
    reference_start_pos = inputs_embeds.size(1)
    for _ in range(args.decode_tokens):
        reference_generated.append(int(reference_next_token.item()))
        reference_embed = model.embed_tokens(reference_next_token.view(1, 1)).to(torch.float16)
        decode_positions = all_positions[..., reference_start_pos:reference_start_pos + 1]
        reference_logits = model.forward_naive_with_inputs_embeds(
            reference_embed, reference_start_pos, positions=decode_positions)
        reference_next_token = reference_logits.argmax(dim=-1)
        reference_start_pos += 1

    next_token = xlite_logits.argmax(dim=-1)
    generated = []
    decode_ms = []
    start_pos = inputs_embeds.size(1)
    for _ in range(args.decode_tokens):
        generated.append(int(next_token.item()))
        token_embed = model.embed_tokens(next_token.view(1, 1)).to(torch.float16)
        logits, elapsed = _sync_ms(
            lambda token_embed=token_embed, start_pos=start_pos:
                model.forward_xlite_with_inputs_embeds(
                    token_embed, start_pos,
                    positions=all_positions[..., start_pos:start_pos + 1])
        )
        decode_ms.append(elapsed)
        next_token = logits.argmax(dim=-1)
        start_pos += 1

    def _run_complete_xlite_sequence() -> None:
        stability_logits = model.forward_xlite_with_inputs_embeds(
            inputs_embeds, 0, positions=prompt_positions)
        stability_token = stability_logits.argmax(dim=-1)
        stability_position = inputs_embeds.size(1)
        for _ in range(args.decode_tokens):
            stability_embed = model.embed_tokens(stability_token.view(1, 1)).to(torch.float16)
            stability_logits = model.forward_xlite_with_inputs_embeds(
                stability_embed, stability_position,
                positions=all_positions[..., stability_position:stability_position + 1])
            stability_token = stability_logits.argmax(dim=-1)
            stability_position += 1

    _clear_caches(model)
    _run_complete_xlite_sequence()
    torch.npu.synchronize()
    initial_memory = int(torch.npu.memory_allocated())
    for _ in range(args.stability_iters):
        _clear_caches(model)
        _run_complete_xlite_sequence()
    torch.npu.synchronize()
    final_memory = int(torch.npu.memory_allocated())

    runtime_stats = dict(model.xlite_rt.get_stats())
    peak_memory_bytes = int(torch.npu.max_memory_allocated())
    backend_acceptance = {}
    if args.matmul_backend == "ascendc_asr":
        backend_acceptance = {
            "ascendc_asr_projection_hit": runtime_stats["ascendc_asr_requests"] > 0,
            "ascendc_asr_projection_launches_match": (
                runtime_stats["ascendc_asr_kernel_launches"] ==
                runtime_stats["ascendc_asr_requests"]
            ),
            "ascendc_asr_rmsnorm_hit": runtime_stats["ascendc_asr_rmsnorm_requests"] > 0,
            "ascendc_asr_add_rmsnorm_hit": (
                runtime_stats["ascendc_asr_add_rmsnorm_requests"] > 0
            ),
            "ascendc_asr_qk_norm_mrope_cache_hit": (
                runtime_stats["ascendc_asr_qk_norm_mrope_cache_requests"] > 0
            ),
            "ascendc_asr_silu_mul_hit": runtime_stats["ascendc_asr_silu_mul_requests"] > 0,
        }
    if args.decode_attention_backend == "ascendc_asr":
        backend_acceptance.update({
            "ascendc_asr_decode_attention_hit": (
                runtime_stats["ascendc_asr_decode_attention_requests"] > 0
            ),
            "ascendc_asr_decode_attention_launches_match": (
                runtime_stats["ascendc_asr_decode_attention_launches"] ==
                runtime_stats["ascendc_asr_decode_attention_requests"]
            ),
            "ascendc_asr_zero_legacy_decode": (
                runtime_stats["legacy_decode_attention_requests"] == 0
            ),
            "ascendc_asr_zero_decode_kv_gather": (
                runtime_stats["decode_kv_gather_bytes"] == 0
            ),
        })

    # Keep the production-path counters above uncontaminated.  Diagnostics
    # deliberately run both legacy and AscendC backends after the snapshot.
    decode_diagnostics = None
    if args.decode_diagnostics:
        decode_diagnostics = _build_decode_diagnostics(
            model, args.decode_attention_backend, inputs_embeds,
            prompt_positions, all_positions, reference_generated)

    report = {
        "device": device_name,
        "dtype": "float16",
        "xlite_extension": str(Path(xlite_c.__file__).resolve()),
        "xlite_build_info": dict(xlite_c.get_build_info()),
        "weight_load": model.weight_load_report,
        "batch_size": 1,
        "num_layers": model_args.n_layers,
        "input_mode": args.input_mode,
        "positions_shape": list(all_positions.shape),
        "mrope_section": model_args.mrope_section,
        "mrope_interleaved": model_args.mrope_interleaved,
        "prompt_tokens": int(inputs_embeds.size(1)),
        "decode_tokens": args.decode_tokens,
        "generated_token_ids": generated,
        "reference_token_ids": reference_generated,
        "hidden_cosine": hidden_cosine,
        "logits_cosine": logits_cosine,
        "greedy_first_token_match": bool(
            reference_generated[0] == generated[0]
        ),
        "greedy_sequence_match": reference_generated == generated,
        "reference_prefill_ms": reference_prefill_ms,
        "xlite_prefill_ms": xlite_prefill_ms,
        "decode_ms": decode_ms,
        "mean_decode_ms": sum(decode_ms) / len(decode_ms),
        "peak_memory_bytes": peak_memory_bytes,
        "stability_iterations": args.stability_iters,
        "memory_growth_bytes": final_memory - initial_memory,
        "xlite_runtime_stats": runtime_stats,
        "decode_diagnostics": decode_diagnostics,
        "acceptance": {
            "hidden_cosine_gte_0_999": hidden_cosine >= 0.999,
            "logits_cosine_gte_0_999": logits_cosine >= 0.999,
            "first_token_match": bool(
                reference_generated[0] == generated[0]
            ),
            "greedy_16_token_match": reference_generated == generated,
            "no_memory_growth": final_memory <= initial_memory,
            **backend_acceptance,
        },
    }
    report["passed"] = all(report["acceptance"].values())
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
