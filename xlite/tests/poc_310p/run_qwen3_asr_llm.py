#!/usr/bin/env python3
"""Ascend 310P single-card FP16 Qwen3-ASR decoder penetration test."""

from __future__ import annotations

import argparse
import hashlib
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
import torch_npu
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
    for key_cache, value_cache in getattr(model, "xlite_native_kv_cache", ()):
        key_cache.zero_()
        value_cache.zero_()


def _cpu_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _device_name() -> str:
    get_name = getattr(torch.npu, "get_device_name", None)
    return str(get_name(0)) if get_name is not None else "unknown"


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
        choices=("ascendc_asr_nz", "ascendc_asr_perf", "ascendc_asr", "m200_asr_prefill",
                 "m200_asr", "aclnn"),
        default="m200_asr",
        help=("310P MatMul backend; ascendc_asr_nz selects strict FP16 "
              "ND x FRACTAL_NZ Prefill/Decode kernels with no fallback; "
              "ascendc_asr_perf selects performance-gated "
              "Decode projection/SiLU/LM Head candidates while ascendc_asr "
              "keeps the production baselines"),
    )
    parser.add_argument(
        "--decode-attention-backend",
        choices=("ascendc_asr_nz", "ascendc_asr", "direct_atb", "native_atb",
                 "batched_aclnn", "legacy"),
        default="legacy",
        help="310P Decode Attention backend; Prefill remains on its validated path",
    )
    parser.add_argument(
        "--save-decode-diagnostic-bundle", type=Path,
        help=("save the exact CPU prompt, positions and reference tokens for "
              "the isolated Decode diagnostic workers"),
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

    # The caller owns logical-to-physical device mapping and the current NPU
    # context.  Do not re-select logical device 0 from a validation worker.
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
    if args.decode_attention_backend == "ascendc_asr_nz":
        native_cache = []
        for key_cache, _ in model.xlite_kv_cache:
            cache_shape = (key_cache.shape[0], 64, 128, 16)
            native_cache.append([
                torch_npu.empty_with_format(
                    size=cache_shape, dtype=torch.float16, device="npu", acl_format=29),
                torch_npu.empty_with_format(
                    size=cache_shape, dtype=torch.float16, device="npu", acl_format=29),
            ])
        model.xlite_native_kv_cache = native_cache
        model.xlite_model.set_native_kv_cache_310p(native_cache)
    if args.matmul_backend in ("ascendc_asr_nz", "ascendc_asr", "ascendc_asr_perf"):
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
        if args.decode_attention_backend == "ascendc_asr_nz":
            required_nz_attention_stats = {
                "ascendc_asr_nz_cache_write_requests",
                "ascendc_asr_nz_decode_attention_requests",
                "ascendc_asr_nz_decode_attention_launches",
            }
            missing_nz_attention_stats = sorted(
                required_nz_attention_stats - available_stats
            )
            if missing_nz_attention_stats:
                raise RuntimeError(
                    "loaded Xlite extension is stale for ascendc_asr_nz attention: "
                    f"extension={Path(xlite_c.__file__).resolve()}, "
                    f"missing runtime stats={missing_nz_attention_stats}"
                )
        if args.matmul_backend == "ascendc_asr_perf":
            required_perf_stats = {
                "ascendc_asr_perf_projection_requests",
                "ascendc_asr_perf_lm_head_requests",
                "ascendc_asr_perf_silu_mul_requests",
            }
            missing_perf_stats = sorted(required_perf_stats - available_stats)
            if missing_perf_stats:
                raise RuntimeError(
                    "loaded Xlite extension is stale for ascendc_asr_perf: "
                    f"extension={Path(xlite_c.__file__).resolve()}, "
                    f"missing runtime stats={missing_perf_stats}; rebuild with "
                    "'pip install -v -e . --no-build-isolation'"
                )
        if args.matmul_backend == "ascendc_asr_nz":
            required_nz_stats = {
                "ascendc_asr_nz_projection_requests",
                "ascendc_asr_nz_lm_head_requests",
            }
            missing_nz_stats = sorted(required_nz_stats - available_stats)
            if missing_nz_stats:
                raise RuntimeError(
                    "loaded Xlite extension is stale for ascendc_asr_nz: "
                    f"extension={Path(xlite_c.__file__).resolve()}, "
                    f"missing runtime stats={missing_nz_stats}; rebuild with "
                    "'pip install -v -e . --no-build-isolation'"
                )
    if args.decode_attention_backend in ("ascendc_asr", "ascendc_asr_nz"):
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
    # Freeze the replay contract before either implementation can touch model
    # state.  Version 2 bundles are only produced from independently cleared
    # Naive and Xlite cache lifetimes.
    diagnostic_inputs_cpu = inputs_embeds.detach().cpu().contiguous()
    diagnostic_positions_cpu = all_positions.detach().cpu().contiguous()

    _clear_caches(model)
    (reference_prefill_logits,
     reference_hidden), reference_prefill_ms = _sync_ms(
        lambda: model.forward_naive_with_inputs_embeds(
            inputs_embeds, 0, return_hidden=True, positions=prompt_positions)
    )
    # Keep the Prefill comparison tensors independent from any temporary
    # storage reused by subsequent Decode calls.
    reference_prefill_logits = reference_prefill_logits.clone()
    reference_hidden = reference_hidden.clone()

    _require_nonzero_finite("reference hidden state", reference_hidden)
    _require_nonzero_finite("reference logits", reference_prefill_logits)
    reference_next_token = reference_prefill_logits.argmax(dim=-1)
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

    # The two implementations must never inherit each other's KV state.  The
    # former ordering ran Xlite Prefill before completing the Naive sequence,
    # so the serialized reference could not be reproduced from its own saved
    # inputs in a fresh process.
    _clear_caches(model)
    (xlite_logits, xlite_hidden), xlite_prefill_ms = _sync_ms(
        lambda: model.forward_xlite_with_inputs_embeds(
            inputs_embeds, 0, return_hidden=True, positions=prompt_positions)
    )

    _require_nonzero_finite("Xlite hidden state", xlite_hidden)
    _require_nonzero_finite("Xlite logits", xlite_logits)

    hidden_cosine = _cosine(reference_hidden, xlite_hidden)
    logits_cosine = _cosine(reference_prefill_logits, xlite_logits)

    if args.save_decode_diagnostic_bundle is not None:
        args.save_decode_diagnostic_bundle.parent.mkdir(
            parents=True, exist_ok=True)
        torch.save({
            "format_version": 2,
            "reference_protocol": "clean_naive_cache_then_clean_xlite_cache",
            "inputs_embeds": diagnostic_inputs_cpu,
            "inputs_sha256": _cpu_sha256(diagnostic_inputs_cpu),
            "all_positions": diagnostic_positions_cpu,
            "positions_sha256": _cpu_sha256(diagnostic_positions_cpu),
            "reference_token_ids": reference_generated,
            "prompt_tokens": int(inputs_embeds.size(1)),
            "decode_tokens": args.decode_tokens,
        }, args.save_decode_diagnostic_bundle)

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
    if args.matmul_backend in ("ascendc_asr_nz", "ascendc_asr", "ascendc_asr_perf"):
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
    if args.matmul_backend == "ascendc_asr_perf":
        backend_acceptance.update({
            "ascendc_asr_perf_projection_hit": (
                runtime_stats["ascendc_asr_perf_projection_requests"] > 0
            ),
            "ascendc_asr_perf_lm_head_hit": (
                runtime_stats["ascendc_asr_perf_lm_head_requests"] > 0
            ),
            "ascendc_asr_perf_silu_mul_hit": (
                runtime_stats["ascendc_asr_perf_silu_mul_requests"] > 0
            ),
        })
    if args.decode_attention_backend in ("ascendc_asr", "ascendc_asr_nz"):
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
        if args.decode_attention_backend == "ascendc_asr_nz":
            backend_acceptance.update({
                "ascendc_asr_nz_cache_write_hit": (
                    runtime_stats["ascendc_asr_nz_cache_write_requests"] > 0
                ),
                "ascendc_asr_nz_attention_hit": (
                    runtime_stats["ascendc_asr_nz_decode_attention_requests"] > 0
                ),
            })

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
        "decode_diagnostic_bundle": (
            str(args.save_decode_diagnostic_bundle.resolve())
            if args.save_decode_diagnostic_bundle is not None else None
        ),
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
