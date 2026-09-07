#!/usr/bin/env python3
"""Ascend 310P single-card FP16 Qwen3-ASR decoder penetration test."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("FORWARD_BACKEND", "xlite")
os.environ.setdefault("XLITE_DP_SIZE", "1")
os.environ.setdefault("XLITE_TP_SIZE", "1")
os.environ.setdefault("XLITE_WEIGHT_NZ", "0")

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer

from tests.models.llama import Llama
from tests.models.qwen3 import Qwen3ModelArgs
from xlite.poc_310p import load_qwen3_asr_llm_args


def _sync_ms(callable_):
    torch.npu.synchronize()
    started = time.perf_counter_ns()
    result = callable_()
    torch.npu.synchronize()
    return result, (time.perf_counter_ns() - started) / 1_000_000


def _cosine(left: torch.Tensor, right: torch.Tensor) -> float:
    return float(F.cosine_similarity(left.float().flatten(), right.float().flatten(), dim=0).cpu())


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

    report = {
        "device": device_name,
        "dtype": "float16",
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
        "peak_memory_bytes": int(torch.npu.max_memory_allocated()),
        "stability_iterations": args.stability_iters,
        "memory_growth_bytes": final_memory - initial_memory,
        "acceptance": {
            "hidden_cosine_gte_0_999": hidden_cosine >= 0.999,
            "logits_cosine_gte_0_999": logits_cosine >= 0.999,
            "first_token_match": bool(
                reference_generated[0] == generated[0]
            ),
            "greedy_16_token_match": reference_generated == generated,
            "no_memory_growth": final_memory <= initial_memory,
        },
    }
    report["passed"] = all(report["acceptance"].values())
    args.report.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return 0 if report["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
