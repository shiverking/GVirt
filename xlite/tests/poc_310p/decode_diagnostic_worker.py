#!/usr/bin/env python3
"""Run one teacher-forced Decode backend in a fresh process and Runtime."""

from __future__ import annotations

import argparse
import hashlib
import os
import sys
from pathlib import Path

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
import torch_npu

from tests.models.llama import Llama
from tests.models.qwen3 import Qwen3ModelArgs
from xlite.poc_310p import load_qwen3_asr_llm_args


def _clear_caches(model: Llama) -> None:
    for layer in model.layers:
        layer.self_attn.k_cache.zero_()
        layer.self_attn.v_cache.zero_()
    for key_cache, value_cache in model.xlite_kv_cache:
        key_cache.zero_()
        value_cache.zero_()


def _copy_snapshot(
    logits: torch.Tensor, hidden: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    torch.npu.synchronize()
    return logits.detach().cpu().clone(), hidden.detach().cpu().clone()


def _cpu_sha256(value: torch.Tensor) -> str:
    contiguous = value.detach().cpu().contiguous()
    return hashlib.sha256(contiguous.numpy().tobytes()).hexdigest()


def _to_npu_nd(value: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Upload after Runtime initialization and force ACL_FORMAT_ND (2)."""
    uploaded = value.to(device="npu", dtype=dtype).contiguous()
    uploaded = torch_npu.npu_format_cast(uploaded, 2)
    return uploaded.contiguous()


def _npu_format(value: torch.Tensor) -> int | None:
    getter = getattr(torch_npu, "get_npu_format", None)
    return int(getter(value)) if getter is not None else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument(
        "--backend", choices=("legacy", "ascendc_asr"), required=True)
    parser.add_argument("--matmul-backend", default="ascendc_asr")
    parser.add_argument("--max-seq-len", type=int, required=True)
    parser.add_argument("--num-layers", type=int)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    torch.npu.set_device(0)
    torch.set_default_dtype(torch.float16)
    torch.manual_seed(20260907)

    bundle = torch.load(args.bundle, map_location="cpu", weights_only=True)
    if bundle.get("format_version") != 1:
        raise RuntimeError("unsupported Decode diagnostic bundle")
    inputs_embeds_cpu = bundle["inputs_embeds"].to(
        device="cpu", dtype=torch.float16).contiguous()
    all_positions_cpu = bundle["all_positions"].to(
        device="cpu", dtype=torch.int64).contiguous()
    reference_tokens = [int(token) for token in bundle["reference_token_ids"]]

    model_config = load_qwen3_asr_llm_args(
        args.checkpoint, max_seq_len=args.max_seq_len, max_batch_size=1)
    if args.num_layers is not None:
        model_config["n_layers"] = args.num_layers
    model_args = Qwen3ModelArgs(**model_config)
    with torch.device("npu"):
        model = Llama(model_args)
    model.load_weights(args.checkpoint)
    model.xlite_rt.set_matmul_backend_310p(args.matmul_backend)
    model.xlite_rt.set_decode_attention_backend(args.backend)

    # Match the generator's allocation order: construct the model and native
    # Runtime first, then materialize replay inputs.  CPU->NPU copies may
    # otherwise select an internal storage format even when logical strides
    # are contiguous; Xlite consumes these buffers as raw ACL_FORMAT_ND.
    inputs_embeds = _to_npu_nd(inputs_embeds_cpu, torch.float16)
    all_positions = _to_npu_nd(all_positions_cpu, torch.int64)

    _clear_caches(model)
    # Cache zeroing and bundle tensor uploads run on the torch_npu stream,
    # while Xlite owns a separate ACL stream.  This diagnostic process has no
    # preceding model call to establish the normal stream handoff, so close
    # the initialization boundary explicitly before the first Xlite Prefill.
    # This synchronization is diagnostic-only and is outside measured serving
    # or Decode paths.
    torch.npu.synchronize()
    prompt_tokens = int(inputs_embeds.size(1))
    prefill_logits, prefill_hidden = model.forward_xlite_with_inputs_embeds(
        inputs_embeds, 0, return_hidden=True,
        positions=all_positions[..., :prompt_tokens])
    prefill_logits, prefill_hidden = _copy_snapshot(
        prefill_logits, prefill_hidden)

    decode_logits = []
    decode_hidden = []
    position = prompt_tokens
    for token in reference_tokens:
        token_tensor = torch.tensor([[token]], dtype=torch.int64, device="npu")
        token_embed = model.embed_tokens(token_tensor).to(torch.float16)
        logits, hidden = model.forward_xlite_with_inputs_embeds(
            token_embed, position, return_hidden=True,
            positions=all_positions[..., position:position + 1])
        logits_cpu, hidden_cpu = _copy_snapshot(logits, hidden)
        decode_logits.append(logits_cpu)
        decode_hidden.append(hidden_cpu)
        position += 1

    replay_input = {
        "inputs_shape": list(inputs_embeds_cpu.shape),
        "inputs_sha256": _cpu_sha256(inputs_embeds_cpu),
        "inputs_npu_format": _npu_format(inputs_embeds),
        "positions_shape": list(all_positions_cpu.shape),
        "positions_sha256": _cpu_sha256(all_positions_cpu),
        "positions_npu_format": _npu_format(all_positions),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "format_version": 1,
        "backend": args.backend,
        "prefill_logits": prefill_logits,
        "prefill_hidden": prefill_hidden,
        "decode_logits": torch.stack(decode_logits),
        "decode_hidden": torch.stack(decode_hidden),
        "runtime_stats": dict(model.xlite_rt.get_stats()),
        "replay_input": replay_input,
    }, args.output)
    print(f"replay input contract: {replay_input}")
    print(f"isolated Decode diagnostic worker PASS: backend={args.backend}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
