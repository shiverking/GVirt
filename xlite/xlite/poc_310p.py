"""Configuration helpers for the Ascend 310P Qwen3-ASR LLM-only POC."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any


_REQUIRED_TEXT_FIELDS = (
    "hidden_size",
    "intermediate_size",
    "num_hidden_layers",
    "num_attention_heads",
    "num_key_value_heads",
    "vocab_size",
)


def extract_text_config(config: dict[str, Any]) -> dict[str, Any]:
    """Return the Qwen3 decoder config from a Qwen3-ASR or text-only config."""
    candidates = (
        config.get("thinker_config", {}).get("text_config"),
        config.get("text_config"),
        config,
    )
    text_config = next((item for item in candidates if isinstance(item, dict)), None)
    if text_config is None:
        raise ValueError("Qwen3-ASR config does not contain thinker_config.text_config")

    missing = [field for field in _REQUIRED_TEXT_FIELDS if field not in text_config]
    if missing:
        raise ValueError(f"Qwen3 text config is missing required fields: {', '.join(missing)}")
    return text_config


def load_qwen3_asr_llm_args(
    checkpoint: str | Path,
    *,
    max_seq_len: int = 512,
    max_batch_size: int = 1,
) -> dict[str, Any]:
    """Map a Hugging Face Qwen3-ASR config to ``Qwen3ModelArgs`` fields."""
    checkpoint = Path(checkpoint)
    config_path = checkpoint if checkpoint.is_file() else checkpoint / "config.json"
    with config_path.open("r", encoding="utf-8") as config_file:
        text = extract_text_config(json.load(config_file))

    n_heads = int(text["num_attention_heads"])
    hidden_size = int(text["hidden_size"])
    head_dim = int(text.get("head_dim", hidden_size // n_heads))
    if hidden_size != n_heads * head_dim:
        raise ValueError("310P POC requires hidden_size == num_attention_heads * head_dim")

    rope_parameters = text.get("rope_parameters") or text.get("rope_scaling") or {}
    if not isinstance(rope_parameters, dict):
        raise ValueError("rope_parameters/rope_scaling must be an object")
    mrope_section = rope_parameters.get("mrope_section", [])
    if mrope_section is None:
        mrope_section = []
    if not isinstance(mrope_section, list) or any(int(item) < 0 for item in mrope_section):
        raise ValueError("mrope_section must be a list of non-negative integers")
    if mrope_section and sum(int(item) for item in mrope_section) != head_dim // 2:
        raise ValueError("mrope_section must cover head_dim / 2 rotary pairs")

    return {
        "max_batch_size": max_batch_size,
        "max_seq_len": max_seq_len,
        "dim": hidden_size,
        "head_dim": head_dim,
        "inter_dim": int(text["intermediate_size"]),
        "vocab_size": int(text["vocab_size"]),
        "n_layers": int(text["num_hidden_layers"]),
        "n_heads": n_heads,
        "n_kv_heads": int(text["num_key_value_heads"]),
        "norm_eps": float(text.get("rms_norm_eps", 1e-6)),
        "rope_theta": float(text.get("rope_theta", 1_000_000.0)),
        "rope_type": str(rope_parameters.get("rope_type", rope_parameters.get("type", "default"))),
        "mrope_section": [int(item) for item in mrope_section],
        "mrope_interleaved": bool(rope_parameters.get("mrope_interleaved", False)),
        "dtype": "float16",
        "tie_word_embeddings": bool(text.get("tie_word_embeddings", False)),
        "qkv_bias": bool(text.get("attention_bias", False)),
        "qk_norm": True,
        "model_type": "qwen3",
    }
