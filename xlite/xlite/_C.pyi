"""Type stubs for :mod:`xlite._C`.

This file mirrors the symbols exported by `csrc/_C.cpp`.
The typing and docstrings are designed for Python 3.9 to 3.12.
"""

from __future__ import annotations

from enum import Enum
from typing import List, Optional, Sequence, Union

import torch

class Runtime:
    """Ascend runtime handle for streams, communication, and tensor pools.

    Attributes:
        task_id (int): Current task slot used by multi-task parallel execution.
        notify (object): Runtime notify handle used for cross-stream synchronization.
        peer_notify (object): Peer notify handle used by the peer-stream sync path.
        multi_task_parallel (bool): Enables the dual-task scheduling path.
    """

    task_id: int = ...
    """Current task slot used by multi-task parallel execution."""
    notify: object = ...
    """Runtime notify handle used for cross-stream synchronization."""
    peer_notify: object = ...
    """Peer notify handle used by the peer-stream sync path."""
    multi_task_parallel: bool = ...
    """Enables the dual-task scheduling path."""
    aic_num: int = ...
    """Number of AI cores reported by ACL."""
    aiv_num: int = ...
    """Logical vector-kernel launch block count used by GVirt."""
    reported_aiv_num: int = ...
    """Dedicated vector-core count reported by ACL before 310P mapping."""

    def __init__(
        self,
        devid: int,
        size: int = 0,
        rank: int = 0,
        tp_size: int = 1,
        dp_size: int = 1,
        moe_tp_size: int = 1,
        moe_ep_size: int = 1,
    ) -> None:
        """Create a runtime for one Ascend device.

        Args:
            devid (int): Device ID passed to the ACL runtime.
            size (int): Optional tensor-pool size in MB.
            rank (int): Global rank in the distributed group.
            tp_size (int): Tensor-parallel group size.
            dp_size (int): Data-parallel group size.
            moe_tp_size (int): Tensor-parallel size for MoE layers.
            moe_ep_size (int): Expert-parallel size for MoE layers.
        """

    def update_core_num(self, util: float) -> None:
        """Scale active AI/vector core counts by utilization.

        Args:
            util (float): Utilization ratio, typically in `[0, 1]`.

        Returns:
            None: The runtime updates rounded AI/vector core counts.
        """

    def init_tensor_pool(self, size: int) -> int:
        """Initialize the tensor pool for this runtime.

        Args:
            size (int): Requested pool size in megabytes.

        Returns:
            int: `0` on success.

        Note:
            Passing `0` leaves the current pool unchanged.
        """

    def set_current_context(self) -> None:
        """Set the runtime ACL context as current.

        Returns:
            None: Context is updated on the calling thread.
        """

    def configure_swizzle(self, swizzle: int, use_swizzle_table: bool) -> None:
        """Configure swizzle parameters for matrix multiplication

        Args:
            swizzle(int): new default swizzle valuey.
            use_swizzle_table(bool): Whether to use precomputed swizzle values from xlite.

        Returns:
            None: The runtime updates swizzle configuration.
        """

class ModelConfig:
    """Model hyperparameters and layout used by :class:`Model`.

    Attributes:
        vocab_size (int): Vocabulary size for embedding and LM head.
        hidden_size (int): Transformer hidden width.
        n_layers (int): Total number of transformer layers.
        attn_type (AttnType): Attention family.
        n_heads (int): Number of attention heads.
        n_kv_heads (int): Number of key/value heads.
        head_dim (int): Per-head dimension for MHA.
        nope_head_dim (int): Non-RoPE query dimension for MLA/DSA.
        rope_head_dim (int): RoPE query/value dimension.
        v_head_dim (int): Value projection dimension.
        q_lora_rank (int): LoRA rank for query projection.
        kv_lora_rank (int): LoRA rank for key/value projection.
        quant_attn_weight_transpose (bool): Whether quanted attention weights are transposed.
        quant_attn_weight_nz (bool): Whether quanted attention weights are in NZ layout.
        norm_eps (float): RMSNorm/LayerNorm epsilon.
        rope_theta (float): Rotary base frequency.
        softmax_scale (float): Attention softmax scale.
        n_dense_layers (int): Number of dense FFN layers.
        n_routed_experts (int): Number of routed experts.
        n_shared_experts (int): Number of shared experts.
        n_expert_groups (int): Number of expert groups.
        n_limited_groups (int): Number of limited routing groups.
        n_act_experts (int): Number of active experts.
        intermediate_size (int): Dense FFN intermediate width.
        moe_intermediate_size (int): MoE FFN intermediate width.
        route_scale (float): Routing score scale.
        def_tp_size (int): Default tensor-parallel size.
        def_dp_size (int): Default data-parallel size.
        moe_ep_size (int): Expert-parallel size.
        moe_tp_size (int): tensor-parallel size in MoE.
        max_seq_len (int): Maximum sequence length.
        max_batch_size (int): Maximum batch size.
        max_m (int): Maximum token count batched.
        max_num_batched_tokens (int): Maximum token count batched.
        block_size (int): KV block size.(deprecated)
        block_sizes (List[int]): Per-attention-type KV block sizes. Single-element for
            non-CXA; for CXA one entry per 5-tuple cache in order
            ``(indexer_state, indexer_k, compress_kv, state, swa_kv)``. Seeded from
            ``block_size`` when empty at construction.
        weight_nz (bool): Whether weights are in NZ layout.
        experts_weight_transpose (bool): Whether expert weights are transposed.
        experts_weight_nz (bool): Whether expert weights are in NZ layout.
        gate_captured(bool): Whether gate layer is captured by vllm-ascend.
        qkv_bias (bool): Whether MHA QKV has bias.
        qk_norm (bool): Whether MHA applies Q/K norm.
        qk_norm_full (bool): Whether MHA applies Q/K norm full.
        attn_output_gate (bool): Whether full MHA applies sigmoid output gate (Qwen3.5).
            When true, fused mha_qkv layout is [Q | K | V | Gate].
        scoring_func (ScoringFuncType): MoE scoring function.
        norm_topk_prob (bool): Whether top-k probabilities are normalized.
        mrope_section (List[int]): mRoPE section layout values.
        mrope_interleaved (bool): Whether mRoPE layout is interleaved.
        deepstack_num_level (int): Number of deepstack levels.
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count.
        index_topk (int): Indexer top-k size.
        index_softmax_scale (float): Indexer softmax scale.(deprecated)
        index_rope_interleaved (bool): Whether indexer RoPE is interleaved.
        o_groups (int): Output projection group count (DeepSeek-V4).
        o_lora_rank (int): Output projection LoRA rank (DeepSeek-V4).
        window_size (int): Sliding window attention size (DeepSeek-V4).
        compress_rope_theta (float): KV compressor RoPE base (DeepSeek-V4).
        original_seq_len (int): YaRN original sequence length (DeepSeek-V4).
        rope_factor (float): YaRN extension factor (DeepSeek-V4).
        beta_fast (int): YaRN beta_fast (DeepSeek-V4).
        beta_slow (int): YaRN beta_slow (DeepSeek-V4).
        hc_mult (int): Multi-stage Hyper-Connections multiplier (DeepSeek-V4).
        hc_sinkhorn_iters (int): MHC Sinkhorn iterations (DeepSeek-V4).
        hc_eps (float): MHC Sinkhorn epsilon (DeepSeek-V4).
        swiglu_limit (float): SwiGLU clamp limit; 0 disables (DeepSeek-V4).
        n_hash_layers (int): Number of hash-routed MoE layers (DeepSeek-V4).
        compress_ratios (List[int]): Per-layer KV compression ratios (DeepSeek-V4).
    """

    vocab_size: int = ...
    """Vocabulary size for embedding and LM head."""
    hidden_size: int = ...
    """Transformer hidden width."""
    n_layers: int = ...
    """Total number of transformer layers."""
    attn_type: AttnType = ...
    """Attention family."""
    rope_type: RopeType = ...
    """Rotary embedding layout (NeoX or GPT-J)."""
    n_heads: int = ...
    """Number of attention heads."""
    n_kv_heads: int = ...
    """Number of key/value heads."""
    head_dim: int = ...
    """Per-head dimension for MHA."""
    nope_head_dim: int = ...
    """Non-RoPE query dimension for MLA/DSA."""
    rope_head_dim: int = ...
    """RoPE query/value dimension."""
    v_head_dim: int = ...
    """Value projection dimension."""
    q_lora_rank: int = ...
    """LoRA rank for query projection."""
    kv_lora_rank: int = ...
    """LoRA rank for key/value projection."""
    quant_attn_weight_transpose: bool = ...
    """Whether quanted attention weights are transposed."""
    quant_attn_weight_nz: bool = ...
    """Whether quanted attention weights are in NZ layout."""
    norm_eps: float = ...
    """RMSNorm/LayerNorm epsilon."""
    rope_theta: float = ...
    """Rotary base frequency."""
    softmax_scale: float = ...
    """Attention softmax scale."""
    n_dense_layers: int = ...
    """Number of dense FFN layers."""
    n_routed_experts: int = ...
    """Number of routed experts."""
    n_shared_experts: int = ...
    """Number of shared experts."""
    n_expert_groups: int = ...
    """Number of expert groups."""
    n_limited_groups: int = ...
    """Number of limited routing groups."""
    n_act_experts: int = ...
    """Number of active experts."""
    intermediate_size: int = ...
    """Dense FFN intermediate width."""
    moe_intermediate_size: int = ...
    """MoE FFN intermediate width."""
    route_scale: float = ...
    """Routing score scale."""
    def_tp_size: int = ...
    """Default tensor-parallel size."""
    def_dp_size: int = ...
    """Default data-parallel size."""
    moe_ep_size: int = ...
    """Expert-parallel size."""
    moe_tp_size: int = ...
    """Tensor-parallel size in MoE."""
    max_seq_len: int = ...
    """Maximum sequence length."""
    max_batch_size: int = ...
    """Maximum batch size."""
    max_m: int = ...
    """Maximum token count batched."""
    max_num_batched_tokens: int = ...
    """Maximum token count batched."""
    block_size: int = ...
    """KV block size.(deprecated)"""
    block_sizes: List[int] = ...
    """Per-attention-type KV block sizes. Single-element for non-CXA; for CXA one entry
    per 5-tuple cache in order (indexer_state, indexer_k, compress_kv, state, swa_kv).
    Seeded from ``block_size`` when empty at construction."""
    weight_nz: bool = ...
    """Whether weights are in NZ layout."""
    experts_weight_transpose: bool = ...
    """Whether expert weights are transposed."""
    experts_weight_nz: bool = ...
    """Whether expert weights are in NZ layout."""
    gate_captured: bool = ...
    """Whether gate layer is captured by vllm-ascend."""
    qkv_bias: bool = ...
    """Whether MHA QKV has bias."""
    qk_norm: bool = ...
    """Whether MHA applies Q/K norm."""
    qk_norm_full: bool = ...
    """Whether MHA applies Q/K norm full."""
    attn_output_gate: bool = ...
    """Whether full MHA applies sigmoid output gate (Qwen3.5)."""
    linear_num_k_heads: int = ...
    """Number of key heads for linear attention layers."""
    linear_num_v_heads: int = ...
    """Number of value heads for linear attention layers."""
    linear_key_head_dim: int = ...
    """Key head dim for linear attention layers."""
    linear_value_head_dim: int = ...
    """Value head dim for linear attention layers."""
    linear_conv_kernel_dim: int = ...
    """Causal conv1d kernel size for linear attention layers."""
    full_attention_interval: int = ...
    """Interval between full-attention layers in hybrid models."""
    scoring_func: ScoringFuncType = ...
    """MoE scoring function."""
    norm_topk_prob: bool = ...
    """Whether top-k probabilities are normalized."""
    mrope_section: List[int] = ...
    """mRoPE section layout values."""
    mrope_interleaved: bool = ...
    """Whether mRoPE layout is interleaved."""
    deepstack_num_level: int = ...
    """Number of deepstack levels."""
    index_head_dim: int = ...
    """Indexer head dimension."""
    index_n_heads: int = ...
    """Indexer head count."""
    index_topk: int = ...
    """Indexer top-k size."""
    index_softmax_scale: float = ...
    """Indexer softmax scale.(deprecated)"""
    index_rope_interleaved: bool = ...
    """Whether indexer RoPE is interleaved."""
    index_full_mask: List[bool] = ...
    """Per-layer indexer mask. ``True``: full indexer layer; ``False``: shared indexer layer. Pass as an empty list to
    enable all full indexer layers; otherwise, the list length must match :data:`n_layers`."""
    o_groups: int = ...
    """Output projection group count (DeepSeek-V4)."""
    o_lora_rank: int = ...
    """Output projection LoRA rank (DeepSeek-V4)."""
    window_size: int = ...
    """Sliding window attention size (DeepSeek-V4)."""
    compress_rope_theta: float = ...
    """KV compressor RoPE base (DeepSeek-V4)."""
    original_seq_len: int = ...
    """YaRN original sequence length (DeepSeek-V4)."""
    rope_factor: float = ...
    """YaRN extension factor (DeepSeek-V4)."""
    beta_fast: int = ...
    """YaRN beta_fast (DeepSeek-V4)."""
    beta_slow: int = ...
    """YaRN beta_slow (DeepSeek-V4)."""
    hc_mult: int = ...
    """Multi-stage Hyper-Connections multiplier (DeepSeek-V4)."""
    hc_sinkhorn_iters: int = ...
    """MHC Sinkhorn iterations (DeepSeek-V4)."""
    hc_eps: float = ...
    """MHC Sinkhorn epsilon (DeepSeek-V4)."""
    swiglu_limit: float = ...
    """SwiGLU clamp limit; 0 disables (DeepSeek-V4)."""
    n_hash_layers: int = ...
    """Number of hash-routed MoE layers (DeepSeek-V4)."""
    compress_ratios: List[int] = ...
    """Per-layer KV compression ratios (DeepSeek-V4)."""

class AttnMeta:
    """Attention metadata for the native runtime forward path.

    This path reuses host block-table lists while taking `positions` directly
    from the provided tensor for attention position indexing.

    Attributes:
        lens (List[int]): Per-sample query lengths.
        cached_lens (List[int]): Per-sample cached lengths.
        block_tables_cpu (List[List[int]]): Per-sample block tables on host.
        positions (torch.Tensor): Position tensor for version-1 attention metadata.
    """

    lens: List[int] = ...
    """Per-sample query lengths."""
    cached_lens: List[int] = ...
    """Per-sample cached lengths."""
    block_tables_cpu: List[List[int]] = ...
    """Per-sample block tables on host."""
    positions: torch.Tensor = ...
    """Position tensor for version-1 attention metadata."""


class AttnMetaV2:
    """Device-tensor attention metadata for the V2 forward path.

    Unlike :class:`AttnMeta` (V1), this variant carries ``lens``,
    ``cached_lens``, ``query_start_loc``, ``slot_mapping`` and
    ``block_tables`` as pre-built device tensors. The C++ side skips host
    computation and H2D copies, only shape-checking and aliasing these tensors
    (zero-copy). Per-sample block tables are flattened into a 1D tensor padded
    to ``max_num_blocks``. The host ``lens_cpu``/``cached_lens_cpu`` lists are
    still required for C++-side tile-size selection (``GetTileSizeOfCachedKV``).

    Attributes:
        lens (torch.Tensor): Per-sample query lengths, shape ``[batch]`` int32 device.
        cached_lens (torch.Tensor): Per-sample cached lengths, shape ``[batch]`` int32 device.
        positions (torch.Tensor): Position tensor, shape ``[batched_tokens]`` int64.
        lens_cpu (List[int]): Per-sample query lengths (host, for tile-size selection).
        cached_lens_cpu (List[int]): Per-sample cached lengths (host, for tile-size selection).
        query_start_loc (torch.Tensor): Prefix-sum of lens, shape ``[batch]`` int32.
        slot_mapping (Sequence[torch.Tensor]): Slot indices, each shape
            ``[batched_tokens]`` int32 device, one per kv cache.
        block_tables (Sequence[torch.Tensor]): Padded block tables, each shape
            ``[batch, max_num_blocks]`` int32 device, one per kv cache.
    """

    lens: torch.Tensor = ...
    """Per-sample query lengths, shape ``[batch]`` int32 device."""
    cached_lens: torch.Tensor = ...
    """Per-sample cached lengths, shape ``[batch]`` int32 device."""
    positions: torch.Tensor = ...
    """Position tensor, shape ``[batched_tokens]`` int64."""
    lens_cpu: List[int] = ...
    """Per-sample query lengths (host, for tile-size selection)."""
    cached_lens_cpu: List[int] = ...
    """Per-sample cached lengths (host, for tile-size selection)."""
    query_start_loc: torch.Tensor = ...
    """Prefix-sum of lens, shape ``[batch]`` int32."""
    slot_mapping: Sequence[torch.Tensor] = ...
    """Slot indices, each shape ``[batched_tokens]`` int32 device, per kv cache."""
    block_tables: Sequence[torch.Tensor] = ...
    """Padded block tables, each shape ``[batch, max_num_blocks]`` int32 device, per kv cache."""

class AttnType(Enum):
    """Attention type enum exported by the native extension."""

    AttnMHA = ...
    """Standard multi-head attention."""
    AttnMLA = ...
    """Multi-head latent attention."""
    AttnDSA = ...
    """Dual sparse attention."""
    AttnHybrid = ...
    """Hybrid full + linear attention (Qwen3.5)."""
    AttnCxA = ...
    """C4A / C128A attention (DeepSeek-V4)."""

class RopeType(Enum):
    """Rotary embedding layout enum exported by the native extension."""

    RopeNeox = ...
    RopeGptj = ...

class ScoringFuncType(Enum):
    """MoE scoring function enum exported by the native extension."""

    ScoringFuncSoftmax = ...
    """Softmax-based expert routing."""
    ScoringFuncSigmoid = ...
    """Sigmoid-based expert routing."""
    ScoringFuncSqrtsoftplus = ...
    """Sqrtsoftplus-based expert routing (hash or topk)."""

AttnMHA: AttnType = ...
"""Alias for :attr:`AttnType.AttnMHA`."""
AttnHybrid: AttnType = ...
"""Alias for :attr:`AttnType.AttnHybrid`."""
AttnMLA: AttnType = ...
"""Alias for :attr:`AttnType.AttnMLA`."""
AttnDSA: AttnType = ...
"""Alias for :attr:`AttnType.AttnDSA`."""
AttnCxA: AttnType = ...
"""Alias for :attr:`AttnType.AttnCxA`."""
RopeNeox: RopeType = ...
"""Alias for :attr:`RopeType.RopeNeox`."""
RopeGptj: RopeType = ...
"""Alias for :attr:`RopeType.RopeGptj`."""

ScoringFuncSoftmax: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSoftmax`."""
ScoringFuncSigmoid: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSigmoid`."""
ScoringFuncSqrtsoftplus: ScoringFuncType = ...
"""Alias for :attr:`ScoringFuncType.ScoringFuncSqrtsoftplus`."""

class Model:
    """Model weights and forward methods.

    Attributes:
        embed (torch.Tensor): Embedding weights.
        norm (torch.Tensor): Final normalization weights.
        norm_bias (torch.Tensor): Final normalization bias.
        head (torch.Tensor): LM head weights.
        attn_norm (List[torch.Tensor]): Attention norm weights per layer.
        attn_norm_bias (List[torch.Tensor]): Attention norm bias per layer.
        attn_out (List[torch.Tensor]): Attention output projection weights per layer.
        attn_out_input_scale (List[torch.Tensor]): Attn output quantization input scale per layer.
        attn_out_input_offset (List[torch.Tensor]): Attn output quantization input offset per layer.
        attn_out_quant_bias (List[torch.Tensor]): Attn output quantization bias per layer.
        attn_out_deq_scale (List[torch.Tensor]): Attn output dequantization scale per layer.
        mha_qkv (List[torch.Tensor]): MHA QKV weights per layer.
        mha_qkv_bias (List[torch.Tensor]): MHA QKV bias per layer.
        mha_qkv_input_scale (List[torch.Tensor]): MHA QKV quantization input scale per layer.
        mha_qkv_input_offset (List[torch.Tensor]): MHA QKV quantization input offset per layer.
        mha_qkv_quant_bias (List[torch.Tensor]): MHA QKV quantization bias per layer.
        mha_qkv_deq_scale (List[torch.Tensor]): MHA QKV dequantization scale per layer.
        mha_q_norm (List[torch.Tensor]): MHA Q norm weights per layer.
        mha_q_norm_bias (List[torch.Tensor]): MHA Q norm bias per layer.
        mha_k_norm (List[torch.Tensor]): MHA K norm weights per layer.
        mha_k_norm_bias (List[torch.Tensor]): MHA K norm bias per layer.
        mla_qkv_a (List[torch.Tensor]): MLA QA KVA weights per layer.
        mla_qkv_a_input_scale (List[torch.Tensor]): MLA QA KVA quantization input scale per layer.
        mla_qkv_a_input_offset (List[torch.Tensor]): MLA QA KVA quantization input offset per layer.
        mla_qkv_a_quant_bias (List[torch.Tensor]): MLA QA KVA quantization bias per layer.
        mla_qkv_a_deq_scale (List[torch.Tensor]): MLA QA KVA dequantization scale per layer.
        mla_q_b (List[torch.Tensor]): MLA QB weights per layer.
        mla_q_b_input_scale (List[torch.Tensor]): MLA QB quantization input scale per layer.
        mla_q_b_input_offset (List[torch.Tensor]): MLA QB quantization input offset per layer.
        mla_q_b_quant_bias (List[torch.Tensor]): MLA QB quantization bias per layer.
        mla_q_b_deq_scale (List[torch.Tensor]): MLA QB dequantization scale per layer.
        mla_q_norm (List[torch.Tensor]): MLA Q norm weights per layer.
        mla_q_norm_bias (List[torch.Tensor]): MLA Q norm bias per layer.
        mla_wuv (List[torch.Tensor]): MLA W_UV weights per layer, shape (n_local_heads, kv_lora_rank, v_head_dim).
        mla_wuk_t (List[torch.Tensor]): MLA W_UK^T weights per layer, shape (n_local_heads, qk_nope_head_dim, kv_lora_rank).
        mla_kv_norm (List[torch.Tensor]): MLA KV norm weights per layer.
        mla_kv_norm_bias (List[torch.Tensor]): MLA KV norm bias per layer.
        index_q_b (List[torch.Tensor]): DSA index QB weights per layer.
        index_q_b_input_scale (List[torch.Tensor]): DSA index QB quantization input scale per layer.
        index_q_b_input_offset (List[torch.Tensor]): DSA index QB quantization input offset per layer.
        index_q_b_quant_bias (List[torch.Tensor]): DSA index QB quantization bias per layer.
        index_q_b_deq_scale (List[torch.Tensor]): DSA index QB dequantization scale per layer.
        index_k_weights_proj (List[torch.Tensor]): DSA index K and weights projection combined per layer.
        index_k_norm (List[torch.Tensor]): DSA index K norm weights per layer.
        index_k_norm_bias (List[torch.Tensor]): DSA index K norm bias per layer.
        linear_in_proj_qkv (List[torch.Tensor]): Linear attention QKV projection weights per layer.
        linear_in_proj_z (List[torch.Tensor]): Linear attention Z projection weights per layer.
        linear_in_proj_b (List[torch.Tensor]): Linear attention B projection weights per layer.
        linear_in_proj_a (List[torch.Tensor]): Linear attention A projection weights per layer.
        linear_conv1d (List[torch.Tensor]): Linear attention conv1d weights per layer.
        linear_a_log (List[torch.Tensor]): Linear attention A_log parameters per layer.
        linear_dt_bias (List[torch.Tensor]): Linear attention dt_bias parameters per layer.
        linear_norm (List[torch.Tensor]): Linear attention gated RMSNorm weights per layer.
        linear_out_proj (List[torch.Tensor]): Linear attention output projection weights per layer.
        mlp_norm (List[torch.Tensor]): MLP norm weights per layer.
        mlp_norm_bias (List[torch.Tensor]): MLP norm bias per layer.
        mlp_up_gate (List[torch.Tensor]): Dense up-gate weights per layer.
        mlp_up_gate_input_scale (List[torch.Tensor]): Dense up-gate quantization input scale per layer.
        mlp_up_gate_input_offset (List[torch.Tensor]): Dense up-gate quantization input offset per layer.
        mlp_up_gate_quant_bias (List[torch.Tensor]): Dense up-gate quantization bias per layer.
        mlp_up_gate_deq_scale (List[torch.Tensor]): Dense up-gate dequantization scale per layer.
        mlp_down (List[torch.Tensor]): Dense down weights per layer.
        mlp_down_input_scale (List[torch.Tensor]): Dense down quantization input scale per layer.
        mlp_down_input_offset (List[torch.Tensor]): Dense down quantization input offset per layer.
        mlp_down_quant_bias (List[torch.Tensor]): Dense down quantization bias per layer.
        mlp_down_deq_scale (List[torch.Tensor]): Dense down dequantization scale per layer.
        gate (List[torch.Tensor]): MoE gate weights per layer.
        gate_bias (List[torch.Tensor]): MoE gate bias per layer.
        se_up_gate (List[torch.Tensor]): Shared-expert up-gate weights per layer.
        se_up_gate_deq_scale (List[torch.Tensor]): Shared-expert up-gate scales per layer.
        se_down (List[torch.Tensor]): Shared-expert down weights per layer.
        se_down_deq_scale (List[torch.Tensor]): Shared-expert down scales per layer.
        se_gate (List[torch.Tensor]): Optional shared-expert sigmoid gate, shape [1, hidden] per MoE layer.
        re_up_gate (List[torch.Tensor]): Routed-expert up-gate weights.
        re_up_gate_scale (List[torch.Tensor]): Routed-expert up-gate scales(deprecated).
        re_up_gate_deq_scale (List[torch.Tensor]): Routed-expert up-gate scales.
        re_down (List[torch.Tensor]): Routed-expert down weights.
        re_down_scale (List[torch.Tensor]): Routed-expert down scales(deprecated).
        re_down_deq_scale (List[torch.Tensor]): Routed-expert down scales.
        attn_sink (List[torch.Tensor]): Per-head attention sink (DeepSeek-V4).
        attn_wq_a (List[torch.Tensor]): Per-layer attention wq_a (DeepSeek-V4).
        attn_wq_a_input_scale (List[torch.Tensor]): Attn wq_a quantization input scale per layer.
        attn_wq_a_input_offset (List[torch.Tensor]): Attn wq_a quantization input offset per layer.
        attn_wq_a_quant_bias (List[torch.Tensor]): Attn wq_a quantization bias per layer.
        attn_wq_a_deq_scale (List[torch.Tensor]): Attn wq_a dequantization scale per layer.
        attn_wo_a (List[torch.Tensor]): Per-layer output projection wo_a (DeepSeek-V4).
        attn_wo_b (List[torch.Tensor]): Per-layer output projection wo_b (DeepSeek-V4).
        attn_wkv (List[torch.Tensor]): Per-layer attention wkv (DeepSeek-V4).
        attn_wkv_input_scale (List[torch.Tensor]): Attn wkv quantization input scale per layer.
        attn_wkv_input_offset (List[torch.Tensor]): Attn wkv quantization input offset per layer.
        attn_wkv_quant_bias (List[torch.Tensor]): Attn wkv quantization bias per layer.
        attn_wkv_deq_scale (List[torch.Tensor]): Attn wkv dequantization scale per layer.
        comp_ape (List[torch.Tensor]): Compressor ape per layer (DeepSeek-V4).
        comp_w_kv (List[torch.Tensor]): Compressor wkv per layer (DeepSeek-V4, fp32).
        comp_w_gate (List[torch.Tensor]): Compressor wgate per layer (DeepSeek-V4, fp32).
        comp_norm (List[torch.Tensor]): Compressor RMSNorm weight per layer (DeepSeek-V4).
        idx_wq_b (List[torch.Tensor]): Indexer wq_b per layer (DeepSeek-V4).
        idx_wq_b_input_scale (List[torch.Tensor]): Indexer wq_b quantization input scale per layer.
        idx_wq_b_input_offset (List[torch.Tensor]): Indexer wq_b quantization input offset per layer.
        idx_wq_b_quant_bias (List[torch.Tensor]): Indexer wq_b quantization bias per layer.
        idx_wq_b_deq_scale (List[torch.Tensor]): Indexer wq_b dequantization scale per layer.
        idx_weights_proj (List[torch.Tensor]): Indexer weights_proj per layer (DeepSeek-V4).
        idx_comp_ape (List[torch.Tensor]): Indexer compressor ape per layer (DeepSeek-V4).
        idx_comp_w_kv (List[torch.Tensor]): Indexer compressor wkv per layer (fp32).
        idx_comp_w_gate (List[torch.Tensor]): Indexer compressor wgate per layer (fp32).
        idx_comp_norm (List[torch.Tensor]): Indexer compressor norm per layer (DeepSeek-V4).
        hc_attn_fn (List[torch.Tensor]): MHC attn fn per layer (DeepSeek-V4).
        hc_ffn_fn (List[torch.Tensor]): MHC ffn fn per layer (DeepSeek-V4).
        hc_attn_base (List[torch.Tensor]): MHC attn base per layer (DeepSeek-V4).
        hc_ffn_base (List[torch.Tensor]): MHC ffn base per layer (DeepSeek-V4).
        hc_attn_scale (List[torch.Tensor]): MHC attn scale per layer (DeepSeek-V4).
        hc_ffn_scale (List[torch.Tensor]): MHC ffn scale per layer (DeepSeek-V4).
        hc_head_fn (torch.Tensor): MHC head fn (Transformer-level, DeepSeek-V4).
        hc_head_base (torch.Tensor): MHC head base (Transformer-level, DeepSeek-V4).
        hc_head_scale (torch.Tensor): MHC head scale (Transformer-level, DeepSeek-V4).
    """

    embed: torch.Tensor = ...
    """Embedding weights."""
    norm: torch.Tensor = ...
    """Final normalization weights."""
    norm_bias: torch.Tensor = ...
    """Final normalization bias."""
    head: torch.Tensor = ...
    """LM head weights."""
    attn_norm: List[torch.Tensor] = ...
    """Attention norm weights per layer."""
    attn_norm_bias: List[torch.Tensor] = ...
    """Attention norm bias per layer."""
    attn_out: List[torch.Tensor] = ...
    """Attention output projection weights per layer."""
    attn_out_input_scale: List[torch.Tensor] = ...
    """Attn output quantization input scale per layer."""
    attn_out_input_offset: List[torch.Tensor] = ...
    """Attn output quantization input offset per layer."""
    attn_out_quant_bias: List[torch.Tensor] = ...
    """Attn output quantization bias per layer."""
    attn_out_deq_scale: List[torch.Tensor] = ...
    """Attn output dequantization scale per layer."""
    mha_qkv: List[torch.Tensor] = ...
    """MHA QKV weights per layer."""
    mha_qkv_bias: List[torch.Tensor] = ...
    """MHA QKV bias per layer."""
    mha_qkv_input_scale: List[torch.Tensor] = ...
    """MHA QKV quantization input scale per layer."""
    mha_qkv_input_offset: List[torch.Tensor] = ...
    """MHA QKV quantization input offset per layer."""
    mha_qkv_quant_bias: List[torch.Tensor] = ...
    """MHA QKV quantization bias per layer."""
    mha_qkv_deq_scale: List[torch.Tensor] = ...
    """MHA QKV dequantization scale per layer."""
    mha_q_norm: List[torch.Tensor] = ...
    """MHA Q norm weights per layer."""
    mha_q_norm_bias: List[torch.Tensor] = ...
    """MHA Q norm bias per layer."""
    mha_k_norm: List[torch.Tensor] = ...
    """MHA K norm weights per layer."""
    mha_k_norm_bias: List[torch.Tensor] = ...
    """MHA K norm bias per layer."""
    mla_qkv_a: List[torch.Tensor] = ...
    """MLA fused Q A and KV A weights per layer."""
    mla_qkv_a_input_scale: List[torch.Tensor] = ...
    """MLA QKVA 每层量化输入缩放因子"""
    mla_qkv_a_input_offset: List[torch.Tensor] = ...
    """MLA QKVA 每层量化输入偏移"""
    mla_qkv_a_quant_bias: List[torch.Tensor] = ...
    """MLA QKVA 每层量化偏置"""
    mla_qkv_a_deq_scale: List[torch.Tensor] = ...
    """MLA QKVA 每层反量化缩放因子"""
    mla_q_b: List[torch.Tensor] = ...
    """MLA QB weights per layer."""
    mla_q_b_input_scale: List[torch.Tensor] = ...
    """MLA QB 每层量化输入缩放因子"""
    mla_q_b_input_offset: List[torch.Tensor] = ...
    """MLA QB 每层量化输入偏移"""
    mla_q_b_quant_bias: List[torch.Tensor] = ...
    """MLA QB 每层量化偏置"""
    mla_q_b_deq_scale: List[torch.Tensor] = ...
    """MLA QB 每层反量化缩放因子"""
    mla_q_norm: List[torch.Tensor] = ...
    """MLA Q norm weights per layer."""
    mla_q_norm_bias: List[torch.Tensor] = ...
    """MLA Q norm bias per layer."""
    mla_wuv: List[torch.Tensor] = ...
    """MLA W_UV weights per layer, shape (n_local_heads, kv_lora_rank, v_head_dim)."""
    mla_wuk_t: List[torch.Tensor] = ...
    """MLA W_UK^T weights per layer, shape (n_local_heads, qk_nope_head_dim, kv_lora_rank)."""
    mla_kv_norm: List[torch.Tensor] = ...
    """MLA KV norm weights per layer."""
    mla_kv_norm_bias: List[torch.Tensor] = ...
    """MLA KV norm bias per layer."""
    index_q_b: List[torch.Tensor] = ...
    """DSA index QB weights per layer."""
    index_q_b_input_scale: List[torch.Tensor] = ...
    """DSA index QB 每层量化输入缩放因子"""
    index_q_b_input_offset: List[torch.Tensor] = ...
    """DSA index QB 每层量化输入偏移"""
    index_q_b_quant_bias: List[torch.Tensor] = ...
    """DSA index QB 每层量化偏置"""
    index_q_b_deq_scale: List[torch.Tensor] = ...
    """DSA index QB 每层反量化缩放因子"""
    index_k_weights_proj: List[torch.Tensor] = ...
    """DSA index K and weights projection combined per layer."""
    index_k_norm: List[torch.Tensor] = ...
    """DSA index K norm weights per layer."""
    index_k_norm_bias: List[torch.Tensor] = ...
    """DSA index K norm bias per layer."""
    linear_in_proj_qkv: List[torch.Tensor] = ...
    """Linear attention QKV projection weights per layer."""
    linear_in_proj_z: List[torch.Tensor] = ...
    """Linear attention Z projection weights per layer."""
    linear_in_proj_b: List[torch.Tensor] = ...
    """Linear attention B projection weights per layer."""
    linear_in_proj_a: List[torch.Tensor] = ...
    """Linear attention A projection weights per layer."""
    linear_conv1d: List[torch.Tensor] = ...
    """Linear attention conv1d weights per layer."""
    linear_a_log: List[torch.Tensor] = ...
    """Linear attention A_log parameters per layer."""
    linear_dt_bias: List[torch.Tensor] = ...
    """Linear attention dt_bias parameters per layer."""
    linear_norm: List[torch.Tensor] = ...
    """Linear attention gated RMSNorm weights per layer."""
    linear_out_proj: List[torch.Tensor] = ...
    """Linear attention output projection weights per layer."""
    mlp_norm: List[torch.Tensor] = ...
    """MLP norm weights per layer."""
    mlp_norm_bias: List[torch.Tensor] = ...
    """MLP norm bias per layer."""
    mlp_up_gate: List[torch.Tensor] = ...
    """Dense up-gate weights per layer."""
    mlp_up_gate_input_scale: List[torch.Tensor] = ...
    """Dense up-gate quantization input scale per layer."""
    mlp_up_gate_input_offset: List[torch.Tensor] = ...
    """Dense up-gate quantization input offset per layer."""
    mlp_up_gate_quant_bias: List[torch.Tensor] = ...
    """Dense up-gate quantization bias per layer."""
    mlp_up_gate_deq_scale: List[torch.Tensor] = ...
    """Dense up-gate dequantization scale per layer."""
    mlp_down: List[torch.Tensor] = ...
    """Dense down weights per layer."""
    mlp_down_input_scale: List[torch.Tensor] = ...
    """Dense down quantization input scale per layer."""
    mlp_down_input_offset: List[torch.Tensor] = ...
    """Dense down quantization input offset per layer."""
    mlp_down_quant_bias: List[torch.Tensor] = ...
    """Dense down quantization bias per layer."""
    mlp_down_deq_scale: List[torch.Tensor] = ...
    """Dense down dequantization scale per layer."""
    gate: List[torch.Tensor] = ...
    """MoE gate weights per layer."""
    gate_bias: List[torch.Tensor] = ...
    """MoE gate bias per layer."""
    tid2eid: List[torch.Tensor] = ...
    """MoE hash layers: token-id->expert lookup table ``[vocab_size, n_act_experts]`` per layer."""
    se_up_gate: List[torch.Tensor] = ...
    """Shared-expert up-gate weights per layer."""
    se_up_gate_deq_scale: List[torch.Tensor] = ...
    """Shared-expert up-gate scales per layer."""
    se_down: List[torch.Tensor] = ...
    """Shared-expert down weights per layer."""
    se_down_deq_scale: List[torch.Tensor] = ...
    """Shared-expert down scales per layer."""
    se_gate: List[torch.Tensor] = ...
    """Optional shared-expert sigmoid gate weights per MoE layer, shape [1, hidden]."""
    re_up_gate: List[torch.Tensor] = ...
    """Routed-expert up-gate weights."""
    re_up_gate_scale: List[torch.Tensor] = ...
    """Routed-expert up-gate scales.(deprecated)"""
    re_up_gate_deq_scale: List[torch.Tensor] = ...
    """Routed-expert up-gate scales."""
    re_down: List[torch.Tensor] = ...
    """Routed-expert down weights."""
    re_down_scale: List[torch.Tensor] = ...
    """Routed-expert down scales.(deprecated)"""
    re_down_deq_scale: List[torch.Tensor] = ...
    """Routed-expert down scales."""

    # DeepSeek-V4 (CxA)
    attn_sink: List[torch.Tensor] = ...
    """Per-head attention sink (DeepSeek-V4)."""
    attn_wq_a: List[torch.Tensor] = ...
    """Per-layer attention wq_a (DeepSeek-V4)."""
    attn_wq_a_input_scale: List[torch.Tensor] = ...
    """Attn wq_a quantization input scale per layer."""
    attn_wq_a_input_offset: List[torch.Tensor] = ...
    """Attn wq_a quantization input offset per layer."""
    attn_wq_a_quant_bias: List[torch.Tensor] = ...
    """Attn wq_a quantization bias per layer."""
    attn_wq_a_deq_scale: List[torch.Tensor] = ...
    """Attn wq_a dequantization scale per layer."""
    attn_wo_a: List[torch.Tensor] = ...
    """Per-layer output projection wo_a (DeepSeek-V4)."""
    attn_wo_b: List[torch.Tensor] = ...
    """Per-layer output projection wo_b (DeepSeek-V4)."""
    attn_wkv: List[torch.Tensor] = ...
    """Per-layer attention wkv (DeepSeek-V4)."""
    attn_wkv_input_scale: List[torch.Tensor] = ...
    """Attn wkv quantization input scale per layer."""
    attn_wkv_input_offset: List[torch.Tensor] = ...
    """Attn wkv quantization input offset per layer."""
    attn_wkv_quant_bias: List[torch.Tensor] = ...
    """Attn wkv quantization bias per layer."""
    attn_wkv_deq_scale: List[torch.Tensor] = ...
    """Attn wkv dequantization scale per layer."""
    comp_ape: List[torch.Tensor] = ...
    """Compressor ape per layer (DeepSeek-V4)."""
    comp_w_kv: List[torch.Tensor] = ...
    """Compressor wkv per layer (DeepSeek-V4, fp32)."""
    comp_w_gate: List[torch.Tensor] = ...
    """Compressor wgate per layer (DeepSeek-V4, fp32)."""
    comp_norm: List[torch.Tensor] = ...
    """Compressor RMSNorm weight per layer (DeepSeek-V4)."""
    # Indexer.wq_b is v4-specific (different shape from DSA's index_q_b).
    idx_wq_b: List[torch.Tensor] = ...
    """Indexer wq_b per layer (DeepSeek-V4)."""
    idx_wq_b_input_scale: List[torch.Tensor] = ...
    """Indexer wq_b quantization input scale per layer."""
    idx_wq_b_input_offset: List[torch.Tensor] = ...
    """Indexer wq_b quantization input offset per layer."""
    idx_wq_b_quant_bias: List[torch.Tensor] = ...
    """Indexer wq_b quantization bias per layer."""
    idx_wq_b_deq_scale: List[torch.Tensor] = ...
    """Indexer wq_b dequantization scale per layer."""
    idx_weights_proj: List[torch.Tensor] = ...
    """Indexer weights_proj per layer (DeepSeek-V4)."""
    idx_comp_ape: List[torch.Tensor] = ...
    """Indexer compressor ape per layer (DeepSeek-V4)."""
    idx_comp_w_kv: List[torch.Tensor] = ...
    """Indexer compressor wkv per layer (fp32)."""
    idx_comp_w_gate: List[torch.Tensor] = ...
    """Indexer compressor wgate per layer (fp32)."""
    idx_comp_norm: List[torch.Tensor] = ...
    """Indexer compressor norm per layer (DeepSeek-V4)."""
    hc_attn_fn: List[torch.Tensor] = ...
    """MHC attn fn per layer (DeepSeek-V4)."""
    hc_ffn_fn: List[torch.Tensor] = ...
    """MHC ffn fn per layer (DeepSeek-V4)."""
    hc_attn_base: List[torch.Tensor] = ...
    """MHC attn base per layer (DeepSeek-V4)."""
    hc_ffn_base: List[torch.Tensor] = ...
    """MHC ffn base per layer (DeepSeek-V4)."""
    hc_attn_scale: List[torch.Tensor] = ...
    """MHC attn scale per layer (DeepSeek-V4)."""
    hc_ffn_scale: List[torch.Tensor] = ...
    """MHC ffn scale per layer (DeepSeek-V4)."""
    hc_head_fn: torch.Tensor = ...
    """MHC head fn (Transformer-level, DeepSeek-V4)."""
    hc_head_base: torch.Tensor = ...
    """MHC head base (Transformer-level, DeepSeek-V4)."""
    hc_head_scale: torch.Tensor = ...
    """MHC head scale (Transformer-level, DeepSeek-V4)."""

    def init(self, config: ModelConfig, rank: int = 0) -> None:
        """Initialize native model state from Python-provided weights.

        Args:
            config (ModelConfig): Native model configuration.
            rank (int): Model-parallel rank.

        Returns:
            None: Native model state is created in place.

        Raises:
            ValueError: On configuration or parameter count mismatch.
            RuntimeError: If native initialization fails.
        """

    def forward(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass with host/vLLM-compatible attention metadata (V1).

        ``freqs_cis`` is a single tensor shared across all layers. For per-layer
        freqs tensors or device-side attention metadata, use :meth:`forward_v2`.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor shared by all layers.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid shapes or cache mismatch.
        """

    def forward_get_logits(
        self,
        rt: Runtime,
        input: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run logits-only path.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input hidden-state tensor.
            indices (torch.Tensor): Logits indices.
            output (torch.Tensor): Output logits tensor.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: If the native logits path fails.
        """

    def forward_and_get_logits(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        indices: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass and materialize logits (host metadata, V1).

        ``freqs_cis`` is a single tensor shared across all layers. For per-layer
        freqs tensors or device-side attention metadata, use
        :meth:`forward_and_get_logits_v2`.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (Union[torch.Tensor, Sequence[torch.Tensor]]): Rotary
                frequency tensor shared by all layers, or a per-layer sequence
            indices (torch.Tensor): Logits indices.
            output (torch.Tensor): Output logits buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On invalid KV-cache layout or other native execution failures.
        """

    def forward_with_inputs_embeds(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMeta,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
        deepstack_input: Sequence[torch.Tensor] = ...,
        input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Run forward pass with deepstack input embeddings (host metadata).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMeta): Host-side attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.
            deepstack_input (Sequence[torch.Tensor], default empty): Extra deepstack embeddings.
            input_ids (Optional[torch.Tensor], default None): Token ids for the sqrtsoftplus MoE
                hash-gate path (tid2eid[input_ids]); required only when scoring_func is
                sqrtsoftplus, otherwise may be left unset (defaults to None).

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On KV-cache/deepstack shape mismatch or other native execution failures.
        """

    def forward_v2(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMetaV2,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: Sequence[torch.Tensor],
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass with device-tensor attention metadata (V2).

        Unlike :meth:`forward` (V1), this path takes ``query_start_loc``,
        ``slot_mapping`` and ``block_tables`` as pre-built device tensors on
        ``attn_meta``; the native side skips host computation and H2D copies.

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMetaV2): Device-tensor attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (Sequence[torch.Tensor]): Per-layer rotary frequency tensors.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On shape mismatch or other native execution failures.
        """

    def forward_and_get_logits_v2(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMetaV2,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: Sequence[torch.Tensor],
        indices: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
    ) -> None:
        """Run forward pass and materialize logits (device metadata, V2).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMetaV2): Device-tensor attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (Sequence[torch.Tensor]): Per-layer rotary frequency tensors.
            indices (torch.Tensor): Logits indices.
            output (torch.Tensor): Output logits buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On shape mismatch or other native execution failures.
        """

    def forward_with_inputs_embeds_v2(
        self,
        rt: Runtime,
        input: torch.Tensor,
        attn_meta: AttnMetaV2,
        kv_cache: Sequence[Sequence[torch.Tensor]],
        freqs_cis: torch.Tensor,
        output: torch.Tensor,
        curr_stream: int = 0,
        deepstack_input: Sequence[torch.Tensor] = ...,
        input_ids: Optional[torch.Tensor] = None,
    ) -> None:
        """Run forward pass with deepstack input embeddings (device metadata, V2).

        Args:
            rt (Runtime): Native runtime handle.
            input (torch.Tensor): Input token tensor.
            attn_meta (AttnMetaV2): Device-tensor attention metadata.
            kv_cache (Sequence[Sequence[torch.Tensor]]): Per-layer KV cache.
            freqs_cis (torch.Tensor): Rotary frequency tensor.
            output (torch.Tensor): Output hidden-state buffer.
            curr_stream (int, default=0): Optional ACL stream pointer cast to integer.
            deepstack_input (Sequence[torch.Tensor], default empty): Extra deepstack embeddings.
            input_ids (Optional[torch.Tensor], default None): Token ids for the sqrtsoftplus MoE
                hash-gate path (tid2eid[input_ids]); required only when scoring_func is
                sqrtsoftplus, otherwise may be left unset (defaults to None).

        Returns:
            None: Output is written in place.

        Raises:
            RuntimeError: On shape mismatch or other native execution failures.
        """

    def get_tensor_pool_size(self, dbg: int = 0) -> int:
        """Get tensor-pool usage information.

        Args:
            dbg (int, default=0): Debug level forwarded to the native model.

        Returns:
            int: Current tensor pool size metric returned by native code.
        """

class CoreAssigner:
    """Helper to split hardware cores between prefill and decode work."""

    def __init__(self, prefill_ratio: float) -> None:
        """Create a core assigner.

        Args:
            prefill_ratio (float): Prefill-to-decode core split ratio.
        """

    def assign_core(self, is_decode: bool) -> float:
        """Assign cores for one request phase.

        Args:
            is_decode (bool): `True` for decode phase, `False` for prefill.

        Returns:
            float: Assigned core ratio.
        """

    def release_core(self, is_decode: bool) -> None:
        """Release cores previously assigned for one phase.

        Args:
            is_decode (bool): `True` for decode phase, `False` for prefill.

        Returns:
            None: Internal assignment state is updated.
        """

"""Low-level collective and kernel operator bindings."""

def all_gather(rt: Runtime, out: torch.Tensor, in_: torch.Tensor, comm_type: int = 0) -> None:
    """Collectively gather tensors from all ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for gathered values.
        in_ (torch.Tensor): Local input shard.
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def reduce_scatter(rt: Runtime, out: torch.Tensor, in_: torch.Tensor, comm_type: int = 0) -> None:
    """Reduce then scatter tensors across ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for the reduced local shard.
        in_ (torch.Tensor): Input tensor to reduce across ranks.
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def all_reduce(rt: Runtime, out: torch.Tensor, in_: torch.Tensor, comm_type: int = 0) -> None:
    """Collectively reduce tensors across all ranks.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output tensor for reduced results.
        in_ (torch.Tensor): Input tensor to reduce.
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If the native collective call fails.
    """
    ...

def alltoallv(
    rt: Runtime,
    out: torch.Tensor,
    in_: torch.Tensor,
    send_counts: torch.Tensor,
    recv_counts: torch.Tensor,
    sdispls: torch.Tensor,
    rdispls: torch.Tensor,
    comm_type: int = 0,
) -> None:
    """All-to-all vectorized collective communication.

    Each rank sends `send_counts[i]` elements starting at `sdispls[i]` to
    rank *i* and receives `recv_counts[i]` elements starting at `rdispls[i]`
    from rank *i*.

    Args:
        rt (Runtime): Native runtime handle.
        out (torch.Tensor): Output buffer for received data.
        in_ (torch.Tensor): Input tensor with data to send.
        send_counts (torch.Tensor): Per-rank send element counts.
        recv_counts (torch.Tensor): Per-rank receive element counts.
        sdispls (torch.Tensor): Per-rank send displacement offsets.
        rdispls (torch.Tensor): Per-rank receive displacement offsets.
        comm_type (int, default=0): Communication domain selector.
            ``0`` (TP), ``1`` (DP), ``2`` (EP).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If ``in_.dtype != out.dtype`` or the HCCL call fails.
    """
    ...

def probe_310p(rt: Runtime, out: torch.Tensor, value: int = 0x03102002) -> None:
    """Launch one 310P block and write ``value`` directly to an INT32 GM tensor."""
    ...

def add(rt: Runtime, x: torch.Tensor, y: torch.Tensor, z: torch.Tensor) -> None:
    """Elementwise add two tensors into output.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left operand.
        y (torch.Tensor): Right operand.
        z (torch.Tensor): Output tensor.

    Returns:
        None: `z` is written in place.

    Raises:
        RuntimeError: If tensor shapes or dtypes are unsupported by the kernel.
    """
    ...

def matmul(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    weight_nz: bool = False,
    transpose: bool = False,
) -> None:
    """Matrix multiplication with optional transpose/layout flags.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Right matrix/weight.
        z (torch.Tensor): Output matrix.
        weight_nz (bool): Whether `y` uses NZ weight layout.
        transpose (bool): Whether to transpose the right matrix in compute.

    Returns:
        None: `z` is written in place.

    Raises:
        RuntimeError: If the native kernel launch fails.
    """
    ...

def matmul_bench(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    x_warmup: torch.Tensor,
    y_warmup: torch.Tensor,
    z_warmup: torch.Tensor,
    iterations: int,
    warmup_iterations: int,
    weight_nz: bool = False,
    transpose: bool = False,
) -> int:
    """Matmul benchmark. Measures average time of a matmul operation.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Right matrix/weight.
        z (torch.Tensor): Output matrix.
        x_warmup (torch.Tensor): Left matrix for warmup.
        y_warmup (torch.Tensor): Right matrix/weight for warmup.
        z_warmup (torch.Tensor): Output matrix for warmup.
        iterations: Number of matmul iterations to run
        weight_nz (bool): Whether `y` uses NZ weight layout.
        transpose (bool): Whether to transpose the right matrix in compute.

    Returns:
        int: Measured average time for matmul operation in nanoseconds.

    Raises:
        RuntimeError: If the native kernel launch fails.
    """
    ...

def matmul_with_bias(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    z: torch.Tensor,
    bias: torch.Tensor,
    weight_nz: bool = False,
) -> None:
    """Matrix multiplication followed by bias add.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Right matrix/weight.
        z (torch.Tensor): Output matrix.
        bias (torch.Tensor): Bias tensor added to output.
        weight_nz (bool): Whether `y` uses NZ weight layout.

    Returns:
        None: `z` is written in place.

    Raises:
        RuntimeError: If the native kernel launch fails.
    """
    ...

def embed(
    rt: Runtime,
    weight: torch.Tensor,
    in_: torch.Tensor,
    out: torch.Tensor,
    start: int,
    end: int,
) -> None:
    """Embedding lookup over the provided token range.

    Args:
        rt (Runtime): Native runtime handle.
        weight (torch.Tensor): Embedding table.
        in_ (torch.Tensor): Token IDs.
        out (torch.Tensor): Output embedding tensor.
        start (int): Start token index (inclusive).
        end (int): End token index (exclusive).

    Returns:
        None: `out` is written in place.
    """
    ...

def rmsnorm_variance_only(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
    cnt_per_token: int = 1,
    in_start_offset: int = 0,
    out_start_offset: int = 0,
) -> None:
    """Compute variance for RMSNorm.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.

    Returns:
        None: `out` is written in place.
    """

def rmsnorm(
    rt: Runtime,
    in_: torch.Tensor,
    norm: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
    cnt_per_token: int = 1,
    in_start_offset: int = 0,
    out_start_offset: int = 0,
    variance: Optional[torch.Tensor] = None,
) -> None:
    """Apply RMSNorm with optional offsets.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        norm (torch.Tensor): RMSNorm weight tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.
        variance (Optional[torch.Tensor]): Optional output tensor for variance values.

    Returns:
        None: `out` is written in place.
    """
    ...

def rmsnorm_with_bias(
    rt: Runtime,
    in_: torch.Tensor,
    norm: torch.Tensor,
    norm_bias: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
    cnt_per_token: int = 1,
    in_start_offset: int = 0,
    out_start_offset: int = 0,
) -> None:
    """Apply RMSNorm with optional offsets.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        norm (torch.Tensor): RMSNorm weight tensor.
        norm_bias (torch.Tensor): RMSNorm Bias tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.
        cnt_per_token (int): Number of contiguous segments per token.
        in_start_offset (int): Input offset for segmented normalization.
        out_start_offset (int): Output offset for segmented normalization.

    Returns:
        None: `out` is written in place.
    """
    ...

def qk_rmsnorm_310p(
    rt: Runtime,
    in_: torch.Tensor,
    q_norm: torch.Tensor,
    k_norm: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    n_heads: int = 16,
    n_kv_heads: int = 8,
    head_dim: int = 128,
) -> None:
    """Run the fixed-shape Q/K RMSNorm gate used by the 310P POC."""
    ...

def layernorm(
    rt: Runtime,
    in_: torch.Tensor,
    norm: torch.Tensor,
    norm_bias: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int,
) -> None:
    """Apply LayerNorm with learned weight and bias.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        norm (torch.Tensor): LayerNorm weight tensor.
        norm_bias (torch.Tensor): LayerNorm bias tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width.

    Returns:
        None: `out` is written in place.
    """
    ...

def l2norm(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
    norm_dim: int = 0,
) -> None:
    """Apply L2 Norm.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.
        norm_dim (int): Normalization width. `0` lets native code infer it.

    Returns:
        None: `out` is written in place.
    """
    ...

def add_bias(rt: Runtime, in_: torch.Tensor, weight: torch.Tensor, out: torch.Tensor) -> None:
    """Add bias tensor to input tensor.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        weight (torch.Tensor): Bias tensor.
        out (torch.Tensor): Output tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def silu_and_mul(rt: Runtime, in_: torch.Tensor, out: torch.Tensor) -> None:
    """Apply SiLU and gated multiply.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        out (torch.Tensor): Output tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def sigmoid_gate_mul(rt: Runtime, attn: torch.Tensor, gate: torch.Tensor, out: torch.Tensor) -> None:
    """Compute out = attn * sigmoid(gate).

    Args:
        rt (Runtime): Native runtime handle.
        attn (torch.Tensor): Attention output, shape [num_tokens, dim].
        gate (torch.Tensor): Gate logits, shape [num_tokens, dim] (elementwise) or
            [num_tokens, 1] (broadcast per token).
        out (torch.Tensor): Output tensor, shape [num_tokens, dim]. May alias `attn`.

    Returns:
        None: `out` is written in place.
    """
    ...

def rope_and_cache(
    rt: Runtime,
    inout: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    position: torch.Tensor,
    cosin: torch.Tensor,
    slot_mapping: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    rot_dim: int,
    block_size: int,
    is_neox: bool,
    mrope_mask_h: int = 0,
    mrope_mask_w: int = 0,
) -> None:
    """Apply RoPE transform and update KV cache.

    Args:
        rt (Runtime): Native runtime handle.
        inout (torch.Tensor): Input/output QKV tensor.
        k_cache (torch.Tensor): Key cache tensor.
        v_cache (torch.Tensor): Value cache tensor.
        position (torch.Tensor): Position indices.
        cosin (torch.Tensor): Rotary cosine/sine tensor.
        slot_mapping (torch.Tensor): Slot mapping for paged cache writes.
        n_heads (int): Number of query heads.
        n_kv_heads (int): Number of KV heads.
        head_dim (int): Head dimension.
        rot_dim (int): Rotary dimension.
        block_size (int): KV cache block size.
        is_neox (bool): Whether to use NeoX rotary layout.
        mrope_mask_h (int): Optional mRoPE height mask.
        mrope_mask_w (int): Optional mRoPE width mask.

    Returns:
        None: Inputs/caches are updated in place.
    """
    ...

def attention(
    rt: Runtime,
    qkv: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    n_kv_heads: int,
    head_dim: int,
    block_size: int,
    batch: int,
    enable_flash_attention: bool = False,
    tile_size_of_cached_kv: int = 8192,
) -> None:
    """Run paged attention for cached KV tensors.

    Args:
        rt (Runtime): Native runtime handle.
        qkv (torch.Tensor): Query tensor.
        k_cache (torch.Tensor): Key cache tensor.
        v_cache (torch.Tensor): Value cache tensor.
        output (torch.Tensor): Attention output tensor.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of query heads.
        n_kv_heads (int): Number of KV heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size.
        batch (int): Batch size.
        enable_flash_attention (bool): Whether to use flash attention kernels.
        tile_size_of_cached_kv (int): Tile size for cached KV in flash attention.

    Returns:
        None: `output` is written in place.
    """
    ...

def add_and_rmsnorm(
    rt: Runtime,
    in_: torch.Tensor,
    add_in_out: torch.Tensor,
    norm: torch.Tensor,
    out: torch.Tensor,
    norm_eps: float,
) -> None:
    """Residual add followed by RMSNorm.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Residual input tensor.
        add_in_out (torch.Tensor): In/out tensor for residual accumulation.
        norm (torch.Tensor): RMSNorm weight tensor.
        out (torch.Tensor): Output tensor.
        norm_eps (float): Numerical epsilon used in normalization.

    Returns:
        None: `add_in_out`/`out` are updated in place per kernel behavior.
    """
    ...

def softmax_topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    out_weights: torch.Tensor,
    out_routing: torch.Tensor,
    top_k: int,
    norm_top_k_prob: bool,
) -> None:
    """Compute top-k routing with softmax scores.

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Routing score tensor.
        indices (torch.Tensor): Output top-k index tensor.
        out_weights (torch.Tensor): Output top-k probability tensor.
        out_routing (torch.Tensor): Output routing mask tensor.
        top_k (int): Number of experts selected per token.
        norm_top_k_prob (bool): Whether to normalize selected probabilities.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def sigmoid_topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor,
    scale: float,
    out_weights: torch.Tensor,
    out_routing: torch.Tensor,
    n_group: int,
    n_topk_group: int,
    top_k: int,
    norm_top_k_prob: bool,
) -> None:
    """Compute top-k routing with sigmoid scores.

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Routing score tensor.
        indices (torch.Tensor): Output top-k index tensor.
        bias (torch.Tensor): Bias tensor applied before top-k selection.
        scale (float): Scale factor applied to scores.
        out_weights (torch.Tensor): Output top-k weight tensor.
        out_routing (torch.Tensor): Output routing mask tensor.
        n_group (int): Number of routing groups.
        n_topk_group (int): Number of groups participating in top-k.
        top_k (int): Number of experts selected per token.
        norm_top_k_prob (bool): Whether to normalize selected probabilities.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def sqrtsoftplus_hash_topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    bias: torch.Tensor,
    input_ids: torch.Tensor,
    tid2eid: torch.Tensor,
    out_weights: torch.Tensor,
    routing_map: torch.Tensor,
    scale: float,
    top_k: int,
    hash: bool,
) -> None:
    """Compute V4 MoE gate routing (the part below the matmul), sqrtsoftplus only.

    The caller feeds pre-activation ``scores`` (the ``F.linear`` output); this op does
    the sqrtsoftplus activation, top-k (or hash lookup), gather, normalize, scale.

    Two outputs: a sparse ``[M, n_routed_experts]`` weight row (only the top-k expert
    slots hold a value, rest 0) and a ``[M, n_routed_experts]`` BIT1 routing bitmap (one
    bit per selected expert). The dense top-k indices are internal and NOT a GM output.

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Pre-activation routing scores ``[M, n_routed_experts]``
            (bf16 or fp32).
        indices (torch.Tensor): Identity helper ``[n_routed_experts]`` int32 (``0..N-1``),
            used as the vbitsort sort-key payload in the non-hash top-k path.
        bias (torch.Tensor): Per-expert bias ``[n_routed_experts]`` fp32. Pass an empty
            tensor on hash layers (bias is unused there).
        input_ids (torch.Tensor): Token ids ``[M]`` int32. Only read when ``hash`` is True;
            pass an empty tensor on non-hash layers.
        tid2eid (torch.Tensor): Token-id->expert lookup table
            ``[vocab_size, n_activated_experts]`` int32. Only read when ``hash`` is True;
            pass an empty tensor on non-hash layers.
        scale (float): ``route_scale`` multiplier applied after normalization.
        out_weights (torch.Tensor): Output SPARSE weights ``[M, n_routed_experts]`` (same
            dtype as ``scores``) — only the top-k selected experts hold a nonzero weight,
            the rest are 0.
        routing_map (torch.Tensor): Output routing bitmap ``[M, n_routed_experts]`` uint32
            (BIT1, one bit per expert; set for the top-k selected experts).
        top_k (int): Number of experts selected per token (``n_activated_experts``).
        hash (bool): If True, route via ``tid2eid[input_ids]`` (first ``n_hash_layers``);
            else route via ``scores.topk``.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def topk(
    rt: Runtime,
    scores: torch.Tensor,
    indices: torch.Tensor,
    outIndices: torch.Tensor,
    query_lens: torch.Tensor,
    cached_lens: torch.Tensor,
    k: int,
) -> None:
    """Select top-k elements by scores in batches

    Args:
        rt (Runtime): Native runtime handle.
        scores (torch.Tensor): Routing score tensor.
        indices (torch.Tensor): Tensor of indices that match scores.
        outIndices (torch.Tensor): Output top-k index tensor.
        query_lens (torch.Tensor): Vector of query lengths for each batch.
        cached_lens (torch.Tensor): Vector of cached KV lengths for each batch.
        k (int): Number of experts selected per token.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def cast_up(rt: Runtime, in_: torch.Tensor, out: torch.Tensor) -> None:
    """Cast tensor values to a higher-precision type.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        out (torch.Tensor): Output tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def permutation(
    rt: Runtime,
    in_: torch.Tensor,
    routing: torch.Tensor,
    start: int,
    end: int,
    out: torch.Tensor,
    unp_idx: torch.Tensor,
    counts: torch.Tensor,
) -> None:
    """Permute token rows into expert-local layout.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input token tensor.
        routing (torch.Tensor): Routing/expert assignment tensor.
        start (int): Start expert index.
        end (int): End expert index.
        out (torch.Tensor): Permuted output tensor.
        unp_idx (torch.Tensor): Unpermutation index tensor.
        counts (torch.Tensor): Per-expert count tensor.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def unpermutation(
    rt: Runtime,
    in_: torch.Tensor,
    routing: torch.Tensor,
    weights: torch.Tensor,
    start: int,
    end: int,
    out: torch.Tensor,
    unp_idx: torch.Tensor,
) -> None:
    """Restore original row order after expert routing.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Permuted input tensor.
        routing (torch.Tensor): Routing/expert assignment tensor.
        weights (torch.Tensor): Routing weight tensor.
        start (int): Start expert index.
        end (int): End expert index.
        out (torch.Tensor): Unpermuted output tensor.
        unp_idx (torch.Tensor): Unpermutation index tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def group_matmul(
    rt: Runtime,
    in_: torch.Tensor,
    weights: Sequence[torch.Tensor],
    scales: Sequence[torch.Tensor],
    counts: torch.Tensor,
    start: int,
    end: int,
    out_dim: int,
    in_dim: int,
    output: torch.Tensor,
    weight_nz: bool,
    transpose: bool,
) -> None:
    """Run grouped matmul for per-expert weights.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor.
        weights (Sequence[torch.Tensor]): Per-group weight tensors.
        scales (Sequence[torch.Tensor]): Optional per-group scale tensors.
        counts (torch.Tensor): Per-group token count tensor.
        start (int): Start group index.
        end (int): End group index.
        out_dim (int): Output dimension.
        in_dim (int): Input dimension.
        output (torch.Tensor): Output tensor.
        weight_nz (bool): Whether weight tensors use NZ layout.
        transpose (bool): Whether grouped weights are transposed.

    Returns:
        None: `output` is written in place.
    """
    ...

def softmax(rt: Runtime, x: torch.Tensor, calc_len: int, is_long: bool) -> None:
    """Apply softmax over the configured dimension.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input/output tensor.
        calc_len (int): Effective softmax length.
        is_long (bool): Whether to use the long-sequence kernel path.

    Returns:
        None: `x` is updated in place.
    """
    ...

def rope_complex(
    rt: Runtime,
    n_local_heads: int,
    step_dim: int,
    rope_dim: int,
    input_with_r: torch.Tensor,
    freqs: torch.Tensor,
    position: torch.Tensor,
    output: torch.Tensor,
    inverse: bool = False,
    out_interleaved: bool = False,
) -> None:
    """Apply complex-domain rotary embedding helper.

    Args:
        rt (Runtime): Native runtime handle.
        n_local_heads (int): Number of local heads.
        step_dim (int): Per-step hidden dimension.
        rope_dim (int): Rotary dimension.
        input_with_r (torch.Tensor): Input tensor with real/imag layout.
        freqs (torch.Tensor): Rotary frequency tensor.
        position (torch.Tensor): Position tensor.
        output (torch.Tensor): Output tensor.
        inverse (bool): If True, apply the conjugate (reverse) rotation.
        out_interleaved (bool): If True, write the rope result interleaved
            ``[r0,i0,r1,i1,...]`` (matches torch ``view_as_real().flatten``);
            otherwise write the deinterleaved half layout
            ``[r0..r(half-1) | i0..i(half-1)]`` (MLA/DSA kv-cache convention).

    Returns:
        None: Output is produced in place according to kernel contract.
    """
    ...

def mla_prepare(
    rt: Runtime,
    attn_qkvc: torch.Tensor,
    q_norm: torch.Tensor,
    q_norm_bias: torch.Tensor,
    attn_norm_qc: torch.Tensor,
    kv_norm: torch.Tensor,
    kv_norm_bias: torch.Tensor,
    attn_norm_kvc: torch.Tensor,
    freqs: torch.Tensor,
    position: torch.Tensor,
    q_lora_rank: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    block_size: int,
    k_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_eps: float,
) -> None:
    """Fused MLA prepare: two RMSNorm passes followed by rope_complex_and_cache.

    Args:
        rt (Runtime): Native runtime handle.
        attn_qkvc (torch.Tensor): Concatenated [q_lora_rank | kv_lora_rank | rope_head_dim] per token.
        q_norm (torch.Tensor): RMSNorm weight for the q-lora slice.
        q_norm_bias (torch.Tensor): RMSNorm bias for the q-lora slice.
        attn_norm_qc (torch.Tensor): Output RMSNormed q-lora slice.
        kv_norm (torch.Tensor): RMSNorm weight for the kv-lora slice.
        kv_norm_bias (torch.Tensor): RMSNorm bias for the kv-lora slice.
        attn_norm_kvc (torch.Tensor): Output RMSNormed kv-lora slice; also used as the `key` written into k_cache.
        freqs (torch.Tensor): Precomputed rotary freqs_cis (TTTWWW layout).
        position (torch.Tensor): Per-token position ids (int64).
        q_lora_rank (int): q-lora rank dimension.
        kv_lora_rank (int): kv-lora rank dimension.
        rope_head_dim (int): Rotary head dimension.
        block_size (int): Paged kv-cache block size; 0 disables cache writes.
        k_cache (torch.Tensor): Output paged k-cache.
        pe_cache (torch.Tensor): Output paged pe-cache (RoPE'd rope slice).
        slot_mapping (torch.Tensor): Per-token paged-cache slot mapping.
        norm_eps (float): RMSNorm epsilon.

    Returns:
        None: Outputs are written in place into attn_norm_qc, attn_norm_kvc, k_cache, pe_cache, and the rope slice of attn_qkvc.
    """
    ...

def indexer_prepare(
    rt: Runtime,
    kw: torch.Tensor,
    k_norm: torch.Tensor,
    k_norm_bias: torch.Tensor,
    freqs: torch.Tensor,
    position: torch.Tensor,
    index_head_dim: int,
    index_n_heads: int,
    rope_head_dim: int,
    block_size: int,
    index_k_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    norm_eps: float,
    q: torch.Tensor,
    scale: float,
    top_k: int,
    is_long: bool,
) -> None:
    """Fused DSA indexer prepare: LayerNorm + rope_complex_and_cache, optional rope_complex(q) + muls(kw).

    Always runs:
      * LayerNorm over ``kw[:, :index_head_dim]`` (in place).
      * rope_complex_and_cache on ``kw[:, :index_head_dim+index_n_heads]`` writing the
        rotary slice of ``kw`` and scattering ``index_head_dim`` slice into ``index_k_cache``.

    When ``is_long`` is true, additionally runs:
      * rope_complex on ``q`` (``index_n_heads`` heads, each ``index_head_dim`` wide), in place.
      * muls on ``kw[:, index_head_dim:index_head_dim+index_n_heads]`` by ``scale`` (in place).

    Args:
        rt (Runtime): Native runtime handle.
        kw (torch.Tensor): ``[token_num, index_head_dim + index_n_heads]`` per token.
        k_norm (torch.Tensor): LayerNorm weight ``[index_head_dim]``.
        k_norm_bias (torch.Tensor): LayerNorm bias ``[index_head_dim]``.
        freqs (torch.Tensor): Precomputed rotary freqs_cis (TTTWWW layout).
        position (torch.Tensor): Per-token position ids (int64).
        index_head_dim (int): Indexer head dimension.
        index_n_heads (int): Indexer head count (also the muls width).
        rope_head_dim (int): Rotary head dimension.
        block_size (int): Paged k-cache block size; 0 disables cache writes.
        index_k_cache (torch.Tensor): Output paged indexer k-cache.
        slot_mapping (torch.Tensor): Per-token paged-cache slot mapping (int32).
        norm_eps (float): LayerNorm epsilon.
        q (torch.Tensor): ``[token_num, index_n_heads * index_head_dim]`` query tensor; only
            touched when ``is_long`` is true. The rotary slice is written in place.
        scale (float): Scalar applied to ``kw[:, index_head_dim:]`` when ``is_long`` is true.
        top_k (int): Number of top-k tokens to select in the indexer/sparse attention.
        is_long (bool): Whether to run the rope_complex(q) + muls(kw) tail.

    Returns:
        None: ``kw`` (norm+rope slices, optional muls slice), ``index_k_cache``, and (when
        ``is_long``) the rotary slice of ``q`` are written in place.
    """
    ...

def quant(
    rt: Runtime,
    x: torch.Tensor,
    scale_reciprocal: torch.Tensor,
    offset: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Quantize tensor using explicit reciprocal scale and offset.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor.
        scale_reciprocal (torch.Tensor): Reciprocal scale tensor.
        offset (torch.Tensor): Quantization offset tensor.
        out (torch.Tensor): Quantized output tensor.

    Returns:
        None: `out` is written in place.
    """
    ...

def quant_dynamic(rt: Runtime, x: torch.Tensor, scale: torch.Tensor, out: torch.Tensor) -> None:
    """Dynamically quantize tensor values and emit scale.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor.
        scale (torch.Tensor): Output scale tensor.
        out (torch.Tensor): Quantized output tensor.

    Returns:
        None: `scale` and `out` are written in place.
    """
    ...

def matmul_dequant(
    rt: Runtime,
    x: torch.Tensor,
    y: torch.Tensor,
    bias: torch.Tensor,
    deq_scale: torch.Tensor,
    z: torch.Tensor,
    weight_nz: bool = False,
    transpose: bool = False,
) -> None:
    """Matmul on quantized weights with dequantization.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Left matrix.
        y (torch.Tensor): Quantized right matrix.
        bias (torch.Tensor): Bias tensor.
        deq_scale (torch.Tensor): Dequantization scale tensor.
        z (torch.Tensor): Output matrix.
        weight_nz (bool): Whether `y` uses NZ layout.
        transpose (bool): Whether to transpose the right matrix.

    Returns:
        None: `z` is written in place.
    """
    ...

def msd_merge_dequant(
    rt: Runtime,
    y_merged: torch.Tensor,
    scale_biases: Sequence[torch.Tensor],
    counts: torch.Tensor,
    per_token_scale: torch.Tensor,
    out: torch.Tensor,
) -> None:
    """Merge MSD (W4A8) row-merged int8 result and per-token dequantize.

    Used by the MSD W4A8 MoE post-stage to turn a mid-stage row-merged result
    into the final BF16 output. ``y_merged`` packs two halves of a 4-bit weight
    matmul: rows ``[0, m)`` hold the low nibble and rows ``[m, 2m)`` hold the
    high nibble (each in int8 form). The kernel reconstructs the full value
    ``Y_high * 16 + Y_low``, compensates the low-nibble ``-8`` bias, adds a
    per-column ``scale_bias``, and scales by a per-token ``per_token_scale``::

        Y = (Y_high * 16 + Y_low + scale_bias) * perTokenScale

    Args:
        rt (Runtime): Native runtime handle.
        y_merged (torch.Tensor): Row-merged int8 mid-stage result, shape
            ``[2*m, n]`` (float16). Rows ``[0, m)`` are the low nibble, rows
            ``[m, 2m)`` are the high nibble.
        scale_bias (torch.Tensor): Per-column bias added after merge, shape
            ``[n]`` (float32).
        per_token_scale (torch.Tensor): Per-token dequantization scale, shape
            ``[m]`` (float32).
        out (torch.Tensor): Output tensor, shape ``[m, n]`` (bfloat16).

    Returns:
        None: `out` is written in place.

    Raises:
        RuntimeError: If dtypes/shapes are unsupported
            (requires ``y_merged`` float16, ``scale_bias``/``per_token_scale``
            float32, ``out`` bfloat16, and ``y_merged.shape[0]`` even).
    """
    ...

def dequant(rt: Runtime, in_: torch.Tensor, scale: torch.Tensor, out: torch.Tensor, has_scale: bool) -> None:
    """Dequantize tensor values into output precision.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Quantized input tensor.
        scale (torch.Tensor): Scale tensor.
        out (torch.Tensor): Dequantized output tensor.
        has_scale (bool): Whether scale should be applied.

    Returns:
        None: `out` is written in place.
    """
    ...

def mla_v2(
    rt: Runtime,
    q_with_qr: torch.Tensor,
    qr: torch.Tensor,
    k_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    wuk_t: torch.Tensor,
    wuv: torch.Tensor,
    output: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    rope_head_dim: int,
    nope_head_dim: int,
    v_head_dim: int,
    kv_lora_rank: int,
    block_size: int,
    batch: int,
    scale: float,
    topk_indices: torch.Tensor,
    top_k: int = 0,
    nz: bool = False,
    enable_flash_attention: bool = False,
    tile_size_of_cached_kv: int = 8192,
) -> None:
    """Run MLA v2 path as three kernels: wuk einsum + mla_v2 attention + wuv einsum.

    Compared to :func:`mla`, the WUK absorb and WUV projection are moved out of
    the fused kernel and run as separate einsum operators surrounding the
    ``mla_v2`` attention kernel. The final ``output`` (v_head_dim) is the same
    as :func:`mla`'s output.

    Args:
        rt (Runtime): Native runtime handle.
        q_with_qr (torch.Tensor): Query tensor with rotary components, shape
            (total_query_tokens, n_heads, nope_head_dim + rope_head_dim).
        qr (torch.Tensor): Pre-rotated q_rope slice, contiguous, shape
            (total_query_tokens, n_heads, rope_head_dim).
        k_cache (torch.Tensor): Paged KV cache (kv_lora_rank slice).
        pe_cache (torch.Tensor): Paged RoPE key cache (rope_head_dim slice).
        wuk_t (torch.Tensor): MLA W_UK^T weight, shape
            (n_heads, nope_head_dim, kv_lora_rank).
        wuv (torch.Tensor): MLA W_UV weight, shape
            (n_heads, kv_lora_rank, v_head_dim).
        output (torch.Tensor): Output tensor (v_head_dim); written in place.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of query heads.
        rope_head_dim (int): Rotary head dimension.
        nope_head_dim (int): Non-rotary head dimension.
        v_head_dim (int): Value head dimension.
        kv_lora_rank (int): KV LoRA rank.
        block_size (int): KV block size.
        batch (int): Batch size.
        scale (float): Attention scaling factor.
        topk_indices (torch.Tensor): Top-k indices tensor for sparse attention
            (may be empty when ``top_k == 0``).
        top_k (int): Number of top-k indices; 0 disables sparse attention.
        nz (bool): Whether to use nz weights.
        enable_flash_attention (bool): Whether to use the flash MLA v2 kernel.
        tile_size_of_cached_kv (int): Tile size for cached KV in flash MLA v2.

    Returns:
        None: `output` is written in place.
    """
    ...

def gather_sparse_kv_cache(
    rt: Runtime,
    k_cache: torch.Tensor,
    pe_cache: torch.Tensor,
    block_tables: torch.Tensor,
    topk_indices: torch.Tensor,
    query_lens: torch.Tensor,
    cached_lens: torch.Tensor,
    k_dense_cache: torch.Tensor,
    pe_dense_cache: torch.Tensor,
    batch: int,
    index_topk: int,
    block_size: int,
    kv_lora_rank: int,
    rope_head_dim: int,
    kv_heads: int = 1,
) -> None:
    """Gather sparse KV cache into a contiguous dense cache per batch.

    For each ``(b, i)`` pair, reads the token index ``tok = topk_indices[b, i]``
    (token-index semantics), maps it to the physical block via
    ``block_tables[b * max_num_blocks + tok // block_size]`` and offset
    ``tok % block_size``, and copies one row from paged ``k_cache`` /
    ``pe_cache`` into the contiguous ``k_dense_cache`` / ``pe_dense_cache``.

    Only the first ``min(query_lens[b] + cached_lens[b], index_topk)`` slots per
    batch are written; slots beyond that (topk_indices padding tail) are
    **skipped** (left as-is, typically zero-initialized by the caller).
    :func:`mla_v3` reads only those valid slots and masks the rest in softmax,
    so the skipped slots do not affect output.

    Used by the decode + DSA long-sequence path to feed a contiguous dense cache
    to :func:`mla_v3` instead of the paged layout.

    Args:
        rt (Runtime): Native runtime handle.
        k_cache (torch.Tensor): Paged KV cache (kv_lora_rank slice), shape
            (kvcache_block_num, block_size, kv_heads, kv_lora_rank).
        pe_cache (torch.Tensor): Paged RoPE key cache (rope_head_dim slice),
            shape (kvcache_block_num, block_size, kv_heads, rope_head_dim).
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        topk_indices (torch.Tensor): Top-k token indices from IndexerTopK, shape
            (batch, index_topk), dtype int32.
        query_lens (torch.Tensor): Per-batch current query lengths, shape
            (batch,), dtype int32. Decode = 1 per batch.
        cached_lens (torch.Tensor): Per-batch cached token lengths, shape
            (batch,), dtype int32. Used with query_lens to derive the valid
            slot count per batch.
        k_dense_cache (torch.Tensor): Output contiguous K cache, shape
            (batch, index_topk, kv_heads, kv_lora_rank); written in place.
        pe_dense_cache (torch.Tensor): Output contiguous PE cache, shape
            (batch, index_topk, kv_heads, rope_head_dim); written in place.
        batch (int): Batch size.
        index_topk (int): Number of top-k tokens per batch (dense length).
        block_size (int): KV block size.
        kv_lora_rank (int): KV LoRA rank.
        rope_head_dim (int): Rotary head dimension.
        kv_heads (int): Number of KV heads (must be 1; defaults to 1 for MLA).

    Returns:
        None: `k_dense_cache` and `pe_dense_cache` are written in place.
    """
    ...

def mla_v3(
    rt: Runtime,
    q_absorb: torch.Tensor,
    qr: torch.Tensor,
    k_dense_cache: torch.Tensor,
    pe_dense_cache: torch.Tensor,
    o_absorb: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    n_heads: int,
    rope_head_dim: int,
    kv_lora_rank: int,
    batch: int,
    index_topk: int,
    scale: float,
) -> None:
    """Run MLA v3 attention over a contiguous dense KV cache.

    Variant of :func:`mla_v2` that reads from a contiguous
    ``k_dense_cache`` / ``pe_dense_cache`` (produced by
    :func:`gather_sparse_kv_cache`) instead of a paged block-table layout.
    Drops the block-table indirection and the top-k vgather mask: every dense
    token participates in softmax (no causal mask), since the dense cache is
    already the top-k selected tokens. The QK/SV n-tile size is a fixed
    internal constant (no ``block_size`` argument needed, unlike :func:`mla_v2`).

    The ``qk`` workspace is allocated internally from the runtime tensor pool.

    Args:
        rt (Runtime): Native runtime handle.
        q_absorb (torch.Tensor): Pre-absorbed query, shape
            (total_query_tokens, n_heads, kv_lora_rank).
        qr (torch.Tensor): Pre-rotated q_rope slice, shape
            (total_query_tokens, n_heads, rope_head_dim).
        k_dense_cache (torch.Tensor): Contiguous K cache, shape
            (batch, index_topk, kv_heads, kv_lora_rank).
        pe_dense_cache (torch.Tensor): Contiguous PE cache, shape
            (batch, index_topk, kv_heads, rope_head_dim).
        o_absorb (torch.Tensor): Output absorb tensor, shape
            (total_query_tokens, n_heads, kv_lora_rank); written in place.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        n_heads (int): Number of query heads.
        rope_head_dim (int): Rotary head dimension.
        kv_lora_rank (int): KV LoRA rank.
        block_size (int): KV block size (used as the n0 tile in QK/SV mmad).
        batch (int): Batch size.
        index_topk (int): Number of top-k tokens per batch (dense length).
        scale (float): Attention scaling factor.

    Returns:
        None: `o_absorb` is written in place. Apply the WUV projection on the
        host side to obtain the final output.
    """
    ...

def indexer_scores(
    rt: Runtime,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weight: torch.Tensor,
    scores: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    head_dim: int,
    block_size: int,
    batch: int,
) -> None:
    """Compute DSA indexer scores over cached keys.

    Args:
        rt (Runtime): Native runtime handle.
        q (torch.Tensor): Query tensor.
        k_cache (torch.Tensor): Key cache tensor.
        weight (torch.Tensor): Indexer weight tensor.
        scores (torch.Tensor): Output score tensor.
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size.
        batch (int): Batch size.

    Returns:
        None: `scores` is written in place.
    """
    ...

def indexer_topk(
    rt: Runtime,
    q: torch.Tensor,
    k_cache: torch.Tensor,
    weight: torch.Tensor,
    indices: torch.Tensor,
    topk_indices: torch.Tensor,
    query_start_loc: torch.Tensor,
    lens: torch.Tensor,
    cached_lens: torch.Tensor,
    block_tables: torch.Tensor,
    n_heads: int,
    head_dim: int,
    block_size: int,
    batch: int,
    top_k: int,
) -> None:
    """Fused DSA indexer scores + top-k selection over cached keys.

    Combines :func:`indexer_scores` and :func:`topk` into a single kernel
    launch with pingpong buffers. Scratch buffers (scores, last_topk, sync)
    are allocated internally by the runtime.

    Args:
        rt (Runtime): Native runtime handle.
        q (torch.Tensor): Query tensor ``[total_query_len, n_heads, head_dim]``.
        k_cache (torch.Tensor): Key cache tensor ``[max_num_block*batch, block_size, head_dim]``.
        weight (torch.Tensor): Indexer weight tensor
            ``[total_query_len, head_dim + n_heads]`` (last ``n_heads`` columns are
            the indexer weights).
        indices (torch.Tensor): Input index tensor ``[max_seq_len]`` (int32),
            pre-filled with ``0..max_seq_len-1``.
        topk_indices (torch.Tensor): Output top-k indices tensor
            ``[total_query_len, top_k]`` (int32).
        query_start_loc (torch.Tensor): Prefix-sum prompt lengths.
        lens (torch.Tensor): Current token lengths.
        cached_lens (torch.Tensor): Cached token lengths.
        block_tables (torch.Tensor): Block table, 1-D ``[batch * max_num_blocks]``
            (legacy flattened) or 2-D ``[batch, max_num_blocks]`` int32. The
            per-request max_num_blocks is derived internally from the shape
            (2-D: shape[1]; 1-D: len // batch).
        n_heads (int): Number of heads.
        head_dim (int): Head dimension.
        block_size (int): KV block size (must be <= 128).
        batch (int): Batch size.
        top_k (int): Number of top-k indices to select (must be <= 2048).

    Returns:
        None: ``topk_indices`` is written in place.
    """
    ...

def muls(rt: Runtime, input: torch.Tensor, scale: float, output: torch.Tensor) -> None:
    """Multiply tensor by scalar and write to output.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): Input tensor.
        scale (float): Scalar multiplier.
        output (torch.Tensor): Output tensor.

    Returns:
        None: `output` is written in place.
    """
    ...

def experts_counts_sum(
    rt: Runtime,
    experts_counts_input: torch.Tensor,
    tokens_per_epgroup: torch.Tensor,
    experts_counts_output: torch.Tensor,
    n_routed_experts: int,
) -> None:
    """Compute two reductions over a per-DP-rank per-expert token count matrix.

    Given an input ``experts_counts_input`` of shape ``[ep_size, n_routed_experts]``
    where row *i* is the dispatch count vector from DP rank *i*, the kernel writes:

    * ``tokens_per_epgroup[dp_idx, ep_id]`` — total tokens from DP rank ``dp_idx``
      destined for experts in EP group ``ep_id``.  Shape ``[ep_size, ep_size]``.
    * ``experts_counts_output[expert]`` — total tokens for expert ``expert``,
      summed across all DP ranks.  Shape ``[n_routed_experts]``.

    Args:
        rt (Runtime): Native runtime handle.
        experts_counts_input (torch.Tensor): Per-DP-rank per-expert count matrix
            ``[ep_size, n_routed_experts]`` (int32).
        tokens_per_epgroup (torch.Tensor): Output buffer for per-DP-rank
            per-EP-group token sums ``[ep_size, ep_size]`` (int32).
        experts_counts_output (torch.Tensor): Output buffer for per-expert total
            counts ``[n_routed_experts]`` (int32).
        n_routed_experts (int): Total number of routed experts.

    Returns:
        None: ``tokens_per_epgroup`` and ``experts_counts_output`` are written
        in place.
    """
    ...

def reorder_moe(
    rt: Runtime,
    in_: torch.Tensor,
    out: torch.Tensor,
    counts: torch.Tensor,
    hidden_size: int,
    local_start: int,
    local_end: int,
    forward: bool,
) -> None:
    """Permute token rows between source-grouped and expert-grouped layouts.

    The kernel handles two directions based on ``forward``:

    * **forward=True**: source-grouped → expert-grouped.  Input tokens are
      grouped by source EP rank; output tokens are grouped by target expert
      index, ready for per-expert computation.
    * **forward=False**: expert-grouped → source-grouped.  The inverse
      permutation that restores the original source-grouped order.

    The ``counts`` tensor of shape ``[moe_ep_size, n_routed_experts]``
    (int32) specifies how many tokens each source rank sends to each expert.
    ``local_start`` and ``local_end`` select a contiguous range of local
    experts (the shard owned by the current EP rank).

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input token tensor ``[total_tokens, hidden_size]``.
        out (torch.Tensor): Output token tensor ``[total_tokens, hidden_size]``.
        counts (torch.Tensor): Per-source per-expert token count matrix
            ``[moe_ep_size, n_routed_experts]`` (int32).
        hidden_size (int): Hidden dimension per token.
        local_start (int): First local expert index (inclusive).
        local_end (int): Last local expert index (exclusive).
        forward (bool): ``True`` for forward permutation, ``False`` for reverse.

    Returns:
        None: ``out`` is written in place.
    """
    ...

def linear_att_proj(
    rt: Runtime,
    x: torch.Tensor,
    W_qkv: torch.Tensor,
    W_z: torch.Tensor,
    W_b: torch.Tensor,
    W_a: torch.Tensor,
    mix_qkv: torch.Tensor,
    z: torch.Tensor,
    b: torch.Tensor,
    a: torch.Tensor,
    m: int,
    n: int,
    v: int,
    h: int,
    k: int,
) -> None:
    """Linear attention projection.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Input tensor.
        W_qkv (torch.Tensor): QKV weight tensor.
        W_z (torch.Tensor): Z weight tensor.
        W_b (torch.Tensor): B weight tensor.
        W_a (torch.Tensor): A weight tensor.
        mix_qkv (torch.Tensor): Output mixed QKV tensor.
        z (torch.Tensor): Output z(gating parameters) tensor.
        b (torch.Tensor): Output b(beta input) tensor.
        a (torch.Tensor): Output a(decay input) tensor.
        m (int): Dimension of the input x(batch*seqlen).
        n (int): QKV weight dimension.
        v (int): Z weight dimension.
        h (int): B,A weight dimension.
        k (int): Hidden layer dimension.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def transpose_1_2(rt: Runtime, input: torch.Tensor, eye: torch.Tensor, output: torch.Tensor) -> None:
    """Transpose input 3D tensor along dimensions 1 and 2.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): Input tensor of shape (b, m, n).
        output (torch.Tensor): Output tensor of shape (b, n, m).

    Returns:
        None: `output` is written in place.
    """
    ...

def linear_att_conv_and_silu(
    rt: Runtime,
    mix_qkv: torch.Tensor,
    conv_state: torch.Tensor,
    weight: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Fused causal conv1d + SiLU for linear attention (no host concat).

    Args:
        rt (Runtime): Native runtime handle.
        mix_qkv (torch.Tensor): Input mixed QKV tensor, shape [B, S, C].
        conv_state (torch.Tensor): Convolution state tensor, shape [B, C, K].
        weight (torch.Tensor): Kernel weight tensor, shape [C, 1, K] or [C, K].
        output (torch.Tensor): Output tensor, shape [B, S, C].

    Returns:
        None: `output` is written in place. State is always updated.
    """
    ...

def split_col(rt: Runtime, in_: torch.Tensor, outputs: List[torch.Tensor]) -> None:
    """Split tensor along column dimension into multiple outputs.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Input tensor to split.
        outputs (List[torch.Tensor]): List of output tensors.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def concat(rt: Runtime, inputs: List[torch.Tensor], out: torch.Tensor) -> None:
    """Concatenate input tensors into one contiguous byte buffer (1D pack).

    The inputs are laid out end-to-end by raw bytes into ``out``. ``out`` is
    treated as a flat byte buffer: its total byte size must equal the sum of the
    input byte sizes. Element dtype / shape of the inputs need not match; only
    bytes are packed. Used by the MoE packed-send path to stage one AllGather.

    Args:
        rt (Runtime): Native runtime handle.
        inputs (List[torch.Tensor]): Input tensors to pack (each viewed as bytes).
        out (torch.Tensor): Flat output buffer holding all inputs concatenated.

    Returns:
        None: ``out`` is written in place.
    """
    ...

def split(
    rt: Runtime,
    in_: torch.Tensor,
    outputs: List[torch.Tensor],
    sizes: List[int],
    num_packets: int,
) -> None:
    """Split a contiguous byte buffer into outputs, repeated across packets.

    ``in_`` holds ``num_packets`` interleaved packets, each ``sum(sizes)``
    bytes. For packet ``i`` and output ``j``, ``sizes[j]`` bytes are copied from
    ``in_ + i*sum(sizes) + offset_j`` into ``outputs[j] + i*sizes[j]``. Each
    output is treated as a flat byte buffer of size ``num_packets * sizes[j]``
    bytes. Used by the MoE packed-recv path to deinterleave one AllGather result.

    Args:
        rt (Runtime): Native runtime handle.
        in_ (torch.Tensor): Flat input buffer holding all packets.
        outputs (List[torch.Tensor]): Output buffers, one per segment.
        sizes (List[int]): Byte size of each segment within a single packet.
        num_packets (int): Number of interleaved packets in ``in_``.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def beta_decay(
    rt: Runtime,
    b: torch.Tensor,
    a: torch.Tensor,
    A_log: torch.Tensor,
    dt_bias: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    bsz: int,
    seqlen: int,
    num_v_heads: int,
) -> None:
    """Calculate beta and decay for linear attention.

    Args:
        rt (Runtime): Native runtime handle.
        b (torch.Tensor): b input tensor.
        a (torch.Tensor): a input tensor.
        A_log (torch.Tensor): Learnable Decay parameters.
        dt_bias (torch.Tensor): Time bias.
        beta (torch.Tensor): beta tensor.
        g (torch.Tensor): g(decay) tensor.
        bsz (int): Batch size.
        seqlen (int): Sequence length.
        num_v_heads (int): Number of value heads.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def recurrent_gated_delta_rule(
    rt: Runtime,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    beta: torch.Tensor,
    g: torch.Tensor,
    state: torch.Tensor,
    out: torch.Tensor,
    batch: int,
    seqlen: int,
    num_heads: int,
    k_dim: int,
    v_dim: int,
) -> None:
    """Recurrent gated delta rule (GDN linear-attention core).

    Args:
        rt (Runtime): Native runtime handle.
        query (torch.Tensor): [B*S, H*k_dim], L2-normalized.
        key (torch.Tensor): [B*S, H*k_dim], L2-normalized.
        value (torch.Tensor): [B*S, H*v_dim].
        beta (torch.Tensor): [B*S, H].
        g (torch.Tensor): [B*S, H], log-space decay (kernel applies exp).
        state (torch.Tensor): [B, H, k_dim, v_dim], updated in-place.
        out (torch.Tensor): [B*S, H*v_dim].
        batch (int): Batch size.
        seqlen (int): Sequence length.
        num_heads (int): Number of heads.
        k_dim (int): Key head dim (<=128).
        v_dim (int): Value head dim (<=128).

    Returns:
        None: Output and state are written in place.
    """
    ...

def einsum_mht_hdt_mhd(
    rt: Runtime,
    mht: torch.Tensor,
    hdt: torch.Tensor,
    mhd: torch.Tensor,
    m: int,
    h: int,
    t: int,
    d: int,
    weight_nz: bool = False,
) -> None:
    """Batched matmul for ``mhd = einsum("mht,hdt->mhd", mht, hdt)``.

    The right operand ``hdt`` has a head-major layout (``[h, d, t]``), which the
    kernel consumes via the matmul transpose path. The output ``mhd`` is laid
    out as ``[m, h, d]`` with the head dimension ``h`` kept as an outer loop.

    Args:
        rt (Runtime): Native runtime handle.
        mht (torch.Tensor): Left operand of shape ``[m, h, t]``.
        hdt (torch.Tensor): Right operand of shape ``[h, d, t]``.
        mhd (torch.Tensor): Output tensor of shape ``[m, h, d]``.
        m (int): Outer batch dimension (token count).
        h (int): Head dimension.
        t (int): Inner reduction dimension.
        d (int): Output feature dimension.
        weight_nz (bool): Whether ``hdt`` is in NZ weight layout.

    Returns:
        None: ``mhd`` is written in place.
    """
    ...

def einsum_mht_htd_mhd(
    rt: Runtime,
    mht: torch.Tensor,
    htd: torch.Tensor,
    mhd: torch.Tensor,
    m: int,
    h: int,
    t: int,
    d: int,
    weight_nz: bool = False,
) -> None:
    """Batched matmul for ``mhd = einsum("mht,htd->mhd", mht, htd)``.

    The right operand ``htd`` has a head-row layout (``[h, t, d]``), which the
    kernel consumes directly via the standard matmul path. The output ``mhd``
    is laid out as ``[m, h, d]`` with the head dimension ``h`` kept as an outer
    loop.

    Args:
        rt (Runtime): Native runtime handle.
        mht (torch.Tensor): Left operand of shape ``[m, h, t]``.
        htd (torch.Tensor): Right operand of shape ``[h, t, d]``.
        mhd (torch.Tensor): Output tensor of shape ``[m, h, d]``.
        m (int): Outer batch dimension (token count).
        h (int): Head dimension.
        t (int): Inner reduction dimension.
        d (int): Output feature dimension.
        weight_nz (bool): Whether ``htd`` is in NZ weight layout.

    Returns:
        None: ``mhd`` is written in place.
    """
    ...

def unpack_activation(
    rt: Runtime,
    input: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Split int8 tensor to low/high int4 tensor.

    Args:
        rt (Runtime): Native runtime handle.
        input (torch.Tensor): input int8 tensor.
        output (torch.Tensor): output low/high int4 tensor.

    Returns:
        None: Output tensors are written in place.
    """
    ...

def print(x: torch.Tensor, name: str = "", row: int = 6, col: int = 6) -> None:
    """Print a tensor preview for debugging.

    Args:
        x (torch.Tensor): Tensor to print.
        name (str): Optional label shown in output.
        row (int): Number of rows to print.
        col (int): Number of columns to print.

    Returns:
        None: Output is emitted to native stdout.
    """
    ...

def get_tile_size_of_cached_kv(
    cached_lens: List[int],
    query_lens: List[int],
    head_num_in_group: int,
    n_kv_heads: int,
    block_size: int,
    aic_num: int,
) -> int:
    """Get optimal tile size for cached KV based on workload.

    This function computes the optimal tile size for flash attention
    based on the current workload characteristics including cached KV
    lengths and query lengths.

    Args:
        cached_lens (List[int]): Per-sample cached KV token lengths.
        query_lens (List[int]): Per-sample query token lengths.
        head_num_in_group (int): Number of heads in each attention group.
        n_kv_heads (int): Number of key/value heads.
        block_size (int): KV cache block size.
        aic_num (int): Number of AI cores available.

    Returns:
        int: Optimal tile size for cached KV in flash attention.
    """
    ...

def hc_act(
    rt: Runtime,
    mixes: torch.Tensor,
    hc_scale: torch.Tensor,
    hc_base: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    hc_mult: int,
    eps: float,
    sinkhorn_iters: int,
    x_resid: torch.Tensor,
    output: torch.Tensor,
) -> None:
    """Hyper-Connection gate activation.

    Computes pre/post/comb gates from `mixes`:
      pre  = sigmoid(mixes[:, :K]       * scale[0] + base[:K])           + eps
      post = 2 * sigmoid(mixes[:, K:2K] * scale[1] + base[K:2K])
      comb = sinkhorn(softmax(mixes[:, 2K:] * scale[2] + base[2K:]) + eps)
    where K = hc_mult. `mixes` [n, mix_hc] (mix_hc = (2+K)*K), `hc_scale` [3],
    `hc_base` [mix_hc]; all fp32. Head mode is auto-detected when
    `hc_base.numel() == hc_mult` (only pre runs, post/comb untouched). `pre` is
    consumed internally and never written to GM.

    Args:
        rt (Runtime): Native runtime handle.
        mixes (torch.Tensor): Gate pre-activation [n, mix_hc] fp32.
        hc_scale (torch.Tensor): Per-segment scale [3] fp32 (or [1] in head mode).
        hc_base (torch.Tensor): Per-segment bias [mix_hc] fp32 (or [hc_mult] head).
        post (torch.Tensor): Output post gate [n, hc_mult] fp32.
        comb (torch.Tensor): Output comb [n, hc_mult*hc_mult] fp32.
        hc_mult (int): Hyper-connection multiplier K.
        eps (float): Epsilon added to pre and every Sinkhorn denominator.
        sinkhorn_iters (int): Sinkhorn normalization iterations.
        x_resid (torch.Tensor): Merge input [n, hc_mult, hidden] bf16.
        output (torch.Tensor): Merge output [n, hidden] bf16.

    Returns:
        None: `post`/`comb`/`output` written in place.
    """
    ...

def hc_post(
    rt: Runtime,
    x: torch.Tensor,
    post: torch.Tensor,
    comb: torch.Tensor,
    residual: torch.Tensor,
    y: torch.Tensor,
    m: int,
    hc_mult: int,
    hidden: int,
) -> None:
    """Hyper-Connection post-activation merge (DeepSeek-V4).

    y[m,H,D] = post[m,H]*x[m,D] (broadcast) + sum_k comb[m,H,k]*residual[m,k,D].
    `x` [m, hidden] bf16, `post` [m, hc_mult] fp32, `comb` [m, hc_mult*hc_mult] fp32,
    `residual` [m, hc_mult, hidden] bf16, `y` [m, hc_mult, hidden] bf16. `residual`
    may alias `y` (in-place): all sources are read before any output is written.

    Args:
        rt (Runtime): Native runtime handle.
        x (torch.Tensor): Submodule output [m, hidden] bf16.
        post (torch.Tensor): Post gate [m, hc_mult] fp32.
        comb (torch.Tensor): Comb matrix [m, hc_mult*hc_mult] fp32.
        residual (torch.Tensor): Residual stream [m, hc_mult, hidden] bf16.
        y (torch.Tensor): Output [m, hc_mult, hidden] bf16.
        m (int): Token count.
        hc_mult (int): Hyper-connection multiplier H.
        hidden (int): Hidden dimension D.

    Returns:
        None: `y` written in place.
    """
    ...