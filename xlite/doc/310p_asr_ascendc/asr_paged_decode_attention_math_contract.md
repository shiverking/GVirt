# `asr_paged_decode_attention_fp16` math contract

Status: `production_runtime_candidate`. The direct 4-D BSHD paging, low-level
QK/PV MMAD blocks, FP32 partition state, and scratch-free stable merge have
passed the independent device probe. Runtime routing remains an explicit,
non-default backend until the full-model token gate passes.

For request `r`, query head `h`, logical token `t`:

```text
kv_head        = h / 2
physical_block = block_table[r, t / 128]
token_in_block = t % 128
cache_offset   = ((physical_block * 128 + token_in_block) * 8 + kv_head) * 128
score_tile     = Mmad(q[r,h,:], k_cache[logical_tile,:,:]^T)
```

Q has already been scaled by `1/sqrt(128)` by the fused QK/MRoPE/cache kernel.
The attention kernel must not scale it again. Only tokens in `[0, kv_len)` are
read. Block-table padding and unused cache slots are never read.

Each 512-token partition maintains FP32 `(m_p, l_p, a_p[128])`. QK and PV use
audited low-level `LoadData -> Mmad` blocks, then the at-most-four partition
states are merged on chip:

```text
m' = max(m, score)
c  = exp(m - m')
w  = exp(score - m')
l' = l * c + w
a' = a * c + v * w
M  = max(M, m_p)
L' = L * exp(M_old - M) + l_p * exp(m_p - M)
A' = A * exp(M_old - M) + a_p * exp(m_p - M)
output = fp16(A / L)
```

Each partition starts from deterministic FP32 zero/max state. Exponent input
and output use separate UB buffers. Vector-to-scalar and scalar-to-vector
handoffs use dynamically allocated `V_S` and `S_V` events.
Runtime scalar weights are materialized with `vector_dup` into dedicated/reused
FP32 UB vectors after `S_V`, then consumed by `vmul`; they are never passed as
runtime scalar operands to `vmuls` on CANN 9.1 beta1. The `vector_dup` scalar is
an exact mutable stack-local `float`; CANN 9.1 rejects the `const float` form.

## Acceptance

- Batch 1/2/8/20 and KV lengths 1/16/127/128/129/512/2048.
- Per-request mixed lengths and non-contiguous physical blocks.
- FP16 output cosine >= 0.999 and max absolute error <= 0.01 against a CPU FP32
  stable-softmax reference.
- Output guards remain unchanged and K/V cache is bitwise unchanged.
- No high-level Matmul, ACLNN, ATB, dynamic device allocation, host metadata
  readback, or fixed event ID exists in the kernel.

The kernel uses 18,240 bytes UB, 12,288 bytes L1, 4,096 bytes each of L0A/L0B,
and 8,192 bytes L0C. It writes no GM partition state and launches once per
attention layer. The next gate is Runtime routing followed by 129-token Prefill
plus 16-step greedy-token equivalence.
