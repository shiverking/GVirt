# `asr_paged_decode_attention_fp16` math contract

Status: `probe_paged_addressing_online_softmax`. This probe is not eligible for
Runtime routing or performance claims. It validates direct 4-D BSHD paging and
the FP32 online-softmax state before QK/PV are replaced by low-level MMAD blocks.

For request `r`, query head `h`, logical token `t`:

```text
kv_head        = h / 2
physical_block = block_table[r, t / 128]
token_in_block = t % 128
cache_offset   = ((physical_block * 128 + token_in_block) * 8 + kv_head) * 128
score_t        = sum_d fp32(q[r,h,d]) * fp32(k_cache[cache_offset+d])
```

Q has already been scaled by `1/sqrt(128)` by the fused QK/MRoPE/cache kernel.
The attention kernel must not scale it again. Only tokens in `[0, kv_len)` are
read. Block-table padding and unused cache slots are never read.

Online softmax maintains FP32 `(m, l, a[128])`:

```text
m' = max(m, score)
c  = exp(m - m')
w  = exp(score - m')
l' = l * c + w
a' = a * c + v * w
output = fp16(a / l)
```

The first token initializes `m=score`, `l=1`, `a=v`. Exponent input and output
use separate UB buffers. The 128-lane dot reduction is a power-of-two fold with
no vector mask wider than 64 lanes. Vector-to-scalar and scalar-to-vector
handoffs use dynamically allocated `V_S` and `S_V` events.
Runtime scalar weights are materialized with `vector_dup` into dedicated/reused
FP32 UB vectors after `S_V`, then consumed by `vmul`; they are never passed as
runtime scalar operands to `vmuls` on CANN 9.1 beta1. The `vector_dup` scalar is
an exact mutable stack-local `float`; CANN 9.1 rejects the `const float` form.

## Probe acceptance

- Batch 1/2/8/20 and KV lengths 1/16/127/128/129/512/2048.
- Per-request mixed lengths and non-contiguous physical blocks.
- FP16 output cosine >= 0.999 and max absolute error <= 0.01 against a CPU FP32
  stable-softmax reference.
- Output guards remain unchanged and K/V cache is bitwise unchanged.
- No high-level Matmul, ACLNN, ATB, dynamic device allocation, host metadata
  readback, or fixed event ID exists in the kernel.

After this contract passes, the vector QK/PV blocks are replaced by isolated
low-level `LoadData -> Mmad` blocks and KV>512 partition/merge is validated in a
new commit. Only that implementation can advance to `production_decode_attention`.
