# Ascend310P3 ASR production-kernel performance audit

This audit covers the fixed Qwen3-ASR-1.7B Decode path only.  It identifies
structural performance risks from the production source; device timing remains
the deciding gate because the CANN 9.0 cost model is not authoritative for the
CANN 9.1 beta1 target runtime.

## Classification

### Critical: paged Decode Attention

Source evidence (`csrc/kernels/310p/asr_paged_decode_attention_fp16.cpp`):

- Work is assigned per `(request, query_head)` (line 151), so both query heads in one GQA
  group independently stage the same K/V head.
- `StageCacheRows` now coalesces the 16 fixed-head BSHD rows into one ND-to-NZ
  GM-to-L1 burst, splitting only at a physical 128-token block boundary.
- Full 16-token tiles overwrite the complete L1 region and skip zero-fill;
  only a short tail is explicitly initialized to preserve I20.
- QK, scalar online softmax, probability conversion, PV and accumulator update
  are serialized with no double buffering.
- Only the eight available AICs execute up to `batch * 16` head tasks by
  striding through them.

Consequences:

- Cache traffic is duplicated for the two Q heads sharing each KV head.
- The former per-token transfer setup and unconditional zero-fill have been
  removed. Device timing must still confirm the resulting MTE2 reduction.
- Long and high-batch KV shapes scale almost linearly with total head-token
  work.  The current implementation is a correctness baseline, not yet a
  high-performance attention kernel.

Required follow-up after strict token correctness:

1. Process the two Q heads of one GQA group together and stage each K/V tile
   once.
2. Calibrate the block-aware burst against the saved scalar-staging timing for
   KV 16/128/512/2048 and Batch 1/8/20.
3. Calibrate 16/32-token tiles on CANN 9.1 beta1 and introduce TBuf ping-pong
   only after the event graph is measured.
4. Profile MTE2, M, V and MTE3 separately; do not select tiling from kernel
   wall time alone.

### Critical: projection kernel

Source evidence (`csrc/kernels/310p/asr_m200_projection_fp16.cpp`):

- The outer loop is N-tile first (line 95). Every output N tile restages the
  same input activation in the K-tile loop (line 115).
- A/B staging, Mmad and output drain are serialized.
- There is no activation reuse across the QKV/Gate-Up N tiles and no
  load/compute ping-pong.

This is especially costly for Gate/Up (`N=12288`) and QKV (`N=4096`).  The
kernel is a valid low-level-Mmad implementation, but should not be considered
performance-complete merely because it removes ACLNN launch overhead.

Required follow-up:

1. Add per-shape device timing against `m200_asr` and ACLNN for M=1/8/20.
2. Reorder or group N tiles so an activation K tile is reused across multiple
   output tiles when L1 permits.
3. Evaluate A/B L1 ping-pong with the CANN 9.1 measured pipe costs.

### Critical boundary: LM Head

`asr_m200_lm_head_fp16` is not implemented.  Decode still executes ACLNN LM
Head and synchronizes once per token.  Pure AscendC eager and full Decode Graph
cannot close until the fixed `[M,2048] x [2048,151936]` kernel exists.

### Medium: QK-Norm/MRoPE/Cache

The fused kernel removes several launches, but every query/key head loads the
three position-dependent cosine/sine rows independently
(`csrc/kernels/310p/asr_qk_norm_mrope_cache_fp16.cpp:88-106`) and uses a serial
vector pipeline with many full `PIPE_V` barriers. Position rows and norm
weights should be staged/reused per token after correctness profiling proves
this kernel is material in Decode time.

### Medium: RMSNorm, Add-RMSNorm and SiLU-Mul

These kernels are correct production vector paths, but remain launch-oriented:

- RMSNorm and Add-RMSNorm reload weights independently on each active core
  (`asr_rmsnorm_fp16.cpp:98-109`, `asr_add_rmsnorm_fp16.cpp:113-124`).
- SiLU-Mul processes three 2048-element tiles serially without ping-pong
  (`asr_silu_mul_fp16.cpp:46-75`).
- Each remains a separate kernel launch in eager Decode.

Their individual payloads are small, so the next optimization is graph replay
or carefully justified fusion, not a speculative rewrite.  They should only be
retiled if profiling places one of them in the top three device-time costs.

## Decision gates

1. Do not optimize an operator that is outside the top three measured Decode
   costs.
2. Do not promote paged Attention while teacher-forced 16-token equivalence
   fails, even when standalone cosine/max-absolute-error thresholds pass.
3. Every optimized variant keeps the current kernel as an explicit selectable
   baseline until operator, full-model and real-ASR gates pass.
4. Every new tiling records UB/L1/L0 use, dynamic event pairs and measured
   CANN 9.1 beta1 timings for M or Batch 1/8/20.
