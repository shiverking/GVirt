# Ascend310P3 ASR production-kernel performance audit

This audit covers the fixed Qwen3-ASR-1.7B Decode path only.  It identifies
structural performance risks from the production source; device timing remains
the deciding gate because the CANN 9.0 cost model is not authoritative for the
CANN 9.1 beta1 target runtime.

## Classification

### Critical: paged Decode Attention

Source evidence (`csrc/kernels/310p/asr_paged_decode_attention_fp16.cpp`):

- Work is assigned per `(request, kv_head)`. Both query heads in one GQA group
  reuse the same staged K tile and the same staged V tile.
- `StageCacheRows` now coalesces the 16 fixed-head BSHD rows into one ND-to-NZ
  GM-to-L1 burst, splitting only at a physical 128-token block boundary.
- Full 16-token tiles overwrite the complete L1 region and skip zero-fill;
  only a short tail is explicitly initialized to preserve I20.
- QK, scalar online softmax, probability conversion, PV and accumulator update
  are serialized with no double buffering.
- Only the eight available AICs execute up to `batch * 16` head tasks by
  striding through them.

Consequences:

- The former duplicate K/V cache traffic for the two Q heads has been removed;
  CANN 9.1 device timing remains the promotion gate.
- The former per-token transfer setup and unconditional zero-fill have been
  removed. Device timing must still confirm the resulting MTE2 reduction.
- Long and high-batch KV shapes scale almost linearly with total head-token
  work.  The current implementation is a correctness baseline, not yet a
  high-performance attention kernel.

Required follow-up after strict token correctness:

1. Calibrate the GQA-paired block-aware burst against the saved scalar-staging timing for
   KV 16/128/512/2048 and Batch 1/8/20.
2. Calibrate 16/32-token tiles on CANN 9.1 beta1 and introduce TBuf ping-pong
   only after the event graph is measured.
3. Profile MTE2, M, V and MTE3 separately; do not select tiling from kernel
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

1. The experimental `asr_m200_projection_cached_fp16` now stages every A K
   tile once per AIC and reuses it across assigned N tiles.  The production
   kernel remains the explicit baseline.
2. Run the same-build full-shape A/B gate for all four projection classes and
   M=1/2/4/6/8/12/16/20.  Promotion requires the recorded variance-aware gate;
   source-level traffic reduction alone is not performance evidence.
3. Only after that gate, compare the winning projection against `m200_asr` and
   ACLNN using model weights.  Evaluate B ping-pong only if profiling still
   places projection among the top three Decode costs.

### Critical boundary: LM Head

`asr_m200_lm_head_fp16` now has an isolated correctness microprobe, but is not
runtime eligible yet.  It executes the fixed `[M,2048] x [2048,151936]`
contract in one launch with eight AICs, FP32 accumulation and direct full-logit
output.  The initial schedule intentionally matches the already proven
low-level projection data layout.  It restages A for each N tile and therefore
must not replace ACLNN until M=1/8/20 correctness and CANN 9.1 timing pass;
grouped N-tile activation reuse is the next measured optimization.

### External device-state diagnostic limitation

The isolated attention suite can fail before any kernel launch when its
required `aclrtSetDevice(0)` returns `507033`.  This is classified as external
device/context availability, not kernel evidence.  Device visibility and
logical-to-physical mapping are deployment-owned and must not be rewritten by
the test or runtime.  Re-run the affected probe only after the existing device
environment is healthy; this known issue does not block source-only LM Head
development.

### Medium: QK-Norm/MRoPE/Cache

The fused kernel removes several launches, but every query/key head loads the
three position-dependent cosine/sine rows independently
(`csrc/kernels/310p/asr_qk_norm_mrope_cache_fp16.cpp:88-106`) and uses a serial
vector pipeline with many full `PIPE_V` barriers. Position rows and norm
weights should be staged/reused per token after correctness profiling proves
this kernel is material in Decode time.

The experimental `asr_qk_norm_mrope_cache_grouped_fp16` addresses only those
measured structural redundancies: each AIV owns two Q heads and one K/V head,
converts Q/K weights once per launch, and composes the three position rows once
per token for all three heads.  The production kernel remains the baseline;
promotion requires the eight-shape same-build device A/B gate followed by
model-weight token equivalence.

That gate rejected the grouped schedule: all correctness runs passed, but
M=2--20 regressed 1.18--4.53% and no shape achieved the required 5% gain.
The original head-parallel production kernel therefore remains the measured
winner.

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

RMSNorm and Add-RMSNorm already load and convert weights once per active core
and reuse them across assigned rows, so no additional eager candidate is
justified before profiling.  SiLU-Mul still performs three complete
load/event/store rounds per row; `asr_silu_mul_row_fp16` is an isolated
whole-row candidate subject to the same eight-shape device A/B gate.

## Decision gates

1. Do not optimize an operator that is outside the top three measured Decode
   costs.
2. Do not promote paged Attention while teacher-forced 16-token equivalence
   fails, even when standalone cosine/max-absolute-error thresholds pass.
3. Every optimized variant keeps the current kernel as an explicit selectable
   baseline until operator, full-model and real-ASR gates pass.
4. Every new tiling records UB/L1/L0 use, dynamic event pairs and measured
   CANN 9.1 beta1 timings for M or Batch 1/8/20.

## Explicit performance-runtime slice

The complete projection, whole-row SiLU-Mul and cached LM Head microprobe
gates have passed. They are compiled into the isolated AIC/AIV libraries but
are selected only by `matmul_backend=ascendc_asr_perf`. The existing
`ascendc_asr` backend remains the baseline and the global default remains
`m200_asr`.

`ascendc_asr_perf` applies only to fixed pure-Decode shapes with M in [1,20].
Dynamic Prefill retains the established ACLNN path and every such boundary is
counted. Unknown Decode MatMul shapes still fail instead of silently falling
back. Runtime promotion now requires random/model-weight correctness, exact
16-token whole-model equivalence and a later repeated real-ASR performance
gate.
