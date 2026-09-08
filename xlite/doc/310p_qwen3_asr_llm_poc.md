# Ascend 310P Qwen3-ASR LLM-only POC

This branch validates only the dense Qwen3 decoder on one Ascend 310P device.
The audio tower, vLLM integration, tensor parallelism, quantization and graph
capture are outside the POC boundary.

## Build

Use a 310P CANN development container and build the dedicated capability
profile:

```bash
cd xlite
# Xlite accepts either case, then passes the canonical CANN product name to
# AscendC. Prefer the canonical spelling in build commands.
export SOC_VERSION=Ascend310P3
export XLITE_KERNEL_SET=llm_fp16
pip install -v -e . --no-build-isolation
```

The profile defines the project-owned `XLITE_ARCH_310P` macro; it never fakes
`__DAV_C220_VEC__` or `__DAV_C220_CUBE__`. Device compilation additionally
requires the compiler-provided `__NPU_ARCH__` to equal `2002`. The profile
enforces rank 0, TP1, DP1, dense MHA/GQA and disables the 910B3 matmul swizzle
table. Unsupported parallel, MoE and quantized model configs fail during
runtime/model initialization instead of entering an unverified kernel path.

The build stops at configure time if the target CANN does not provide
`aclnnMatmul`, `aclnnPromptFlashAttention`, or `aclnnIncreFlashAttention` headers.
MatMul and attention keep the Xlite ABI but execute through ACLNN. ACLNN
workspace is borrowed from `XTensorPool`. On 310P, ACLNN work is submitted
asynchronously to the single Xlite runtime stream and the workspace is returned
to the pool immediately. Reuse is safe because every later use of that device
address is ordered on the same stream. MatMul currently retains the validated
synchronization fallback by default; select `XLITE_310P_ASYNC_MATMUL=1` to
exercise asynchronous MatMul after the mask upload ordering fix below.

For asynchronous-lifetime diagnosis, synchronization can be restricted to one
operator family with `XLITE_310P_FORCE_SYNC_MATMUL=1` or
`XLITE_310P_FORCE_SYNC_ATTENTION=1`. The global switch takes precedence. These
Explicit force-sync switches override the asynchronous MatMul selection.

Attention metadata is retained on the host and reused by all decoder layers.
`PrepareAttn()` copies lens, cached lens, query starts, block tables, slot
mapping, and version-0 positions from page-locked staging buffers. The 310P
ACLNN attention backend therefore performs no per-layer metadata D2H copy.
Online forward paths exchange ownership with the PyTorch current stream through
two dedicated `ACL_EVENT_SYNC` events: one for PyTorch-to-Xlite input readiness
and one for Xlite-to-PyTorch output readiness. The events are not reused in
opposite directions. The original async model failed on physical 310P, and
forward-boundary synchronization did not help. Source inspection found an
unordered write in Prefill Attention: `MemcpyH2D` used synchronous
`aclrtMemcpy` to upload a mask into TensorPool storage that an earlier queued
MatMul could still be using as workspace. A synchronous host API return does
not place that write after work on the Xlite stream. Per-MatMul synchronization
masked this hazard; the observations do not establish a CANN MatMul defect.

Mask upload now uses `aclrtMemcpy2dAsync` on `rt.stream` from a runtime-owned,
immutable 2048-by-2048 pinned causal template (4 MiB host memory). A row offset
of `cachedLength` implements chunked prefill. Padded query rows are unmasked
and discarded. All writes to the pooled mask are stream ordered, and the host
template is freed only after the destructor drains the stream. This avoids
both per-layer host allocation and mutation of host buffers referenced by DMA.
The validated MatMul synchronization fallback remains the default until the
fixed async model passes hardware acceptance. `XLITE_310P_FORCE_SYNC_FORWARD=1`
is optional and disabled by default.

Runtime counters are available through `Runtime.get_stats()` and through the
vLLM-Ascend Xlite runtime statistics. `attention_metadata_d2h_bytes` must remain
zero. `forced_sync_launches` records the MatMul correctness fallback; when
async MatMul is explicitly selected and no force flags are set, it is zero.
`forward_boundary_synchronizations` is zero in the default path.

`op.cpp` currently exposes all launch stubs through one shared host ABI. The
POC therefore retains those stubs in the build even though only the FP16 dense
decoder path is accepted at runtime. Splitting the host operator library is a
follow-up requirement before this profile can become a size-minimal package.

Before building, record the exact environment and ACLNN header hashes:

```bash
python tests/poc_310p/probe_environment.py \
  --report /tmp/qwen3_asr_310p_environment.json
```

## Run the penetration test

```bash
cd xlite
export FORWARD_BACKEND=xlite
export XLITE_WEIGHT_NZ=0
python tests/poc_310p/run_qwen3_asr_llm.py \
  --checkpoint /models/Qwen3-ASR-1.7B \
  --input-mode tokens \
  --decode-tokens 16 \
  --stability-iters 50 \
  --report /tmp/qwen3_asr_310p_tokens.json

python tests/poc_310p/run_qwen3_asr_llm.py \
  --checkpoint /models/Qwen3-ASR-1.7B \
  --input-mode synthetic \
  --prompt-tokens 32 \
  --decode-tokens 16 \
  --report /tmp/qwen3_asr_310p_embeds.json

python tests/poc_310p/run_qwen3_asr_llm.py \
  --checkpoint /models/Qwen3-ASR-1.7B \
  --input-mode file \
  --embeds-file /data/audio_embeddings.npy \
  --positions-file /data/audio_positions.npy \
  --decode-tokens 16 \
  --report /tmp/qwen3_asr_310p_audio.json
```

The runner extracts `thinker_config.text_config`, skips audio-tower weights,
uses the existing four-dimensional per-layer K/V cache and compares Xlite with
the torch_npu reference path using the same weights. It records final hidden
and logits cosine similarity, generated token IDs, prefill/decode latency,
peak memory and a 50-iteration memory-stability check.

For mRoPE checkpoints, `rope_parameters`/`rope_scaling`, `mrope_section`,
`mrope_interleaved`, RoPE type and theta are read from
`thinker_config.text_config`. A positions file must cover both prompt and decode
steps and have shape `[prompt+decode]` or `[3,prompt+decode]`. Real audio
embeddings are rejected without the positions emitted by the same processor;
plain `arange` positions are not substituted.

For per-layer tensor diagnostics, build editable mode with
`XLITE_DEBUG_ON=forward`; the existing forward debug instrumentation prints
layer boundaries and NaN/Inf checks.

## Acceptance and stop gates

The generated report passes when:

- hidden and logits cosine similarity are both at least 0.999;
- the first greedy token equals the torch_npu result;
- at least 16 decode steps complete;
- the stability loop does not leave additional allocated device memory.

Before claiming hardware support, run `tests/poc_310p/run_kernel_smoke.sh`. Its
vector tests prefill outputs with NaN sentinels, so an empty kernel fails before
the numeric comparison. MatMul covers M=1/8/127/128/129. Attention covers
prefill 1/127/128/129 and decode valid lengths 16/127/128/129/512/2048, including
cache block boundaries. If MatMul, RoPE/cache or decode attention
cannot compile for `ascend310p3`, stop the full-model run and record the first
compiler diagnostic plus the affected kernel and shape. Do not silently fall
back to BF16, quantized kernels, multi-card communication or vLLM's 5D NZ KV
cache.

Gate 5 uses the same runner with `--num-layers 1`; the full acceptance script
runs that comparison before the five full-model text cases and the real-audio
case.

The P1 asynchronous workspace checks can be run independently after the normal
operator sweep:

```bash
python tests/poc_310p/test_matmul.py --async-stress-iters 1000
python tests/kernels/attention.py --async-stress-iters 1000

XLITE_310P_FORCE_SYNC_ACLNN=1 \
  python tests/poc_310p/test_matmul.py --shape 20 2048 2048
XLITE_310P_FORCE_SYNC_ACLNN=1 XLITE_ATTENTION_CASE_INDEX=6 \
  python tests/kernels/attention.py
```

The two stress options set the test-only
`XLITE_310P_STRESS_WORKSPACE_REUSE=1` switch. After every ACLNN launch, the
released workspace is immediately reacquired and overwritten on the same
runtime stream. This verifies stream-ordered reuse and must not be enabled in
serving.

## Known unverified items

- AscendC compilation and execution require a Linux 310P host and were not run
  in the Windows development workspace.
- The conservative 310P UB profile and disabled swizzle still require silicon
  validation and performance tuning.
- The ACLNN cache view currently accepts only batch 1 and consecutive physical
  block IDs; non-contiguous blocks fail explicitly. Chunked prefill also
  fails explicitly. A bounded gather fallback remains future work.
- The legacy monolithic host ABI still causes non-POC launcher libraries to be
  packaged. Those paths are rejected by model/runtime gates but are not yet
  physically removed from the wheel.
- Exact 16-token agreement must be repeated over five fixed prompts before the
  POC is promoted beyond penetration-test status.
