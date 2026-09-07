# Ascend 310P Qwen3-ASR LLM-only POC

This branch validates only the dense Qwen3 decoder on one Ascend 310P device.
The audio tower, vLLM integration, tensor parallelism, quantization and graph
capture are outside the POC boundary.

## Build

Use a 310P CANN development container and build the dedicated capability
profile:

```bash
cd xlite
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
workspace is borrowed from `XTensorPool`; the correctness backend synchronizes
before returning it. This synchronization is intentional and is not a
performance design.

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
cannot compile for `Ascend310P3`, stop the full-model run and record the first
compiler diagnostic plus the affected kernel and shape. Do not silently fall
back to BF16, quantized kernels, multi-card communication or vLLM's 5D NZ KV
cache.

Gate 5 uses the same runner with `--num-layers 1`; the full acceptance script
runs that comparison before the five full-model text cases and the real-audio
case.

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
