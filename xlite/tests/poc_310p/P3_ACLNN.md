# P3 ACLNN MatMul / LM Head

Status: opt-in implementation, CANN build and device acceptance pending. Attention
must remain `legacy`. MatMul synchronization flags and Sampling are unchanged.

## Implemented paths

- `Runtime.set_matmul_optimization("legacy" | "p3_aclnn")`; legacy default.
- P3 M1 LM Head writes each N chunk directly into final logits. M>1 packed output
  is the default; direct strided output is selected only by offline probing.
- Runtime precomputes dimensions/strides/chunk plans for the five Qwen3 projection
  shapes at M1–20. One input ACL descriptor per logical P3 call, fresh executor
  for every ACLNN query/execute. No executor cache and no persistent duplicate weights.
- Candidates: N12288/24576/49152/151936, workspace cap remains 512 MiB.
  Pool planning adds 2 MiB for the largest packed output tile; this is staging,
  not a larger ACLNN workspace allowance, and is included in startup estimation.
- Stats classify the verified `(N,K)` shapes as QKV/O/Gate-Up/Down/LM Head;
  unfamiliar shapes are `other`, use legacy and are not guessed from tensor names.
- No online fallback after an execute failure. Unsupported layouts are rejected
  by isolated offline workers, and the policy retains a measured safe alternative.

## First and only next command

In the existing CANN/torch_npu environment, from GVirt/xlite:

```bash
bash tests/poc_310p/validate_p3_310p.sh
```

Build plus P3-only screen/validation. Each independent candidate has a JSON/log
under `p3_matmul_report/`; `summary.json` retains all rejections. Larger N/layout
rejections are expected outcomes, not an instruction to increase memory. The
screen first uses M1/8/20, then checks the nearest anchor winner against legacy
at each M1–20; projections check M1/8/20. Slower/invalid winners stay legacy.
The checks include 10 retained output buffers, guards, finite values, cosine
>=0.999 and rtol/atol=1e-2. Each timing includes submission and output copies.

`--rerun-failed` on the script skips rebuild and reuses passed worker reports.
Policy export refuses mixed build/environment identities. After rebuilding, use
a fresh report directory for comparable timings; retained old tests remain
diagnostic evidence, not a strategy valid for a different binary.

Generated `policy.json` binds SoC/build, CANN package version, torch_npu, sync
flags and shapes. Set `ASCEND_CANN_PACKAGE_PATH` to the active CANN installation
if automatic version discovery fails. The file records package metadata, not a
claim to independently verify every dynamically loaded driver library.

## After successful screening

Inspect `optimized shapes` first. If zero, stop: do not run full ASR expecting a
speedup. If nonzero, preserve the binary and sync flags and run:

```bash
python3 tests/poc_310p/run_qwen3_asr_llm.py \
  --checkpoint /home/models/Qwen3-ASR-1.7B \
  --decode-attention-backend legacy --matmul-optimization p3_aclnn \
  --matmul-policy p3_matmul_report/policy.json \
  --input-mode synthetic --prompt-tokens 129 --decode-tokens 16 \
  --max-seq-len 512 --stability-iters 1 \
  --report poc_310p_results/p3_full28_synthetic129.json
```

Require `passed=true`, reference token equality and hidden/logits cosine>=0.999.
Then use vllm-ascend's P3 command, without Native/Attention/Audio reruns.

## Short diagnostics, not serving instrumentation

The tuner accepts `--diagnostics` for a separate diagnostic pass (excluded from
selection timings): `host_prepare_ms` includes descriptor/workspace preparation;
`device_ms` is the ACL event interval around MatMul execution, not a breakdown
of its internal kernels. This mode adds event waits explicitly. It is off in serving.
`--profile` captures short torch_npu CPU/NPU traces for LM Head screening workers.
After screening, use `python3 tests/poc_310p/tune_p3_matmul.py --profile-selected`
to trace only legacy and selected LM Head at M1/20 without rerunning the matrix
or overwriting the policy; reports are in `p3_matmul_report/selected_profile/`.
Inspect internal format conversions/copies in the trace; an external `copy_bytes=0`
alone does not establish zero-copy ACLNN execution. No profile claims are made here.

P3 target: >=15% throughput gain at c1 or c20, <=5% loss in the other, exact
transcripts and no request failures. Default remains legacy until real acceptance.
