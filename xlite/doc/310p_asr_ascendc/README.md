# Ascend310P3 Qwen3-ASR AscendC backend

This directory records the fixed contract and provenance for the experimental
Qwen3-ASR-only AscendC backend. The files under `D:/Downloads/shared` are design
inputs only: the build must not read that directory and production kernels must
not include generated templates from it.

The target is Ascend310P3 (`__NPU_ARCH__ == 2002`), CANN 9.1 beta1, FP16, TP1,
batch 1--20, hidden size 2048, Q16/KV8/D128, intermediate size 6144, block size
128, sequence length at most 2048, and vocabulary size 151936.

New production sources are named `asr_*.cpp`. They use explicit low-level
`Mmad`, `LoadData`, data-movement and vector operations. The high-level AscendC
Matmul interface is deliberately prohibited. The existing
`m200_matmul_float16.cpp` is a historical fallback and is not an implementation
of this backend.

`asr_paged_decode_attention_fp16.cpp` is currently an explicitly non-production
correctness probe. Its vector QK/PV blocks validate paged addressing and online
softmax only; `kernel_resources.json` keeps `runtime_eligible=false` until the
audited low-level MMAD and long-KV partition/merge implementation passes.

Before committing a production kernel, fill in its exact UB use and event-pair
count in `kernel_resources.json`, then run:

```bash
python3 tests/poc_310p/check_ascendc_asr_gates.py --require-resources
```

To verify that a local copy of the read-only design assets still matches the
reviewed snapshot, add `--shared-root /path/to/shared`.
