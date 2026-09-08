# P2: opt-in 310P paged Decode

Status: implementation awaiting CANN compilation and NPU accuracy/performance acceptance.
Default remains `legacy`. No P2 speedup or transcription equivalence is claimed.

## Scope and memory

- FP16, Q16/KV8, D128, block128, batch1–20, valid KV1–2048.
- `query_len == 1` goes through one batched kernel; >512 maximum Decode KV
  uses four partitions plus one merge kernel. Mixed multi-token requests retain
  the existing prefill path. Q is already scaled by RoPE.
- Tile32, explicit UB layout: 26,912 bytes (32 KiB budget), merge <4 KiB.
- Partial record: 136 FP32 words, including max/sum, padding and 128 accumulators.
  Required scratch is **696,320 bytes**, not the proposed 256 KiB.
  A permanent 1 MiB TensorPool reservation includes scratch and 1,600-byte metadata;
  model pool planning includes this reservation. No Decode-time allocation.
- Metadata staging waits only for its preceding upload event before modification.
  This is a host event wait, not a Decoder stream synchronize; queued uploads may
  still wait behind preceding stream work. It is not claimed to be wait-free.
- MatMul, Audio Graph, RoPE and cache format are unchanged.

## Only next command after installing these source commits

From `GVirt/xlite`, with the existing working CANN/torch_npu environment:

```bash
bash tests/poc_310p/validate_p2_310p.sh
```

Failures are aggregated under `attention_paged_310p_report/`. After a source fix,
rebuild the extension, then rerun only failed shapes:

```bash
bash tests/poc_310p/validate_p2_310p.sh --rerun-failed
```

The test preserves all asynchronous outputs, checks guards/cache integrity and
reports cosine/max absolute error. CPU `test_paged_decode_math.py` checks equations
and addressing only, and does not establish kernel correctness.

## After the new attention gate passes

```bash
python3 tests/poc_310p/run_qwen3_asr_llm.py \
  --checkpoint /home/models/Qwen3-ASR-1.7B \
  --decode-attention-backend paged_310p \
  --input-mode synthetic --prompt-tokens 129 --decode-tokens 16 \
  --max-seq-len 512 --stability-iters 1 \
  --report poc_310p_results/p2_full28_synthetic129.json
```

Require hidden/logits cosine >=0.999 and the complete reference token sequence.
Then update vllm-ascend (Python only, no native rebuild), use its P2 benchmark
documentation. Do not change MatMul flags between baseline and P2.

## Promotion gate

Historical Xlite: c1 0.28 req/s, TPOT83.66ms; c20 2.83 req/s, TPOT145.31ms;
both 100 successful requests, 3,985 generated tokens. These are rounded historical
figures, not a replacement for per-request text comparison or sync-policy evidence.
Target c20 >=3.40 req/s and c1 >=0.266 req/s, with identical transcripts and no
Decode fallback/gather. Keep legacy default until device results meet these gates.
If throughput misses, profile the same P2 configuration briefly; do not infer the
bottleneck solely from aggregate throughput, and do not promote the default.
