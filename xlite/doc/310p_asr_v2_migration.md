# 310P ASR Xlite V2 migration boundary

This branch starts from AtomGit GVirt `master` at `a916ced` and carries only
the minimum proven 310P Qwen3-ASR path.

Included:

- canonical `Ascend310P3`/`llm_fp16` build and ABI capability reporting;
- FP16 Add, Embedding, RMSNorm, Q/K RMSNorm, SiLU-and-Mul and MRoPE/cache
  kernels that have real M200 entries and no-op detection;
- ACLNN FP16 ND MatMul plus PromptFlashAttention correctness backends;
- 4D BSHD KV cache, batch 1-20, arbitrary block tables and chunked prefill;
- Qwen3-ASR checkpoint parsing, decoder-only weight loading,
  `inputs_embeds`, explicit three-axis positions and 16-token decode tests.

Excluded on purpose:

- P2 native vector-only paged decode attention;
- P3 offline ACLNN MatMul policies and direct-output experiments;
- BF16, quantization, MoE, MLA, communication, TP/DP and other model adapters;
- decoder graph capture and production performance claims.

The current ACLNN MatMul/attention implementation is a correctness bootstrap,
not the V2 performance destination. After the migration gates pass on silicon,
the next implementation target is a fixed-shape M200 Cube LM Head based on the
new upstream `TaskTilesInit`/`RunTileByIdx` MatMul structure, followed by the
five dense decoder projections.
