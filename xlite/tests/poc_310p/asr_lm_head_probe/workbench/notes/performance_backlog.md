# LM Head performance backlog

## 2026-09-20 — LMHEAD-L1-A: pending device measurements

User reports baseline M=1/8/20 passed at commit 1feb60e. Original source is
preserved by that Git snapshot and by the baseline template specialization.
No measured latency was supplied; this candidate is NOT performance accepted.

Source: asr_m200_lm_head_fp16.cpp ProcessTile previously copies the same A
K-slice for each of 1187 vocabulary tiles. Candidate Process preloads 16
independent NZ slices per core. ProcessTile selects that read-only slot.
Weight staging, accumulation order and full-logit scatter remain identical.
Eight cores reduce A GM-to-L1 calls from 18992 to 128 per launch; weight
traffic remains 622396416 bytes and can still dominate wall time.

L1 map: cached A [0,131072), B [131072,163840); baseline A [0,8192),
B [8192,40960). Hardware reference: shared/hardware/intrinsic-support-matrix/
Ascend310P3.json AICoreSpec.l1_size=1048576. UB and L0 allocations unchanged.
No shared template is copied. Existing proven ND2NZ/LoadData blocks are reused.

Event audit: preload MTE2_MTE1 Set/Wait finishes before any cached read.
Cached A is never overwritten. B retains the existing MTE1_MTE2 prime,
per-K wait/set and terminal drain. Other seven event-channel pairs and the
Mmad accumulation order are unchanged; all event IDs come from TPipe.
Inactive padded A rows cannot affect valid Mmad output rows.

Validation command: test_asr_lm_head_probe.sh CANN all compare 3.
Each variant runs in a fresh process, ordering alternates. K/row/column
dependent dyadic inputs exercise slice selection and scatter; checks include
all logits, finite values, cosine>=0.999, rtol/atol=0.01, prefix/suffix guards.
This structured rank-one weight test is not a substitute for random/model
weight tests. Before runtime promotion, extend those checks and run full
(M=1..20), then whole-model token comparison.

Summary records raw runs, median latency and CV. Require >=5% reduction on
at least one shape, <=2% regression on all others, >=8 shapes and three runs
with CV<=5%. All three quick shapes passing is only a development gate.
No production default or build capability is changed.
