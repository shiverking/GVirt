# Projection performance backlog

## 2026-09-20 — PROJECTION-L1-A: device A/B gate pending

The production `asr_m200_projection_fp16` remains the baseline.  The
experimental `asr_m200_projection_cached_fp16` changes one scheduling choice:
all activation K tiles are staged in L1 once per AIC and reused across that
core's N tiles.  B staging, Mmad accumulation order, L0C readback and final NZ
scatter remain structurally equivalent to the proven baseline.

The maximum Down allocation is L1 425984 B (A 393216 B plus B 32768 B), UB
24576 B, L0A 8192 B, L0B 32768 B and L0C 16384 B.  This is below the pinned
Ascend310P3 L1 limit of 1048576 B and the project UB design budget of 196608 B.
The candidate uses eight event channels obtained from `FetchEventID`; the A
preload completes its MTE2-to-MTE1 edge before any cached L1 read.

The probe uses K-, row- and N-dependent dyadic data, checks every output,
prefix/suffix guards, finite values, cosine >= 0.999 and rtol/atol 0.01.  A/B
runs use fresh processes and alternate order.  The full gate covers QKV, O,
Gate/Up and Down for M=1/2/4/6/8/12/16/20 with three repeats.  It requires CV
<=5%, at least one >=5% latency reduction, and no shape regression beyond the
larger of 2% baseline latency or three measured standard deviations.

The structured microprobe is not runtime promotion.  Model-weight projection
checks, 16-token greedy equivalence and real-ASR performance remain required.
