# QK Norm/MRoPE/Cache performance backlog

## 2026-09-20 — QK-MROPE-GROUPED: rejected by device A/B gate

The production kernel schedules `tokens * 24` independent head tasks.  Each
task loads the same Q or K norm weight and all three token-position frequency
rows.  The grouped candidate assigns AIV `h` Q heads `2h` and `2h+1`, K/V head
`h`, preserving independent output addresses while enabling reuse.

Per launch and AIV, Q/K weights are each loaded and converted once.  Per token
and AIV, the three frequency rows are loaded and composed once for all three
heads.  Thus frequency-row GM transfers fall from 72 to 24 per token and norm
weight transfers become independent of token count.  RMSNorm FP32 arithmetic,
MRoPE masks, Q scaling and BSHD K/V writes are unchanged.

Candidate resources are UB 4864 B, no L1/L0 and seven dynamic event channels.
The UB remains below the 196608 B design budget.  MTE2/V/MTE3 and V/S edges use
`FetchEventID`; the V staging buffer follows a direct MTE2-to-MTE3 dependency.

The A/B command covers tokens 1/2/4/6/8/12/16/20 in fresh processes, alternates
variant order and records three measurements.  It checks the full QKV tensor,
K/V cache values, untouched cache sentinels, finite values, cosine and maximum
absolute error.  Performance acceptance requires CV <=5%, at least one >=5%
latency reduction, and no shape regression beyond max(2%, three sigma).

Passing this synthetic microprobe does not change runtime routing.  Model
weights, 16-step greedy equivalence and real-ASR performance remain mandatory.

Measured result: all 48 correctness runs passed.  M=1 improved only 1.93%,
while M=2/4/6/8/12/16/20 regressed 1.18/3.63/2.52/4.53/4.22/4.48/4.38%.
M=8 grouped CV was 8.26%.  The required performance gate therefore failed.
The grouped kernel is retained only as an explicit experiment and is not
eligible for runtime promotion.  The production baseline remains selected.
