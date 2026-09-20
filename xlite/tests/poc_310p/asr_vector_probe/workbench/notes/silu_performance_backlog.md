# SiLU-Mul performance backlog

## 2026-09-20 — SILU-ROW: device A/B gate pending

The production kernel divides each 6144-element gate/up row into three 2048
tiles.  Each tile performs a separate pair of GM loads, event round-trip,
Vector sequence and GM store.  The whole-row candidate keeps the exact FP16
operation order but uses one 6144-element iteration.

Candidate resources are UB 36864 B, no L1/L0 and three dynamic event channels.
The 48 Vector repeats and 384 DMA blocks are within the proven M200 instruction
field ranges.  Floating scalar operands are function-local.  No unsupported
copy or high-level Matmul API is used.

The A/B gate covers M=1/2/4/6/8/12/16/20 in fresh processes with alternating
order and three measurements.  It validates every output, finite values,
cosine >=0.999, max absolute error <=0.04 and prefix/suffix guards.  Promotion
requires CV <=5%, at least one >=5% latency reduction and no regression beyond
max(2%, three sigma).  Passing remains a microprobe result, not runtime
promotion.

The first 48-run correctness set passed, but its 20-launch timing window was
only about 0.4--0.9 ms.  Observed CV ranged from 6.9% to 107%, so the apparent
-109.91% to +53.75% changes are invalid as performance evidence.  The gate now
first retried with 50 warmups, 2000 timed launches and a 20 ms minimum window.
No kernel decision was made from the noisy run.

The 2000-launch retry again passed 48/48 correctness, but M=2 row measured only
15.412 ms total and M=1 CV remained 12--15%.  The final calibration therefore
uses 1000 warmups, 30000 timed launches and a hard 200 ms minimum synchronized
window.  Shorter samples are invalid input, not a failed optimization.

The calibrated full experiment subsequently passed its correctness, minimum
window, variance, improvement and no-regression gates.  The row candidate is
microprobe-performance accepted but remains runtime-ineligible until the
whole-model token and performance gates pass.
