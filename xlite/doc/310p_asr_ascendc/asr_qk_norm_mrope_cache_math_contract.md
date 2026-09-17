# asr_qk_norm_mrope_cache_fp16 — Math Contract

Fixed Qwen3-ASR-1.7B decode operator for Ascend310P3. Input QKV is FP16
`[tokens,4096]`, split as Q16/K8/V8 with head dimension 128. Each Q/K head
is RMS-normalized in FP32 and rounded to FP16 before the three-axis
interleaved NeoX MRoPE operation. Frequency-pair indices `<60` select axes
T/H/W by `index % 3`; the final four pairs use T. Q is scaled in FP16 by
`1/sqrt(128)`. Post-RoPE K and unchanged V are written only to BSHD slots
named by `slot_mapping`.

The independent device probe covers tokens 1/2/4/6/8/12/16/20, distinct
three-axis positions, non-contiguous physical slots, untouched-cache
sentinels, finite output, cosine >=0.999 and maximum absolute error <=0.05.
