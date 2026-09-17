# ASR paged KV staging block

This isolated Ascend310P3 probe converts one logical 16-token KV tile from the
model's paged `[num_blocks,128,8,128]` BSHD cache into the two native B operands
consumed by the verified attention MMAD blocks:

```text
K_tile[t,d]  = K_cache[block_table[(start+t)/128], (start+t)%128, head, d]
Vt_tile[d,t] = V_cache[block_table[(start+t)/128], (start+t)%128, head, d]
```

The remaining rows/columns for `t >= valid_tokens` are explicitly zero. K is
copied by token; V uses eight basic `16×16` `vtranspose` operations. The block
does not use `DataCopyPad`, `LoadDataWithTranspose`, gather matrices or any
high-level Matmul API. Acceptance is bitwise exact, with output guards intact
and both cache arrays bitwise unchanged.

`vTransposeUb` is a cyclic V-to-MTE3 buffer. Every dimension-block iteration
waits for `MTE3_V` **before** `vtranspose` writes that buffer; waiting only
before the subsequent GM store would permit V to overwrite data still being
read by the preceding MTE3 operation.
