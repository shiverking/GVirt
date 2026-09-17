# ASR attention low-level MMAD blocks

This is an isolated Ascend310P3 block contract. It does not alter the paged
attention kernel or qualify for Runtime routing.

## QK block

For a 16-token KV tile, Q is replicated to the cube minimum M dimension:

```text
C[m,n] = sum(k=0..127) Q[m,k] * K[n,k]
M=16, N=16, K=128
```

## PV block

The 16 attention weights are replicated to M=16 and the BSHD V tile is staged
as `[output_dimension, token]`:

```text
C[m,n] = sum(k=0..15) P[m,k] * V_transposed[n,k]
M=16, N=128, K=16
```

Inputs and outputs are FP16; L0C accumulation is FP32 and is cast to FP16 only
after MATRIX-mode L0C-to-UB transfer. Acceptance requires cosine >= 0.999,
maximum absolute error <= 0.02, finite output and unchanged 32-byte guards.

The implementation is a unified-core flow using explicit ND-to-NZ transfer,
`LoadData`, `Mmad`, MATRIX-mode enhanced `DataCopy` and NZ-to-ND scatter. Every
cross-pipe dependency uses `FetchEventID`; fixed events, high-level Matmul,
FixPipe, `DataCopyPad` and `LoadDataWithTranspose` are forbidden.
