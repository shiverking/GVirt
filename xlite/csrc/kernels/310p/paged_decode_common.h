#pragma once
#include "entry_guard.h"
#include "../../paged_decode_310p.h"

// All storage is explicitly placed in UB. No C220 reductions/instructions.
namespace Paged310P {
constexpr unsigned Tile = 32;
constexpr unsigned UbBytes = 32768;
static_assert(UbBytes <= 64 * 1024, "310P per-core UB budget");
__aicore__ inline void mask(unsigned n = 64)
{
    set_vector_mask(0, n == 64 ? uint64_t(-1) : (uint64_t(1) << n) - 1);
}
__aicore__ inline void vs()
{
    set_flag(PIPE_V, PIPE_S, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_S, EVENT_ID0);
}
__aicore__ inline void sv()
{
    set_flag(PIPE_S, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_S, PIPE_V, EVENT_ID0);
}
__aicore__ inline void load(__ubuf__ float16_t *dst, __gm__ float16_t *src, unsigned rows)
{
    // One head from each BSHD token: seven intervening heads = 56 DMA blocks.
    copy_gm_to_ubuf(dst, src, 0, rows, 8, 56, 0);
    set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);
}
__aicore__ inline void output(__gm__ float16_t *dst, __ubuf__ float16_t *half,
                             __ubuf__ float *acc, float sum)
{
    mask();
    vmuls(acc, acc, 1.0f / sum, 2, 1, 1, 8, 8);
    pipe_barrier(PIPE_V);
    vconv_f322f16(half, acc, 2, 1, 1, 4, 8);
    set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);
    copy_ubuf_to_gm(dst, half, 0, 1, 8, 0, 0);
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
}
}
