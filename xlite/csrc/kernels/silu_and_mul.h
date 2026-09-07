/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#if defined(__DAV_C220_VEC__) || defined(XLITE_ARCH_310P)

#define GMA(T) (__gm__ T *)
#define UBA(T) (__ubuf__ T *)
#define MAX_HIDDENSIZE_PER_PIECE \
    38912  // MAX_HIDDENSIZE_PER_PIECE = UB_SIZE(196608) / buffer_num(5)

// 本算子由小艺团队贡献，参考论文《XY-Serve: End-to-End Versatile Production Serving for Dynamic LLM
// Workloads》 [ASPLOS 2026]
template <typename SrcType, typename CalType>
__aicore__ void silu_and_mul(GM_ADDR x, GM_ADDR y, GM_ADDR pnum_tokens, uint32_t num_tokens,
                             uint32_t dim, float swiglu_limit = 0.0f)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    if (pnum_tokens) {
        uint32_t pnum_tokens_val = *((__gm__ uint32_t *)pnum_tokens);
        num_tokens = pnum_tokens_val < num_tokens ? pnum_tokens_val : num_tokens;
    }

    // split the matrix in a row-major manner such that each row (2 * dim elements) is contained in
    // a single block
    int32_t block_dims = get_block_num();
    int32_t tokens_per_block = DIV_ROUND_UP(num_tokens, block_dims);

    // calculate actual batch size == how many tokens (rows) are handled in current block
    int32_t tokens_copied = tokens_per_block * get_block_idx();
    int32_t tokens_to_copy = tokens_per_block;
    if (tokens_copied + tokens_to_copy > num_tokens) {
        tokens_to_copy = num_tokens - tokens_copied;
    }

    if (tokens_to_copy <= 0) {
        return;
    }

    int32_t x_offset = tokens_copied * dim * 2;
    int32_t y_offset = tokens_copied * dim;
    __gm__ SrcType *x_start = (__gm__ SrcType *)x + x_offset;
    __gm__ SrcType *y_start = (__gm__ SrcType *)y + y_offset;

    // Double buffer, each needs 2 sub-buffer and needs 5 buffer in total
    uint64_t ubuf_unit_size = MAX_HIDDENSIZE_PER_PIECE;
    __ubuf__ CalType *ubuf0 = (__ubuf__ CalType *)(uintptr_t)0x00000;
    __ubuf__ CalType *ubuf1 = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size);
    __ubuf__ CalType *ubuf2 = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size * 2);
    __ubuf__ CalType *ubuf3 = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size * 3);
    __ubuf__ CalType *cal_ubuf = (__ubuf__ CalType *)(uintptr_t)(ubuf_unit_size * 4);

    int32_t token_len = dim * sizeof(CalType);
    int32_t tokens_per_tile = ubuf_unit_size / token_len;

    // Config for vector operation
    constexpr uint64_t dst_stride = 1;
    constexpr uint64_t src0_stride = 1;
    constexpr uint64_t src1_stride = 1;
    constexpr uint64_t dst_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    constexpr uint64_t src0_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    constexpr uint64_t src1_repeat_stride = VECTOR_MAX_BYTESIZE / BLOCK_SIZE;
    uint64_t repeat_dtype =
        DIV_ROUND_UP(tokens_per_tile * dim, VECTOR_MAX_BYTESIZE / sizeof(CalType));
    uint64_t vector_config3ops =
        set_vector_xt(dst_repeat_stride, src0_repeat_stride, src1_repeat_stride, dst_stride,
                      src0_stride, src1_stride, repeat_dtype);
    uint64_t vector_config2ops = set_vector_1src_xt(dst_repeat_stride, src0_repeat_stride,
                                                    dst_stride, src0_stride, repeat_dtype);

    // Config for copy operation
    constexpr uint8_t sid = 0;
    constexpr uint16_t n_burst = 1;
    constexpr uint16_t src_gap = 0;
    constexpr uint16_t dst_gap = 0;
    uint64_t len_burst = DIV_ROUND_UP(tokens_per_tile * token_len, BLOCK_SIZE);
    uint64_t out_copy_config = __set_dmi_config(sid, n_burst, len_burst, src_gap, dst_gap);
    uint64_t token_len_burst = DIV_ROUND_UP(token_len, BLOCK_SIZE);
    uint64_t token_copy_config =
        __set_dmi_config(sid, tokens_per_tile, token_len_burst, token_len_burst, dst_gap);

    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    set_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    for (int32_t copied = 0, ping = 1; copied < tokens_to_copy; copied += tokens_per_tile) {
        __ubuf__ CalType *x_ubuf = ping ? ubuf0 : ubuf2;
        __ubuf__ CalType *y_ubuf = ping ? ubuf1 : ubuf3;
        auto event_id = ping ? EVENT_ID0 : EVENT_ID1;

        int32_t to_copy = tokens_per_tile;
        if (copied + to_copy > tokens_to_copy) {
            to_copy = tokens_to_copy - copied;
            len_burst = DIV_ROUND_UP(to_copy * token_len, BLOCK_SIZE);
            out_copy_config = __set_dmi_config(sid, n_burst, len_burst, src_gap, dst_gap);
            token_copy_config =
                __set_dmi_config(sid, to_copy, token_len_burst, token_len_burst, dst_gap);
        }

        wait_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        copy_gm_to_ubuf(x_ubuf, x_start, token_copy_config);
        copy_gm_to_ubuf(y_ubuf, x_start + dim, token_copy_config);

        x_start += 2 * dim * to_copy;
        set_flag(PIPE_MTE2, PIPE_V, event_id);
        wait_flag(PIPE_MTE2, PIPE_V, event_id);

        // SwiGLU clamp: up = clamp(up, -L, L), gate = clamp(gate, max=L).
        // x_ubuf = gate, y_ubuf = up.
        if (swiglu_limit > 0.0f) {
            vmins(x_ubuf, x_ubuf, (CalType)swiglu_limit, vector_config2ops);
            vmins(y_ubuf, y_ubuf, (CalType)swiglu_limit, vector_config2ops);
            pipe_barrier(PIPE_V);
            vmaxs(y_ubuf, y_ubuf, (CalType)(-swiglu_limit), vector_config2ops);
            pipe_barrier(PIPE_V);
        }

        // -x
        vmuls(cal_ubuf, x_ubuf, (CalType)-1.0, vector_config2ops);
        pipe_barrier(PIPE_V);

        // e^-x
        vexp(cal_ubuf, cal_ubuf, vector_config2ops);
        pipe_barrier(PIPE_V);

        // 1 + e^-x
        vadds(cal_ubuf, cal_ubuf, (CalType)1.0, vector_config2ops);
        pipe_barrier(PIPE_V);

        // silu = x / (1 + e^-x)
        vdiv(cal_ubuf, x_ubuf, cal_ubuf, vector_config3ops);
        pipe_barrier(PIPE_V);

        // x * silu(x) = y * x / (1 + e^-x)
        vmul(x_ubuf, y_ubuf, cal_ubuf, vector_config3ops);
        pipe_barrier(PIPE_V);

        set_flag(PIPE_V, PIPE_MTE3, event_id);
        wait_flag(PIPE_V, PIPE_MTE3, event_id);

        copy_ubuf_to_gm(y_start, x_ubuf, out_copy_config);
        set_flag(PIPE_MTE3, PIPE_MTE2, event_id);

        ping = 1 - ping;
        y_start += to_copy * dim;
    }
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_MTE3, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define SILU_AND_MUL_FUNC_DEFINE(dtype, cast_type)                                      \
    extern "C" __global__ __aicore__ void silu_and_mul_##dtype(                         \
        GM_ADDR x, GM_ADDR y, GM_ADDR pnum_tokens, uint32_t n_tokens, uint32_t dim,     \
        float swiglu_limit)                                                             \
    {                                                                                   \
        silu_and_mul<dtype, cast_type>(x, y, pnum_tokens, n_tokens, dim, swiglu_limit); \
    }

#else
#define SILU_AND_MUL_FUNC_DEFINE(dtype, cast_type)                                  \
    extern "C" __global__ __aicore__ void silu_and_mul_##dtype(                     \
        GM_ADDR x, GM_ADDR y, GM_ADDR pnum_tokens, uint32_t n_tokens, uint32_t dim, \
        float swiglu_limit)                                                         \
    {                                                                               \
    }
#endif
