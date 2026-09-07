/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#define UBA(T) (__ubuf__ T *)
#define GMA(T) (__gm__ T *)

#if defined(__DAV_C220_VEC__) || defined(XLITE_ARCH_310P)
// 2维矩阵加法
template <typename Dtype>
__aicore__ void add(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t x_numel, uint32_t y_numel)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __ubuf__ uint8_t *t1_dtype = (__ubuf__ uint8_t *)get_imm(0);
    __ubuf__ uint8_t *t2_dtype = (__ubuf__ uint8_t *)get_imm(y_numel * 2);
    __ubuf__ uint8_t *t1_float = (__ubuf__ uint8_t *)get_imm(y_numel * 4);
    __ubuf__ uint8_t *t2_float = (__ubuf__ uint8_t *)get_imm(y_numel * 8);

    uint32_t process_num = x_numel;
    uint16_t vec_repeat_float = DIV_ROUND_UP(y_numel * sizeof(float), 256);

    // Initial flags: allow the first iteration to proceed without waiting
    set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // t2_float is free (initial)
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // t1_dtype is free (initial)
    set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // t2_dtype is free (initial)

    for (uint32_t process = block_idx; process < process_num; process += uint32_t(block_num)) {
        // --- MTE2 pipeline: load x from GM to UB ---
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // wait for V to release t1_dtype
        copy_gm_to_ubuf(UBA(Dtype) t1_dtype, GMA(Dtype) x + process * y_numel, 0, 1,
                        DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);  // notify V: t1_dtype is loaded

        // --- MTE2 pipeline: load y from GM to UB ---
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // wait for V to release t2_dtype
        copy_gm_to_ubuf(UBA(Dtype) t2_dtype, GMA(Dtype) y + process * y_numel, 0, 1,
                        DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);  // notify V: t2_dtype is loaded

        // --- V pipeline: convert x to float32 ---
        // This overlaps with MTE2 loading y (different pipelines, flag-based sync)
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0);  // wait for x to be loaded
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(UBA(float) t1_float, UBA(Dtype) t1_dtype, vec_repeat_float, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(UBA(float) t1_float, UBA(Dtype) t1_dtype, vec_repeat_float, 1, 1, 8, 4);
        }
        pipe_barrier(PIPE_V);  // ensure vconv completes before signaling t1_dtype is free
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);  // notify MTE2: t1_dtype is free for next iter

        // --- V pipeline: convert y to float32 ---
        wait_flag(PIPE_MTE3, PIPE_V,
                  EVENT_ID0);  // wait for t2_float to be free from previous store
        wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID1);  // wait for y to be loaded
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f162f32(UBA(float) t2_float, UBA(Dtype) t2_dtype, vec_repeat_float, 1, 1, 8, 4);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_bf162f32(UBA(float) t2_float, UBA(Dtype) t2_dtype, vec_repeat_float, 1, 1, 8, 4);
        }
        pipe_barrier(PIPE_V);  // ensure vconv completes before signaling t2_dtype is free
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);  // notify MTE2: t2_dtype is free for next iter

        // --- V pipeline: add ---
        vadd(UBA(float) t1_float, UBA(float) t1_float, UBA(float) t2_float, vec_repeat_float, 1, 1,
             1, 8, 8, 8);
        pipe_barrier(PIPE_V);  // ensure vadd completes before vconv back

        // --- V pipeline: convert result back to dtype ---
        if constexpr (std::is_same_v<Dtype, float16_t>) {
            vconv_f322f16r(UBA(Dtype) t2_float, UBA(float) t1_float, vec_repeat_float, 1, 1, 4, 8);
        } else if constexpr (std::is_same_v<Dtype, bfloat16_t>) {
            vconv_f322bf16r(UBA(Dtype) t2_float, UBA(float) t1_float, vec_repeat_float, 1, 1, 4, 8);
        }
        pipe_barrier(PIPE_V);  // ensure vconv completes before MTE3 stores

        // --- MTE3 pipeline: store z from UB to GM ---
        set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);   // notify MTE3: t2_float is ready
        wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0);  // MTE3 waits for V
        copy_ubuf_to_gm(GMA(Dtype) z + process * y_numel, UBA(Dtype) t2_float, 0, 1,
                        DIV_ROUND_UP(y_numel * sizeof(Dtype), BLOCK_SIZE), 0, 0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);  // notify V: t2_float is free for next iter
    }

    // Cleanup: ensure all pipelines have completed before kernel exit
    wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
    pipe_barrier(PIPE_ALL);
}

#define ADD_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void add_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z,    \
                                                      uint32_t x_numel, uint32_t y_numel) \
    {                                                                                     \
        add<dtype>(x, y, z, x_numel, y_numel);                                            \
    }
#else
#define ADD_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void add_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z,    \
                                                      uint32_t x_numel, uint32_t y_numel) \
    {                                                                                     \
    }
#endif
