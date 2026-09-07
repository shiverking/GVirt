/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#if defined(__DAV_C220_VEC__) || defined(XLITE_ARCH_310P)
template <typename Dtype>
__aicore__ void embed_kernel(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t dim, uint32_t batch_size,
                             uint32_t emb_start_idx, uint32_t emb_end_idx, uint32_t tp_size)
{
    set_atomic_none();
    set_mask_norm();
    set_vector_mask((uint64_t)-1, (uint64_t)-1);

    __gm__ Dtype *x_gm_addr = ((__gm__ Dtype *)x);
    __gm__ uint32_t *y_gm_addr = ((__gm__ uint32_t *)y);
    __gm__ Dtype *z_gm_addr = ((__gm__ Dtype *)z);
    auto *ub_addr = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)0);
    auto *zero_ub_addr = reinterpret_cast<__ubuf__ Dtype *>((uintptr_t)(dim * 2));
    int start = block_idx;
    uint16_t n_burst = 1;
    uint16_t len_burst = dim / 16;  // Expected embedding dimension size could be divided by 16

    if (tp_size > 1) {
        vector_dup(zero_ub_addr, (Dtype)0, dim / VECTOR_MAX_NUM_OF_FP16, 1, 1, 8, 0);
    }
    pipe_barrier(PIPE_ALL);
    for (int index = start; index < batch_size; index += block_num) {
        uint32_t row = *(y_gm_addr + index);
        if ((row >= emb_end_idx) || (row < emb_start_idx)) {
            copy_ubuf_to_gm(z_gm_addr + index * dim, zero_ub_addr, 0, n_burst, len_burst, 0, 0);
        } else {
            auto *row_addr = x_gm_addr + (row - emb_start_idx) * dim;
            copy_gm_to_ubuf(ub_addr, row_addr, 0, n_burst, len_burst, 0, 0);
            pipe_barrier(PIPE_ALL);
            copy_ubuf_to_gm(z_gm_addr + index * dim, ub_addr, 0, n_burst, len_burst, 0, 0);
        }
        pipe_barrier(PIPE_ALL);
    }
}

#define EMBED_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void embed_kernel_##dtype(                             \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t dim, uint32_t batch_size,                 \
        uint32_t emb_start_idx, uint32_t emb_end_idx, uint32_t tp_size)                     \
    {                                                                                       \
        embed_kernel<dtype>(x, y, z, dim, batch_size, emb_start_idx, emb_end_idx, tp_size); \
    }
#else
#define EMBED_FUNC_DEFINE(dtype)                                            \
    extern "C" __global__ __aicore__ void embed_kernel_##dtype(             \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t dim, uint32_t batch_size, \
        uint32_t emb_start_idx, uint32_t emb_end_idx, uint32_t tp_size)     \
    {                                                                       \
    }
#endif
