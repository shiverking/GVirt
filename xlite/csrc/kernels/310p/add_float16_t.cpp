/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

// Correctness-first M200 implementation.  Do not reuse the C220 UB/Event
// pipeline here: those legacy copy/vector intrinsics compile for dav-m200 but
// have been observed to leave GM untouched on 310P.
extern "C" __global__ __aicore__ void add_float16_t(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                     uint32_t rows, uint32_t cols)
{
    auto *xGm = reinterpret_cast<__gm__ float16_t *>(x);
    auto *yGm = reinterpret_cast<__gm__ float16_t *>(y);
    auto *zGm = reinterpret_cast<__gm__ float16_t *>(z);
    for (uint32_t row = block_idx; row < rows; row += static_cast<uint32_t>(block_num)) {
        uint64_t offset = static_cast<uint64_t>(row) * cols;
        for (uint32_t col = 0; col < cols; ++col) {
            float value = static_cast<float>(xGm[offset + col]) +
                          static_cast<float>(yGm[offset + col]);
            zGm[offset + col] = static_cast<float16_t>(value);
        }
    }
}
