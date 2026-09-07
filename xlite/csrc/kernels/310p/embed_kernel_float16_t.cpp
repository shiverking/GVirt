/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

// Scalar GM path used to establish a real 310P correctness baseline before
// introducing an M200-native LocalTensor/DataCopy implementation.
extern "C" __global__ __aicore__ void embed_kernel_float16_t(
    GM_ADDR weight, GM_ADDR ids, GM_ADDR output, uint32_t dim, uint32_t tokenCount,
    uint32_t embStart, uint32_t embEnd, uint32_t tpSize)
{
    auto *weightGm = reinterpret_cast<__gm__ float16_t *>(weight);
    auto *idsGm = reinterpret_cast<__gm__ uint32_t *>(ids);
    auto *outputGm = reinterpret_cast<__gm__ float16_t *>(output);
    for (uint32_t token = block_idx; token < tokenCount;
         token += static_cast<uint32_t>(block_num)) {
        uint32_t row = idsGm[token];
        uint64_t outputOffset = static_cast<uint64_t>(token) * dim;
        if (row < embStart || row >= embEnd) {
            for (uint32_t col = 0; col < dim; ++col) {
                outputGm[outputOffset + col] = static_cast<float16_t>(0.0f);
            }
            continue;
        }
        uint64_t weightOffset = static_cast<uint64_t>(row - embStart) * dim;
        for (uint32_t col = 0; col < dim; ++col) {
            outputGm[outputOffset + col] = weightGm[weightOffset + col];
        }
    }
    (void)tpSize;
}
