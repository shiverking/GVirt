/*
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "matmul.h"

// kM: tokens of one expert
// kN: moe intermediate size
// kK: hidden size
// [start_idx, end_idx]: Range of expert index involved this time
// deqScale: uint64_t, 低 32 位 TF32 格式有效, 1 符号位, 8 指数位, 10 尾数位, 后 13 位不参与计算
template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ void group_matmul_kernel(GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales,
                                    GM_ADDR counts, uint32_t n, int64_t kN, int64_t kK, uint64_t m0,
                                    uint64_t n0, uint64_t k0, uint32_t startIdx, uint32_t endIdx,
                                    bool weightNZ, bool transpose, uint64_t swizzle)
{
    Matmul<Dtype, MatDtype, OutDtype> matmul_op;

    bool useDequant = (deqScales != nullptr);

    int dtypeBits = std::is_same<Dtype, int4b_t>::value ? 4 : (sizeof(Dtype) * BYTE_BITS);

    matmul_op.Init(m0, n0, k0, false, useDequant, transpose, weightNZ, swizzle);
    matmul_op.SetFlags();

    uint32_t off = 0;
    uint32_t coreOffset = 0;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    for (uint32_t i = startIdx; i < endIdx && i < n; i++) {
        uint32_t kM = *((__gm__ uint32_t *)(counts + i * sizeof(uint32_t)));
        if (kM <= 0) {
            continue;
        }
        // For w4a8, INT8 activation will be packed into [2m, k/2] INT4 matrix, so counts
        // should be 2 * kM in int4 case
        if (std::is_same<Dtype, int4b_t>::value) {
            kM = kM * 2;
        }

        uint64_t weightAddr = *((__gm__ uint64_t *)(ws + i * sizeof(void *)));
        uint64_t deqScaleAddr = (uint64_t)nullptr;
        if (useDequant) {
            deqScaleAddr = *((__gm__ uint64_t *)(deqScales + i * sizeof(void *)));
        }
        __gm__ uint8_t *w = (__gm__ uint8_t *)weightAddr;
        __gm__ uint8_t *deqScale = (__gm__ uint8_t *)deqScaleAddr;

        uint64_t xOffBytes = off * kK * dtypeBits / BYTE_BITS;
        uint64_t zOffBytes = off * kN * sizeof(OutDtype);

        int64_t tiles = matmul_op.TaskTilesInit(x + xOffBytes, w, z + zOffBytes, (GM_ADDR) nullptr,
                                                (GM_ADDR)deqScale, kM, kN, kK);
        int64_t first = (blockIdx + blockNum - coreOffset) % blockNum;
        for (int64_t idx = first; idx < tiles; idx += blockNum) {
            matmul_op.RunTileByIdx(idx);
        }
        coreOffset = (coreOffset + tiles) % blockNum;
        off += kM;
    }
    matmul_op.WaitFlags();
}

#if defined(XLITE_DEVICE_310P)
#define GROUPMATMUL_FUNC_DEFINE(dtype)                                                            \
    extern "C" __global__ __aicore__ void group_matmul_##dtype(                                  \
        GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales, GM_ADDR counts, uint32_t n,          \
        int64_t kN, int64_t kK, uint64_t m0, uint64_t n0, uint64_t k0, uint32_t startIdx,         \
        uint32_t endIdx, bool weightNZ, bool transpose, uint64_t swizzle)                         \
    {                                                                                             \
    }
#else
#define GROUPMATMUL_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void group_matmul_##dtype(                                    \
        GM_ADDR x, GM_ADDR ws, GM_ADDR z, GM_ADDR deqScales, GM_ADDR counts, uint32_t n,           \
        int64_t kN, int64_t kK, uint64_t m0, uint64_t n0, uint64_t k0, uint32_t startIdx,          \
        uint32_t endIdx, bool weightNZ, bool transpose, uint64_t swizzle)                          \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                            \
        if constexpr (std::is_same<dtype, int8_t>::value || std::is_same<dtype, int4b_t>::value) { \
            group_matmul_kernel<dtype, int32_t, half>(x, ws, z, deqScales, counts, n, kN, kK, m0,  \
                                                      n0, k0, startIdx, endIdx, weightNZ,          \
                                                      transpose, swizzle);                         \
        } else {                                                                                   \
            group_matmul_kernel<dtype, float, dtype>(x, ws, z, deqScales, counts, n, kN, kK, m0,   \
                                                     n0, k0, startIdx, endIdx, weightNZ,           \
                                                     transpose, swizzle);                          \
        }                                                                                          \
    }
#endif
