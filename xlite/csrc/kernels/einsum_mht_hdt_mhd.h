/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "matmul.h"

// mhd = torch.einsum("mht,hdt->mhd", mht, hdt)
template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ void einsum_mht_hdt_mhd(GM_ADDR mht, GM_ADDR hdt, GM_ADDR mhd, uint32_t m, uint32_t h,
                                   uint32_t t, uint32_t d, uint64_t m0, uint64_t n0, uint64_t k0,
                                   bool weightNZ, uint64_t swizzle, int T = -1, int D = -1)
{
    Matmul<Dtype, MatDtype, OutDtype> matmul_op;

    if (T == -1) {
        T = t;
    }
    if (D == -1) {
        D = d;
    }
    matmul_op.Init(m0, n0, k0, false, false, 0, weightNZ, swizzle);
    matmul_op.SetFlags();

    int xStride = T * sizeof(Dtype);
    int yStride = d * t * sizeof(Dtype);
    int zStride = D * sizeof(Dtype);
    int srcDStride = h * T;
    int dstDStride = h * D;
    uint32_t coreOffset = 0;
    int blockNum = GetBlockNum();
    int blockIdx = GetBlockIdx();
    for (int hIdx = 0; hIdx < h; hIdx++) {
        GM_ADDR x = mht + hIdx * xStride;
        GM_ADDR y = hdt + hIdx * yStride;
        GM_ADDR z = mhd + hIdx * zStride;
        int64_t tiles =
            matmul_op.TaskTilesInit(x, y, z, nullptr, nullptr, m, d, t, srcDStride, dstDStride);
        int64_t first = (blockIdx + blockNum - coreOffset) % blockNum;
        for (int64_t idx = first; idx < tiles; idx += blockNum) {
            matmul_op.RunTileByIdx(idx);
        }
        coreOffset = (coreOffset + tiles) % blockNum;
    }
    matmul_op.WaitFlags();
}

#if defined(XLITE_DEVICE_310P)
#define EINSUM_MHT_HDT_MHD_FUNC_DEFINE(dtype)                                                   \
    extern "C" __global__ __aicore__ void einsum_mht_hdt_mhd_##dtype(                          \
        GM_ADDR mht, GM_ADDR hdt, GM_ADDR mhd, uint32_t m, uint32_t h, uint32_t t, uint32_t d,  \
        uint64_t m0, uint64_t n0, uint64_t k0, bool weightNZ, uint64_t swizzle, int T, int D)   \
    {                                                                                           \
    }
#else
#define EINSUM_MHT_HDT_MHD_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void einsum_mht_hdt_mhd_##dtype(                            \
        GM_ADDR mht, GM_ADDR hdt, GM_ADDR mhd, uint32_t m, uint32_t h, uint32_t t, uint32_t d,   \
        uint64_t m0, uint64_t n0, uint64_t k0, bool weightNZ, uint64_t swizzle, int T, int D)    \
    {                                                                                            \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                          \
        einsum_mht_hdt_mhd<dtype, float, dtype>(mht, hdt, mhd, m, h, t, d, m0, n0, k0, weightNZ, \
                                                swizzle, T, D);                                  \
    }
#endif
