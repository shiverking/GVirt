/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "matmul.h"

// mhd = torch.einsum("mht,htd->mhd", mht, htd)
template <typename Dtype, typename MatDtype, typename OutDtype>
__aicore__ void einsum_mht_htd_mhd(GM_ADDR mht, GM_ADDR htd, GM_ADDR mhd, uint32_t m, uint32_t h,
                                   uint32_t t, uint32_t d, uint64_t m0, uint64_t n0, uint64_t k0,
                                   bool weightNZ, uint64_t swizzle, int T = -1, int D = -1)
{
    Matmul<Dtype, MatDtype, OutDtype> matmul_op;

    int coreOffset = 0;
    int nextCoreOffset = 0;
    if (T == -1) {
        T = t;
    }
    if (D == -1) {
        D = d;
    }
    int xStride = T * sizeof(Dtype);
    int yStride = d * t * sizeof(Dtype);
    int zStride = D * sizeof(Dtype);
    int srcDStride = h * T;
    int dstDStride = h * D;
    for (int hIdx = 0; hIdx < h; hIdx++) {
        GM_ADDR x = mht + hIdx * xStride;
        GM_ADDR y = htd + hIdx * yStride;
        GM_ADDR z = mhd + hIdx * zStride;
        matmul_op.Init(x, y, z, nullptr, nullptr, m, d, t, weightNZ, 1, m0, n0, k0, swizzle,
                       coreOffset, &nextCoreOffset, srcDStride, dstDStride);
        matmul_op.Run();
        coreOffset = nextCoreOffset;
    }
}

#if defined(XLITE_DEVICE_310P)
#define EINSUM_MHT_HTD_MHD_FUNC_DEFINE(dtype)                                                   \
    extern "C" __global__ __aicore__ void einsum_mht_htd_mhd_##dtype(                          \
        GM_ADDR mht, GM_ADDR htd, GM_ADDR mhd, uint32_t m, uint32_t h, uint32_t t, uint32_t d,  \
        uint64_t m0, uint64_t n0, uint64_t k0, bool weightNZ, uint64_t swizzle, int T, int D)   \
    {                                                                                           \
    }
#else
#define EINSUM_MHT_HTD_MHD_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void einsum_mht_htd_mhd_##dtype(                            \
        GM_ADDR mht, GM_ADDR htd, GM_ADDR mhd, uint32_t m, uint32_t h, uint32_t t, uint32_t d,   \
        uint64_t m0, uint64_t n0, uint64_t k0, bool weightNZ, uint64_t swizzle, int T, int D)    \
    {                                                                                            \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                          \
        einsum_mht_htd_mhd<dtype, float, dtype>(mht, htd, mhd, m, h, t, d, m0, n0, k0, weightNZ, \
                                                swizzle, T, D);                                  \
    }
#endif
