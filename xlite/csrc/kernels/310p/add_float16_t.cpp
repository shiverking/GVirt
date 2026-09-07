/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

using namespace AscendC;

namespace {

// Match the official Ascend310P Add kernel pattern: FP16 queues and a native
// vector Add.  The three queue buffers consume only 24 KiB.
constexpr uint32_t ADD_TILE_ELEMENTS = 4096;
constexpr uint32_t FP16_BLOCK_ELEMENTS = 32 / sizeof(half);

class KernelAdd310P
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t rows, uint32_t cols)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(y));
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(z));
        totalElements = static_cast<uint64_t>(rows) * cols;

        pipe.InitBuffer(xQueue, 1, ADD_TILE_ELEMENTS * sizeof(half));
        pipe.InitBuffer(yQueue, 1, ADD_TILE_ELEMENTS * sizeof(half));
        pipe.InitBuffer(zQueue, 1, ADD_TILE_ELEMENTS * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        // DataCopyPad is not functional on the target dav-m200 toolchain.
        // Non-aligned shapes are a correctness-only boundary case for this
        // POC, so handle them with scalar GM access. Qwen3's production
        // dimensions remain on the vectorized path below.
        if ((totalElements % FP16_BLOCK_ELEMENTS) != 0) {
            for (uint64_t index = GetBlockIdx(); index < totalElements;
                 index += GetBlockNum()) {
                float value = static_cast<float>(xGm.GetValue(index)) +
                              static_cast<float>(yGm.GetValue(index));
                zGm.SetValue(index, static_cast<half>(value));
            }
            return;
        }

        const uint64_t blockStride =
            static_cast<uint64_t>(ADD_TILE_ELEMENTS) * GetBlockNum();
        for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * ADD_TILE_ELEMENTS;
             offset < totalElements; offset += blockStride) {
            uint32_t count = static_cast<uint32_t>(
                totalElements - offset < ADD_TILE_ELEMENTS ? totalElements - offset
                                                           : ADD_TILE_ELEMENTS);
            LocalTensor<half> xFp16 = xQueue.AllocTensor<half>();
            LocalTensor<half> yFp16 = yQueue.AllocTensor<half>();
            CopyIn(xFp16, xGm[offset], count);
            CopyIn(yFp16, yGm[offset], count);
            xQueue.EnQue(xFp16);
            yQueue.EnQue(yFp16);

            xFp16 = xQueue.DeQue<half>();
            yFp16 = yQueue.DeQue<half>();
            LocalTensor<half> zFp16 = zQueue.AllocTensor<half>();
            Add(zFp16, xFp16, yFp16, count);
            zQueue.EnQue(zFp16);
            xQueue.FreeTensor(xFp16);
            yQueue.FreeTensor(yFp16);

            zFp16 = zQueue.DeQue<half>();
            CopyOut(zGm[offset], zFp16, count);
            zQueue.FreeTensor(zFp16);
        }
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<half> dst, GlobalTensor<half> src,
                                  uint32_t count)
    {
        DataCopy(dst, src, count);
    }

    __aicore__ inline void CopyOut(GlobalTensor<half> dst, LocalTensor<half> src,
                                   uint32_t count)
    {
        DataCopy(dst, src, count);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> zQueue;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
    uint64_t totalElements = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void add_float16_t(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                     uint32_t rows, uint32_t cols)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelAdd310P op;
    op.Init(x, y, z, rows, cols);
    op.Process();
}
