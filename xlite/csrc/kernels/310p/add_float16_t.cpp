/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

using namespace AscendC;

namespace {

// Three FP16 and three FP32 tiles consume 72 KiB, safely below the 310P UB budget.
constexpr uint32_t ADD_TILE_ELEMENTS = 4096;
constexpr uint32_t FP16_BLOCK_ELEMENTS = 32 / sizeof(float16_t);

class KernelAdd310P
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t rows, uint32_t cols)
    {
        xGm.SetGlobalBuffer(reinterpret_cast<__gm__ float16_t *>(x));
        yGm.SetGlobalBuffer(reinterpret_cast<__gm__ float16_t *>(y));
        zGm.SetGlobalBuffer(reinterpret_cast<__gm__ float16_t *>(z));
        totalElements = static_cast<uint64_t>(rows) * cols;

        pipe.InitBuffer(xQueue, 1, ADD_TILE_ELEMENTS * sizeof(float16_t));
        pipe.InitBuffer(yQueue, 1, ADD_TILE_ELEMENTS * sizeof(float16_t));
        pipe.InitBuffer(zQueue, 1, ADD_TILE_ELEMENTS * sizeof(float16_t));
        pipe.InitBuffer(xFp32Buf, ADD_TILE_ELEMENTS * sizeof(float));
        pipe.InitBuffer(yFp32Buf, ADD_TILE_ELEMENTS * sizeof(float));
        pipe.InitBuffer(zFp32Buf, ADD_TILE_ELEMENTS * sizeof(float));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float> xFp32 = xFp32Buf.Get<float>();
        LocalTensor<float> yFp32 = yFp32Buf.Get<float>();
        LocalTensor<float> zFp32 = zFp32Buf.Get<float>();

        const uint64_t blockStride =
            static_cast<uint64_t>(ADD_TILE_ELEMENTS) * GetBlockNum();
        for (uint64_t offset = static_cast<uint64_t>(GetBlockIdx()) * ADD_TILE_ELEMENTS;
             offset < totalElements; offset += blockStride) {
            uint32_t count = static_cast<uint32_t>(
                totalElements - offset < ADD_TILE_ELEMENTS ? totalElements - offset
                                                           : ADD_TILE_ELEMENTS);
            LocalTensor<float16_t> xFp16 = xQueue.AllocTensor<float16_t>();
            LocalTensor<float16_t> yFp16 = yQueue.AllocTensor<float16_t>();
            CopyIn(xFp16, xGm[offset], count);
            CopyIn(yFp16, yGm[offset], count);
            xQueue.EnQue(xFp16);
            yQueue.EnQue(yFp16);

            xFp16 = xQueue.DeQue<float16_t>();
            yFp16 = yQueue.DeQue<float16_t>();
            LocalTensor<float16_t> zFp16 = zQueue.AllocTensor<float16_t>();

            Cast(xFp32, xFp16, RoundMode::CAST_NONE, count);
            Cast(yFp32, yFp16, RoundMode::CAST_NONE, count);
            PipeBarrier<PIPE_V>();
            Add(zFp32, xFp32, yFp32, count);
            PipeBarrier<PIPE_V>();
            Cast(zFp16, zFp32, RoundMode::CAST_RINT, count);
            zQueue.EnQue(zFp16);
            xQueue.FreeTensor(xFp16);
            yQueue.FreeTensor(yFp16);

            zFp16 = zQueue.DeQue<float16_t>();
            CopyOut(zGm[offset], zFp16, count);
            zQueue.FreeTensor(zFp16);
        }
    }

private:
    __aicore__ inline void CopyIn(LocalTensor<float16_t> dst,
                                  GlobalTensor<float16_t> src, uint32_t count)
    {
        if ((count % FP16_BLOCK_ELEMENTS) == 0) {
            DataCopy(dst, src, count);
            return;
        }
        DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = count * sizeof(float16_t);
        DataCopyPadParams padParams;
        padParams.isPad = false;
        DataCopyPad(dst, src, params, padParams);
    }

    __aicore__ inline void CopyOut(GlobalTensor<float16_t> dst,
                                   LocalTensor<float16_t> src, uint32_t count)
    {
        if ((count % FP16_BLOCK_ELEMENTS) == 0) {
            DataCopy(dst, src, count);
            return;
        }
        DataCopyParams params;
        params.blockCount = 1;
        params.blockLen = count * sizeof(float16_t);
        DataCopyPad(dst, src, params);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, 1> xQueue;
    TQue<QuePosition::VECIN, 1> yQueue;
    TQue<QuePosition::VECOUT, 1> zQueue;
    TBuf<TPosition::VECCALC> xFp32Buf;
    TBuf<TPosition::VECCALC> yFp32Buf;
    TBuf<TPosition::VECCALC> zFp32Buf;
    GlobalTensor<float16_t> xGm;
    GlobalTensor<float16_t> yGm;
    GlobalTensor<float16_t> zGm;
    uint64_t totalElements = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void add_float16_t(GM_ADDR x, GM_ADDR y, GM_ADDR z,
                                                     uint32_t rows, uint32_t cols)
{
    KernelAdd310P op;
    op.Init(x, y, z, rows, cols);
    op.Process();
}
