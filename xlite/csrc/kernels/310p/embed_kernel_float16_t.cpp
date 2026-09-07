/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

using namespace AscendC;

namespace {

constexpr uint32_t EMBED_TILE_ELEMENTS = 4096;
constexpr uint32_t FP16_BLOCK_ELEMENTS = 32 / sizeof(float16_t);

class KernelEmbedding310P
{
public:
    __aicore__ inline void Init(GM_ADDR weight, GM_ADDR ids, GM_ADDR output, uint32_t dim,
                                uint32_t tokenCount, uint32_t embStart, uint32_t embEnd)
    {
        weightGm.SetGlobalBuffer(reinterpret_cast<__gm__ float16_t *>(weight));
        idsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(ids));
        outputGm.SetGlobalBuffer(reinterpret_cast<__gm__ float16_t *>(output));
        embeddingDim = dim;
        tokens = tokenCount;
        start = embStart;
        end = embEnd;
        pipe.InitBuffer(inputQueue, 1, EMBED_TILE_ELEMENTS * sizeof(float16_t));
        pipe.InitBuffer(outputQueue, 1, EMBED_TILE_ELEMENTS * sizeof(float16_t));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t token = GetBlockIdx(); token < tokens; token += GetBlockNum()) {
            uint32_t row = idsGm.GetValue(token);
            bool localRow = row >= start && row < end;
            uint64_t outputOffset = static_cast<uint64_t>(token) * embeddingDim;
            uint64_t weightOffset =
                localRow ? static_cast<uint64_t>(row - start) * embeddingDim : 0;

            for (uint32_t column = 0; column < embeddingDim;
                 column += EMBED_TILE_ELEMENTS) {
                uint32_t count = embeddingDim - column < EMBED_TILE_ELEMENTS
                                     ? embeddingDim - column
                                     : EMBED_TILE_ELEMENTS;
                LocalTensor<float16_t> outputLocal =
                    outputQueue.AllocTensor<float16_t>();
                if (localRow) {
                    LocalTensor<float16_t> inputLocal =
                        inputQueue.AllocTensor<float16_t>();
                    CopyIn(inputLocal, weightGm[weightOffset + column], count);
                    inputQueue.EnQue(inputLocal);
                    inputLocal = inputQueue.DeQue<float16_t>();
                    Adds(outputLocal, inputLocal, static_cast<float16_t>(0.0f), count);
                    inputQueue.FreeTensor(inputLocal);
                } else {
                    Duplicate(outputLocal, static_cast<float16_t>(0.0f), count);
                }
                outputQueue.EnQue(outputLocal);
                outputLocal = outputQueue.DeQue<float16_t>();
                CopyOut(outputGm[outputOffset + column], outputLocal, count);
                outputQueue.FreeTensor(outputLocal);
            }
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
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    GlobalTensor<float16_t> weightGm;
    GlobalTensor<uint32_t> idsGm;
    GlobalTensor<float16_t> outputGm;
    uint32_t embeddingDim = 0;
    uint32_t tokens = 0;
    uint32_t start = 0;
    uint32_t end = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void embed_kernel_float16_t(
    GM_ADDR weight, GM_ADDR ids, GM_ADDR output, uint32_t dim, uint32_t tokenCount,
    uint32_t embStart, uint32_t embEnd, uint32_t tpSize)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelEmbedding310P op;
    op.Init(weight, ids, output, dim, tokenCount, embStart, embEnd);
    op.Process();
    (void)tpSize;
}
