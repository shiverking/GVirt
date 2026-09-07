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
        pipe.InitBuffer(rowBuf, EMBED_TILE_ELEMENTS * sizeof(float16_t));
    }

    __aicore__ inline void Process()
    {
        LocalTensor<float16_t> rowLocal = rowBuf.Get<float16_t>();
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
                if (localRow) {
                    CopyIn(rowLocal, weightGm[weightOffset + column], count);
                    PipeBarrier<PIPE_ALL>();
                } else {
                    Duplicate(rowLocal, static_cast<float16_t>(0.0f), count);
                    PipeBarrier<PIPE_V>();
                }
                CopyOut(outputGm[outputOffset + column], rowLocal, count);
                PipeBarrier<PIPE_ALL>();
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
    TBuf<TPosition::VECCALC> rowBuf;
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
    KernelEmbedding310P op;
    op.Init(weight, ids, output, dim, tokenCount, embStart, embEnd);
    op.Process();
    (void)tpSize;
}
