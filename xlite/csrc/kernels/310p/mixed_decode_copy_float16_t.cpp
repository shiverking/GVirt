/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

using namespace AscendC;

namespace {

constexpr uint32_t kDecodeRowElements = 16 * 128;

template <bool Scatter>
class KernelMixedDecodeCopy310P
{
public:
    __aicore__ inline void Init(GM_ADDR source, GM_ADDR destination,
                                GM_ADDR queryOffsets, uint32_t rows)
    {
        sourceGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(source));
        destinationGm.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(destination));
        offsetsGm.SetGlobalBuffer(reinterpret_cast<__gm__ uint32_t *>(queryOffsets));
        rowCount = rows;
        pipe.InitBuffer(inputQueue, 1, kDecodeRowElements * sizeof(half));
        pipe.InitBuffer(outputQueue, 1, kDecodeRowElements * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        for (uint32_t localRow = GetBlockIdx(); localRow < rowCount;
             localRow += GetBlockNum()) {
            const uint32_t packedRow = localRow;
            const uint32_t flattenedRow = offsetsGm.GetValue(localRow);
            const uint64_t sourceRow = Scatter ? packedRow : flattenedRow;
            const uint64_t destinationRow = Scatter ? flattenedRow : packedRow;

            LocalTensor<half> input = inputQueue.AllocTensor<half>();
            DataCopy(input, sourceGm[sourceRow * kDecodeRowElements],
                     kDecodeRowElements);
            inputQueue.EnQue(input);
            input = inputQueue.DeQue<half>();
            LocalTensor<half> output = outputQueue.AllocTensor<half>();
            Adds(output, input, static_cast<half>(0.0f), kDecodeRowElements);
            outputQueue.EnQue(output);
            inputQueue.FreeTensor(input);
            output = outputQueue.DeQue<half>();
            DataCopy(destinationGm[destinationRow * kDecodeRowElements], output,
                     kDecodeRowElements);
            outputQueue.FreeTensor(output);
        }
    }

private:
    TPipe pipe;
    TQue<QuePosition::VECIN, 1> inputQueue;
    TQue<QuePosition::VECOUT, 1> outputQueue;
    GlobalTensor<half> sourceGm;
    GlobalTensor<half> destinationGm;
    GlobalTensor<uint32_t> offsetsGm;
    uint32_t rowCount = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void mixed_decode_compact_float16_t(
    GM_ADDR source, GM_ADDR destination, GM_ADDR queryOffsets, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelMixedDecodeCopy310P<false> op;
    op.Init(source, destination, queryOffsets, rows);
    op.Process();
}

extern "C" __global__ __aicore__ void mixed_decode_scatter_float16_t(
    GM_ADDR source, GM_ADDR destination, GM_ADDR queryOffsets, uint32_t rows)
{
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    KernelMixedDecodeCopy310P<true> op;
    op.Init(source, destination, queryOffsets, rows);
    op.Process();
}
