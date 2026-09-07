/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

using namespace AscendC;

namespace {

// Keep these constants identical to Ascend's AddKernelInvocationNeo sample.
constexpr uint32_t TOTAL_LENGTH = 8 * 2048;
constexpr uint32_t USE_CORE_NUM = 8;
constexpr uint32_t BLOCK_LENGTH = TOTAL_LENGTH / USE_CORE_NUM;
constexpr uint32_t TILE_NUM = 8;
constexpr uint32_t BUFFER_NUM = 2;
constexpr uint32_t TILE_LENGTH = BLOCK_LENGTH / TILE_NUM / BUFFER_NUM;

class KernelOfficialAddProbe310P
{
public:
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z)
    {
        xGm.SetGlobalBuffer((__gm__ half *)x + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        yGm.SetGlobalBuffer((__gm__ half *)y + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        zGm.SetGlobalBuffer((__gm__ half *)z + BLOCK_LENGTH * GetBlockIdx(), BLOCK_LENGTH);
        pipe.InitBuffer(inQueueX, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(inQueueY, BUFFER_NUM, TILE_LENGTH * sizeof(half));
        pipe.InitBuffer(outQueueZ, BUFFER_NUM, TILE_LENGTH * sizeof(half));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t loopCount = TILE_NUM * BUFFER_NUM;
        for (uint32_t progress = 0; progress < loopCount; ++progress) {
            CopyIn(progress);
            Compute(progress);
            CopyOut(progress);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t progress)
    {
        LocalTensor<half> xLocal = inQueueX.AllocTensor<half>();
        LocalTensor<half> yLocal = inQueueY.AllocTensor<half>();
        DataCopy(xLocal, xGm[progress * TILE_LENGTH], TILE_LENGTH);
        DataCopy(yLocal, yGm[progress * TILE_LENGTH], TILE_LENGTH);
        inQueueX.EnQue(xLocal);
        inQueueY.EnQue(yLocal);
    }

    __aicore__ inline void Compute(uint32_t)
    {
        LocalTensor<half> xLocal = inQueueX.DeQue<half>();
        LocalTensor<half> yLocal = inQueueY.DeQue<half>();
        LocalTensor<half> zLocal = outQueueZ.AllocTensor<half>();
        Add(zLocal, xLocal, yLocal, TILE_LENGTH);
        outQueueZ.EnQue(zLocal);
        inQueueX.FreeTensor(xLocal);
        inQueueY.FreeTensor(yLocal);
    }

    __aicore__ inline void CopyOut(uint32_t progress)
    {
        LocalTensor<half> zLocal = outQueueZ.DeQue<half>();
        DataCopy(zGm[progress * TILE_LENGTH], zLocal, TILE_LENGTH);
        outQueueZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueX;
    TQue<QuePosition::VECIN, BUFFER_NUM> inQueueY;
    TQue<QuePosition::VECOUT, BUFFER_NUM> outQueueZ;
    GlobalTensor<half> xGm;
    GlobalTensor<half> yGm;
    GlobalTensor<half> zGm;
};

}  // namespace

extern "C" __global__ __aicore__ void xlite_official_add_probe_310p(GM_ADDR x, GM_ADDR y,
                                                                      GM_ADDR z)
{
    // Deliberately omit task-type overrides: this mirrors the official sample.
    KernelOfficialAddProbe310P op;
    op.Init(x, y, z);
    op.Process();
}

#ifndef ASCENDC_CPU_DEBUG
// Match AddKernelInvocationNeo's proven host launch path.  This wrapper is
// compiled by Bisheng's host pass and exported by the AscendC kernel library;
// it intentionally bypasses the generated aclrtlaunch_* entry used elsewhere
// in Xlite so the two mechanisms can be distinguished on Ascend310P.
void xlite_official_add_probe_310p_do(uint32_t blockDim, void *stream, uint8_t *x,
                                      uint8_t *y, uint8_t *z)
{
    xlite_official_add_probe_310p<<<blockDim, nullptr, stream>>>(x, y, z);
}
#endif
