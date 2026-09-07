/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

using namespace AscendC;

constexpr int32_t BUFFER_NUM = 2;  // tensor num for each queue

template <typename T>
class KernelAddBias
{
public:
    __aicore__ inline KernelAddBias()
    {
    }
    __aicore__ inline void Init(GM_ADDR x, GM_ADDR y, GM_ADDR z, uint32_t xNumelIn,
                                uint32_t yNumelIn)
    {
        xGm.SetGlobalBuffer((__gm__ T *)x);
        yGm.SetGlobalBuffer((__gm__ T *)y);
        zGm.SetGlobalBuffer((__gm__ T *)z);
        yNumel = yNumelIn;
        rowNum = xNumelIn / yNumelIn;
        pipe.InitBuffer(queInX, BUFFER_NUM, yNumel * sizeof(T));
        pipe.InitBuffer(queOutZ, BUFFER_NUM, yNumel * sizeof(T));
        pipe.InitBuffer(yBuf, yNumel * sizeof(T));
        if constexpr (!std::is_same<T, float>::value) {
            pipe.InitBuffer(yFp32Buf, yNumel * sizeof(float));
            pipe.InitBuffer(xFp32Buf, yNumel * sizeof(float));
            pipe.InitBuffer(zFp32Buf, yNumel * sizeof(float));
        }
    }
    __aicore__ inline void Process()
    {
        LocalTensor<float> yFp32Local;
        LocalTensor<T> yLocal = yBuf.Get<T>();
        DataCopy(yLocal, yGm, ROUND_UP(yNumel * sizeof(T), BLOCK_SIZE) / sizeof(T));
        PipeBarrier<PIPE_ALL>();
        if constexpr (!std::is_same<T, float>::value) {
            yFp32Local = yFp32Buf.Get<float>();
            Cast(yFp32Local, yLocal, RoundMode::CAST_NONE, yNumel);
            PipeBarrier<PIPE_V>();
        } else {
            yFp32Local = yLocal;
        }

        for (uint32_t loop = GetBlockIdx(); loop < rowNum; loop += GetBlockNum()) {
            CopyIn(loop);
            Compute(loop, yFp32Local);
            CopyOut(loop);
        }
    }

private:
    __aicore__ inline void CopyIn(uint32_t loop)
    {
        LocalTensor<T> xLocal = queInX.AllocTensor<T>();
        DataCopy(xLocal, xGm[loop * yNumel], ROUND_UP(yNumel * sizeof(T), BLOCK_SIZE) / sizeof(T));
        queInX.EnQue(xLocal);
    }

    __aicore__ inline void Compute(uint32_t loop, LocalTensor<float> yFp32Local)
    {
        LocalTensor<T> xLocal = queInX.DeQue<T>();
        LocalTensor<T> zLocal = queOutZ.AllocTensor<T>();

        if constexpr (std::is_same<T, float>::value) {
            Add(zLocal, xLocal, yFp32Local, yNumel);
        } else {
            LocalTensor<float> xFp32Local = xFp32Buf.Get<float>();
            LocalTensor<float> zFp32Local = zFp32Buf.Get<float>();
            Cast(xFp32Local, xLocal, RoundMode::CAST_NONE, yNumel);
            Add(zFp32Local, xFp32Local, yFp32Local, yNumel);
            Cast(zLocal, zFp32Local, RoundMode::CAST_RINT, yNumel);
        }

        queOutZ.EnQue<T>(zLocal);
        queInX.FreeTensor(xLocal);
    }

    __aicore__ inline void CopyOut(uint32_t loop)
    {
        LocalTensor<T> zLocal = queOutZ.DeQue<T>();

        if (((yNumel * sizeof(T)) & (BLOCK_SIZE - 1)) == 0) {
            DataCopy(zGm[loop * yNumel], zLocal, yNumel);
        } else {
            DataCopyParams copyParams;
            copyParams.blockLen = yNumel * sizeof(T);
            copyParams.blockCount = 1;
            DataCopyPad(zGm[loop * yNumel], zLocal, copyParams);
        }

        queOutZ.FreeTensor(zLocal);
    }

    TPipe pipe;
    TQue<QuePosition::VECIN, BUFFER_NUM> queInX;
    TQue<QuePosition::VECOUT, BUFFER_NUM> queOutZ;
    GlobalTensor<T> xGm;
    GlobalTensor<T> yGm;
    GlobalTensor<T> zGm;
    TBuf<TPosition::VECCALC> yBuf, yFp32Buf, xFp32Buf, zFp32Buf;
    uint32_t rowNum;
    uint32_t yNumel;
};

#if defined(XLITE_DEVICE_310P)
#define ADDBIAS_FUNC_DEFINE(dtype)                                                          \
    extern "C" __global__ __aicore__ void add_bias_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z, \
                                                            uint32_t xNumel, uint32_t yNumel) \
    {                                                                                       \
    }
#else
#define ADDBIAS_FUNC_DEFINE(dtype)                                                           \
    extern "C" __global__ __aicore__ void add_bias_##dtype(GM_ADDR x, GM_ADDR y, GM_ADDR z,  \
                                                           uint32_t xNumel, uint32_t yNumel) \
    {                                                                                        \
        KernelAddBias<dtype> op;                                                             \
        op.Init(x, y, z, xNumel, yNumel);                                                    \
        op.Process();                                                                        \
    }
#endif
