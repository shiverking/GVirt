/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "softmax_attn_aiv.h"

#if defined(XLITE_DEVICE_310P)
#define SOFTMAX_FUNC_DEFINE(Dtype)                                                           \
    extern "C" __global__ __aicore__ void softmax_##Dtype(GM_ADDR x, uint32_t m, uint32_t n, \
                                                            uint32_t contextLen)             \
    {                                                                                        \
    }
#else
#define SOFTMAX_FUNC_DEFINE(Dtype)                                                           \
    extern "C" __global__ __aicore__ void softmax_##Dtype(GM_ADDR x, uint32_t m, uint32_t n, \
                                                          uint32_t contextLen)               \
    {                                                                                        \
        RunAivSoftmaxPingPong((__gm__ Dtype *)x, m, n, contextLen);                          \
    }
#endif

SOFTMAX_FUNC_DEFINE(float16_t);
SOFTMAX_FUNC_DEFINE(bfloat16_t);

#if defined(XLITE_DEVICE_310P)
#define SOFTMAX_LONG_FUNC_DEFINE(Dtype)                                                \
    extern "C" __global__ __aicore__ void softmax_long_##Dtype(                       \
        GM_ADDR x, GM_ADDR expBuf, uint32_t m, uint32_t n, uint32_t contextLen)        \
    {                                                                                  \
    }
#else
#define SOFTMAX_LONG_FUNC_DEFINE(Dtype)                                                 \
    extern "C" __global__ __aicore__ void softmax_long_##Dtype(                         \
        GM_ADDR x, GM_ADDR expBuf, uint32_t m, uint32_t n, uint32_t contextLen)         \
    {                                                                                   \
        RunAivSoftmaxLong((__gm__ Dtype *)x, (__gm__ float *)expBuf, m, n, contextLen); \
    }
#endif

SOFTMAX_LONG_FUNC_DEFINE(float16_t);
SOFTMAX_LONG_FUNC_DEFINE(bfloat16_t);
