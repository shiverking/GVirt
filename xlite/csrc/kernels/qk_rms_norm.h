/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "norm.h"

#if defined(__DAV_C220_VEC__) || defined(XLITE_ARCH_310P)

// Fuses the two independent RmsNorm passes (Q segment + K segment) applied to
// disjoint column ranges of the same qkv buffer into a single device kernel.
// The two segments have no data dependency (disjoint read/write/weight ranges),
// so they are issued back-to-back through the coreOffset relay (same pattern as
// mla_prepare) instead of as two separate launches. The relay assigns the two
// segments to different core ranges; it does not introduce a serial dependency.
//
// Each segment carries its own (normDim, cntPerToken) so the kernel can express
// both normalization granularities used by MHA:
//   - plain qkNorm:        Q (normDim=headDim,      cnt=qHeads), K (headDim,     kHeads)
//   - qkNormFull variance:  Q (normDim=headDim*qHeads, cnt=1),    K (headDim*kHeads, 1)
//   - qkNormFull apply:     same as variance stage, but useNorm=true reading pre-reduced variance
//
// useNorm = true  -> apply RmsNorm (compute variance inline if variance is null,
//                    otherwise consume the pre-reduced variance, e.g. qkNormFull apply stage)
// useNorm = false -> variance-only stage (write per-token variance to qVariance/kVariance,
//                    e.g. qkNormFull variance stage before the AllReduce)
template <typename Dtype>
__aicore__ void qk_rms_norm(GM_ADDR input, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR qOut,
                            GM_ADDR kNorm, GM_ADDR kNormBias, GM_ADDR kOut, uint32_t token_num,
                            uint32_t q_norm_dim, uint32_t q_cnt_per_token, uint32_t k_norm_dim,
                            uint32_t k_cnt_per_token, uint32_t in_step, uint32_t out_step,
                            float norm_eps, uint32_t k_start_offset, bool useNorm,
                            GM_ADDR qVariance, GM_ADDR kVariance, uint32_t tpSize)
{
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Rms);
    int coreOffset = 0;
    int nextCoreOffset = 0;

    // Q segment: starts at column 0.
    norm<Dtype>(input, nullptr, qNorm, qNormBias, qOut, token_num, q_norm_dim, norm_eps, kind,
                q_cnt_per_token, in_step, out_step, 0, 0, useNorm, qVariance, tpSize, nullptr,
                nullptr, 0, coreOffset, &nextCoreOffset);

    coreOffset = nextCoreOffset;
    // K segment: offset to the K column range of the same row.
    uint32_t k_out_start_offset = useNorm ? k_start_offset : 0;
    norm<Dtype>(input, nullptr, kNorm, kNormBias, kOut, token_num, k_norm_dim, norm_eps, kind,
                k_cnt_per_token, in_step, out_step, k_start_offset, k_out_start_offset, useNorm,
                kVariance, tpSize, nullptr, nullptr, 0, coreOffset, &nextCoreOffset);
}

#define QK_RMS_NORM_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void qk_rms_norm_##dtype(                                     \
        GM_ADDR input, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR qOut, GM_ADDR kNorm,              \
        GM_ADDR kNormBias, GM_ADDR kOut, uint32_t token_num, uint32_t q_norm_dim,                  \
        uint32_t q_cnt_per_token, uint32_t k_norm_dim, uint32_t k_cnt_per_token, uint32_t in_step, \
        uint32_t out_step, float norm_eps, uint32_t k_start_offset, bool useNorm,                  \
        GM_ADDR qVariance, GM_ADDR kVariance, uint32_t tpSize)                                     \
    {                                                                                              \
        qk_rms_norm<dtype>(input, qNorm, qNormBias, qOut, kNorm, kNormBias, kOut, token_num,       \
                           q_norm_dim, q_cnt_per_token, k_norm_dim, k_cnt_per_token, in_step,      \
                           out_step, norm_eps, k_start_offset, useNorm, qVariance, kVariance,      \
                           tpSize);                                                                \
    }
#else
#define QK_RMS_NORM_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void qk_rms_norm_##dtype(                                     \
        GM_ADDR input, GM_ADDR qNorm, GM_ADDR qNormBias, GM_ADDR qOut, GM_ADDR kNorm,              \
        GM_ADDR kNormBias, GM_ADDR kOut, uint32_t token_num, uint32_t q_norm_dim,                  \
        uint32_t q_cnt_per_token, uint32_t k_norm_dim, uint32_t k_cnt_per_token, uint32_t in_step, \
        uint32_t out_step, float norm_eps, uint32_t k_start_offset, bool useNorm,                  \
        GM_ADDR qVariance, GM_ADDR kVariance, uint32_t tpSize)                                     \
    {                                                                                              \
    }
#endif
