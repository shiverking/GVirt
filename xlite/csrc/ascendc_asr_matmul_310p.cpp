/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "ascendc_asr_matmul_310p.h"

#include <algorithm>
#include <stdexcept>

#include "aclrtlaunch_all.h"
#include "ascend.h"

namespace {

constexpr uint32_t kMaxDecodeBatch = 20;
constexpr uint32_t kLmHeadN = 151936;

bool IsProjectionShape(uint32_t n, uint32_t k)
{
    return (n == 4096 && k == 2048) || (n == 2048 && k == 2048) ||
           (n == 12288 && k == 2048) || (n == 2048 && k == 6144);
}

bool IsCommonContract(const XTensor &in, const XTensor &weight,
                      const XTensor &out, bool weightNZ, const XTensor &bias,
                      const XTensor &deqScale, bool transpose)
{
    if (weightNZ || transpose || bias.ptr != nullptr || deqScale.ptr != nullptr ||
        in.dtype != FP16 || weight.dtype != FP16 || out.dtype != FP16 ||
        in.shape.size() != 2 || weight.shape.size() != 2 || out.shape.size() != 2) {
        return false;
    }
    const uint64_t m = in.shape[0];
    const uint64_t k = in.shape[1];
    const uint64_t n = weight.shape[0];
    return m >= 1 && weight.shape[1] == k && out.shape[0] == m &&
           out.shape[1] == n;
}

bool IsNzContract(const XTensor &in, const XTensor &weight,
                  const XTensor &out, bool weightNZ, const XTensor &bias,
                  const XTensor &deqScale, bool transpose)
{
    if (!weightNZ || transpose || bias.ptr != nullptr || deqScale.ptr != nullptr ||
        in.dtype != FP16 || weight.dtype != FP16 || out.dtype != FP16 ||
        in.shape.size() != 2 || weight.shape.size() != 2 || out.shape.size() != 2) {
        return false;
    }
    const uint64_t m = in.shape[0];
    const uint64_t k = in.shape[1];
    const uint64_t n = weight.shape[0];
    return m >= 1 && m <= 4096 && weight.shape[1] == k &&
           out.shape[0] == m && out.shape[1] == n &&
           (IsProjectionShape(static_cast<uint32_t>(n), static_cast<uint32_t>(k)) ||
            (n == kLmHeadN && k == 2048));
}

}  // namespace

bool XliteAscendCAsrProjection310PSupported(const XTensor &in,
                                             const XTensor &weight,
                                             const XTensor &out, bool weightNZ,
                                             const XTensor &bias,
                                             const XTensor &deqScale,
                                             bool transpose)
{
    if (!IsCommonContract(in, weight, out, weightNZ, bias, deqScale, transpose)) {
        return false;
    }
    const uint32_t m = static_cast<uint32_t>(in.shape[0]);
    const uint32_t k = static_cast<uint32_t>(in.shape[1]);
    const uint32_t n = static_cast<uint32_t>(weight.shape[0]);
    return m <= kMaxDecodeBatch && IsProjectionShape(n, k);
}

bool XliteAscendCAsrKnownMatmul310P(const XTensor &in, const XTensor &weight,
                                    const XTensor &out, bool weightNZ,
                                    const XTensor &bias,
                                    const XTensor &deqScale, bool transpose)
{
    if (!IsCommonContract(in, weight, out, weightNZ, bias, deqScale, transpose)) {
        return false;
    }
    const uint32_t k = static_cast<uint32_t>(in.shape[1]);
    const uint32_t n = static_cast<uint32_t>(weight.shape[0]);
    return IsProjectionShape(n, k) || (n == kLmHeadN && k == 2048);
}

void XliteAscendCAsrProjection310P(XRuntime &rt, XTensor &in,
                                   XTensor &weight, XTensor &out, bool cached)
{
    const uint32_t m = static_cast<uint32_t>(in.shape[0]);
    const uint32_t k = static_cast<uint32_t>(in.shape[1]);
    const uint32_t n = static_cast<uint32_t>(weight.shape[0]);
    const uint32_t blockDim = std::min<uint32_t>(8, (n + 127) / 128);
    if (blockDim == 0) {
        throw std::runtime_error("AscendC ASR projection computed zero blockDim");
    }
    rt.RecordAscendCAsrMatmul310P(m);
    if (cached) {
        ACLRT_LAUNCH_KERNEL(asr_m200_projection_cached_fp16)
        (blockDim, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k);
        ++rt.ascendcAsrPerfProjectionRequests;
    } else {
        ACLRT_LAUNCH_KERNEL(asr_m200_projection_fp16)
        (blockDim, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k);
    }
    ++rt.ascendcAsrMatmulKernelLaunches;
}

bool XliteAscendCAsrLmHead310PSupported(const XTensor &in,
                                        const XTensor &weight,
                                        const XTensor &out, bool weightNZ,
                                        const XTensor &bias,
                                        const XTensor &deqScale,
                                        bool transpose)
{
    if (!IsCommonContract(in, weight, out, weightNZ, bias, deqScale, transpose)) {
        return false;
    }
    return in.shape[0] <= kMaxDecodeBatch && in.shape[1] == 2048 &&
           weight.shape[0] == kLmHeadN;
}

void XliteAscendCAsrLmHead310P(XRuntime &rt, XTensor &in,
                               XTensor &weight, XTensor &out)
{
    const uint32_t m = static_cast<uint32_t>(in.shape[0]);
    rt.RecordAscendCAsrMatmul310P(m);
    ACLRT_LAUNCH_KERNEL(asr_m200_lm_head_cached_fp16)
    (8, rt.stream, in.ptr, weight.ptr, out.ptr, m);
    ++rt.ascendcAsrMatmulKernelLaunches;
    ++rt.ascendcAsrPerfLmHeadRequests;
}

bool XliteAscendCAsrNzMatmul310PSupported(const XTensor &in,
                                          const XTensor &weight,
                                          const XTensor &out, bool weightNZ,
                                          const XTensor &bias,
                                          const XTensor &deqScale,
                                          bool transpose)
{
    return IsNzContract(in, weight, out, weightNZ, bias, deqScale, transpose);
}

void XliteAscendCAsrNzMatmul310P(XRuntime &rt, XTensor &in,
                                 XTensor &weight, XTensor &out)
{
    const uint32_t m = static_cast<uint32_t>(in.shape[0]);
    const uint32_t n = static_cast<uint32_t>(weight.shape[0]);
    const uint32_t k = static_cast<uint32_t>(weight.shape[1]);
    const uint32_t blockDim = std::min<uint32_t>(8, n / 128);
    if (blockDim == 0) {
        throw std::runtime_error("AscendC ASR NZ MatMul computed zero blockDim");
    }
    rt.RecordAscendCAsrMatmul310P(m);
    ACLRT_LAUNCH_KERNEL(asr_m200_matmul_nz_fp16)
    (blockDim, rt.stream, in.ptr, weight.ptr, out.ptr, m, n, k);
    ++rt.ascendcAsrMatmulKernelLaunches;
    if (n == kLmHeadN) {
        ++rt.ascendcAsrNzLmHeadRequests;
    } else {
        ++rt.ascendcAsrNzProjectionRequests;
    }
}
