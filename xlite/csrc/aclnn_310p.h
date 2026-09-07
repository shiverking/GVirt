/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_ACLNN_310P_H_
#define _XLITE_ACLNN_310P_H_

#include <cstddef>

#include "base.h"
#include "runtime.h"

constexpr size_t XLITE_310P_ACLNN_WORKSPACE_BYTES = 512ULL * 1024ULL * 1024ULL;
// The 310P ACLNN implementation can select a workspace larger than the POC
// budget for the 151936-column LM head.  Projection sizes up to 12288 have
// been validated on the target, so split only wider, non-transposed outputs.
constexpr size_t XLITE_310P_MATMUL_N_CHUNK = 12288;

// Correctness-first Ascend 310P backend. These entry points intentionally expose
// the existing Xlite tensor ABI while delegating cube/attention work to ACLNN.
void XliteAclnn310PMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out,
                          bool weightNZ, const XTensor &bias, const XTensor &deqScale,
                          bool transpose);

void XliteAclnn310PAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                             XTensor &output, XTensor &lens, XTensor &cachedLens,
                             XTensor &blockTables, uint32_t maxNumBlock,
                             uint32_t nHeads, uint32_t nKvHeads,
                             uint32_t headDim, uint32_t blockSize, uint32_t batch,
                             bool decode);

#endif
