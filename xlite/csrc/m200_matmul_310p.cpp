/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "m200_matmul_310p.h"

#include <algorithm>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <tuple>
#include <unordered_map>
#include <vector>

#include "aclrtlaunch_all.h"
#include "ascend.h"
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace {

constexpr uint32_t kMaxDecodeBatch = 20;
constexpr uint32_t kLmHeadN = 151936;
constexpr uint32_t kLmHeadChunkN = 12288;

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

bool IsAsrProjection(uint32_t n, uint32_t k)
{
    return (n == 4096 && k == 2048) || (n == 2048 && k == 2048) ||
           (n == 12288 && k == 2048) || (n == 2048 && k == 6144) ||
           (n == kLmHeadN && k == 2048);
}

struct TilingEntry {
    std::vector<uint8_t> host;
    void *device = nullptr;
    uint32_t usedCores = 0;
    uint64_t systemWorkspaceBytes = 0;
};

struct M200MatmulState {
    void *systemWorkspace = nullptr;
    uint64_t systemWorkspaceBytes = 0;
    std::unordered_map<uint64_t, TilingEntry> tilings;
};

uint64_t TilingKey(uint32_t m, uint32_t n, uint32_t k)
{
    return (static_cast<uint64_t>(m) << 48) | (static_cast<uint64_t>(n) << 24) | k;
}

TilingEntry BuildTiling(uint32_t m, uint32_t n, uint32_t k)
{
    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(XLITE_BUILD_SOC);
    const uint32_t availableCores = platform->GetCoreNumAic();
    if (availableCores == 0) {
        throw std::runtime_error("Ascend310P reported zero Cube cores");
    }
    const uint32_t targetCores = std::min(availableCores, CeilDiv(n, 256U));
    const uint32_t singleCoreN = AlignUp(CeilDiv(n, targetCores), 256U);
    const uint32_t requestedCores = CeilDiv(n, singleCoreN);

    MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    tilingApi.SetOrgShape(m, n, k);
    tilingApi.SetShape(m, n, k);
    tilingApi.SetSingleShape(32, singleCoreN, -1);
    tilingApi.SetDim(requestedCores);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tiling;
    if (tilingApi.GetTiling(tiling) == -1) {
        throw std::runtime_error("Ascend310P M200 MatMul tiling failed");
    }
    TilingEntry result;
    result.usedCores = static_cast<uint32_t>(tiling.get_usedCoreNum());
    if (result.usedCores == 0 || result.usedCores > availableCores) {
        throw std::runtime_error("Ascend310P M200 MatMul generated an invalid core count");
    }
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    result.systemWorkspaceBytes = static_cast<uint64_t>(platform->GetLibApiWorkSpaceSize());
    const uint32_t tilingBytes = tiling.GetDataSize();
    if (tilingBytes > sizeof(optiling::TCubeTiling)) {
        throw std::runtime_error("Ascend310P TCubeTiling exceeds its ABI storage");
    }
    result.host.resize(sizeof(optiling::TCubeTiling) + sizeof(uint64_t), 0);
    tiling.SaveToBuffer(result.host.data(), tilingBytes);
    std::memcpy(result.host.data() + sizeof(optiling::TCubeTiling), &ubBytes, sizeof(ubBytes));
    return result;
}

M200MatmulState &GetState(XRuntime &rt)
{
    if (rt._m200MatmulState == nullptr) {
        rt._m200MatmulState = new M200MatmulState();
    }
    return *static_cast<M200MatmulState *>(rt._m200MatmulState);
}

TilingEntry &GetTiling(XRuntime &rt, uint32_t m, uint32_t n, uint32_t k)
{
    M200MatmulState &state = GetState(rt);
    const uint64_t key = TilingKey(m, n, k);
    auto found = state.tilings.find(key);
    if (found != state.tilings.end()) {
        return found->second;
    }
    auto inserted = state.tilings.emplace(key, BuildTiling(m, n, k));
    TilingEntry &entry = inserted.first->second;
    CHECK_ACL(aclrtMalloc(&entry.device, entry.host.size(), ACL_MEM_MALLOC_NORMAL_ONLY));
    // One-time initialization. A synchronous copy makes the cache independent
    // of unordered_map rehashing and host-vector lifetime during async DMA.
    CHECK_ACL(aclrtMemcpy(entry.device, entry.host.size(), entry.host.data(), entry.host.size(),
                         ACL_MEMCPY_HOST_TO_DEVICE));
    if (state.systemWorkspaceBytes < entry.systemWorkspaceBytes) {
        if (state.systemWorkspace != nullptr) {
            CHECK_ACL(aclrtSynchronizeStream(rt.stream));
            CHECK_ACL(aclrtFree(state.systemWorkspace));
        }
        CHECK_ACL(aclrtMalloc(&state.systemWorkspace, entry.systemWorkspaceBytes,
                             ACL_MEM_MALLOC_NORMAL_ONLY));
        state.systemWorkspaceBytes = entry.systemWorkspaceBytes;
    }
    return entry;
}

void Launch(XRuntime &rt, void *a, void *b, void *c, uint32_t m, uint32_t n, uint32_t k)
{
    TilingEntry &tiling = GetTiling(rt, m, n, k);
    M200MatmulState &state = GetState(rt);
    ACLRT_LAUNCH_KERNEL(xlite_m200_matmul_float16)
    (tiling.usedCores, rt.stream, a, b, c, state.systemWorkspace, tiling.device);
    ++rt.m200MatmulKernelLaunches;
}

}  // namespace

bool XliteM200Matmul310PSupported(const XTensor &in, const XTensor &weight,
                                  const XTensor &out, bool weightNZ,
                                  const XTensor &bias, const XTensor &deqScale,
                                  bool transpose)
{
    if (weightNZ || transpose || bias.ptr != nullptr || deqScale.ptr != nullptr ||
        in.dtype != FP16 || weight.dtype != FP16 || out.dtype != FP16 ||
        in.shape.size() != 2 || weight.shape.size() != 2 || out.shape.size() != 2) {
        return false;
    }
    const uint32_t m = static_cast<uint32_t>(in.shape[0]);
    const uint32_t k = static_cast<uint32_t>(in.shape[1]);
    const uint32_t n = static_cast<uint32_t>(weight.shape[0]);
    return m >= 1 && m <= kMaxDecodeBatch && weight.shape[1] == k &&
           out.shape[0] == m && out.shape[1] == n && IsAsrProjection(n, k);
}

void XliteM200Matmul310P(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out)
{
    ++rt.m200MatmulRequests;
    const uint32_t m = static_cast<uint32_t>(in.shape[0]);
    const uint32_t k = static_cast<uint32_t>(in.shape[1]);
    const uint32_t n = static_cast<uint32_t>(weight.shape[0]);
    if (n != kLmHeadN) {
        Launch(rt, in.ptr, weight.ptr, out.ptr, m, n, k);
        return;
    }

    XTensor *chunkOutput = nullptr;
    if (m > 1) {
        chunkOutput = &rt.GetTensor({m, kLmHeadChunkN}, FP16, DBG_LOC);
    }
    for (uint32_t offset = 0; offset < n; offset += kLmHeadChunkN) {
        const uint32_t currentN = std::min(kLmHeadChunkN, n - offset);
        void *weightData = static_cast<void *>(
            static_cast<uint16_t *>(weight.ptr) + static_cast<size_t>(offset) * k);
        void *outputData = m == 1
            ? static_cast<void *>(static_cast<uint16_t *>(out.ptr) + offset)
            : chunkOutput->ptr;
        Launch(rt, in.ptr, weightData, outputData, m, currentN, k);
        if (m > 1) {
            void *destination = static_cast<void *>(static_cast<uint16_t *>(out.ptr) + offset);
            CHECK_ACL(aclrtMemcpy2dAsync(destination, static_cast<size_t>(n) * sizeof(uint16_t),
                                         chunkOutput->ptr,
                                         static_cast<size_t>(currentN) * sizeof(uint16_t),
                                         static_cast<size_t>(currentN) * sizeof(uint16_t), m,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
    }
    if (chunkOutput != nullptr) {
        rt.PutTensor(*chunkOutput);
    }
}

void XliteM200Matmul310PDestroy(XRuntime &rt)
{
    auto *state = static_cast<M200MatmulState *>(rt._m200MatmulState);
    if (state == nullptr) {
        return;
    }
    if (rt.stream != nullptr) {
        (void)aclrtSynchronizeStream(rt.stream);
    }
    for (auto &item : state->tilings) {
        if (item.second.device != nullptr) {
            (void)aclrtFree(item.second.device);
        }
    }
    if (state->systemWorkspace != nullptr) {
        (void)aclrtFree(state->systemWorkspace);
    }
    delete state;
    rt._m200MatmulState = nullptr;
}
