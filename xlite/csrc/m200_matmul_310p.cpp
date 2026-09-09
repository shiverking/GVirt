/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "m200_matmul_310p.h"

#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <vector>

#include "aclrtlaunch_all.h"
#include "ascend.h"
#include "m200_tiling_310p_c.h"

namespace {

constexpr uint32_t kMaxDecodeBatch = 20;
constexpr uint32_t kMaxLmHeadCubeBatch = 8;
constexpr uint32_t kLmHeadN = 151936;
constexpr uint32_t kLmHeadChunkN = 12288;
constexpr uint32_t kLmHeadRowGroup = 8;

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
    std::vector<void *> systemWorkspaces;
    uint64_t systemWorkspaceBytes = 0;
    std::unordered_map<uint64_t, TilingEntry> tilings;
};

uint64_t TilingKey(uint32_t m, uint32_t n, uint32_t k)
{
    return (static_cast<uint64_t>(m) << 48) | (static_cast<uint64_t>(n) << 24) | k;
}

TilingEntry BuildTiling(uint32_t m, uint32_t n, uint32_t k)
{
    TilingEntry result;
    result.host.resize(4096);
    size_t written = 0;
    char error[256] = {};
    if (XliteM200BuildTiling310P(m, n, k, result.host.data(), result.host.size(), &written,
                                 &result.usedCores, &result.systemWorkspaceBytes,
                                 error, sizeof(error)) != 0) {
        throw std::runtime_error(std::string("Ascend310P M200 tiling: ") + error);
    }
    result.host.resize(written);
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
        if (!state.systemWorkspaces.empty()) {
            CHECK_ACL(aclrtSynchronizeStream(rt.stream));
            for (void *workspace : state.systemWorkspaces) {
                CHECK_ACL(aclrtFree(workspace));
            }
            state.systemWorkspaces.clear();
        }
        state.systemWorkspaceBytes = entry.systemWorkspaceBytes;
    }
    return entry;
}

void EnsureWorkspaceSlots(M200MatmulState &state, uint32_t slots)
{
    while (state.systemWorkspaces.size() < slots) {
        void *workspace = nullptr;
        CHECK_ACL(aclrtMalloc(&workspace, state.systemWorkspaceBytes,
                             ACL_MEM_MALLOC_NORMAL_ONLY));
        state.systemWorkspaces.push_back(workspace);
    }
}

void Launch(XRuntime &rt, void *a, void *b, void *c, uint32_t m, uint32_t n, uint32_t k,
            uint32_t workspaceSlot = 0)
{
    TilingEntry &tiling = GetTiling(rt, m, n, k);
    M200MatmulState &state = GetState(rt);
    EnsureWorkspaceSlots(state, workspaceSlot + 1);
    ACLRT_LAUNCH_KERNEL(xlite_m200_matmul_float16)
    (tiling.usedCores, rt.stream, a, b, c, state.systemWorkspaces[workspaceSlot], tiling.device);
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
    const bool safeLmHeadBatch = n != kLmHeadN || m <= kMaxLmHeadCubeBatch;
    return m >= 1 && m <= kMaxDecodeBatch && safeLmHeadBatch && weight.shape[1] == k &&
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

    if (m == 1) {
        uint32_t launchIndex = 0;
        for (uint32_t offset = 0; offset < n; offset += kLmHeadChunkN) {
            const uint32_t currentN = std::min(kLmHeadChunkN, n - offset);
            void *weightData = static_cast<void *>(
                static_cast<uint16_t *>(weight.ptr) + static_cast<size_t>(offset) * k);
            void *outputData = static_cast<void *>(static_cast<uint16_t *>(out.ptr) + offset);
            Launch(rt, in.ptr, weightData, outputData, 1, currentN, k, launchIndex++);
        }
        return;
    }

    const uint32_t rowGroups = (m + kLmHeadRowGroup - 1) / kLmHeadRowGroup;
    const uint32_t columnChunks = (n + kLmHeadChunkN - 1) / kLmHeadChunkN;
    std::vector<XTensor *> chunkOutputs;
    chunkOutputs.reserve(static_cast<size_t>(rowGroups) * columnChunks);
    for (uint32_t rowStart = 0; rowStart < m; rowStart += kLmHeadRowGroup) {
        const uint32_t groupM = std::min(kLmHeadRowGroup, m - rowStart);
        for (uint32_t offset = 0; offset < n; offset += kLmHeadChunkN) {
            const uint32_t currentN = std::min(kLmHeadChunkN, n - offset);
            chunkOutputs.push_back(&rt.GetTensor({groupM, currentN}, FP16, DBG_LOC));
        }
    }

    uint32_t launchIndex = 0;
    for (uint32_t rowStart = 0; rowStart < m; rowStart += kLmHeadRowGroup) {
        const uint32_t groupM = std::min(kLmHeadRowGroup, m - rowStart);
        void *inputData = static_cast<void *>(
            static_cast<uint16_t *>(in.ptr) + static_cast<size_t>(rowStart) * k);
        for (uint32_t offset = 0; offset < n; offset += kLmHeadChunkN) {
            const uint32_t currentN = std::min(kLmHeadChunkN, n - offset);
            void *weightData = static_cast<void *>(
                static_cast<uint16_t *>(weight.ptr) + static_cast<size_t>(offset) * k);
            XTensor *chunkOutput = chunkOutputs[launchIndex];
            Launch(rt, inputData, weightData, chunkOutput->ptr, groupM, currentN, k,
                   launchIndex);
            void *destination = static_cast<void *>(
                static_cast<uint16_t *>(out.ptr) + static_cast<size_t>(rowStart) * n + offset);
            CHECK_ACL(aclrtMemcpy2dAsync(destination, static_cast<size_t>(n) * sizeof(uint16_t),
                                         chunkOutput->ptr,
                                         static_cast<size_t>(currentN) * sizeof(uint16_t),
                                         static_cast<size_t>(currentN) * sizeof(uint16_t), groupM,
                                         ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            ++launchIndex;
        }
    }
    for (XTensor *chunkOutput : chunkOutputs) {
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
    for (void *workspace : state->systemWorkspaces) {
        if (workspace != nullptr) {
            (void)aclrtFree(workspace);
        }
    }
    delete state;
    rt._m200MatmulState = nullptr;
}
