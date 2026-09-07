/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */

// CANN 9.1 keeps the v1 attention APIs available but marks them for a future
// December 2026 removal.  This correctness POC intentionally targets those
// APIs until the V3/V4 signatures are probed on the physical 310P environment.
// Keep the repository-wide -Werror policy while limiting the exception to this
// backend translation unit.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wcpp"
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
#include "aclnn_310p.h"

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclnnop/aclnn_incre_flash_attention.h"
#include "aclnnop/aclnn_matmul.h"
#include "aclnnop/aclnn_prompt_flash_attention.h"
#include "ascend.h"

namespace {

class AclTensorGuard {
public:
    explicit AclTensorGuard(aclTensor *tensor) : tensor_(tensor) {}
    ~AclTensorGuard()
    {
        if (tensor_ != nullptr) {
            (void)aclDestroyTensor(tensor_);
        }
    }
    aclTensor *get() const
    {
        return tensor_;
    }
    void release()
    {
        tensor_ = nullptr;
    }

private:
    aclTensor *tensor_;
};

class AclIntArrayGuard {
public:
    explicit AclIntArrayGuard(aclIntArray *array) : array_(array) {}
    ~AclIntArrayGuard()
    {
        if (array_ != nullptr) {
            (void)aclDestroyIntArray(array_);
        }
    }
    aclIntArray *get() const
    {
        return array_;
    }

private:
    aclIntArray *array_;
};

class AclTensorListGuard {
public:
    explicit AclTensorListGuard(aclTensorList *list) : list_(list) {}
    ~AclTensorListGuard()
    {
        if (list_ != nullptr) {
            (void)aclDestroyTensorList(list_);
        }
    }
    aclTensorList *get() const
    {
        return list_;
    }

private:
    aclTensorList *list_;
};

static aclTensor *CreateTensor(const std::vector<int64_t> &dims, const std::vector<int64_t> &strides,
                               aclDataType dtype, void *data,
                               const std::vector<int64_t> &storageDims = {})
{
    const std::vector<int64_t> &storage = storageDims.empty() ? dims : storageDims;
    aclTensor *tensor = aclCreateTensor(dims.data(), dims.size(), dtype, strides.data(), 0,
                                        ACL_FORMAT_ND, storage.data(), storage.size(), data);
    if (tensor == nullptr) {
        throw std::runtime_error("aclCreateTensor returned nullptr");
    }
    return tensor;
}

static std::vector<int64_t> ContiguousStrides(const std::vector<int64_t> &dims)
{
    std::vector<int64_t> strides(dims.size(), 1);
    for (size_t i = dims.size(); i > 1; --i) {
        strides[i - 2] = strides[i - 1] * dims[i - 1];
    }
    return strides;
}

static void ValidateFp16(const char *op, std::initializer_list<const XTensor *> tensors)
{
    for (const XTensor *tensor : tensors) {
        if (tensor->dtype != FP16) {
            throw std::runtime_error(std::string(op) + " on Ascend310P only supports FP16");
        }
    }
}

static XTensor *GetWorkspace(XRuntime &rt, uint64_t workspaceSize)
{
    if (workspaceSize == 0) {
        return nullptr;
    }
    if (workspaceSize > XLITE_310P_ACLNN_WORKSPACE_BYTES) {
        throw std::runtime_error("ACLNN workspace request " + std::to_string(workspaceSize) +
                                 " bytes exceeds the 512 MiB reserved 310P TensorPool budget");
    }
    return &rt.GetTensor({static_cast<size_t>(workspaceSize)}, INT8, DBG_LOC);
}

static void FinishAclnn(XRuntime &rt, XTensor *workspace)
{
    // Correctness POC: synchronize before ACL descriptor destruction and before
    // returning workspace storage to TensorPool. Performance work may remove this.
    CHECK_ACL(aclrtSynchronizeStream(rt.stream));
    if (workspace != nullptr) {
        rt.PutTensor(*workspace);
    }
}

static void ReadAttentionMetadata(XRuntime &rt, XTensor &lens, XTensor &cachedLens,
                                  XTensor &blockTables, uint32_t maxNumBlock, uint32_t batch)
{
    if (batch != 1) {
        throw std::runtime_error("Ascend310P attention POC requires batch=1");
    }
    rt.Synchronize();
    rt._lensHost.resize(1);
    rt._cachedLensHost.resize(1);
    rt._blockTablesHost.resize(maxNumBlock);
    rt.MemcpyD2H(rt._lensHost.data(), lens.ptr, sizeof(uint32_t));
    rt.MemcpyD2H(rt._cachedLensHost.data(), cachedLens.ptr, sizeof(uint32_t));
    rt.MemcpyD2H(rt._blockTablesHost.data(), blockTables.ptr,
                 static_cast<size_t>(maxNumBlock) * sizeof(uint32_t));
}

static uint32_t RequireContiguousBlocks(const XRuntime &rt, uint32_t totalLength,
                                        uint32_t blockSize)
{
    const uint32_t blocks = (totalLength + blockSize - 1) / blockSize;
    if (rt._blockTablesHost.size() < blocks) {
        throw std::runtime_error("Ascend310P block table is smaller than the valid KV length");
    }
    const uint32_t firstBlock = rt._blockTablesHost[0];
    for (uint32_t i = 0; i < blocks; ++i) {
        if (rt._blockTablesHost[i] != firstBlock + i) {
            throw std::runtime_error(
                "Ascend310P correctness backend currently requires contiguous physical KV blocks");
        }
    }
    return firstBlock;
}

static XTensor &ExtractQuery(XRuntime &rt, XTensor &qkv, uint32_t tokens, uint32_t nHeads,
                             uint32_t nKvHeads, uint32_t headDim)
{
    const size_t qElements = static_cast<size_t>(nHeads) * headDim;
    const size_t rowElements = static_cast<size_t>(nHeads + 2 * nKvHeads) * headDim;
    XTensor &query = rt.GetTensor({tokens, qElements}, FP16, DBG_LOC);
    const size_t qBytes = qElements * sizeof(uint16_t);
    const size_t rowBytes = rowElements * sizeof(uint16_t);
    CHECK_ACL(aclrtMemcpy2dAsync(query.ptr, qBytes, qkv.ptr, rowBytes, qBytes, tokens,
                                ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
    return query;
}

}  // namespace

void XliteAclnn310PMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out,
                          bool weightNZ, const XTensor &bias, const XTensor &deqScale,
                          bool transpose)
{
    ValidateFp16("aclnnMatmul", {&in, &weight, &out});
    if (weightNZ || bias.ptr != nullptr || deqScale.ptr != nullptr) {
        throw std::runtime_error(
            "aclnnMatmul Ascend310P POC requires ND weight, no bias and no dequant scale");
    }
    if (in.shape.size() != 2 || weight.shape.size() != 2 || out.shape.size() != 2) {
        throw std::runtime_error("aclnnMatmul Ascend310P POC requires rank-2 tensors");
    }

    const int64_t m = static_cast<int64_t>(in.shape[0]);
    const int64_t k = static_cast<int64_t>(in.shape[1]);
    const int64_t n = static_cast<int64_t>(transpose ? weight.shape[1] : weight.shape[0]);
    const int64_t weightK = static_cast<int64_t>(transpose ? weight.shape[0] : weight.shape[1]);
    if (weightK != k || out.shape[0] != static_cast<size_t>(m) ||
        out.shape[1] != static_cast<size_t>(n)) {
        throw std::runtime_error("aclnnMatmul Ascend310P shape mismatch");
    }

    const auto runMatmul = [&](int64_t currentN, void *weightData, void *outputData,
                               const std::vector<int64_t> &weightStorage,
                               const std::vector<int64_t> &weightStrides) {
        const std::vector<int64_t> inDims{m, k};
        const std::vector<int64_t> weightDims{k, currentN};
        const std::vector<int64_t> outDims{m, currentN};
        AclTensorGuard aclIn(
            CreateTensor(inDims, ContiguousStrides(inDims), ACL_FLOAT16, in.ptr));
        AclTensorGuard aclWeight(CreateTensor(weightDims, weightStrides, ACL_FLOAT16, weightData,
                                              weightStorage));
        AclTensorGuard aclOut(
            CreateTensor(outDims, ContiguousStrides(outDims), ACL_FLOAT16, outputData));

        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        CHECK_ACL(aclnnMatmulGetWorkspaceSize(aclIn.get(), aclWeight.get(), aclOut.get(), 0,
                                              &workspaceSize, &executor));
        XTensor *workspace = GetWorkspace(rt, workspaceSize);
        CHECK_ACL(aclnnMatmul(workspace == nullptr ? nullptr : workspace->ptr, workspaceSize,
                              executor, rt.stream));
        FinishAclnn(rt, workspace);
    };

    if (!transpose && n > static_cast<int64_t>(XLITE_310P_MATMUL_N_CHUNK)) {
        const int64_t chunkLimit = static_cast<int64_t>(XLITE_310P_MATMUL_N_CHUNK);
        // A contiguous temporary is necessary because an N slice of [M,N] is
        // strided for M>1.  Copy each completed chunk into the final output.
        XTensor &chunkOutput =
            rt.GetTensor({static_cast<size_t>(m), XLITE_310P_MATMUL_N_CHUNK}, FP16, DBG_LOC);
        for (int64_t offset = 0; offset < n; offset += chunkLimit) {
            const int64_t currentN = std::min(chunkLimit, n - offset);
            void *weightData = static_cast<void *>(
                static_cast<uint16_t *>(weight.ptr) + static_cast<size_t>(offset * k));
            runMatmul(currentN, weightData, chunkOutput.ptr, {currentN, k}, {1, k});
            void *outputData = static_cast<void *>(
                static_cast<uint16_t *>(out.ptr) + static_cast<size_t>(offset));
            CHECK_ACL(aclrtMemcpy2dAsync(
                outputData, static_cast<size_t>(n) * sizeof(uint16_t), chunkOutput.ptr,
                static_cast<size_t>(currentN) * sizeof(uint16_t),
                static_cast<size_t>(currentN) * sizeof(uint16_t), static_cast<size_t>(m),
                ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
        CHECK_ACL(aclrtSynchronizeStream(rt.stream));
        rt.PutTensor(chunkOutput);
        return;
    }

    const std::vector<int64_t> weightStorage{
        static_cast<int64_t>(weight.shape[0]), static_cast<int64_t>(weight.shape[1])};
    // Xlite linear weights are normally [N,K], hence the logical [K,N] view.
    const std::vector<int64_t> weightStrides =
        transpose ? std::vector<int64_t>{n, 1} : std::vector<int64_t>{1, k};
    runMatmul(n, weight.ptr, out.ptr, weightStorage, weightStrides);
}

void XliteAclnn310PAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                             XTensor &output, XTensor &lens, XTensor &cachedLens,
                             XTensor &blockTables, uint32_t maxNumBlock,
                             uint32_t nHeads, uint32_t nKvHeads,
                             uint32_t headDim, uint32_t blockSize, uint32_t batch,
                             bool decode)
{
    ValidateFp16("ACLNN attention", {&qkv, &kCache, &vCache, &output});
    ReadAttentionMetadata(rt, lens, cachedLens, blockTables, maxNumBlock, batch);
    if (headDim != 128 || nHeads != 16 || nKvHeads != 8 || blockSize != 128) {
        throw std::runtime_error(
            "Ascend310P attention POC is fixed to head_dim=128, Q16/KV8 and block_size=128");
    }
    const uint32_t queryLength = rt._lensHost[0];
    const uint32_t cachedLength = rt._cachedLensHost[0];
    const uint32_t totalLength = queryLength + cachedLength;
    (void)decode;
    const bool isDecode = queryLength == 1 && cachedLength > 0;
    const uint32_t firstBlock = RequireContiguousBlocks(rt, totalLength, blockSize);
    if (kCache.shape.empty() || vCache.shape.empty() ||
        static_cast<size_t>(firstBlock) + (totalLength + blockSize - 1) / blockSize >
            kCache.shape[0] || kCache.shape[0] != vCache.shape[0]) {
        throw std::runtime_error("Ascend310P KV block table points outside cache storage");
    }

    XTensor &query = ExtractQuery(rt, qkv, queryLength, nHeads, nKvHeads, headDim);
    const std::vector<int64_t> qDims{1, queryLength, nHeads, headDim};
    const std::vector<int64_t> kvDims{1, totalLength, nKvHeads, headDim};
    const std::vector<int64_t> outDims{1, queryLength, nHeads, headDim};
    AclTensorGuard aclQuery(
        CreateTensor(qDims, ContiguousStrides(qDims), ACL_FLOAT16, query.ptr));
    const size_t blockElements = static_cast<size_t>(blockSize) * nKvHeads * headDim;
    const size_t byteOffset = static_cast<size_t>(firstBlock) * blockElements * sizeof(uint16_t);
    void *keyData = reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(kCache.ptr) + byteOffset);
    void *valueData =
        reinterpret_cast<void *>(reinterpret_cast<uintptr_t>(vCache.ptr) + byteOffset);
    AclTensorGuard aclKey(CreateTensor(
        kvDims, ContiguousStrides(kvDims), ACL_FLOAT16, keyData,
        kvDims));
    AclTensorGuard aclValue(CreateTensor(
        kvDims, ContiguousStrides(kvDims), ACL_FLOAT16, valueData,
        kvDims));
    AclTensorGuard aclOut(
        CreateTensor(outDims, ContiguousStrides(outDims), ACL_FLOAT16, output.ptr));
    // Xlite's RoPE-and-Cache kernel already scales Q by 1/sqrt(head_dim).
    // ACLNN must therefore use identity scaling or attention would be scaled twice.
    const double scale = 1.0;
    const int64_t actualLengthValue = totalLength;
    AclIntArrayGuard actualLengths(aclCreateIntArray(&actualLengthValue, 1));
    if (actualLengths.get() == nullptr) {
        throw std::runtime_error("aclCreateIntArray returned nullptr");
    }

    uint64_t workspaceSize = 0;
    aclOpExecutor *executor = nullptr;
    XTensor *workspace = nullptr;
    if (isDecode) {
        const aclTensor *keyItems[] = {aclKey.get()};
        const aclTensor *valueItems[] = {aclValue.get()};
        AclTensorListGuard keys(aclCreateTensorList(keyItems, 1));
        if (keys.get() == nullptr) {
            throw std::runtime_error("aclCreateTensorList returned nullptr");
        }
        aclKey.release();
        AclTensorListGuard values(aclCreateTensorList(valueItems, 1));
        if (values.get() == nullptr) {
            throw std::runtime_error("aclCreateTensorList returned nullptr");
        }
        // aclDestroyTensorList also destroys its member aclTensor descriptors.
        aclValue.release();
        CHECK_ACL(aclnnIncreFlashAttentionGetWorkspaceSize(
            aclQuery.get(), keys.get(), values.get(), nullptr, nullptr, actualLengths.get(), nHeads,
            scale, const_cast<char *>("BSND"), nKvHeads, aclOut.get(), &workspaceSize, &executor));
        workspace = GetWorkspace(rt, workspaceSize);
        CHECK_ACL(aclnnIncreFlashAttention(workspace == nullptr ? nullptr : workspace->ptr,
                                           workspaceSize, executor, rt.stream));
        FinishAclnn(rt, workspace);
    } else {
        // The cache contains all prompt K/V after RoPE-and-Cache.  On Atlas inference
        // products preTokens/nextTokens are ignored when attenMask is nullptr, so a
        // real BOOL upper-triangular mask is required for causal prefill.
        if (cachedLength != 0) {
            throw std::runtime_error(
                "Ascend310P PromptFlashAttention POC does not support chunked prefill");
        }
        constexpr uint32_t BOOL_MASK_ALIGNMENT = 32;
        const uint32_t maskKvLength =
            (totalLength + BOOL_MASK_ALIGNMENT - 1) / BOOL_MASK_ALIGNMENT * BOOL_MASK_ALIGNMENT;
        const size_t maskElements =
            static_cast<size_t>(queryLength) * static_cast<size_t>(maskKvLength);
        // Padded columns must also be masked.  Clear only the causal prefix in
        // each row, leaving the future and padding columns set to true.
        std::vector<uint8_t> hostMask(maskElements, 1);
        for (uint32_t row = 0; row < queryLength; ++row) {
            for (uint32_t col = 0; col <= row; ++col) {
                hostMask[static_cast<size_t>(row) * maskKvLength + col] = 0;
            }
        }
        XTensor &causalMask = rt.GetTensor({queryLength, maskKvLength}, INT8, DBG_LOC);
        rt.MemcpyH2D(causalMask.ptr, hostMask.data(), maskElements);
        const std::vector<int64_t> maskDims{queryLength, maskKvLength};
        AclTensorGuard aclMask(CreateTensor(maskDims, ContiguousStrides(maskDims), ACL_BOOL,
                                            causalMask.ptr));
        std::vector<int64_t> queryLengths{static_cast<int64_t>(queryLength)};
        AclIntArrayGuard actualQueryLengths(
            aclCreateIntArray(queryLengths.data(), queryLengths.size()));
        if (actualQueryLengths.get() == nullptr) {
            throw std::runtime_error("aclCreateIntArray returned nullptr");
        }
        CHECK_ACL(aclnnPromptFlashAttentionGetWorkspaceSize(
            aclQuery.get(), aclKey.get(), aclValue.get(), nullptr, aclMask.get(),
            actualQueryLengths.get(), nHeads, scale, 2147483647, 0, const_cast<char *>("BSND"),
            nKvHeads, aclOut.get(), &workspaceSize, &executor));
        workspace = GetWorkspace(rt, workspaceSize);
        CHECK_ACL(aclnnPromptFlashAttention(workspace == nullptr ? nullptr : workspace->ptr,
                                            workspaceSize, executor, rt.stream));
        FinishAclnn(rt, workspace);
        rt.PutTensor(causalMask);
    }
    rt.PutTensor(query);
}

#pragma GCC diagnostic pop
