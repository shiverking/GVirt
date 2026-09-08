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
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
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
    if (batch == 0 || batch > XLITE_310P_MAX_BATCH) {
        throw std::runtime_error("Ascend310P attention requires batch in [1, " +
                                 std::to_string(XLITE_310P_MAX_BATCH) + "]");
    }
    if (maxNumBlock == 0) {
        throw std::runtime_error("Ascend310P attention requires a non-empty block table");
    }
    if (lens.numel < batch || cachedLens.numel < batch ||
        blockTables.numel < static_cast<size_t>(batch) * maxNumBlock) {
        throw std::runtime_error("Ascend310P attention metadata tensor is smaller than batch");
    }
    rt.Synchronize();
    rt._lensHost.resize(batch);
    rt._cachedLensHost.resize(batch);
    rt._blockTablesHost.resize(static_cast<size_t>(batch) * maxNumBlock);
    rt.MemcpyD2H(rt._lensHost.data(), lens.ptr, static_cast<size_t>(batch) * sizeof(uint32_t));
    rt.MemcpyD2H(rt._cachedLensHost.data(), cachedLens.ptr,
                 static_cast<size_t>(batch) * sizeof(uint32_t));
    rt.MemcpyD2H(rt._blockTablesHost.data(), blockTables.ptr,
                 static_cast<size_t>(batch) * maxNumBlock * sizeof(uint32_t));
}

static std::vector<uint32_t> GetRequestBlocks(const XRuntime &rt, uint32_t request,
                                              uint32_t totalLength, uint32_t blockSize,
                                              uint32_t maxNumBlock, size_t cacheBlocks)
{
    const uint32_t blocks = (totalLength + blockSize - 1) / blockSize;
    if (blocks > maxNumBlock) {
        throw std::runtime_error(
            "Ascend310P block table is smaller than the valid KV length: total_kv=" +
            std::to_string(totalLength) + ", required_blocks=" + std::to_string(blocks) +
            ", supplied_blocks=" + std::to_string(maxNumBlock));
    }
    std::vector<uint32_t> result;
    result.reserve(blocks);
    const size_t rowOffset = static_cast<size_t>(request) * maxNumBlock;
    for (uint32_t i = 0; i < blocks; ++i) {
        const uint32_t physicalBlock = rt._blockTablesHost[rowOffset + i];
        if (physicalBlock >= cacheBlocks) {
            throw std::runtime_error("Ascend310P KV block table points outside cache storage");
        }
        result.push_back(physicalBlock);
    }
    return result;
}

static XTensor &ExtractQuery(XRuntime &rt, XTensor &qkv, size_t rowOffset, uint32_t tokens,
                             uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim)
{
    const size_t qElements = static_cast<size_t>(nHeads) * headDim;
    const size_t rowElements = static_cast<size_t>(nHeads + 2 * nKvHeads) * headDim;
    XTensor &query = rt.GetTensor({tokens, qElements}, FP16, DBG_LOC);
    const size_t qBytes = qElements * sizeof(uint16_t);
    const size_t rowBytes = rowElements * sizeof(uint16_t);
    void *source = static_cast<void *>(static_cast<uint8_t *>(qkv.ptr) + rowOffset * rowBytes);
    CHECK_ACL(aclrtMemcpy2dAsync(query.ptr, qBytes, source, rowBytes, qBytes, tokens,
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
    (void)decode;
    if (kCache.shape.size() != 4 || vCache.shape != kCache.shape ||
        kCache.shape[1] != blockSize || kCache.shape[2] != nKvHeads ||
        kCache.shape[3] != headDim) {
        throw std::runtime_error(
            "Ascend310P attention requires identical 4D BSHD K/V caches");
    }
    const size_t qElements = static_cast<size_t>(nHeads) * headDim;
    const size_t rowElements = static_cast<size_t>(nHeads + 2 * nKvHeads) * headDim;
    size_t totalQueryTokens = 0;
    for (uint32_t request = 0; request < batch; ++request) {
        totalQueryTokens += rt._lensHost[request];
    }
    if (qkv.numel < totalQueryTokens * rowElements ||
        output.numel < totalQueryTokens * qElements) {
        throw std::runtime_error("Ascend310P attention QKV/output is smaller than lens sum");
    }

    size_t queryOffset = 0;
    const size_t tokenKvBytes = static_cast<size_t>(nKvHeads) * headDim * sizeof(uint16_t);
    const size_t blockKvBytes = static_cast<size_t>(blockSize) * tokenKvBytes;
    for (uint32_t request = 0; request < batch; ++request) {
        const uint32_t queryLength = rt._lensHost[request];
        const uint32_t cachedLength = rt._cachedLensHost[request];
        const uint32_t totalLength = queryLength + cachedLength;
        if (queryLength == 0 || totalLength > XLITE_310P_MAX_SEQ_LEN) {
            throw std::runtime_error("Ascend310P attention request length is outside [1, " +
                                     std::to_string(XLITE_310P_MAX_SEQ_LEN) + "]");
        }
        const bool isDecode = queryLength == 1 && cachedLength > 0;
        const std::vector<uint32_t> blocks = GetRequestBlocks(
            rt, request, totalLength, blockSize, maxNumBlock, kCache.shape[0]);
        XTensor &query = ExtractQuery(rt, qkv, queryOffset, queryLength,
                                      nHeads, nKvHeads, headDim);

        const uint32_t kvTensorLength =
            isDecode ? totalLength : ROUND_UP(totalLength, blockSize);
        // CANN 9.1 PromptFlashAttention on 310P rejects the GQA chunked-
        // prefill tiling when padded S1 and S2 differ (561103). Pad Q to the
        // total-KV bucket as well. The mask retains chunk semantics and only
        // the real query rows are copied back, so cached tokens do not create
        // synthetic query outputs.
        const uint32_t queryTensorLength = isDecode ? queryLength : kvTensorLength;
        XTensor *paddedQuery = nullptr;
        XTensor *paddedOutput = nullptr;
        void *queryData = query.ptr;
        void *outputData = static_cast<void *>(
            static_cast<uint8_t *>(output.ptr) + queryOffset * qElements * sizeof(uint16_t));
        if (queryTensorLength != queryLength) {
            paddedQuery = &rt.GetTensor({queryTensorLength, qElements}, FP16, DBG_LOC);
            paddedOutput = &rt.GetTensor({queryTensorLength, qElements}, FP16, DBG_LOC);
            CHECK_ACL(aclrtMemsetAsync(paddedQuery->ptr, paddedQuery->bytes, 0,
                                       paddedQuery->bytes, rt.stream));
            CHECK_ACL(aclrtMemcpyAsync(paddedQuery->ptr, paddedQuery->bytes, query.ptr,
                                       query.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            queryData = paddedQuery->ptr;
            outputData = paddedOutput->ptr;
        }

        XTensor &packedKey =
            rt.GetTensor({kvTensorLength, nKvHeads, headDim}, FP16, DBG_LOC);
        XTensor &packedValue =
            rt.GetTensor({kvTensorLength, nKvHeads, headDim}, FP16, DBG_LOC);
        if (kvTensorLength != totalLength) {
            CHECK_ACL(aclrtMemsetAsync(packedKey.ptr, packedKey.bytes, 0,
                                       packedKey.bytes, rt.stream));
            CHECK_ACL(aclrtMemsetAsync(packedValue.ptr, packedValue.bytes, 0,
                                       packedValue.bytes, rt.stream));
        }
        for (size_t logicalBlock = 0; logicalBlock < blocks.size(); ++logicalBlock) {
            const uint32_t tokenOffset = static_cast<uint32_t>(logicalBlock) * blockSize;
            const uint32_t validTokens = std::min(blockSize, totalLength - tokenOffset);
            const size_t copyBytes = static_cast<size_t>(validTokens) * tokenKvBytes;
            const size_t sourceOffset = static_cast<size_t>(blocks[logicalBlock]) * blockKvBytes;
            const size_t destinationOffset = static_cast<size_t>(tokenOffset) * tokenKvBytes;
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(packedKey.ptr) + destinationOffset,
                                       packedKey.bytes - destinationOffset,
                                       static_cast<uint8_t *>(kCache.ptr) + sourceOffset,
                                       copyBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(packedValue.ptr) + destinationOffset,
                                       packedValue.bytes - destinationOffset,
                                       static_cast<uint8_t *>(vCache.ptr) + sourceOffset,
                                       copyBytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }

        const std::vector<int64_t> qDims{1, queryTensorLength, nHeads, headDim};
        const std::vector<int64_t> kvDims{1, kvTensorLength, nKvHeads, headDim};
        const std::vector<int64_t> outDims{1, queryTensorLength, nHeads, headDim};
        AclTensorGuard aclQuery(
            CreateTensor(qDims, ContiguousStrides(qDims), ACL_FLOAT16, queryData));
        AclTensorGuard aclKey(CreateTensor(kvDims, ContiguousStrides(kvDims), ACL_FLOAT16,
                                           packedKey.ptr));
        AclTensorGuard aclValue(CreateTensor(kvDims, ContiguousStrides(kvDims), ACL_FLOAT16,
                                             packedValue.ptr));
        AclTensorGuard aclOut(
            CreateTensor(outDims, ContiguousStrides(outDims), ACL_FLOAT16, outputData));
        // RoPE-and-Cache already scales Q by 1/sqrt(head_dim).
        const double scale = 1.0;
        uint64_t workspaceSize = 0;
        aclOpExecutor *executor = nullptr;
        XTensor *workspace = nullptr;
        if (!isDecode) {
            const size_t maskElements =
                static_cast<size_t>(queryTensorLength) * kvTensorLength;
            std::vector<uint8_t> hostMask(maskElements, 1);
            for (uint32_t row = 0; row < queryLength; ++row) {
                const uint32_t visibleKeys = cachedLength + row + 1;
                for (uint32_t col = 0; col < visibleKeys; ++col) {
                    hostMask[static_cast<size_t>(row) * kvTensorLength + col] = 0;
                }
            }
            for (uint32_t row = queryLength; row < queryTensorLength; ++row) {
                hostMask[static_cast<size_t>(row) * kvTensorLength] = 0;
            }
            XTensor &causalMask =
                rt.GetTensor({queryTensorLength, kvTensorLength}, INT8, DBG_LOC);
            rt.MemcpyH2D(causalMask.ptr, hostMask.data(), maskElements);
            const std::vector<int64_t> maskDims{queryTensorLength, kvTensorLength};
            AclTensorGuard aclMask(CreateTensor(maskDims, ContiguousStrides(maskDims), ACL_BOOL,
                                                causalMask.ptr));
            // The explicit mask already encodes the cached-token offset:
            // row i may attend through cachedLength + i. nextTokens=0 would
            // additionally apply PromptFA's unshifted causal window after Q
            // is padded to the KV bucket, incorrectly hiding cached keys.
            CHECK_ACL(aclnnPromptFlashAttentionGetWorkspaceSize(
                aclQuery.get(), aclKey.get(), aclValue.get(), nullptr, aclMask.get(), nullptr,
                nHeads, scale, 2147483647, 2147483647, const_cast<char *>("BSND"), nKvHeads,
                aclOut.get(), &workspaceSize, &executor));
            workspace = GetWorkspace(rt, workspaceSize);
            CHECK_ACL(aclnnPromptFlashAttention(workspace == nullptr ? nullptr : workspace->ptr,
                                                workspaceSize, executor, rt.stream));
            FinishAclnn(rt, workspace);
            rt.PutTensor(causalMask);
        } else {
            CHECK_ACL(aclnnPromptFlashAttentionGetWorkspaceSize(
                aclQuery.get(), aclKey.get(), aclValue.get(), nullptr, nullptr, nullptr,
                nHeads, scale, 2147483647, 2147483647, const_cast<char *>("BSND"), nKvHeads,
                aclOut.get(), &workspaceSize, &executor));
            workspace = GetWorkspace(rt, workspaceSize);
            CHECK_ACL(aclnnPromptFlashAttention(workspace == nullptr ? nullptr : workspace->ptr,
                                                workspaceSize, executor, rt.stream));
            FinishAclnn(rt, workspace);
        }
        if (paddedOutput != nullptr) {
            const size_t validOutputBytes =
                static_cast<size_t>(queryLength) * qElements * sizeof(uint16_t);
            void *destination = static_cast<void *>(static_cast<uint8_t *>(output.ptr) +
                                                    queryOffset * qElements * sizeof(uint16_t));
            CHECK_ACL(aclrtMemcpyAsync(destination, output.bytes - queryOffset * qElements *
                                                       sizeof(uint16_t),
                                       paddedOutput->ptr, validOutputBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            CHECK_ACL(aclrtSynchronizeStream(rt.stream));
            rt.PutTensor(*paddedOutput);
            rt.PutTensor(*paddedQuery);
        }
        rt.PutTensor(packedValue);
        rt.PutTensor(packedKey);
        rt.PutTensor(query);
        queryOffset += queryLength;
    }
}

#pragma GCC diagnostic pop
