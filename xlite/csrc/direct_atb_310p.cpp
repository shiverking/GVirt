/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "direct_atb_310p.h"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/operation.h>
#include <atb/types.h>

#include <initializer_list>
#include <sstream>
#include <stdexcept>

namespace {
void CheckAtb(atb::Status status, const char *stage)
{
    if (status != atb::NO_ERROR) {
        std::ostringstream message;
        message << "310P direct ATB " << stage << " failed, status="
                << static_cast<int64_t>(status);
        throw std::runtime_error(message.str());
    }
}

atb::Tensor MakeTensor(void *deviceData, aclDataType dtype, aclFormat format,
                       std::initializer_list<int64_t> dims)
{
    atb::Tensor tensor{};
    tensor.deviceData = deviceData;
    tensor.desc.dtype = dtype;
    tensor.desc.format = format;
    tensor.desc.shape.dimNum = static_cast<uint64_t>(dims.size());
    size_t elements = 1;
    size_t index = 0;
    for (int64_t dim : dims) {
        tensor.desc.shape.dims[index++] = dim;
        elements *= static_cast<size_t>(dim);
    }
    size_t elementBytes = 0;
    switch (dtype) {
        case ACL_FLOAT16:
            elementBytes = sizeof(uint16_t);
            break;
        case ACL_INT32:
            elementBytes = sizeof(int32_t);
            break;
        default:
            throw std::invalid_argument("310P direct ATB received an unsupported dtype");
    }
    tensor.dataSize = elements * elementBytes;
    return tensor;
}

void RunOperation(atb::Operation *operation, atb::VariantPack &pack, atb::Context *context,
                  const XliteDirectAtb310P::WorkspaceAcquire &acquire,
                  const XliteDirectAtb310P::WorkspaceRelease &release, uint64_t &setupCount,
                  uint64_t &executeCount)
{
    uint64_t workspaceSize = 0;
    CheckAtb(operation->Setup(pack, workspaceSize, context), "Setup");
    ++setupCount;
    void *workspace = workspaceSize == 0 ? nullptr : acquire(static_cast<size_t>(workspaceSize));
    if (workspaceSize != 0 && workspace == nullptr) {
        throw std::runtime_error("310P direct ATB workspace allocator returned null");
    }
    try {
        CheckAtb(operation->Execute(pack, static_cast<uint8_t *>(workspace), workspaceSize, context),
                 "Execute");
        ++executeCount;
    } catch (...) {
        if (workspace != nullptr) {
            release(workspace);
        }
        throw;
    }
    if (workspace != nullptr) {
        release(workspace);
    }
}
}  // namespace

class XliteDirectAtb310P::Impl
{
public:
    Impl()
    {
        CheckAtb(atb::CreateContext(&context), "CreateContext");

        atb::infer::ReshapeAndCacheParam reshapeParam{};
        CheckAtb(atb::CreateOperation(reshapeParam, &reshape), "CreateReshapeAndCacheOperation");

        atb::infer::PagedAttentionParam pagedParam{};
        pagedParam.headNum = 16;
        pagedParam.kvHeadNum = 8;
        // RoPE-and-cache scales Q before this operation.
        pagedParam.qkScale = 1.0F;
        CheckAtb(atb::CreateOperation(pagedParam, &paged), "CreatePagedAttentionOperation");
    }

    ~Impl()
    {
        if (paged != nullptr) {
            (void)atb::DestroyOperation(paged);
        }
        if (reshape != nullptr) {
            (void)atb::DestroyOperation(reshape);
        }
        if (context != nullptr) {
            (void)atb::DestroyContext(context);
        }
    }

    atb::Context *context = nullptr;
    atb::Operation *reshape = nullptr;
    atb::Operation *paged = nullptr;
    aclrtStream stream = nullptr;
    uint64_t setupCount = 0;
    uint64_t executeCount = 0;
};

XliteDirectAtb310P::XliteDirectAtb310P() : impl_(std::make_unique<Impl>()) {}
XliteDirectAtb310P::~XliteDirectAtb310P() = default;

void XliteDirectAtb310P::SetStream(aclrtStream stream)
{
    if (stream == nullptr) {
        throw std::invalid_argument("310P direct ATB requires a valid Xlite stream");
    }
    if (impl_->stream == stream) {
        return;
    }
    CheckAtb(impl_->context->SetExecuteStream(stream), "SetExecuteStream");
    impl_->stream = stream;
}

void XliteDirectAtb310P::ReshapeAndCache(
    void *key, void *value, uint32_t tokens, void *keyCache, void *valueCache,
    uint32_t cacheBlocks, void *slots, const WorkspaceAcquire &acquire,
    const WorkspaceRelease &release)
{
    atb::VariantPack pack;
    pack.inTensors = {
        MakeTensor(key, ACL_FLOAT16, ACL_FORMAT_ND, {tokens, 8, 128}),
        MakeTensor(value, ACL_FLOAT16, ACL_FORMAT_ND, {tokens, 8, 128}),
        MakeTensor(keyCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16}),
        MakeTensor(valueCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16}),
        MakeTensor(slots, ACL_INT32, ACL_FORMAT_ND, {tokens}),
    };
    pack.outTensors = {pack.inTensors[2], pack.inTensors[3]};
    RunOperation(impl_->reshape, pack, impl_->context, acquire, release, impl_->setupCount,
                 impl_->executeCount);
}

void XliteDirectAtb310P::PagedAttention(
    void *query, uint32_t batch, void *keyCache, void *valueCache, uint32_t cacheBlocks,
    void *blockTables, uint32_t tableColumns, void *contextLens, void *output,
    const WorkspaceAcquire &acquire, const WorkspaceRelease &release)
{
    atb::VariantPack pack;
    pack.inTensors = {
        MakeTensor(query, ACL_FLOAT16, ACL_FORMAT_ND, {batch, 16, 128}),
        MakeTensor(keyCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16}),
        MakeTensor(valueCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16}),
        MakeTensor(blockTables, ACL_INT32, ACL_FORMAT_ND, {batch, tableColumns}),
        MakeTensor(contextLens, ACL_INT32, ACL_FORMAT_ND, {batch}),
    };
    pack.outTensors = {
        MakeTensor(output, ACL_FLOAT16, ACL_FORMAT_ND, {batch, 16, 128}),
    };
    RunOperation(impl_->paged, pack, impl_->context, acquire, release, impl_->setupCount,
                 impl_->executeCount);
}

uint64_t XliteDirectAtb310P::SetupCount() const
{
    return impl_->setupCount;
}

uint64_t XliteDirectAtb310P::ExecuteCount() const
{
    return impl_->executeCount;
}
