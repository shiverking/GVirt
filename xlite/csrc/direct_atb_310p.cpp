/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "direct_atb_310p.h"

#include <acl/acl.h>
#include <atb/atb_infer.h>
#include <atb/operation.h>
#include <atb/types.h>

#include <array>
#include <initializer_list>
#include <sstream>
#include <stdexcept>
#include <vector>

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
    }

    ~Impl()
    {
        for (const std::unique_ptr<LayerPlan> &plan : layers) {
            if (plan != nullptr && plan->paged != nullptr) {
                (void)atb::DestroyOperation(plan->paged);
            }
            if (plan != nullptr && plan->reshape != nullptr) {
                (void)atb::DestroyOperation(plan->reshape);
            }
        }
        if (context != nullptr) {
            (void)atb::DestroyContext(context);
        }
    }

    struct Signature
    {
        std::array<void *, 7> pointers{};
        std::array<uint32_t, 3> dimensions{};
        bool valid = false;

        bool Update(const std::array<void *, 7> &newPointers,
                    const std::array<uint32_t, 3> &newDimensions)
        {
            const bool reused = valid && pointers == newPointers && dimensions == newDimensions;
            pointers = newPointers;
            dimensions = newDimensions;
            valid = true;
            return reused;
        }
    };

    struct LayerPlan
    {
        atb::Operation *reshape = nullptr;
        atb::Operation *paged = nullptr;
        atb::VariantPack reshapePack;
        atb::VariantPack pagedPack;
        Signature reshapeSignature;
        Signature pagedSignature;
    };

    LayerPlan &GetLayer(uint32_t layer)
    {
        if (layers.size() <= layer) {
            layers.resize(static_cast<size_t>(layer) + 1);
        }
        if (layers[layer] == nullptr) {
            layers[layer] = std::make_unique<LayerPlan>();
        }
        return *layers[layer];
    }

    atb::Context *context = nullptr;
    std::vector<std::unique_ptr<LayerPlan>> layers;
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

bool XliteDirectAtb310P::ReshapeAndCache(
    uint32_t layer, void *key, void *value, uint32_t tokens, void *keyCache, void *valueCache,
    uint32_t cacheBlocks, void *slots, const WorkspaceAcquire &acquire,
    const WorkspaceRelease &release)
{
    Impl::LayerPlan &plan = impl_->GetLayer(layer);
    if (plan.reshape == nullptr) {
        atb::infer::ReshapeAndCacheParam reshapeParam{};
        CheckAtb(atb::CreateOperation(reshapeParam, &plan.reshape),
                 "CreateReshapeAndCacheOperation");
    }
    atb::VariantPack &pack = plan.reshapePack;
    pack.inTensors.resize(5);
    pack.outTensors.resize(2);
    pack.inTensors[0] = MakeTensor(key, ACL_FLOAT16, ACL_FORMAT_ND, {tokens, 8, 128});
    pack.inTensors[1] = MakeTensor(value, ACL_FLOAT16, ACL_FORMAT_ND, {tokens, 8, 128});
    pack.inTensors[2] = MakeTensor(
        keyCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16});
    pack.inTensors[3] = MakeTensor(
        valueCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16});
    pack.inTensors[4] = MakeTensor(slots, ACL_INT32, ACL_FORMAT_ND, {tokens});
    pack.outTensors[0] = pack.inTensors[2];
    pack.outTensors[1] = pack.inTensors[3];
    const bool reused = plan.reshapeSignature.Update(
        {key, value, keyCache, valueCache, slots, nullptr, nullptr},
        {tokens, cacheBlocks, 0});
    RunOperation(plan.reshape, pack, impl_->context, acquire, release, impl_->setupCount,
                 impl_->executeCount);
    return reused;
}

bool XliteDirectAtb310P::PagedAttention(
    uint32_t layer, void *query, uint32_t batch, void *keyCache, void *valueCache,
    uint32_t cacheBlocks, void *blockTables, uint32_t tableColumns, void *contextLens, void *output,
    const WorkspaceAcquire &acquire, const WorkspaceRelease &release)
{
    Impl::LayerPlan &plan = impl_->GetLayer(layer);
    if (plan.paged == nullptr) {
        atb::infer::PagedAttentionParam pagedParam{};
        pagedParam.headNum = 16;
        pagedParam.kvHeadNum = 8;
        // RoPE-and-cache scales Q before this operation.
        pagedParam.qkScale = 1.0F;
        CheckAtb(atb::CreateOperation(pagedParam, &plan.paged),
                 "CreatePagedAttentionOperation");
    }
    atb::VariantPack &pack = plan.pagedPack;
    pack.inTensors.resize(5);
    pack.outTensors.resize(1);
    pack.inTensors[0] = MakeTensor(query, ACL_FLOAT16, ACL_FORMAT_ND, {batch, 16, 128});
    pack.inTensors[1] = MakeTensor(
        keyCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16});
    pack.inTensors[2] = MakeTensor(
        valueCache, ACL_FLOAT16, ACL_FORMAT_FRACTAL_NZ, {cacheBlocks, 64, 128, 16});
    pack.inTensors[3] =
        MakeTensor(blockTables, ACL_INT32, ACL_FORMAT_ND, {batch, tableColumns});
    pack.inTensors[4] = MakeTensor(contextLens, ACL_INT32, ACL_FORMAT_ND, {batch});
    pack.outTensors[0] = MakeTensor(output, ACL_FLOAT16, ACL_FORMAT_ND, {batch, 16, 128});
    const bool reused = plan.pagedSignature.Update(
        {query, keyCache, valueCache, blockTables, contextLens, output, nullptr},
        {batch, cacheBlocks, tableColumns});
    RunOperation(plan.paged, pack, impl_->context, acquire, release, impl_->setupCount,
                 impl_->executeCount);
    return reused;
}

uint64_t XliteDirectAtb310P::SetupCount() const
{
    return impl_->setupCount;
}

uint64_t XliteDirectAtb310P::ExecuteCount() const
{
    return impl_->executeCount;
}
