/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/torch.h>
#include <torch/extension.h>
#include <ATen/core/dispatch/Dispatcher.h>
#include <ATen/core/stack.h>
#include <iostream>
#include <limits>
#include <optional>
#include <sstream>
#include "xlite.h"
#include "core_assigner.h"
#include "op.h"
#include "auto_tuner.h"
#include "debug.h"
#ifdef XLITE_DIRECT_ATB_310P
#include "direct_atb_310p.h"
#endif

namespace py = pybind11;

#ifndef XLITE_BUILD_SOC
#define XLITE_BUILD_SOC "unknown"
#endif
#ifndef XLITE_BUILD_KERNEL_SET
#define XLITE_BUILD_KERNEL_SET "unknown"
#endif

struct CModelAttnMeta {
    std::vector<uint32_t> lens;
    std::vector<uint32_t> cachedLens;
    std::vector<std::vector<uint32_t>> blockTablesList;
    at::Tensor positions;
};

struct CModelAttnMetaV2 {
    std::vector<uint32_t> lensCpu;        // [batch] host
    std::vector<uint32_t> cachedLensCpu;  // [batch] host
    at::Tensor positions;                 // [batchedTokens] int64 device
    at::Tensor lens;                      // [batch] int32 device
    at::Tensor cachedLens;                // [batch] int32 device
    at::Tensor queryStartLoc;             // [batch] int32 device, prefix-sum of lens
    std::vector<at::Tensor> slotMapping;  // [batchedTokens] int32 device, per-kv-cache
    std::vector<at::Tensor> blockTables;  // [batch, maxNumBlocks] int32 device, per-kv-cache
};

class _CModel
{
public:
    _CModel() {};
    ~_CModel();
    void Init(struct XModelConfig &c, uint32_t rankId);
    void ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                   at::Tensor &output, uint64_t currStream);
    void ForwardV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                   std::vector<std::vector<at::Tensor>> &kvCache, std::vector<at::Tensor> &freqsCis,
                   at::Tensor &output, uint64_t currStream);
    void ForwardGetLogits(XRuntime &rt, at::Tensor &input, at::Tensor &indices, at::Tensor &output,
                          uint64_t currStream);
    void ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                               std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                               at::Tensor &indices, at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogitsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                               std::vector<std::vector<at::Tensor>> &kvCache,
                               std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                               at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput,
                                   const std::optional<at::Tensor> &inputIds);
    void ForwardWithInputsEmbedsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                   std::vector<std::vector<at::Tensor>> &kvCache,
                                   at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                   std::vector<at::Tensor> &deepstackInput,
                                   const std::optional<at::Tensor> &inputIds);
    size_t GetTensorPoolSize(int dbg);
    void SetNativeKvCache310P(std::vector<std::vector<at::Tensor>> &kvCache);

    enum XModelAttnType attnType = XMODEL_ATTN_MHA;

    // weights
    at::Tensor embed;
    at::Tensor norm;
    at::Tensor normBias;
    at::Tensor head;

    std::vector<at::Tensor> attnNorm;
    std::vector<at::Tensor> attnNormBias;
    std::vector<at::Tensor> attnOut;
    std::vector<at::Tensor> attnOutInputScale;
    std::vector<at::Tensor> attnOutInputOffset;
    std::vector<at::Tensor> attnOutQuantBias;
    std::vector<at::Tensor> attnOutDeqScale;
    std::vector<at::Tensor> mhaQKV;
    std::vector<at::Tensor> mhaQKVBias;
    std::vector<at::Tensor> mhaQKVInputScale;
    std::vector<at::Tensor> mhaQKVInputOffset;
    std::vector<at::Tensor> mhaQKVQuantBias;
    std::vector<at::Tensor> mhaQKVDeqScale;
    std::vector<at::Tensor> mhaQNorm;
    std::vector<at::Tensor> mhaQNormBias;
    std::vector<at::Tensor> mhaKNorm;
    std::vector<at::Tensor> mhaKNormBias;
    std::vector<at::Tensor> mlaQKVA;
    std::vector<at::Tensor> mlaQKVAInputScale;
    std::vector<at::Tensor> mlaQKVAInputOffset;
    std::vector<at::Tensor> mlaQKVAQuantBias;
    std::vector<at::Tensor> mlaQKVADeqScale;
    std::vector<at::Tensor> mlaQB;
    std::vector<at::Tensor> mlaQBInputScale;
    std::vector<at::Tensor> mlaQBInputOffset;
    std::vector<at::Tensor> mlaQBQuantBias;
    std::vector<at::Tensor> mlaQBDeqScale;
    std::vector<at::Tensor> mlaQNorm;
    std::vector<at::Tensor> mlaQNormBias;
    std::vector<at::Tensor> mlaWUV;
    std::vector<at::Tensor> mlaWUKT;
    std::vector<at::Tensor> mlaKVNorm;
    std::vector<at::Tensor> mlaKVNormBias;
    std::vector<at::Tensor> indexQB;
    std::vector<at::Tensor> indexQBInputScale;
    std::vector<at::Tensor> indexQBInputOffset;
    std::vector<at::Tensor> indexQBQuantBias;
    std::vector<at::Tensor> indexQBDeqScale;
    std::vector<at::Tensor> indexKWeightsProj;
    std::vector<at::Tensor> indexKNorm;
    std::vector<at::Tensor> indexKNormBias;

    std::vector<at::Tensor> linearInProjQKV;
    std::vector<at::Tensor> linearInProjZ;
    std::vector<at::Tensor> linearInProjB;
    std::vector<at::Tensor> linearInProjA;
    std::vector<at::Tensor> linearConv1d;
    std::vector<at::Tensor> linearALog;
    std::vector<at::Tensor> linearDtBias;
    std::vector<at::Tensor> linearNorm;
    std::vector<at::Tensor> linearOutProj;

    std::vector<at::Tensor> mlpNorm;
    std::vector<at::Tensor> mlpNormBias;
    std::vector<at::Tensor> mlpUpGate;
    std::vector<at::Tensor> mlpUpGateInputScale;
    std::vector<at::Tensor> mlpUpGateInputOffset;
    std::vector<at::Tensor> mlpUpGateQuantBias;
    std::vector<at::Tensor> mlpUpGateDeqScale;
    std::vector<at::Tensor> mlpDown;
    std::vector<at::Tensor> mlpDownInputScale;
    std::vector<at::Tensor> mlpDownInputOffset;
    std::vector<at::Tensor> mlpDownQuantBias;
    std::vector<at::Tensor> mlpDownDeqScale;

    std::vector<at::Tensor> moeGate;
    std::vector<at::Tensor> moeGateBias;
    std::vector<at::Tensor> moeTid2Eid;
    std::vector<at::Tensor> moeSEUpGate;
    std::vector<at::Tensor> moeSEUpGateDeqScale;
    std::vector<at::Tensor> moeSEDown;
    std::vector<at::Tensor> moeSEDownDeqScale;
    std::vector<at::Tensor> moeSEGate;
    std::vector<at::Tensor> moeREUpGate;
    std::vector<at::Tensor> moeREUpGateDeqScale;
    std::vector<at::Tensor> moeREDown;
    std::vector<at::Tensor> moeREDownDeqScale;
    std::vector<at::Tensor> moeREUpGateScaleBias;
    std::vector<at::Tensor> moeREDownScaleBias;

    // DeepSeek-V4 (CxA)
    std::vector<at::Tensor> attnSink;
    std::vector<at::Tensor> attnWqA;
    std::vector<at::Tensor> attnWqAInputScale;
    std::vector<at::Tensor> attnWqAInputOffset;
    std::vector<at::Tensor> attnWqAQuantBias;
    std::vector<at::Tensor> attnWqADeqScale;
    std::vector<at::Tensor> attnWoA;
    std::vector<at::Tensor> attnWoB;
    std::vector<at::Tensor> attnWKv;
    std::vector<at::Tensor> attnWKvInputScale;
    std::vector<at::Tensor> attnWKvInputOffset;
    std::vector<at::Tensor> attnWKvQuantBias;
    std::vector<at::Tensor> attnWKvDeqScale;
    std::vector<at::Tensor> compApe;
    std::vector<at::Tensor> compWKv;
    std::vector<at::Tensor> compWGate;
    std::vector<at::Tensor> compNorm;
    std::vector<at::Tensor> idxWqB;
    std::vector<at::Tensor> idxWqBInputScale;
    std::vector<at::Tensor> idxWqBInputOffset;
    std::vector<at::Tensor> idxWqBQuantBias;
    std::vector<at::Tensor> idxWqBDeqScale;
    std::vector<at::Tensor> idxWeightsProj;
    std::vector<at::Tensor> idxCompApe;
    std::vector<at::Tensor> idxCompWKv;
    std::vector<at::Tensor> idxCompWGate;
    std::vector<at::Tensor> idxCompNorm;
    std::vector<at::Tensor> hcAttnFn;
    std::vector<at::Tensor> hcFfnFn;
    std::vector<at::Tensor> hcAttnBase;
    std::vector<at::Tensor> hcFfnBase;
    std::vector<at::Tensor> hcAttnScale;
    std::vector<at::Tensor> hcFfnScale;
    at::Tensor hcHeadFn;
    at::Tensor hcHeadBase;
    at::Tensor hcHeadScale;

private:
    XModel *_model = nullptr;
    std::vector<std::vector<XTensor>> _kv;
    std::vector<std::vector<at::Tensor>> _nativeKv310P;
#ifdef XLITE_DIRECT_ATB_310P
    std::unique_ptr<XliteDirectAtb310P> _directAtb310P;
#endif
    at::Tensor _nativeQueryStage310P;
    at::Tensor _nativeKeyStage310P;
    at::Tensor _nativeValueStage310P;
    std::vector<at::Tensor> _nativeDecodeQueryStage310P;
    std::vector<at::Tensor> _nativeDecodeKeyStage310P;
    std::vector<at::Tensor> _nativeDecodeValueStage310P;
    std::vector<at::Tensor> _nativeDecodeOutputStage310P;
    std::vector<uint8_t> _directAtbRopeStaged310P;
    at::Tensor _nativeSlotStage310P;
    at::Tensor _nativeBlockTableStage310P;
    at::Tensor _nativeTotalLensStage310P;
    uint64_t _nativeMaxTokens310P = 0;
    uint64_t _nativeMaxBatch310P = 0;
    uint64_t _nativeMaxBlocks310P = 0;
    std::vector<XTensor> _deepstackInputEmbeds;
    void Forward(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                 std::vector<std::vector<at::Tensor>> &kvCache, std::vector<at::Tensor> &freqsCis,
                 at::Tensor &output, uint64_t currStream);
    void ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                             std::vector<std::vector<at::Tensor>> &kvCache,
                             std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                             at::Tensor &output, uint64_t currStream);
    void ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<at::Tensor>> &kvCache,
                                 at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                 std::vector<at::Tensor> &deepstackInput, at::Tensor &inputIds);
    void InitMatmulWeight(const std::string &name, std::vector<at::Tensor> &w,
                          std::vector<at::Tensor> &iScale, std::vector<at::Tensor> &iOffset,
                          std::vector<at::Tensor> &qBias, std::vector<at::Tensor> &dScale,
                          std::vector<MatmulWeight> &weightsXT, uint32_t currLayer,
                          bool isRowParallel, uint32_t tpRank, uint32_t layerOffset = 0);
#ifdef XLITE_ARCH_310P
    bool PrepareNativeAtbRopeStages310P(XRuntime &rt, XTensor &kCache, uint32_t tokens,
                                        void *&query, void *&key, void *&value);
    bool RunNativeAtbAttention310P(XRuntime &rt, XTensor &qkv, XTensor &kCache,
                                   XTensor &vCache, XTensor &output, XTensor &slotMapping,
                                   XTensor &blockTables, XTensor &totalLens, uint32_t nHeads,
                                   uint32_t nKvHeads, uint32_t headDim, uint32_t batch);
#endif
};

static bool TensorUsable(const at::Tensor &t)
{
    return t.defined() && t.numel() > 0;
}

static void InitOptionalXTensor(XTensor &out, at::Tensor &in)
{
    if (TensorUsable(in)) {
        InitXTensor(out, in);
    }
}

static std::vector<uint32_t> ResolveLayerTypes(const XModelConfig &c)
{
    std::vector<uint32_t> layerTypes;
    if (c.attnType != XMODEL_ATTN_HYBRID) {
        return layerTypes;
    }
    if (c.fullAttentionInterval == 0) {
        throw std::invalid_argument("fullAttentionInterval must be > 0 for hybrid attention");
    }
    layerTypes.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        layerTypes[i] = ((i + 1) % c.fullAttentionInterval == 0) ? XMODEL_LAYER_ATTN_FULL
                                                                 : XMODEL_LAYER_ATTN_LINEAR;
    }
    return layerTypes;
}

static bool IsConfigLayerFullAttention(XModelConfig &c, uint32_t layer,
                                       const std::vector<uint32_t> &layerTypes)
{
    if (c.attnType == XMODEL_ATTN_HYBRID) {
        return layerTypes[layer] == XMODEL_LAYER_ATTN_FULL;
    }
    return c.attnType == XMODEL_ATTN_MHA;
}

namespace
{
using TensorsRef = std::reference_wrapper<std::vector<at::Tensor>>;
struct LayersCheckEntry {
    TensorsRef weights;
    std::string name;
    bool allowEmpty;
    bool predicate;
};

void checkLayersDims(const std::vector<LayersCheckEntry> &table, uint32_t size, uint32_t rankId,
                     const std::string &prefix)
{
    for (const auto &entry : table) {
        const auto &w = entry.weights.get();
        if (!entry.predicate) {
            continue;
        }
        if (entry.allowEmpty && w.empty()) {
            continue;
        }

        if (w.size() != size) {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of layers: " << w.size() << std::endl;
            std::string msg = prefix + " " + entry.name + " parameters";
            throw std::invalid_argument(msg);
        }
    }
}
}  // namespace

void _CModel::Init(struct XModelConfig &c, uint32_t rankId)
{
    uint32_t idx = 0;
    uint32_t nLocalRoutedExperts = c.nRoutedExperts / c.moeEpSize;
    uint32_t expertsStartIdx = c.moeEpSize == 1 ? 0 : rankId / c.moeTPSize * nLocalRoutedExperts;
    uint32_t expertsEndIdx = expertsStartIdx + nLocalRoutedExperts;
    uint32_t numMoeLayers = c.nLayers - c.nDenseLayers;
    uint32_t nRE = numMoeLayers * nLocalRoutedExperts;
    uint32_t tpRank = rankId % c.defTpSize;
#ifdef XLITE_ARCH_310P
    _nativeMaxTokens310P = c.maxBatchedTokens;
    _nativeMaxBatch310P = c.maxBatch;
    _nativeMaxBlocks310P = (c.maxSeqLen + 127) / 128;
#endif

    if (c.nRoutedExperts % c.moeEpSize != 0) {
        {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of routed experts per expert parallel group: " << nLocalRoutedExperts
              << std::endl;
        }
        throw std::invalid_argument(
            "num of routed experts must be divisible by moe expert parallel size");
    }

    attnType = c.attnType;

    if (c.nLayers < c.nDenseLayers) {
        {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of layers: " << c.nLayers << ", num of dense layers: " << c.nDenseLayers
              << std::endl;
        }
        throw std::invalid_argument(
            "num of layers must be greater than or equal to num of dense layers");
    }

    if (c.attnType == XMODEL_ATTN_HYBRID) {
        if (c.linearNumKHeads == 0 || c.linearNumVHeads == 0 || c.linearKeyHeadDim == 0 ||
            c.linearValueHeadDim == 0 || c.linearConvKernelDim == 0) {
            throw std::invalid_argument("Linear attention config is incomplete");
        }
        if (c.fullAttentionInterval == 0) {
            throw std::invalid_argument("fullAttentionInterval must be > 0 for hybrid attention");
        }
    } else if (c.attnType >= XMODEL_ATTN_MAX_TYPE) {
        XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
        s << "invalid attention type: " << c.attnType << std::endl;
        throw std::invalid_argument("Invalid attention type");
    }

    if (c.attnType == XMODEL_ATTN_CXA && c.compressRatios.size() < c.nLayers) {
        throw std::invalid_argument(
            "compressRatios must have at least nLayers elements for CxA attention");
    }

    bool isAttnMLA = c.attnType == XMODEL_ATTN_MLA;
    bool isAttnMHA = c.attnType == XMODEL_ATTN_MHA;
    bool isAttnDSA = c.attnType == XMODEL_ATTN_DSA;
    bool isAttnHybrid = c.attnType == XMODEL_ATTN_HYBRID;
    bool isAttnCxA = c.attnType == XMODEL_ATTN_CXA;
    const std::vector<LayersCheckEntry> layersTable = {
        // Tensors, Message, allowEmpty, Predicate
        // Common
        {std::ref(attnNorm), "attention norm", false, true},
        {std::ref(attnOut), "attention out", false, !isAttnCxA},
        {std::ref(mlpNorm), "mlp norm", false, true},
        {std::ref(attnNormBias), "attention norm bias", true, true},
        {std::ref(mlpNormBias), "MLP norm bias", true, true},
        // MHA
        {std::ref(attnOutInputScale), "attention out input scale", true, isAttnMHA},
        {std::ref(attnOutInputOffset), "attention out input offset", true, isAttnMHA},
        {std::ref(attnOutQuantBias), "attention out quant bias", true, isAttnMHA},
        {std::ref(attnOutDeqScale), "attention out dequant scale", true, isAttnMHA},
        {std::ref(mhaQKVInputScale), "MHA QKV input scale", true, isAttnMHA},
        {std::ref(mhaQKVInputOffset), "MHA QKV input offset", true, isAttnMHA},
        {std::ref(mhaQKVQuantBias), "MHA QKV quant bias", true, isAttnMHA},
        {std::ref(mhaQKVDeqScale), "MHA QKV dequant scale", true, isAttnMHA},
        // MLA
        {std::ref(mlaQKVA), "MLA attention QKVA", false, isAttnMLA},
        {std::ref(mlaQB), "MLA attention QB", false, isAttnMLA},
        {std::ref(mlaQNorm), "MLA attention Q Norm", false, isAttnMLA},
        {std::ref(mlaKVNorm), "MLA attention KV Norm", false, isAttnMLA},
        {std::ref(mlaWUV), "MLA WUV", false, isAttnMLA},
        {std::ref(mlaWUKT), "MLA WUK_T", false, isAttnMLA},
        {std::ref(mlaQKVAInputScale), "MLA QKVA input scale", true, isAttnMLA},
        {std::ref(mlaQKVAInputOffset), "MLA QKVA input offset", true, isAttnMLA},
        {std::ref(mlaQKVAQuantBias), "MLA QKVA quant bias", true, isAttnMLA},
        {std::ref(mlaQKVADeqScale), "MLA QKVA dequant scale", true, isAttnMLA},
        {std::ref(mlaQBInputScale), "MLA QB input scale", true, isAttnMLA},
        {std::ref(mlaQBInputOffset), "MLA QB input offset", true, isAttnMLA},
        {std::ref(mlaQBQuantBias), "MLA QB quant bias", true, isAttnMLA},
        {std::ref(mlaQBDeqScale), "MLA QB dequant scale", true, isAttnMLA},
        // MHA
        {std::ref(mhaQKV), "MHA QKV", false, isAttnMHA},
        {std::ref(mhaQKVBias), "MHA QKV bias", false, isAttnMHA && c.addBias},
        {std::ref(mhaQNorm), "MHA Q norm", false, isAttnMHA && c.qkNorm},
        {std::ref(mhaKNorm), "MHA K norm", false, isAttnMHA && c.qkNorm},
        {std::ref(mhaQNormBias), "MHA Q norm bias", true, isAttnMHA && c.qkNorm},
        {std::ref(mhaKNormBias), "MHA K norm bias", true, isAttnMHA && c.qkNorm},
        // DSA
        {std::ref(mlaQKVA), "DSA attention QKVA", false, isAttnDSA},
        {std::ref(mlaQB), "DSA attention QB", false, isAttnDSA},
        {std::ref(mlaQNorm), "DSA attention Q Norm", false, isAttnDSA},
        {std::ref(mlaKVNorm), "DSA attention KV norm", false, isAttnDSA},
        {std::ref(mlaWUV), "DSA WUV", false, isAttnDSA},
        {std::ref(mlaWUKT), "DSA WUK_T", false, isAttnDSA},
        {std::ref(mlaQNormBias), "DSA attention Q norm bias", true, isAttnDSA},
        {std::ref(mlaKVNormBias), "DSA attention KV norm bias", true, isAttnDSA},
        {std::ref(indexQB), "DSA attention index QB", false, isAttnDSA},
        {std::ref(indexKWeightsProj), "DSA attention index KWeightsProj", false, isAttnDSA},
        {std::ref(indexKNorm), "DSA attention index KNorm", false, isAttnDSA},
        {std::ref(indexKNormBias), "DSA attention index KNorm bias", false, isAttnDSA},
        {std::ref(indexQBInputScale), "DSA index QB input scale", true, isAttnDSA},
        {std::ref(indexQBInputOffset), "DSA index QB input offset", true, isAttnDSA},
        {std::ref(indexQBQuantBias), "DSA index QB quant bias", true, isAttnDSA},
        {std::ref(indexQBDeqScale), "DSA index QB dequant scale", true, isAttnDSA},
        {std::ref(mlaQKVAInputScale), "DSA QKVA input scale", true, isAttnDSA},
        {std::ref(mlaQKVAInputOffset), "DSA QKVA input offset", true, isAttnDSA},
        {std::ref(mlaQKVAQuantBias), "DSA QKVA quant bias", true, isAttnDSA},
        {std::ref(mlaQKVADeqScale), "DSA QKVA dequant scale", true, isAttnDSA},
        {std::ref(mlaQBInputScale), "DSA QB input scale", true, isAttnDSA},
        {std::ref(mlaQBInputOffset), "DSA QB input offset", true, isAttnDSA},
        {std::ref(mlaQBQuantBias), "DSA QB quant bias", true, isAttnDSA},
        {std::ref(mlaQBDeqScale), "DSA QB dequant scale", true, isAttnDSA},
        // Hybrid
        {std::ref(mhaQKV), "Hybrid QKV", false, isAttnHybrid},
        {std::ref(linearInProjQKV), "Linear QKV proj", false, isAttnHybrid},
        {std::ref(linearInProjZ), "Linear Z proj", false, isAttnHybrid},
        {std::ref(linearInProjB), "Linear B proj", false, isAttnHybrid},
        {std::ref(linearInProjA), "Linear A proj", false, isAttnHybrid},
        {std::ref(linearOutProj), "Linear out proj", false, isAttnHybrid},
        {std::ref(linearConv1d), "Linear conv1d", false, isAttnHybrid},
        {std::ref(linearALog), "Linear A log", false, isAttnHybrid},
        {std::ref(linearDtBias), "Linear Dt bias", false, isAttnHybrid},
        {std::ref(linearNorm), "Linear norm", false, isAttnHybrid},
        {std::ref(mhaQNorm), "Hybrid Q norm", false, isAttnHybrid && c.qkNorm},
        {std::ref(mhaKNorm), "Hybrid K norm", false, isAttnHybrid && c.qkNorm},
        // CxA (DeepSeek-V4)
        {std::ref(attnSink), "v4 attn sink", false, isAttnCxA},
        {std::ref(attnWqA), "v4 attn wq_a", false, isAttnCxA},
        {std::ref(attnWqAInputScale), "v4 attn wq_a input scale", true, isAttnCxA},
        {std::ref(attnWqAInputOffset), "v4 attn wq_a input offset", true, isAttnCxA},
        {std::ref(attnWqAQuantBias), "v4 attn wq_a quant bias", true, isAttnCxA},
        {std::ref(attnWqADeqScale), "v4 attn wq_a deq scale", true, isAttnCxA},
        {std::ref(mlaQB), "DSA attention QB", false, isAttnCxA},
        {std::ref(mlaQBInputScale), "DSA QB input scale", true, isAttnCxA},
        {std::ref(mlaQBInputOffset), "DSA QB input offset", true, isAttnCxA},
        {std::ref(mlaQBQuantBias), "DSA QB quant bias", true, isAttnCxA},
        {std::ref(mlaQBDeqScale), "DSA QB dequant scale", true, isAttnCxA},
        {std::ref(mlaQNorm), "DSA attention Q Norm", false, isAttnCxA},
        {std::ref(mlaKVNorm), "DSA attention KV norm", false, isAttnCxA},
        {std::ref(attnWoA), "v4 attn wo_a", false, isAttnCxA},
        {std::ref(attnWoB), "v4 attn wo_b", false, isAttnCxA},
        {std::ref(attnWKv), "v4 attn wkv", false, isAttnCxA},
        {std::ref(attnWKvInputScale), "v4 attn wkv input scale", true, isAttnCxA},
        {std::ref(attnWKvInputOffset), "v4 attn wkv input offset", true, isAttnCxA},
        {std::ref(attnWKvQuantBias), "v4 attn wkv quant bias", true, isAttnCxA},
        {std::ref(attnWKvDeqScale), "v4 attn wkv deq scale", true, isAttnCxA},
        {std::ref(compApe), "v4 compressor ape", false, isAttnCxA},
        {std::ref(compWKv), "v4 compressor wkv", false, isAttnCxA},
        {std::ref(compWGate), "v4 compressor wgate", false, isAttnCxA},
        {std::ref(compNorm), "v4 compressor norm", false, isAttnCxA},
        {std::ref(idxWeightsProj), "v4 indexer weights_proj", false, isAttnCxA},
        {std::ref(idxWqB), "v4 indexer wq_b", false, isAttnCxA},
        {std::ref(idxWqBInputScale), "v4 indexer wq_b input scale", true, isAttnCxA},
        {std::ref(idxWqBInputOffset), "v4 indexer wq_b input offset", true, isAttnCxA},
        {std::ref(idxWqBQuantBias), "v4 indexer wq_b quant bias", true, isAttnCxA},
        {std::ref(idxWqBDeqScale), "v4 indexer wq_b deq scale", true, isAttnCxA},
        {std::ref(idxCompApe), "v4 idx compressor ape", false, isAttnCxA},
        {std::ref(idxCompWKv), "v4 idx compressor wkv", false, isAttnCxA},
        {std::ref(idxCompWGate), "v4 idx compressor wgate", false, isAttnCxA},
        {std::ref(idxCompNorm), "v4 idx compressor norm", false, isAttnCxA},
        {std::ref(hcAttnFn), "v4 hc attn fn", false, isAttnCxA},
        {std::ref(hcFfnFn), "v4 hc ffn fn", false, isAttnCxA},
        {std::ref(hcAttnBase), "v4 hc attn base", false, isAttnCxA},
        {std::ref(hcFfnBase), "v4 hc ffn base", false, isAttnCxA},
        {std::ref(hcAttnScale), "v4 hc attn scale", false, isAttnCxA},
        {std::ref(hcFfnScale), "v4 hc ffn scale", false, isAttnCxA},
    };
    checkLayersDims(layersTable, c.nLayers, rankId, "Mismatched number of layers");

    const std::vector<LayersCheckEntry> denseLayersTable = {
        {std::ref(mlpUpGate), "up gate", false, true},
        {std::ref(mlpDown), "down", false, true},
        {std::ref(mlpUpGateInputScale), "up gate input scale", true, true},
        {std::ref(mlpUpGateInputOffset), "up gate input offset", true, true},
        {std::ref(mlpUpGateQuantBias), "up gate quant bias", true, true},
        {std::ref(mlpUpGateDeqScale), "up gate deq scale", true, true},
        {std::ref(mlpDownInputScale), "down gate input scale", true, true},
        {std::ref(mlpDownInputOffset), "down gate input offset", true, true},
        {std::ref(mlpDownQuantBias), "down gate quant bias", true, true},
        {std::ref(mlpDownDeqScale), "down gate deq scale", true, true},
    };
    checkLayersDims(denseLayersTable, c.nDenseLayers, rankId, "Mismatched number of dense layers");

    const std::vector<LayersCheckEntry> moeLayersTable = {
        {std::ref(moeGate), "gate", false, true},
        {std::ref(moeGateBias), "gate bias", false, c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID},
        {std::ref(moeSEUpGate), "SE up gate", false, c.nSharedExperts != 0},
        {std::ref(moeSEDown), "SE down", false, c.nSharedExperts != 0},
        {std::ref(moeSEGate), "SE gate", true, c.nSharedExperts != 0},
        {std::ref(moeSEUpGateDeqScale), "SE up gate deq scale", true, c.nSharedExperts != 0},
        {std::ref(moeSEDownDeqScale), "SE down deq scale", true, c.nSharedExperts != 0},
    };
    checkLayersDims(moeLayersTable, numMoeLayers, rankId, "Mismatched number of moe layers");
    if (c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS && c.nHashLayers > 0 &&
        moeTid2Eid.size() != c.nHashLayers) {
        XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
        s << "tid2eid num of layers: " << moeTid2Eid.size() << std::endl;
        throw std::invalid_argument("Mismatched number of tid2eid parameters");
    }

    if (moeREUpGate.size() != nRE || moeREDown.size() != nRE) {
        {
            XDebugStream s(rankId, std::string(__func__) + ":" + std::to_string(__LINE__));
            s << "num of routed experts: " << moeREUpGate.size() << std::endl;
        }
        throw std::invalid_argument(
            "Mismatched number of routed experts up gate or down parameters");
    }

    _model = new XModel(c, rankId);

    InitXTensor(_model->embed, embed);
    InitXTensor(_model->norm, norm);
    InitXTensor(_model->head, head);
    if (normBias.defined()) {
        InitXTensor(_model->normBias, normBias);
    }

    std::vector<uint32_t> layerTypes = ResolveLayerTypes(c);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        InitXTensor(_model->attnNorm[i], attnNorm[i]);
        if (!attnNormBias.empty()) {
            InitXTensor(_model->attnNormBias[i], attnNormBias[i]);
        }
        if (!mlpNormBias.empty()) {
            InitXTensor(_model->mlpNormBias[i], mlpNormBias[i]);
        }
        InitXTensor(_model->mlpNorm[i], mlpNorm[i]);

        bool isFullLayer = IsConfigLayerFullAttention(c, i, layerTypes);
        if (c.attnType == XMODEL_ATTN_HYBRID) {
            if (isFullLayer) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mhaQKV", mhaQKV, mhaQKVInputScale, mhaQKVInputOffset,
                                 mhaQKVQuantBias, mhaQKVDeqScale, _model->mhaQKV, i, false, tpRank);
                if (c.addBias) {
                    InitOptionalXTensor(_model->mhaQKVBias[i], mhaQKVBias[i]);
                }
                if (c.qkNorm) {
                    InitOptionalXTensor(_model->mhaQNorm[i], mhaQNorm[i]);
                    InitOptionalXTensor(_model->mhaKNorm[i], mhaKNorm[i]);
                    if (!mhaQNormBias.empty()) {
                        InitOptionalXTensor(_model->mhaQNormBias[i], mhaQNormBias[i]);
                    }
                    if (!mhaKNormBias.empty()) {
                        InitOptionalXTensor(_model->mhaKNormBias[i], mhaKNormBias[i]);
                    }
                }
            } else {
                InitMatmulWeight("linearOutProj", linearOutProj, attnOutInputScale,
                                 attnOutInputOffset, attnOutQuantBias, attnOutDeqScale,
                                 _model->linearOutProj, i, true, tpRank);
                InitMatmulWeight("linearInProjQKV", linearInProjQKV, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjQKV, i, false, tpRank);
                InitMatmulWeight("linearInProjZ", linearInProjZ, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjZ, i, false, tpRank);
                InitMatmulWeight("linearInProjB", linearInProjB, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjB, i, false, tpRank);
                InitMatmulWeight("linearInProjA", linearInProjA, mhaQKVInputScale,
                                 mhaQKVInputOffset, mhaQKVQuantBias, mhaQKVDeqScale,
                                 _model->linearInProjA, i, false, tpRank);
                InitOptionalXTensor(_model->linearConv1d[i], linearConv1d[i]);
                InitOptionalXTensor(_model->linearALog[i], linearALog[i]);
                InitOptionalXTensor(_model->linearDtBias[i], linearDtBias[i]);
                InitOptionalXTensor(_model->linearNorm[i], linearNorm[i]);
            }
        } else {
            if (c.attnType == XMODEL_ATTN_MLA) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                                 mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false,
                                 tpRank);
                InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                                 mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
                InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
                if (!mlaQNormBias.empty()) {
                    InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i]);
                }
                InitXTensor(_model->mlaWUV[i], mlaWUV[i]);
                InitXTensor(_model->mlaWUKT[i], mlaWUKT[i]);
                InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
                if (!mlaKVNormBias.empty()) {
                    InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i]);
                }
            } else if (c.attnType == XMODEL_ATTN_MHA) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mhaQKV", mhaQKV, mhaQKVInputScale, mhaQKVInputOffset,
                                 mhaQKVQuantBias, mhaQKVDeqScale, _model->mhaQKV, i, false, tpRank);
                if (c.addBias) {
                    InitXTensor(_model->mhaQKVBias[i], mhaQKVBias[i]);
                }
                if (c.qkNorm) {
                    InitXTensor(_model->mhaQNorm[i], mhaQNorm[i]);
                    InitXTensor(_model->mhaKNorm[i], mhaKNorm[i]);
                    if (!mhaQNormBias.empty()) {
                        InitXTensor(_model->mhaQNormBias[i], mhaQNormBias[i]);
                    }
                    if (!mhaKNormBias.empty()) {
                        InitXTensor(_model->mhaKNormBias[i], mhaKNormBias[i]);
                    }
                }
            } else if (c.attnType == XMODEL_ATTN_DSA) {
                InitMatmulWeight("attnOut", attnOut, attnOutInputScale, attnOutInputOffset,
                                 attnOutQuantBias, attnOutDeqScale, _model->attnOut, i, true,
                                 tpRank);
                InitMatmulWeight("mlaQKVA", mlaQKVA, mlaQKVAInputScale, mlaQKVAInputOffset,
                                 mlaQKVAQuantBias, mlaQKVADeqScale, _model->mlaQKVA, i, false,
                                 tpRank);
                InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                                 mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
                InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
                if (!mlaQNormBias.empty()) {
                    InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i]);
                }
                InitXTensor(_model->mlaWUV[i], mlaWUV[i]);
                InitXTensor(_model->mlaWUKT[i], mlaWUKT[i]);
                InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
                if (!mlaKVNormBias.empty()) {
                    InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i]);
                }
                // DSA top-k sharing: skip indexer weight binding on shared layers.
                if (i >= c.indexFullMask.size() || c.indexFullMask[i]) {
                    InitMatmulWeight("indexQB", indexQB, indexQBInputScale, indexQBInputOffset,
                                     indexQBQuantBias, indexQBDeqScale, _model->indexQB, i, false,
                                     tpRank);
                    InitXTensor(_model->indexKWeightsProj[i], indexKWeightsProj[i]);
                    InitXTensor(_model->indexKNorm[i], indexKNorm[i]);
                    InitXTensor(_model->indexKNormBias[i], indexKNormBias[i]);
                }
            } else if (c.attnType == XMODEL_ATTN_CXA) {
                InitMatmulWeight("attnWqA", attnWqA, attnWqAInputScale, attnWqAInputOffset,
                                 attnWqAQuantBias, attnWqADeqScale, _model->attnWqA, i, false,
                                 tpRank);
                InitMatmulWeight("attnWKv", attnWKv, attnWKvInputScale, attnWKvInputOffset,
                                 attnWKvQuantBias, attnWKvDeqScale, _model->attnWKv, i, false,
                                 tpRank);
                // wq_b reuses MLA's mlaQB field (same structure as v3 MLA).
                InitMatmulWeight("mlaQB", mlaQB, mlaQBInputScale, mlaQBInputOffset, mlaQBQuantBias,
                                 mlaQBDeqScale, _model->mlaQB, i, false, tpRank);
                // q_norm/kv_norm reuse MLA's mlaQNorm/mlaKVNorm fields.
                InitXTensor(_model->mlaQNorm[i], mlaQNorm[i]);
                if (!mlaQNormBias.empty()) {
                    InitXTensor(_model->mlaQNormBias[i], mlaQNormBias[i]);
                }
                InitXTensor(_model->mlaKVNorm[i], mlaKVNorm[i]);
                if (!mlaKVNormBias.empty()) {
                    InitXTensor(_model->mlaKVNormBias[i], mlaKVNormBias[i]);
                }
                // v4 attention output projection (replaces v3 attnOut).
                InitXTensor(_model->attnWoA[i], attnWoA[i]);
                InitXTensor(_model->attnWoB[i], attnWoB[i]);
                InitXTensor(_model->attnSink[i], attnSink[i]);
                // Compressor (per-layer; may be empty when compress_ratio == 0).
                InitOptionalXTensor(_model->compWKv[i], compWKv[i]);
                InitOptionalXTensor(_model->compWGate[i], compWGate[i]);
                InitOptionalXTensor(_model->compApe[i], compApe[i]);
                InitOptionalXTensor(_model->compNorm[i], compNorm[i]);
                // Indexer's weights (only on compress_ratio==4 layers)
                if (c.compressRatios[i] == 4) {
                    InitMatmulWeight("idxWqB", idxWqB, idxWqBInputScale, idxWqBInputOffset,
                                     idxWqBQuantBias, idxWqBDeqScale, _model->idxWqB, i, false,
                                     tpRank);
                    InitXTensor(_model->idxWeightsProj[i], idxWeightsProj[i]);
                    InitXTensor(_model->idxCompWKv[i], idxCompWKv[i]);
                    InitXTensor(_model->idxCompWGate[i], idxCompWGate[i]);
                    InitXTensor(_model->idxCompApe[i], idxCompApe[i]);
                    InitXTensor(_model->idxCompNorm[i], idxCompNorm[i]);
                }
                // MHC (per-layer).
                InitXTensor(_model->hcAttnFn[i], hcAttnFn[i]);
                InitXTensor(_model->hcFfnFn[i], hcFfnFn[i]);
                InitXTensor(_model->hcAttnBase[i], hcAttnBase[i]);
                InitXTensor(_model->hcFfnBase[i], hcFfnBase[i]);
                InitXTensor(_model->hcAttnScale[i], hcAttnScale[i]);
                InitXTensor(_model->hcFfnScale[i], hcFfnScale[i]);
            }
        }
    }

    for (uint32_t i = 0; i < c.nDenseLayers; i++) {
        InitMatmulWeight("mlpUpGate", mlpUpGate, mlpUpGateInputScale, mlpUpGateInputOffset,
                         mlpUpGateQuantBias, mlpUpGateDeqScale, _model->mlpUpGate, i, false,
                         tpRank);
        InitMatmulWeight("mlpDown", mlpDown, mlpDownInputScale, mlpDownInputOffset,
                         mlpDownQuantBias, mlpDownDeqScale, _model->mlpDown, i, true, tpRank);
    }

    for (uint32_t i = c.nDenseLayers; i < c.nLayers; i++) {
        InitXTensor(_model->moeGate[i], moeGate[i - c.nDenseLayers]);
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
            InitXTensor(_model->moeGateBias[i], moeGateBias[i - c.nDenseLayers]);
        }
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS) {
            InitOptionalXTensor(_model->moeGateBias[i], moeGateBias[i - c.nDenseLayers]);
        }
        if (c.scoringFunc == XMODEL_SCORING_FUNC_SQRTSOFTPLUS &&
            i < c.nDenseLayers + c.nHashLayers) {
            InitOptionalXTensor(_model->moeTid2Eid[i - c.nDenseLayers],
                                moeTid2Eid[i - c.nDenseLayers]);
        }
        std::vector<at::Tensor> emptyWeights = {};
        if (c.nSharedExperts != 0) {
            InitMatmulWeight("moeSEUpGate", moeSEUpGate, emptyWeights, emptyWeights, emptyWeights,
                             moeSEUpGateDeqScale, _model->moeSEUpGate, i, false, tpRank,
                             c.nDenseLayers);
            InitMatmulWeight("moeSEDown", moeSEDown, emptyWeights, emptyWeights, emptyWeights,
                             moeSEDownDeqScale, _model->moeSEDown, i, true, tpRank, c.nDenseLayers);
            uint32_t seIdx = i - c.nDenseLayers;
            if (seIdx < moeSEGate.size()) {
                InitOptionalXTensor(_model->moeSEGate[i], moeSEGate[seIdx]);
            }
        }

        for (uint32_t j = expertsStartIdx; j < expertsEndIdx; j++) {
            InitXTensor(_model->moeREUpGate[i][j], moeREUpGate[idx]);
            InitXTensor(_model->moeREDown[i][j], moeREDown[idx]);
            // for w4a8 quantization: weights will arrive as int4pack (int32 dtype)
            if (_model->moeREUpGate[i][j].dtype == INT32) {
                _model->moeREUpGate[i][j].View(INT4);
            }
            if (_model->moeREDown[i][j].dtype == INT32) {
                _model->moeREDown[i][j].View(INT4);
            }
            if (!moeREUpGateDeqScale.empty()) {
                InitXTensor(_model->moeREUpGateDeqScale[i][j], moeREUpGateDeqScale[idx]);
            }
            if (!moeREDownDeqScale.empty()) {
                InitXTensor(_model->moeREDownDeqScale[i][j], moeREDownDeqScale[idx]);
            }
            if (!moeREUpGateScaleBias.empty()) {
                InitXTensor(_model->moeREUpGateScaleBias[i][j], moeREUpGateScaleBias[idx]);
            }
            if (!moeREDownScaleBias.empty()) {
                InitXTensor(_model->moeREDownScaleBias[i][j], moeREDownScaleBias[idx]);
            }
            idx++;
        }
    }

    if (c.attnType == XMODEL_ATTN_CXA) {
        InitXTensor(_model->hcHeadFn, hcHeadFn);
        InitXTensor(_model->hcHeadBase, hcHeadBase);
        InitXTensor(_model->hcHeadScale, hcHeadScale);
    }

    _model->Init();

    if (rankId % c.defTpSize == 0) {
        XDebugStream s(rankId, "");
        s << "Euler Xlite Model Inited! [tensor paralled(" << c.defTpSize << "), data parallel("
          << c.defDpSize << "), expert parallel(" << c.moeEpSize << ")]" << std::endl;
    }

    _kv.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        if (c.attnType == XMODEL_ATTN_DSA) {
            _kv[i].resize(3);
        } else if (c.attnType == XMODEL_ATTN_CXA) {
            _kv[i].resize(5);
        } else {
            _kv[i].resize(2);
        }
    }
    _deepstackInputEmbeds.resize(c.deepstackNumLevel);
}

_CModel::~_CModel(void)
{
#ifdef XLITE_DIRECT_ATB_310P
    _directAtb310P.reset();
#endif
    if (_model != nullptr) {
        delete _model;
        _model = nullptr;
    }
}

void _CModel::SetNativeKvCache310P(std::vector<std::vector<at::Tensor>> &kvCache)
{
#ifdef XLITE_ARCH_310P
    if (kvCache.size() != _kv.size()) {
        throw std::invalid_argument("native 310P KV cache layer count does not match model");
    }
    for (size_t layer = 0; layer < kvCache.size(); ++layer) {
        if (kvCache[layer].size() != 2) {
            throw std::invalid_argument("native 310P KV cache requires K/V pairs");
        }
        for (const at::Tensor &cache : kvCache[layer]) {
            if (!cache.defined() || cache.scalar_type() != at::kHalf || cache.dim() != 4 ||
                cache.size(1) != 64 || cache.size(2) != 128 || cache.size(3) != 16) {
                throw std::invalid_argument(
                    "native 310P KV cache must be FP16 [num_blocks,64,128,16] FRACTAL_NZ");
            }
        }
    }
    _nativeKv310P = kvCache;
    const auto fp16Options = kvCache[0][0].options().dtype(at::kHalf);
    const auto intOptions = kvCache[0][0].options().dtype(at::kInt);
    _nativeQueryStage310P = at::empty(
        {static_cast<int64_t>(_nativeMaxTokens310P), 16, 128}, fp16Options);
    _nativeKeyStage310P = at::empty(
        {static_cast<int64_t>(_nativeMaxTokens310P), 8, 128}, fp16Options);
    _nativeValueStage310P = at::empty(
        {static_cast<int64_t>(_nativeMaxTokens310P), 8, 128}, fp16Options);
    _nativeDecodeQueryStage310P.clear();
    _nativeDecodeKeyStage310P.clear();
    _nativeDecodeValueStage310P.clear();
    _nativeDecodeOutputStage310P.clear();
    _nativeDecodeQueryStage310P.reserve(kvCache.size());
    _nativeDecodeKeyStage310P.reserve(kvCache.size());
    _nativeDecodeValueStage310P.reserve(kvCache.size());
    _nativeDecodeOutputStage310P.reserve(kvCache.size());
    _directAtbRopeStaged310P.assign(kvCache.size(), 0);
    for (size_t layer = 0; layer < kvCache.size(); ++layer) {
        _nativeDecodeQueryStage310P.emplace_back(at::empty(
            {static_cast<int64_t>(_nativeMaxBatch310P), 16, 128}, fp16Options));
        _nativeDecodeKeyStage310P.emplace_back(at::empty(
            {static_cast<int64_t>(_nativeMaxBatch310P), 8, 128}, fp16Options));
        _nativeDecodeValueStage310P.emplace_back(at::empty(
            {static_cast<int64_t>(_nativeMaxBatch310P), 8, 128}, fp16Options));
        _nativeDecodeOutputStage310P.emplace_back(at::empty(
            {static_cast<int64_t>(_nativeMaxBatch310P), 16, 128}, fp16Options));
    }
    _nativeSlotStage310P =
        at::empty({static_cast<int64_t>(_nativeMaxTokens310P)}, intOptions);
    // PrepareAttn packs only the active block-table columns. Keep the staging
    // storage flat so a variable-width [batch, columns] view has the same row
    // stride as that packed source.
    _nativeBlockTableStage310P = at::empty(
        {static_cast<int64_t>(_nativeMaxBatch310P) *
         static_cast<int64_t>(_nativeMaxBlocks310P)}, intOptions);
    _nativeTotalLensStage310P =
        at::empty({static_cast<int64_t>(_nativeMaxBatch310P)}, intOptions);
#else
    (void)kvCache;
    throw std::runtime_error("native 310P KV cache is unavailable in this build");
#endif
}

#ifdef XLITE_ARCH_310P
namespace {
size_t TorchTensorBytes(const at::Tensor &tensor)
{
    return static_cast<size_t>(tensor.numel()) * tensor.element_size();
}

void CallAtbBoxed(const char *name, c10::Stack &stack)
{
    static const auto reshape =
        c10::Dispatcher::singleton().findSchemaOrThrow("atb::_npu_reshape_and_cache", "");
    static const auto paged =
        c10::Dispatcher::singleton().findSchemaOrThrow("atb::_npu_paged_attention", "");
    if (std::string(name) == "reshape") {
        reshape.callBoxed(&stack);
    } else {
        paged.callBoxed(&stack);
    }
}

bool NativeAtbDebugEnabled()
{
    static const bool enabled =
        isEnvironmentVariableTrue(std::getenv("XLITE_310P_DEBUG_NATIVE_ATB"));
    return enabled;
}

void SyncNativeAtbDebug(XRuntime &rt, const char *stage, size_t layer,
                        uint32_t batch, int64_t tokens)
{
    if (!NativeAtbDebugEnabled()) {
        return;
    }
    const aclError result = aclrtSynchronizeStream(rt.stream);
    if (result != ACL_ERROR_NONE) {
        std::ostringstream message;
        message << "native_atb diagnostic failure: stage=" << stage
                << ", layer=" << layer << ", batch=" << batch
                << ", tokens=" << tokens << ", aclError=" << result;
        throw std::runtime_error(message.str());
    }
}
}  // namespace

bool _CModel::PrepareNativeAtbRopeStages310P(XRuntime &rt, XTensor &kCache, uint32_t tokens,
                                             void *&query, void *&key, void *&value)
{
    size_t layer = _kv.size();
    for (size_t i = 0; i < _kv.size(); ++i) {
        if (_kv[i][0].ptr == kCache.ptr) {
            layer = i;
            break;
        }
    }
    if (layer == _kv.size() || tokens > _nativeMaxTokens310P ||
        layer >= _directAtbRopeStaged310P.size()) {
        return false;
    }
    at::Tensor &queryStage = rt._linearDecodeStep ? _nativeDecodeQueryStage310P[layer]
                                                  : _nativeQueryStage310P;
    at::Tensor &keyStage = rt._linearDecodeStep ? _nativeDecodeKeyStage310P[layer]
                                                : _nativeKeyStage310P;
    at::Tensor &valueStage = rt._linearDecodeStep ? _nativeDecodeValueStage310P[layer]
                                                  : _nativeValueStage310P;
    query = TensorPtr(queryStage);
    key = TensorPtr(keyStage);
    value = TensorPtr(valueStage);
    _directAtbRopeStaged310P[layer] = 1;
    return true;
}

bool _CModel::RunNativeAtbAttention310P(
    XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &output,
    XTensor &slotMapping, XTensor &blockTables, XTensor &totalLens, uint32_t nHeads,
    uint32_t nKvHeads, uint32_t headDim, uint32_t batch)
{
    (void)vCache;
    size_t layer = _kv.size();
    for (size_t i = 0; i < _kv.size(); ++i) {
        if (_kv[i][0].ptr == kCache.ptr) {
            layer = i;
            break;
        }
    }
    if (layer == _kv.size() || layer >= _nativeKv310P.size()) {
        throw std::runtime_error("native_atb could not resolve the decoder layer KV cache");
    }
    if (nHeads != 16 || nKvHeads != 8 || headDim != 128 || qkv.dtype != FP16) {
        throw std::runtime_error("native_atb is specialized for Q16/KV8/D128 FP16");
    }

    at::Tensor &nativeK = _nativeKv310P[layer][0];
    at::Tensor &nativeV = _nativeKv310P[layer][1];
    const int64_t tokens = static_cast<int64_t>(qkv.shape[0]);
    if (blockTables.shape.size() != 2) {
        throw std::runtime_error("native_atb block table must be rank 2");
    }
    const int64_t tableColumns = static_cast<int64_t>(blockTables.shape[1]);
    const int64_t tableElements = static_cast<int64_t>(batch) * tableColumns;
    if (tokens > static_cast<int64_t>(_nativeMaxTokens310P) ||
        batch > _nativeMaxBatch310P ||
        tableColumns > static_cast<int64_t>(_nativeMaxBlocks310P) ||
        tableElements > _nativeBlockTableStage310P.numel()) {
        throw std::runtime_error("native_atb staging capacity exceeded");
    }
    if (blockTables.bytes !=
        static_cast<size_t>(tableElements) * sizeof(int32_t)) {
        throw std::runtime_error("native_atb block table is not tightly packed");
    }
    if (layer == 0) {
        if (rt._lensHost.size() != batch || rt._cachedLensHost.size() != batch ||
            rt._blockTablesHost.size() != static_cast<size_t>(tableElements)) {
            throw std::runtime_error("native_atb retained host metadata size mismatch");
        }
        const uint32_t cacheBlocks = static_cast<uint32_t>(nativeK.size(0));
        uint32_t minTotalLength = std::numeric_limits<uint32_t>::max();
        uint32_t maxTotalLength = 0;
        uint32_t maxPhysicalBlock = 0;
        for (uint32_t request = 0; request < batch; ++request) {
            const uint32_t totalLength =
                rt._cachedLensHost[request] + rt._lensHost[request];
            const uint32_t requiredBlocks = (totalLength + 127) / 128;
            if (requiredBlocks == 0 ||
                requiredBlocks > static_cast<uint32_t>(tableColumns)) {
                throw std::runtime_error("native_atb block table is shorter than valid KV length");
            }
            minTotalLength = std::min(minTotalLength, totalLength);
            maxTotalLength = std::max(maxTotalLength, totalLength);
            for (uint32_t logical = 0; logical < requiredBlocks; ++logical) {
                const uint32_t physical =
                    rt._blockTablesHost[request * tableColumns + logical];
                if (physical >= cacheBlocks) {
                    std::ostringstream message;
                    message << "native_atb physical block out of range: request=" << request
                            << ", logical=" << logical << ", physical=" << physical
                            << ", cache_blocks=" << cacheBlocks;
                    throw std::runtime_error(message.str());
                }
                maxPhysicalBlock = std::max(maxPhysicalBlock, physical);
            }
        }
        if (NativeAtbDebugEnabled()) {
            std::cerr << "[XLITE_NATIVE_ATB_DEBUG] layer=0 batch=" << batch
                      << " tokens=" << tokens << " table_columns=" << tableColumns
                      << " cache_blocks=" << cacheBlocks
                      << " total_lens_min=" << minTotalLength
                      << " total_lens_max=" << maxTotalLength
                      << " max_physical_block=" << maxPhysicalBlock << std::endl;
        }
    }
    const size_t qBytes = static_cast<size_t>(nHeads) * headDim * sizeof(uint16_t);
    const size_t kvBytes = static_cast<size_t>(nKvHeads) * headDim * sizeof(uint16_t);
    const size_t rowBytes = qBytes + 2 * kvBytes;
    if (qkv.bytes != static_cast<size_t>(tokens) * rowBytes) {
        throw std::runtime_error("native_atb QKV input is not tightly packed");
    }
    const auto *qkvBytes = static_cast<const uint8_t *>(qkv.ptr);
    at::Tensor &queryStage = rt._linearDecodeStep ? _nativeDecodeQueryStage310P[layer]
                                                  : _nativeQueryStage310P;
    at::Tensor &keyStage = rt._linearDecodeStep ? _nativeDecodeKeyStage310P[layer]
                                                : _nativeKeyStage310P;
    at::Tensor &valueStage = rt._linearDecodeStep ? _nativeDecodeValueStage310P[layer]
                                                  : _nativeValueStage310P;
    const bool fusedRopeStages = rt.UseDirectAtbDecodeAttention310P() &&
                                 layer < _directAtbRopeStaged310P.size() &&
                                 _directAtbRopeStaged310P[layer] != 0;
    if (!fusedRopeStages) {
        // ATB requires contiguous ND Q/K/V. Keep this fallback for the
        // dispatcher backend and for model variants that do not implement the
        // 310P fused RoPE staging contract.
        CHECK_ACL(aclrtMemcpy2dAsync(TensorPtr(queryStage), qBytes, qkvBytes,
                                     rowBytes, qBytes, tokens,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        CHECK_ACL(aclrtMemcpy2dAsync(TensorPtr(keyStage), kvBytes,
                                     qkvBytes + qBytes, rowBytes, kvBytes, tokens,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        CHECK_ACL(aclrtMemcpy2dAsync(TensorPtr(valueStage), kvBytes,
                                     qkvBytes + qBytes + kvBytes, rowBytes, kvBytes, tokens,
                                     ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        SyncNativeAtbDebug(rt, "qkv_split", layer, batch, tokens);
        if (rt.UseDirectAtbDecodeAttention310P()) {
            rt.RecordDirectAtbStagingBytes(qkv.bytes);
        } else {
            rt.RecordNativeAtbStagingBytes(qkv.bytes);
        }
    } else {
        rt.RecordDirectAtbFusedRopeStaging(qkv.bytes);
    }
    at::Tensor query = queryStage.narrow(0, 0, tokens);
    at::Tensor key = keyStage.narrow(0, 0, tokens);
    at::Tensor value = valueStage.narrow(0, 0, tokens);
    if (layer == 0 && !rt.UseDirectAtbDecodeAttention310P()) {
        CHECK_ACL(aclrtMemcpyAsync(TensorPtr(_nativeSlotStage310P),
                                   TorchTensorBytes(_nativeSlotStage310P), slotMapping.ptr,
                                   slotMapping.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        CHECK_ACL(aclrtMemcpyAsync(TensorPtr(_nativeBlockTableStage310P),
                                   TorchTensorBytes(_nativeBlockTableStage310P), blockTables.ptr,
                                   blockTables.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        CHECK_ACL(aclrtMemcpyAsync(TensorPtr(_nativeTotalLensStage310P),
                                   TorchTensorBytes(_nativeTotalLensStage310P), totalLens.ptr,
                                   totalLens.bytes, ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        rt.RecordNativeAtbStagingBytes(
            slotMapping.bytes + blockTables.bytes + totalLens.bytes);
        SyncNativeAtbDebug(rt, "metadata_copy", layer, batch, tokens);
    }
    at::Tensor slots = _nativeSlotStage310P.narrow(0, 0, tokens);

#ifdef XLITE_DIRECT_ATB_310P
    if (rt.UseDirectAtbDecodeAttention310P()) {
        if (_directAtb310P == nullptr) {
            _directAtb310P = std::make_unique<XliteDirectAtb310P>();
        }
        _directAtb310P->SetStream(rt.stream);
        XTensor *workspaceTensor = nullptr;
        const auto acquire = [&rt, &workspaceTensor](size_t bytes) -> void * {
            workspaceTensor = &rt.GetTensor({bytes}, INT8, DBG_LOC);
            return workspaceTensor->ptr;
        };
        const auto release = [&rt, &workspaceTensor](void *) {
            if (workspaceTensor != nullptr) {
                rt.PutTensor(*workspaceTensor);
                workspaceTensor = nullptr;
            }
        };
        const uint32_t cacheBlocks = static_cast<uint32_t>(nativeK.size(0));
        const bool reshapePlanReused = _directAtb310P->ReshapeAndCache(
            static_cast<uint32_t>(layer), TensorPtr(key), TensorPtr(value),
            static_cast<uint32_t>(tokens), TensorPtr(nativeK), TensorPtr(nativeV), cacheBlocks,
            slotMapping.ptr, acquire, release);
        rt.RecordDirectAtbPlan(reshapePlanReused, false);
        rt.RecordDirectAtbSetup();
        rt.RecordDirectAtbExecute(false);
        rt.RecordNativeAtbCacheWrite();

        // A normal continuous-batching step often contains one fresh ASR
        // prefill and many one-token decodes. Compact only those decode rows
        // and submit one PagedAttention operation instead of forcing every
        // request through the per-request ACLNN fallback.
        const uint32_t decodeBatch =
            static_cast<uint32_t>(rt._decodeRequestIndicesHost.size());
        if (decodeBatch == 0) {
            return false;
        }
        if (rt._queryOffsetsHost.size() != batch ||
            rt._directAtbProcessedRequests.size() != batch ||
            decodeBatch > _nativeMaxBatch310P || !rt._decodeMetadataReady ||
            rt._decodeTableColumns != _nativeMaxBlocks310P ||
            rt._decodeQueryOffsets.numel < decodeBatch ||
            rt._decodeTotalLens.numel < decodeBatch ||
            rt._decodeBlockTables.numel <
                static_cast<size_t>(decodeBatch) * rt._decodeTableColumns) {
            throw std::runtime_error("direct_atb mixed decode metadata size mismatch");
        }
        at::Tensor decodeQuery =
            _nativeDecodeQueryStage310P[layer].narrow(0, 0, decodeBatch);
        at::Tensor decodeOutput =
            _nativeDecodeOutputStage310P[layer].narrow(0, 0, decodeBatch);
        if (!rt._linearDecodeStep) {
            XliteOpMixedDecodeCopy310P(rt, TensorPtr(query), TensorPtr(decodeQuery),
                                       rt._decodeQueryOffsets, decodeBatch, false);
            rt.RecordDirectAtbCompact(static_cast<uint64_t>(decodeBatch) * qBytes);
        }
        const bool pagedPlanReused = _directAtb310P->PagedAttention(
            static_cast<uint32_t>(layer), TensorPtr(decodeQuery), decodeBatch,
            TensorPtr(nativeK), TensorPtr(nativeV), cacheBlocks,
            rt._decodeBlockTables.ptr, rt._decodeTableColumns,
            rt._decodeTotalLens.ptr, TensorPtr(decodeOutput), acquire, release);
        rt.RecordDirectAtbPlan(pagedPlanReused, true);
        rt.RecordDirectAtbSetup();
        rt.RecordDirectAtbExecute(true, decodeBatch);
        if (rt._linearDecodeStep) {
            CHECK_ACL(aclrtMemcpyAsync(output.ptr, output.bytes,
                                       TensorPtr(decodeOutput),
                                       static_cast<size_t>(decodeBatch) * qBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.RecordDirectAtbScatter(static_cast<uint64_t>(decodeBatch) * qBytes);
            return true;
        }
        rt.RecordDirectAtbMixedDecode(decodeBatch);
        XliteOpMixedDecodeCopy310P(rt, TensorPtr(decodeOutput), output.ptr,
                                   rt._decodeQueryOffsets, decodeBatch, true);
        rt.RecordDirectAtbScatter(static_cast<uint64_t>(decodeBatch) * qBytes);
        for (uint32_t request : rt._decodeRequestIndicesHost) {
            rt._directAtbProcessedRequests[request] = 1;
        }
        return false;
    }
#endif

    c10::Stack reshapeStack;
    reshapeStack.emplace_back(key);
    reshapeStack.emplace_back(value);
    reshapeStack.emplace_back(nativeK);
    reshapeStack.emplace_back(nativeV);
    reshapeStack.emplace_back(slots);
    CallAtbBoxed("reshape", reshapeStack);
    SyncNativeAtbDebug(rt, "reshape_and_cache", layer, batch, tokens);
    rt.RecordNativeAtbCacheWrite();

    if (!rt._decodeStep) {
        // Prefill can use thousands of tokens, so retaining a full-size staging
        // set for every decoder layer would waste hundreds of MiB. Drain only
        // this one-time cache population before the shared buffers are reused.
        if (!NativeAtbDebugEnabled()) {
            rt.Synchronize();
        }
        return false;
    }
    if (tokens != batch) {
        throw std::runtime_error("native_atb decode requires exactly one query token per request");
    }
    at::Tensor table =
        _nativeBlockTableStage310P.narrow(0, 0, tableElements)
            .view({static_cast<int64_t>(batch), tableColumns});
    at::Tensor lengths = _nativeTotalLensStage310P.narrow(0, 0, batch);
    at::Tensor out = _nativeDecodeOutputStage310P[layer].narrow(0, 0, tokens);
    c10::Stack pagedStack;
    pagedStack.emplace_back(query);
    pagedStack.emplace_back(nativeK);
    pagedStack.emplace_back(nativeV);
    pagedStack.emplace_back(static_cast<int64_t>(nKvHeads));
    pagedStack.emplace_back(static_cast<int64_t>(nHeads));
    // Xlite RoPE-and-cache has already applied 1/sqrt(head_dim) to Q.
    pagedStack.emplace_back(1.0);
    pagedStack.emplace_back(table);
    pagedStack.emplace_back(lengths);
    pagedStack.emplace_back(out);
    pagedStack.emplace_back(c10::IValue());
    CallAtbBoxed("paged", pagedStack);
    SyncNativeAtbDebug(rt, "paged_attention", layer, batch, tokens);
    CHECK_ACL(aclrtMemcpyAsync(output.ptr, output.bytes, TensorPtr(out), TorchTensorBytes(out),
                               ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
    SyncNativeAtbDebug(rt, "output_copy", layer, batch, tokens);
    rt.RecordNativeAtbStagingBytes(output.bytes);
    rt.RecordNativeAtbAttention(batch);
    return true;
}
#endif

void _CModel::Forward(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                      std::vector<std::vector<at::Tensor>> &kvCache,
                      std::vector<at::Tensor> &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _output;
    std::vector<XTensor> _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    _freqsCis.resize(freqsCis.size());
    for (size_t i = 0; i < freqsCis.size(); i++) {
        InitXTensor(_freqsCis[i], freqsCis[i]);
    }

    if (kvCache.size() < _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) > output.size(0)) {
        throw std::runtime_error(std::string(__func__) + ": input's size 0 > output's size 0");
    }

    if (input.size(0) == 0) {
        return;  // for DP dummy run with MoE, add a padding token to avoid empty input
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j]);
        }
    }

    bool nativeAtb = false;
    bool directAtb = false;
#ifdef XLITE_ARCH_310P
    nativeAtb = rt.UseNativeKvDecodeAttention310P();
    directAtb = rt.UseDirectAtbDecodeAttention310P();
    if (nativeAtb) {
        if (currStream == 0) {
            throw std::runtime_error(
                "native_atb requires the current PyTorch NPU stream from the online runner");
        }
        if (_nativeKv310P.size() != _kv.size()) {
            throw std::runtime_error(
                "native_atb requires one registered 5D/NZ K/V cache pair per decoder layer");
        }
        if (rt.multiTaskParallel) {
            throw std::runtime_error("native_atb does not support Xlite multi-task parallelism");
        }
    }
#endif
    if (currStream != 0 && rt.taskId == 0 && (!nativeAtb || directAtb)) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    aclrtStream savedStream = rt.stream;
#ifdef XLITE_ARCH_310P
    if (nativeAtb) {
        if (!directAtb) {
            currAclStream = reinterpret_cast<aclrtStream>(currStream);
            rt.stream = currAclStream;
        }
        rt.nativeAtbAttentionCallback =
            [this, &rt](XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &out,
                        XTensor &slots, XTensor &tables, XTensor &totalLens, uint32_t nHeads,
                        uint32_t nKvHeads, uint32_t headDim, uint32_t batch) {
                return RunNativeAtbAttention310P(rt, qkv, kCache, vCache, out, slots, tables,
                                                 totalLens, nHeads, nKvHeads, headDim, batch);
            };
        if (directAtb) {
            std::fill(_directAtbRopeStaged310P.begin(),
                      _directAtbRopeStaged310P.end(), 0);
            rt.nativeAtbRopeStageCallback =
                [this, &rt](XTensor &kCache, uint32_t tokens, void *&query,
                            void *&key, void *&value) {
                    return PrepareNativeAtbRopeStages310P(
                        rt, kCache, tokens, query, key, value);
                };
        }
    }
#endif
    try {
        _model->Forward(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis, _output);
    } catch (...) {
#ifdef XLITE_ARCH_310P
        rt.nativeAtbRopeStageCallback = {};
        rt.nativeAtbAttentionCallback = {};
#endif
        rt.stream = savedStream;
        throw;
    }
#ifdef XLITE_ARCH_310P
    rt.nativeAtbRopeStageCallback = {};
    rt.nativeAtbAttentionCallback = {};
#endif
    rt.stream = savedStream;

    if (rt.multiTaskParallel) {
        if (rt.taskId == 0) {
            rt.NotifyRecordPeerStream();
            rt.NotifyWaitPeerStream();
        }
        if (rt.taskId == 1) {
            rt.NotifyRecordPeerStream();
            return;
        }
    }

    if (currStream != 0 && (!nativeAtb || directAtb)) {
        rt.EventRecordCurrStream(currAclStream);
    } else if (currStream == 0) {
        rt.Synchronize();
    }
}

void _CModel::ForwardV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                        std::vector<std::vector<at::Tensor>> &kvCache, at::Tensor &freqsCis,
                        at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lens;
    _attnMeta.cachedLensCpu = attnMeta.cachedLens;
    _attnMeta.blockTablesCpu = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.position, attnMeta.positions);
    std::vector<at::Tensor> freqsCisVec = {freqsCis};
    Forward(rt, input, _attnMeta, kvCache, freqsCisVec, output, currStream);
}

void _CModel::ForwardV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                        std::vector<std::vector<at::Tensor>> &kvCache,
                        std::vector<at::Tensor> &freqsCis, at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 2;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lensCpu;
    _attnMeta.cachedLensCpu = attnMeta.cachedLensCpu;
    InitXTensor(_attnMeta.position, attnMeta.positions);
    InitXTensor(_attnMeta.lens, attnMeta.lens);
    InitXTensor(_attnMeta.cachedLens, attnMeta.cachedLens);
    InitXTensor(_attnMeta.queryStartLoc, attnMeta.queryStartLoc);
    _attnMeta.slotMapping.resize(attnMeta.slotMapping.size());
    for (size_t i = 0; i < attnMeta.slotMapping.size(); i++) {
        InitXTensor(_attnMeta.slotMapping[i], attnMeta.slotMapping[i]);
    }
    _attnMeta.blockTables.resize(attnMeta.blockTables.size());
    for (size_t i = 0; i < attnMeta.blockTables.size(); i++) {
        InitXTensor(_attnMeta.blockTables[i], attnMeta.blockTables[i]);
    }
    Forward(rt, input, _attnMeta, kvCache, freqsCis, output, currStream);
}

void _CModel::ForwardAndGetLogitsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                    std::vector<std::vector<at::Tensor>> &kvCache,
                                    std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                                    at::Tensor &output, uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 2;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lensCpu;
    _attnMeta.cachedLensCpu = attnMeta.cachedLensCpu;
    InitXTensor(_attnMeta.position, attnMeta.positions);
    InitXTensor(_attnMeta.lens, attnMeta.lens);
    InitXTensor(_attnMeta.cachedLens, attnMeta.cachedLens);
    InitXTensor(_attnMeta.queryStartLoc, attnMeta.queryStartLoc);
    _attnMeta.slotMapping.resize(attnMeta.slotMapping.size());
    for (size_t i = 0; i < attnMeta.slotMapping.size(); i++) {
        InitXTensor(_attnMeta.slotMapping[i], attnMeta.slotMapping[i]);
    }
    _attnMeta.blockTables.resize(attnMeta.blockTables.size());
    for (size_t i = 0; i < attnMeta.blockTables.size(); i++) {
        InitXTensor(_attnMeta.blockTables[i], attnMeta.blockTables[i]);
    }
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCis, indices, output, currStream);
}

void _CModel::ForwardWithInputsEmbedsV2(XRuntime &rt, at::Tensor &input, CModelAttnMetaV2 &attnMeta,
                                        std::vector<std::vector<at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput,
                                        const std::optional<at::Tensor> &inputIds)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 2;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lensCpu;
    _attnMeta.cachedLensCpu = attnMeta.cachedLensCpu;
    InitXTensor(_attnMeta.position, attnMeta.positions);
    InitXTensor(_attnMeta.lens, attnMeta.lens);
    InitXTensor(_attnMeta.cachedLens, attnMeta.cachedLens);
    InitXTensor(_attnMeta.queryStartLoc, attnMeta.queryStartLoc);
    _attnMeta.slotMapping.resize(attnMeta.slotMapping.size());
    for (size_t i = 0; i < attnMeta.slotMapping.size(); i++) {
        InitXTensor(_attnMeta.slotMapping[i], attnMeta.slotMapping[i]);
    }
    _attnMeta.blockTables.resize(attnMeta.blockTables.size());
    for (size_t i = 0; i < attnMeta.blockTables.size(); i++) {
        InitXTensor(_attnMeta.blockTables[i], attnMeta.blockTables[i]);
    }
    at::Tensor ids = inputIds.value_or(at::Tensor());
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream,
                            deepstackInput, ids);
}

void _CModel::ForwardGetLogits(XRuntime &rt, at::Tensor &input, at::Tensor &indices,
                               at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _indices, _output;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_indices, indices);
    InitXTensor(_output, output);

    if (input.size(0) == 0) {
        return;
    }

    if (currStream != 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    _model->ForwardGetLogits(rt, _input, _indices, _output);

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardAndGetLogits(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                  std::vector<std::vector<at::Tensor>> &kvCache,
                                  std::vector<at::Tensor> &freqsCis, at::Tensor &indices,
                                  at::Tensor &output, uint64_t currStream)
{
    XTensor _input, _indices, _output;
    std::vector<XTensor> _freqsCis;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_indices, indices);
    InitXTensor(_output, output);
    _freqsCis.resize(freqsCis.size());
    for (size_t i = 0; i < freqsCis.size(); i++) {
        InitXTensor(_freqsCis[i], freqsCis[i]);
    }

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) == 0) {
        return;  // for DP dummy run with MoE, add a padding token to avoid empty input
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j]);
        }
    }

    if (currStream != 0 && rt.taskId == 0) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    _model->ForwardAndGetLogits(rt, _input, attnMeta, _kv, _deepstackInputEmbeds, _freqsCis,
                                _indices, _output);

    if (rt.multiTaskParallel) {
        if (rt.taskId == 0) {
            rt.NotifyRecordPeerStream();
            rt.NotifyWaitPeerStream();
        }
        if (rt.taskId == 1) {
            rt.NotifyRecordPeerStream();
            return;
        }
    }

    if (currStream != 0) {
        rt.EventRecordCurrStream(currAclStream);
    } else {
        rt.Synchronize();
    }
}

void _CModel::ForwardAndGetLogitsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                    std::vector<std::vector<at::Tensor>> &kvCache,
                                    at::Tensor &freqsCis, at::Tensor &indices, at::Tensor &output,
                                    uint64_t currStream)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lens;
    _attnMeta.cachedLensCpu = attnMeta.cachedLens;
    _attnMeta.blockTablesCpu = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.position, attnMeta.positions);
    std::vector<at::Tensor> freqsCisVec = {freqsCis};
    ForwardAndGetLogits(rt, input, _attnMeta, kvCache, freqsCisVec, indices, output, currStream);
}

void _CModel::ForwardWithInputsEmbeds(XRuntime &rt, at::Tensor &input, XModelAttnMeta &attnMeta,
                                      std::vector<std::vector<at::Tensor>> &kvCache,
                                      at::Tensor &freqsCis, at::Tensor &output, uint64_t currStream,
                                      std::vector<at::Tensor> &deepstackInput, at::Tensor &inputIds)
{
    XModelAttnMeta _attnMeta;
    XTensor _input, _output, _freqsCis, _inputIds;
    aclrtStream currAclStream = nullptr;

    InitXTensor(_input, input);
    InitXTensor(_output, output);
    InitXTensor(_freqsCis, freqsCis);
    InitOptionalXTensor(_inputIds, inputIds);

    if (kvCache.size() != _kv.size()) {
        throw std::runtime_error(std::string(__func__) + ": check kv cache failed!");
    }

    if (input.size(0) > output.size(0)) {
        throw std::runtime_error(std::string(__func__) + ": input's size 0 > output's size 0");
    }

    if (input.size(0) == 0) {
        return;  // for DP dummy run with MoE, add a padding token to avoid empty input
    }

    if (deepstackInput.size() != _deepstackInputEmbeds.size()) {
        throw std::runtime_error(std::string(__func__) + ": check deepstack input failed");
    }

    for (uint64_t i = 0; i < _kv.size(); i++) {
        if (kvCache[i].size() != _kv[i].size()) {
            throw std::runtime_error(std::string(__func__) + ": check kv cache failed at layer " +
                                     std::to_string(i));
        }
        for (uint64_t j = 0; j < _kv[i].size(); j++) {
            InitXTensor(_kv[i][j], kvCache[i][j]);
        }
    }

    for (uint32_t i = 0; i < deepstackInput.size(); i++) {
        InitXTensor(_deepstackInputEmbeds[i], deepstackInput[i]);
    }

    bool nativeAtb = false;
    bool directAtb = false;
#ifdef XLITE_ARCH_310P
    nativeAtb = rt.UseNativeKvDecodeAttention310P();
    directAtb = rt.UseDirectAtbDecodeAttention310P();
    if (nativeAtb) {
        if (currStream == 0) {
            throw std::runtime_error(
                "native_atb requires the current PyTorch NPU stream from the ASR runner");
        }
        if (_nativeKv310P.size() != _kv.size()) {
            throw std::runtime_error(
                "native_atb requires one registered 5D/NZ K/V cache pair per decoder layer");
        }
        if (rt.multiTaskParallel) {
            throw std::runtime_error("native_atb does not support Xlite multi-task parallelism");
        }
    }
#endif
    if (currStream != 0 && rt.taskId == 0 && (!nativeAtb || directAtb)) {
        currAclStream = reinterpret_cast<aclrtStream>(currStream);
        rt.EventWaitCurrStream(currAclStream);
    }

    if (rt.multiTaskParallel && rt.taskId == 1) {
        rt.NotifyWaitPeerStream();
    }

    aclrtStream savedStream = rt.stream;
#ifdef XLITE_ARCH_310P
    if (nativeAtb) {
        if (!directAtb) {
            currAclStream = reinterpret_cast<aclrtStream>(currStream);
            rt.stream = currAclStream;
        }
        rt.nativeAtbAttentionCallback =
            [this, &rt](XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &out,
                        XTensor &slots, XTensor &tables, XTensor &totalLens, uint32_t nHeads,
                        uint32_t nKvHeads, uint32_t headDim, uint32_t batch) {
                return RunNativeAtbAttention310P(rt, qkv, kCache, vCache, out, slots, tables,
                                                 totalLens, nHeads, nKvHeads, headDim, batch);
            };
        if (directAtb) {
            std::fill(_directAtbRopeStaged310P.begin(),
                      _directAtbRopeStaged310P.end(), 0);
            rt.nativeAtbRopeStageCallback =
                [this, &rt](XTensor &kCache, uint32_t tokens, void *&query,
                            void *&key, void *&value) {
                    return PrepareNativeAtbRopeStages310P(
                        rt, kCache, tokens, query, key, value);
                };
        }
    }
#endif
    try {
        _model->ForwardWithInputsEmbeds(rt, _input, attnMeta, _kv, _deepstackInputEmbeds,
                                        _freqsCis, _inputIds, _output);
    } catch (...) {
#ifdef XLITE_ARCH_310P
        rt.nativeAtbRopeStageCallback = {};
        rt.nativeAtbAttentionCallback = {};
#endif
        rt.stream = savedStream;
        throw;
    }
#ifdef XLITE_ARCH_310P
    rt.nativeAtbRopeStageCallback = {};
    rt.nativeAtbAttentionCallback = {};
#endif
    rt.stream = savedStream;

    if (rt.multiTaskParallel) {
        if (rt.taskId == 0) {
            rt.NotifyRecordPeerStream();
            rt.NotifyWaitPeerStream();
        }
        if (rt.taskId == 1) {
            rt.NotifyRecordPeerStream();
            return;
        }
    }

    if (currStream != 0 && (!nativeAtb || directAtb)) {
        rt.EventRecordCurrStream(currAclStream);
    } else if (currStream == 0) {
        rt.Synchronize();
    }
}

void _CModel::ForwardWithInputsEmbedsV1(XRuntime &rt, at::Tensor &input, CModelAttnMeta &attnMeta,
                                        std::vector<std::vector<at::Tensor>> &kvCache,
                                        at::Tensor &freqsCis, at::Tensor &output,
                                        uint64_t currStream,
                                        std::vector<at::Tensor> &deepstackInput,
                                        const std::optional<at::Tensor> &inputIds)
{
    XModelAttnMeta _attnMeta;
    _attnMeta.version = 1;
    _attnMeta.attnType = attnType;
    _attnMeta.lensCpu = attnMeta.lens;
    _attnMeta.cachedLensCpu = attnMeta.cachedLens;
    _attnMeta.blockTablesCpu = attnMeta.blockTablesList;
    InitXTensor(_attnMeta.position, attnMeta.positions);
    at::Tensor ids = inputIds.value_or(at::Tensor());
    ForwardWithInputsEmbeds(rt, input, _attnMeta, kvCache, freqsCis, output, currStream,
                            deepstackInput, ids);
}

size_t _CModel::GetTensorPoolSize(int dbg)
{
    return _model->GetTensorPoolSize(dbg);
}

void _CModel::InitMatmulWeight(const std::string &name, std::vector<at::Tensor> &w,
                               std::vector<at::Tensor> &iScale, std::vector<at::Tensor> &iOffset,
                               std::vector<at::Tensor> &qBias, std::vector<at::Tensor> &dScale,
                               std::vector<MatmulWeight> &weightsXT, uint32_t currLayer,
                               bool isRowParallel, uint32_t tpRank, uint32_t layerOffset)
{
    MatmulWeight &wXT = weightsXT[currLayer];
    uint32_t weightLayer = currLayer - layerOffset;

    wXT.name = name + "[" + std::to_string(currLayer) + "]";
    if (weightLayer >= w.size() || !TensorUsable(w[weightLayer])) {
        throw std::invalid_argument(DBG_PREFIX + ": " + wXT.name +
                                    " weight tensor is required but not provided");
    }
    InitXTensor(wXT.weight, w[weightLayer]);
    if (wXT.weight.dtype ==
        INT32) {  // for int4 matmul: weights will arrive as int4pack (int32 dtype)
        wXT.weight.View(INT4);
    }
    if (wXT.weight.dtype == INT8 && dScale.size() <= weightLayer) {
        std::string errStr = DBG_PREFIX + ": " + name +
                             "dequant scale parameters are "
                             "required when weights are quantized";
        throw std::invalid_argument(errStr);
    }
    if (weightLayer < iScale.size()) {
        InitXTensor(wXT.inputScale, iScale[weightLayer]);
    }
    if (weightLayer < iOffset.size()) {
        InitXTensor(wXT.inputOffset, iOffset[weightLayer]);
    }
    // Notice: only tpRank == 0 in RowParallelLinear need to add quant_bias
    if (weightLayer < qBias.size() && (!isRowParallel || tpRank == 0)) {
        InitXTensor(wXT.quantBias, qBias[weightLayer]);
    }
    if (weightLayer < dScale.size()) {
        InitXTensor(wXT.deqScale, dScale[weightLayer]);
    }
}

void AllGather(XRuntime &rt, at::Tensor &out, at::Tensor &in, uint32_t commType = 0)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);

    enum commType type = commType == 1 ? DP : TP;
    XliteOpAllGather(rt, _in, _out, type, type == DP, DBG_LOC);
    rt.Synchronize();
}

void ReduceScatter(XRuntime &rt, at::Tensor &out, at::Tensor &in, uint32_t commType = 0)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);

    enum commType type = commType == 1 ? DP : TP;
    XliteOpReduceScatter(rt, _in, _out, type, type == DP, DBG_LOC);
    rt.Synchronize();
}

void AllReduce(XRuntime &rt, at::Tensor &out, at::Tensor &in, uint32_t commType = 0)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);

    enum commType type = commType == 1 ? DP : TP;
    XliteOpAllReduceSum(rt, _in, _out, type, type == DP, DBG_LOC);
    rt.Synchronize();
}

void AlltoAllV(XRuntime &rt, at::Tensor &out, at::Tensor &in, at::Tensor &sendCounts,
               at::Tensor &recvCounts, at::Tensor &sdispls, at::Tensor &rdispls,
               uint32_t commType = 0)
{
    XTensor _in, _out, _sendCounts, _recvCounts, _sdispls, _rdispls;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_sendCounts, sendCounts);
    InitXTensor(_recvCounts, recvCounts);
    InitXTensor(_sdispls, sdispls);
    InitXTensor(_rdispls, rdispls);

    enum commType type = TP;
    if (commType == 1) {
        type = DP;
    } else if (commType == 2) {
        type = EP;
    }
    XliteOpAlltoAllV(rt, _in, _out, _sendCounts, _recvCounts, _sdispls, _rdispls, type, DBG_LOC);
    rt.Synchronize();
}

void Add(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z)
{
    XTensor _x, _y, _z;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    XliteOpAdd(rt, _x, _y, _z);
    rt.Synchronize();
}

void Probe310P(XRuntime &rt, at::Tensor &out, uint32_t value)
{
    XTensor _out;
    InitXTensor(_out, out);
    XliteOpProbe310P(rt, _out, value);
    rt.Synchronize();
}

void OfficialAddProbe310P(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z)
{
    XTensor _x, _y, _z;
    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    XliteOpOfficialAddProbe310P(rt, _x, _y, _z);
    rt.Synchronize();
}

void Print(at::Tensor &x, const char *name, uint32_t nRow, uint32_t nCol)
{
    XTensor _x;

    InitXTensor(_x, x);
    _x.Print(name, nRow, nCol);
}

void Matmul(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, bool weightNZ,
            bool transpose)
{
    XTensor _x, _y, _z, _bias, _deqScale;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    if (_x.dtype == INT32 && _y.dtype == INT32) {  // for int4 matmul
        _x.View(INT4);
        _y.View(INT4);
    }
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqScale, transpose);
}

uint64_t MatmulBench(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z,
                     at::Tensor &x_warmup, at::Tensor &y_warmup, at::Tensor &z_warmup,
                     int iterations, int warmup_iterations, bool weightNZ, bool transpose)
{
    XTensor _x, _y, _z, _x_warmup, _y_warmup, _z_warmup, _bias, _deqScale;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);

    InitXTensor(_x_warmup, x_warmup);
    InitXTensor(_y_warmup, y_warmup);
    InitXTensor(_z_warmup, z_warmup);

    for (int i = 0; i < warmup_iterations; i++) {
        XliteOpMatmul(rt, _x_warmup, _y_warmup, _z_warmup, weightNZ, _bias, _deqScale, transpose);
    }
    rt.Synchronize();

    auto begin = std::chrono::steady_clock::now();
    for (int i = 0; i < iterations; i++) {
        XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqScale, transpose);
    }
    rt.Synchronize();
    auto end = std::chrono::steady_clock::now();

    auto res = std::chrono::duration_cast<std::chrono::nanoseconds>(end - begin).count();
    return res / iterations;
}

void MatmulWithBias(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &z, at::Tensor &bias,
                    bool weightNZ)
{
    XTensor _x, _y, _z, _bias, _deqSacle;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_z, z);
    InitXTensor(_bias, bias);
    if (_x.dtype == INT32 && _y.dtype == INT32) {  // for int4 matmul
        _x.View(INT4);
        _y.View(INT4);
    }
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqSacle);
    rt.Synchronize();
}

void Embed(XRuntime &rt, at::Tensor &weight, at::Tensor &in, at::Tensor &out, uint32_t start,
           uint32_t end)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_weight, weight);
    XliteOpEmbed(rt, _in, _weight, start, end, _out);
    rt.Synchronize();
}

void RMSNormVarianceOnly(XRuntime &rt, at::Tensor &in, at::Tensor &out, float normEps,
                         uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
                         uint32_t outStartOffset)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpRmsNorm(rt, _in, XTensor(), _out, normEps, normDim == 0 ? _in.shape[1] : normDim, false,
                   XTensor(), cntPerToken, inStartOffset, outStartOffset, XTensor());
    rt.Synchronize();
}

void RMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &out, float normEps,
             uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
             uint32_t outStartOffset, std::optional<at::Tensor> variance)
{
    XTensor _in, _out, _norm, _normBias, _variance;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitOptionalXTensor(_norm, norm);
    if (variance.has_value()) {
#ifdef XLITE_DEBUG_ON_MISC
        {
            XDebugStream s(rt.rankId(), __func__, rt.rankId() == 0);
            s << "variance: " << variance.value() << std::endl;
        }
#endif
        InitXTensor(_variance, variance.value());
    }
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim, true,
                   _normBias, cntPerToken, inStartOffset, outStartOffset, _variance);
    rt.Synchronize();
}

void RMSNormWithBias(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &normBias,
                     at::Tensor &out, float normEps, uint32_t normDim, uint32_t cntPerToken,
                     uint32_t inStartOffset, uint32_t outStartOffset)
{
    XTensor _in, _out, _norm, _normBias;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    InitXTensor(_normBias, normBias);
    XliteOpRmsNorm(rt, _in, _norm, _out, normEps, normDim == 0 ? _in.shape[1] : normDim, true,
                   _normBias, cntPerToken, inStartOffset, outStartOffset);
    rt.Synchronize();
}

void L2Norm(XRuntime &rt, at::Tensor &in, at::Tensor &out, float normEps, uint32_t normDim)
{
    XTensor _in, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpL2Norm(rt, _in, _out, normEps, normDim == 0 ? _in.shape[1] : normDim);

    rt.Synchronize();
}

void LayerNorm(XRuntime &rt, at::Tensor &in, at::Tensor &norm, at::Tensor &normBias,
               at::Tensor &out, float normEps, uint32_t normDim)
{
    XTensor _in, _out, _norm, _normBias;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    InitXTensor(_normBias, normBias);
    XliteOpLayerNorm(rt, _in, _norm, _normBias, _out, normEps, normDim);
    rt.Synchronize();
}

void AddBias(XRuntime &rt, at::Tensor &in, at::Tensor &weight, at::Tensor &out)
{
    XTensor _in, _out, _weight;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_weight, weight);
    XliteOpAddBias(rt, _in, _weight, _out);
    rt.Synchronize();
}

void SiluAndMul(XRuntime &rt, at::Tensor &in, at::Tensor &out, float swigluLimit)
{
    XTensor _in, _out;
    InitXTensor(_in, in);
    InitXTensor(_out, out);

    XliteOpSiluAndMul(rt, _in, _out, XTensor(), swigluLimit);
    rt.Synchronize();
}

void SigmoidGateMul(XRuntime &rt, at::Tensor &attn, at::Tensor &gate, at::Tensor &out)
{
    XTensor _attn, _gate, _out;
    InitXTensor(_attn, attn);
    InitXTensor(_gate, gate);
    InitXTensor(_out, out);

    XliteOpSigmoidGateMul(rt, _attn, _gate, _out);
    rt.Synchronize();
}

void RopeAndCache(XRuntime &rt, at::Tensor &inout, at::Tensor &kCache, at::Tensor &vCache,
                  at::Tensor &position, at::Tensor &cossin, at::Tensor &slotMapping,
                  uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim,
                  uint32_t blockSize, bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW)
{
    XTensor _inout, _kCache, _vCache, _position, _cossin, _slotMapping;

    InitXTensor(_inout, inout);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_position, position);
    InitXTensor(_cossin, cossin);
    InitXTensor(_slotMapping, slotMapping);
    XliteOpRopeCache(rt, _inout, _kCache, _vCache, _position, _cossin, _slotMapping, nHeads,
                     nKvHeads, headDim, rotDim, blockSize, isNeox, mropeMaskH, mropeMaskW);
    rt.Synchronize();
}

void Attention(XRuntime &rt, at::Tensor &qkv, at::Tensor &kCache, at::Tensor &vCache,
               at::Tensor &output, at::Tensor &queryStartLoc, at::Tensor &lens,
               at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads, uint32_t nKvHeads,
               uint32_t headDim, uint32_t blockSize, uint32_t batch, bool enableFlashAttention,
               uint32_t tileSizeOfCachedKV)
{
    XTensor _qkv, _kCache, _vCache, _qk, _output, _queryStartLoc, _lens, _cachedLens, _blockTables;

    InitXTensor(_qkv, qkv);
    InitXTensor(_kCache, kCache);
    InitXTensor(_vCache, vCache);
    InitXTensor(_output, output);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);
    uint32_t maxNumBlock = DeriveMaxNumBlocks(_blockTables, batch);

    if (!enableFlashAttention) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, maxNumBlock * blockSize},
                                   XDtypeOf(qkv), DBG_LOC);
        XliteOpAttention(rt, _qkv, _kCache, _vCache, qk, _output, _queryStartLoc, _lens,
                         _cachedLens, _blockTables, nHeads, nKvHeads, headDim, blockSize, batch);
        rt.Synchronize();
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV},
                                   XDtypeOf(qkv), DBG_LOC);
        XTensor &sv = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, headDim}, XDtypeOf(qkv), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_qkv.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0);
        XliteOpFlashAttention(rt, _qkv, _kCache, _vCache, qk, sv, max, sum, lastMax, lastSum, sync,
                              _output, _queryStartLoc, _lens, _cachedLens, _blockTables, nHeads,
                              nKvHeads, headDim, blockSize, batch, tileSizeOfCachedKV);
        rt.Synchronize();
        rt.PutTensor(sync);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }
}

void QkRmsNorm310P(XRuntime &rt, at::Tensor &in, at::Tensor &qNorm, at::Tensor &kNorm,
                   at::Tensor &out, float normEps, uint32_t nHeads, uint32_t nKvHeads,
                   uint32_t headDim)
{
    XTensor _in, _qNorm, _kNorm, _out;
    InitXTensor(_in, in);
    InitXTensor(_qNorm, qNorm);
    InitXTensor(_kNorm, kNorm);
    InitXTensor(_out, out);
    XliteOpQkRmsNorm(rt, _in, _qNorm, XTensor(), _kNorm, XTensor(), _out, normEps,
                     headDim, nHeads, headDim, nKvHeads, nHeads * headDim, true,
                     XTensor(), XTensor());
    rt.Synchronize();
}

void MLAV2(XRuntime &rt, at::Tensor &qWithQr, at::Tensor &qr, at::Tensor &kCache,
           at::Tensor &peCache, at::Tensor &wukT, at::Tensor &wuv, at::Tensor &output,
           at::Tensor &queryStartLoc, at::Tensor &lens, at::Tensor &cachedLens,
           at::Tensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim, uint32_t nopeHeadDim,
           uint32_t vHeadDim, uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, float scale,
           at::Tensor &topkIndices, uint32_t topK, bool weightNz, bool enableFlashAttention,
           uint32_t tileSizeOfCachedKV)
{
    XTensor _qWithQr, _qr, _kCache, _peCache, _wukT, _wuv, _output, _queryStartLoc, _lens,
        _cachedLens, _blockTables, _topkIndices;
    InitXTensor(_qWithQr, qWithQr);
    InitXTensor(_qr, qr);
    InitXTensor(_kCache, kCache);
    InitXTensor(_peCache, peCache);
    InitXTensor(_wukT, wukT);
    InitXTensor(_wuv, wuv);
    InitXTensor(_output, output);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);
    InitXTensor(_topkIndices, topkIndices);
    uint32_t maxNumBlocks = DeriveMaxNumBlocks(_blockTables, batch);

    XTensor &qAbsorb =
        rt.GetTensor({_qWithQr.shape[0], nHeads * kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
    XliteOpEinsumMhtHtdMhd(rt, _qWithQr, _wukT, qAbsorb, _qWithQr.shape[0], nHeads, nopeHeadDim,
                           kvLoraRank, weightNz, static_cast<int>(nopeHeadDim + ropeHeadDim));

    XTensor &oAbsorb =
        rt.GetTensor({_qWithQr.shape[0], nHeads * kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
    if (!enableFlashAttention) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, maxNumBlocks * blockSize},
                                   XDtypeOf(qWithQr), DBG_LOC);
        XliteOpMLAV2(rt, qAbsorb, _qr, _kCache, _peCache, qk, oAbsorb, _queryStartLoc, _lens,
                     _cachedLens, _blockTables, nHeads, ropeHeadDim, kvLoraRank, blockSize, batch,
                     scale, topK, _topkIndices);
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, tileSizeOfCachedKV},
                                   XDtypeOf(qWithQr), DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, kvLoraRank}, XDtypeOf(qWithQr), DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({_qWithQr.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({_qWithQr.shape[0], nHeads}, FP32, DBG_LOC);
        XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
        sync.Memset(0);

        XliteOpFlashMLAV2(rt, qAbsorb, _qr, _kCache, _peCache, qk, sv, max, sum, lastMax, lastSum,
                          sync, oAbsorb, _queryStartLoc, _lens, _cachedLens, _blockTables, nHeads,
                          ropeHeadDim, kvLoraRank, blockSize, batch, scale, tileSizeOfCachedKV,
                          topK, _topkIndices);
        rt.PutTensor(sync);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }

    XliteOpEinsumMhtHtdMhd(rt, oAbsorb, _wuv, _output, _qWithQr.shape[0], nHeads, kvLoraRank,
                           vHeadDim, weightNz);
    rt.Synchronize();
    rt.PutTensor(qAbsorb);
    rt.PutTensor(oAbsorb);
}

void GatherSparseKVCache(XRuntime &rt, at::Tensor &kCache, at::Tensor &peCache,
                         at::Tensor &blockTables, at::Tensor &topkIndices, at::Tensor &queryLens,
                         at::Tensor &cachedLens, at::Tensor &kDenseCache, at::Tensor &peDenseCache,
                         uint32_t batch, uint32_t indexTopK, uint32_t blockSize,
                         uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kvHeads)
{
    XTensor _kCache, _peCache, _blockTables, _topkIndices, _queryLens, _cachedLens, _kDenseCache,
        _peDenseCache;
    InitXTensor(_kCache, kCache);
    InitXTensor(_peCache, peCache);
    InitXTensor(_blockTables, blockTables);
    InitXTensor(_topkIndices, topkIndices);
    InitXTensor(_queryLens, queryLens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_kDenseCache, kDenseCache);
    InitXTensor(_peDenseCache, peDenseCache);
    XliteOpGatherSparseKVCache(rt, _kCache, _peCache, _blockTables, _topkIndices, _queryLens,
                               _cachedLens, _kDenseCache, _peDenseCache, batch, indexTopK,
                               blockSize, kvLoraRank, ropeHeadDim, kvHeads);
    rt.Synchronize();
}

void MLAV3(XRuntime &rt, at::Tensor &qAbsorb, at::Tensor &qr, at::Tensor &kDenseCache,
           at::Tensor &peDenseCache, at::Tensor &oAbsorb, at::Tensor &queryStartLoc,
           at::Tensor &lens, at::Tensor &cachedLens, uint32_t nHeads, uint32_t ropeHeadDim,
           uint32_t kvLoraRank, uint32_t batch, uint32_t indexTopK, float scale)
{
    XTensor _qAbsorb, _qr, _kDenseCache, _peDenseCache, _oAbsorb, _queryStartLoc, _lens,
        _cachedLens;
    InitXTensor(_qAbsorb, qAbsorb);
    InitXTensor(_qr, qr);
    InitXTensor(_kDenseCache, kDenseCache);
    InitXTensor(_peDenseCache, peDenseCache);
    InitXTensor(_oAbsorb, oAbsorb);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    XTensor &qk =
        rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, indexTopK}, XDtypeOf(qAbsorb), DBG_LOC);
    XliteOpMLAV3(rt, _qAbsorb, _qr, _kDenseCache, _peDenseCache, qk, _oAbsorb, _queryStartLoc,
                 _lens, _cachedLens, nHeads, ropeHeadDim, kvLoraRank, batch, indexTopK, scale);
    rt.PutTensor(qk);
    rt.Synchronize();
}

void AddAndRMSNorm(XRuntime &rt, at::Tensor &in, at::Tensor &addInOut, at::Tensor &norm,
                   at::Tensor &out, float normEps)
{
    XTensor _in, _addInOut, _out, _norm;

    InitXTensor(_in, in);
    InitXTensor(_addInOut, addInOut);
    InitXTensor(_out, out);
    InitXTensor(_norm, norm);
    XliteOpAddAndRmsNorm(rt, _in, _addInOut, _norm, normEps, _out);
    rt.Synchronize();
}

void SoftmaxTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &outWeights,
                 at::Tensor &outRouting, uint32_t topK, bool normTopKProb)
{
    XTensor _scores, _indices, _outWeights, _outRouting;

    InitXTensor(_scores, scores);
    InitXTensor(_indices, indices);
    InitXTensor(_outWeights, outWeights);
    std::vector<size_t> sizes(scores.sizes().vec().begin(), scores.sizes().vec().end());
    _outRouting.Init(sizes, BIT1, TensorPtr(outRouting));
    XliteOpSoftmaxTopK(rt, _scores, _indices, _outWeights, _outRouting, topK, normTopKProb);
    rt.Synchronize();
}

void SigmoidTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &bias,
                 float scale, at::Tensor &outWeights, at::Tensor &outRouting, uint32_t nGroup,
                 uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)
{
    XTensor _scores, _indices, _bias, _outWeights, _outRouting;

    InitXTensor(_scores, scores);
    InitXTensor(_bias, bias);
    InitXTensor(_indices, indices);
    InitXTensor(_outWeights, outWeights);
    std::vector<size_t> sizes(scores.sizes().vec().begin(), scores.sizes().vec().end());
    _outRouting.Init(sizes, BIT1, TensorPtr(outRouting));
    XliteOpSigmoidTopK(rt, _scores, _indices, _bias, scale, _outWeights, _outRouting, nGroup,
                       nTopkGroup, topK, normTopKProb);
    rt.Synchronize();
}

void SqrtsoftplusHashTopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &bias,
                          at::Tensor &inputIds, at::Tensor &tid2eid, at::Tensor &outWeights,
                          at::Tensor &routingMap, float scale, uint32_t topK, bool hash)
{
    XTensor _scores, _indices, _bias, _inputIds, _tid2eid, _outWeights, _routingMap;

    InitXTensor(_scores, scores);
    InitXTensor(_indices, indices);
    InitXTensor(_bias, bias);
    InitXTensor(_inputIds, inputIds);
    InitXTensor(_tid2eid, tid2eid);
    InitXTensor(_outWeights, outWeights);
    auto scoresSizesVec = scores.sizes().vec();
    std::vector<size_t> sizes(scoresSizesVec.begin(), scoresSizesVec.end());
    _routingMap.Init(sizes, BIT1, TensorPtr(routingMap));
    XliteOpSqrtsoftplusHashTopK(rt, _scores, _indices, _bias, _inputIds, _tid2eid, _outWeights,
                                _routingMap, scale, topK, hash);
    rt.Synchronize();
}

void TopK(XRuntime &rt, at::Tensor &scores, at::Tensor &indices, at::Tensor &outIndices,
          at::Tensor &queryLens, at::Tensor &cachedLens, size_t k)
{
    XTensor _scores, _indices, _outIndices, _queryLens, _cachedLens;

    InitXTensor(_scores, scores);
    InitXTensor(_indices, indices);
    InitXTensor(_outIndices, outIndices);
    InitXTensor(_queryLens, queryLens);
    InitXTensor(_cachedLens, cachedLens);

    XliteOpTopK(rt, _scores, _indices, _outIndices, _queryLens, _cachedLens, _queryLens.shape[0],
                k);

    rt.Synchronize();
}

void CastUp(XRuntime &rt, at::Tensor &in, at::Tensor &out)
{
    XTensor _in, _out;
    XTensor &inScale = rt.GetTensor({1}, XDtypeOf(in), DBG_LOC);

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    XliteOpCastUp(rt, _in, inScale, _out);
    rt.Synchronize();
    rt.PutTensor(inScale);
}

void Permutation(XRuntime &rt, at::Tensor &in, at::Tensor &routing, uint32_t start, uint32_t end,
                 at::Tensor &out, at::Tensor &unpIdx, at::Tensor &counts)
{
    XTensor _in, _routing, _out, _unpIdx, _counts;

    InitXTensor(_in, in);
    InitXTensor(_routing, routing);
    InitXTensor(_out, out);
    InitXTensor(_unpIdx, unpIdx);
    InitXTensor(_counts, counts);
    XliteOpPermutation(rt, _in, _routing, start, end, _out, _unpIdx, _counts);
    rt.Synchronize();
}

void UnPermutation(XRuntime &rt, at::Tensor &in, at::Tensor &routing, at::Tensor &weights,
                   uint32_t start, uint32_t end, at::Tensor &out, at::Tensor &unpIdx)
{
    XTensor _in, _routing, _weights, _out, _unpIdx;
    InitXTensor(_in, in);
    InitXTensor(_routing, routing);
    InitXTensor(_weights, weights);
    InitXTensor(_out, out);
    InitXTensor(_unpIdx, unpIdx);

    XliteOpUnpermutation(rt, _in, _unpIdx, _routing, _weights, start, end, _out);
    rt.Synchronize();
}

void GroupMatmul(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &weights,
                 std::vector<at::Tensor> &scales, at::Tensor &counts, uint32_t start, uint32_t end,
                 long outDim, long inDim, at::Tensor &output, bool weightNZ, bool transpose)
{
    XTensor _in, _counts, _output, _scalesTensor;
    XTensor *_scales = &_scalesTensor;
    std::vector<void *> p;
    uint32_t i, num = counts.size(0);
    bool hasScale = (scales.size() == num);

    InitXTensor(_in, in);
    InitXTensor(_counts, counts);
    InitXTensor(_output, output);
    if (!weights.empty() && XDtypeOf(weights[0]) == INT32 &&
        (_in.dtype == INT32 || _in.dtype == INT8)) {
        // for the w4a8 matmul, activation and weight will be viewed as int4
        _in.View(INT4);
    }
    XTensor &_weights = rt.GetTensor({num}, INT64, DBG_LOC);

    p.resize(num);
    for (i = 0; i < num; i++) {
        p[i] = TensorPtr(weights[i]);
    }
    rt.MemcpyH2D(_weights.ptr, reinterpret_cast<void *>(p.data()), num * sizeof(void *));

    if (hasScale) {
        _scales = &rt.GetTensor({num}, INT64, DBG_LOC);
        for (i = 0; i < num; i++) {
            p[i] = TensorPtr(scales[i]);
        }
        rt.MemcpyH2D(_scales->ptr, reinterpret_cast<void *>(p.data()), num * sizeof(void *));
    }

    enum XDtype weightDtype = XDtypeOf(weights[0]);
    if (weightDtype == INT32) {
        weightDtype = INT4;
    }
    XliteOpGroupMatmul(rt, _in, _weights, *_scales, _counts, start, end, weightDtype, outDim, inDim,
                       _output, weightNZ, transpose);
    rt.Synchronize();
    rt.PutTensor(_weights);
    if (hasScale) {
        rt.PutTensor(*_scales);
    }
}

void Softmax(XRuntime &rt, at::Tensor &x, uint32_t calcLen, bool isLong)
{
    XTensor _x;
    InitXTensor(_x, x);
    if (isLong) {
        XTensor &expBuf = rt.GetTensor({1, _x.shape[1]}, FP32, DBG_LOC);
        XliteOpSoftmaxLong(rt, calcLen, _x, expBuf);
        rt.PutTensor(expBuf);
    } else {
        XliteOpSoftmax(rt, calcLen, _x);
    }
    rt.Synchronize();
}

void RopeComplex(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t ropeDim,
                 at::Tensor &inputWithR, at::Tensor &freqs, at::Tensor &position,
                 at::Tensor &output, bool inverse, bool outInterleaved)
{
    XTensor _inputWithR, _freqs, _position, _output;
    InitXTensor(_inputWithR, inputWithR);
    InitXTensor(_freqs, freqs);
    InitXTensor(_position, position);
    InitXTensor(_output, output);
    XliteOpRopeComplex(rt, nLocalHeads, stepDim, ropeDim, ropeDim, stepDim - ropeDim, 0,
                       _inputWithR, _freqs, _position, _output, inverse, outInterleaved);
    rt.Synchronize();
}

void RopeComplexAndCache(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t ropeDim,
                         uint32_t offset, uint32_t vdim, at::Tensor &inputWithR, at::Tensor &freqs,
                         at::Tensor &position, uint32_t blockSize, at::Tensor &vCache,
                         at::Tensor &slotMapping, bool outInterleaved)
{
    XTensor _inputWithR, _freqs, _position, _vCache, _slotMapping;
    InitXTensor(_inputWithR, inputWithR);
    InitXTensor(_freqs, freqs);
    InitXTensor(_position, position);
    InitXTensor(_vCache, vCache);
    InitXTensor(_slotMapping, slotMapping);
    XliteOpRopeComplexAndCache(rt, nLocalHeads, stepDim, ropeDim, offset, vdim, _inputWithR, _freqs,
                               _position, blockSize, _vCache, _slotMapping, outInterleaved);
    rt.Synchronize();
}

void MlaPrepare(XRuntime &rt, at::Tensor &attnQkvc, at::Tensor &qNorm, at::Tensor &qNormBias,
                at::Tensor &attnNormQc, at::Tensor &kvNorm, at::Tensor &kvNormBias,
                at::Tensor &attnNormKvc, at::Tensor &freqs, at::Tensor &position,
                uint32_t qLoraRank, uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t blockSize,
                at::Tensor &kCache, at::Tensor &peCache, at::Tensor &slotMapping, float normEps)
{
    XTensor _attnQkvc, _qNorm, _qNormBias, _attnNormQc, _kvNorm, _kvNormBias, _attnNormKvc, _freqs,
        _position, _kCache, _peCache, _slotMapping;
    InitXTensor(_attnQkvc, attnQkvc);
    InitXTensor(_qNorm, qNorm);
    InitXTensor(_qNormBias, qNormBias);
    InitXTensor(_attnNormQc, attnNormQc);
    InitXTensor(_kvNorm, kvNorm);
    InitXTensor(_kvNormBias, kvNormBias);
    InitXTensor(_attnNormKvc, attnNormKvc);
    InitXTensor(_freqs, freqs);
    InitXTensor(_position, position);
    InitXTensor(_kCache, kCache);
    InitXTensor(_peCache, peCache);
    InitXTensor(_slotMapping, slotMapping);
    XliteOpMlaPrepare(rt, _attnQkvc, _qNorm, _qNormBias, _attnNormQc, _kvNorm, _kvNormBias, _freqs,
                      _position, qLoraRank, kvLoraRank, ropeHeadDim, blockSize, _kCache, _peCache,
                      _slotMapping, normEps, _attnNormKvc);
    rt.Synchronize();
}

void IndexerPrepare(XRuntime &rt, at::Tensor &kw, at::Tensor &kNorm, at::Tensor &kNormBias,
                    at::Tensor &freqs, at::Tensor &position, uint32_t indexHeadDim,
                    uint32_t indexNHeads, uint32_t ropeHeadDim, uint32_t blockSize,
                    at::Tensor &indexKCache, at::Tensor &slotMapping, float normEps, at::Tensor &q,
                    float scale, uint32_t topK, bool isLong)
{
    XTensor _kw, _kNorm, _kNormBias, _freqs, _position, _indexKCache, _slotMapping, _q;
    InitXTensor(_kw, kw);
    InitXTensor(_kNorm, kNorm);
    InitXTensor(_kNormBias, kNormBias);
    InitXTensor(_freqs, freqs);
    InitXTensor(_position, position);
    InitXTensor(_indexKCache, indexKCache);
    InitXTensor(_slotMapping, slotMapping);
    InitXTensor(_q, q);
    XliteOpIndexerPrepare(rt, _kw, _kNorm, _kNormBias, _freqs, _position, indexHeadDim, indexNHeads,
                          ropeHeadDim, blockSize, _indexKCache, _slotMapping, normEps, _q, scale,
                          topK, isLong);
    rt.Synchronize();
}

void Quant(XRuntime &rt, at::Tensor &x, at::Tensor &scaleReciprocal, at::Tensor &offset,
           at::Tensor &out)
{
    XTensor _x, _scaleRec, _offset, _out;
    InitXTensor(_x, x);
    InitXTensor(_scaleRec, scaleReciprocal);
    InitXTensor(_offset, offset);
    InitXTensor(_out, out);
    XliteOpQuant(rt, _x, _scaleRec, _offset, _out);
    rt.Synchronize();
}

void QuantDyn(XRuntime &rt, at::Tensor &x, at::Tensor &scale, at::Tensor &out)
{
    XTensor _x, _scale, _out;
    InitXTensor(_x, x);
    InitXTensor(_scale, scale);
    InitXTensor(_out, out);
    XliteOpQuantDyn(rt, _x, _scale, _out);
    rt.Synchronize();
}

void MSDMergeDequant(XRuntime &rt, at::Tensor &yMerged, std::vector<at::Tensor> &scaleBiases,
                     at::Tensor &counts, at::Tensor &perTokenScale, at::Tensor &out)
{
    XTensor _yMerged, _perTokenScale, _out, _counts;
    InitXTensor(_yMerged, yMerged);
    InitXTensor(_counts, counts);
    InitXTensor(_perTokenScale, perTokenScale);
    InitXTensor(_out, out);

    // Build an INT64 [numExperts] pointer array of per-expert scale_bias pointers
    uint32_t numExperts = scaleBiases.size();
    if (numExperts == 0) {
        throw std::invalid_argument("group matmul need at least one expert");
    }
    XTensor &_scaleBiasPtrs = rt.GetTensor({numExperts}, INT64, DBG_LOC);
    std::vector<void *> p(numExperts);
    for (uint32_t i = 0; i < numExperts; i++) {
        p[i] = TensorPtr(scaleBiases[i]);
    }
    rt.MemcpyH2D(_scaleBiasPtrs.ptr, reinterpret_cast<void *>(p.data()),
                 numExperts * sizeof(void *));

    XliteOpMSDMergeDequant(rt, _yMerged, _scaleBiasPtrs, _counts, 0, numExperts, _perTokenScale,
                           _out);
    rt.Synchronize();
    rt.PutTensor(_scaleBiasPtrs);
}

void MatmulDeQuant(XRuntime &rt, at::Tensor &x, at::Tensor &y, at::Tensor &bias,
                   at::Tensor &deqScale, at::Tensor &z, bool weightNZ, bool transpose)
{
    XTensor _x, _y, _z, _bias, _deqScale;

    InitXTensor(_x, x);
    InitXTensor(_y, y);
    InitXTensor(_bias, bias);
    InitXTensor(_deqScale, deqScale);
    InitXTensor(_z, z);
    if (_x.dtype == INT32 && _y.dtype == INT32) {  // for int4 matmul
        _x.View(INT4);
        _y.View(INT4);
    }
    XliteOpMatmul(rt, _x, _y, _z, weightNZ, _bias, _deqScale, transpose);
    rt.Synchronize();
}

void DeQuant(XRuntime &rt, at::Tensor &in, at::Tensor &scale, at::Tensor &out, bool hasScale)
{
    XTensor _in, _scale, _out;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    if (hasScale) {
        InitXTensor(_scale, scale);
    }
    XliteOpDeQuant(rt, _in, _out, _scale);
    rt.Synchronize();
}

void IndexerScores(XRuntime &rt, at::Tensor &q, at::Tensor &kCache, at::Tensor &weight,
                   at::Tensor &scores, at::Tensor &queryStartLoc, at::Tensor &lens,
                   at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads,
                   uint32_t headDim, uint32_t blockSize, uint32_t batch)
{
    XTensor _q, _kCache, _weight, _scores, _queryStartLoc, _lens, _cachedLens, _blockTables;

    InitXTensor(_q, q);
    InitXTensor(_kCache, kCache);
    InitXTensor(_weight, weight);
    InitXTensor(_scores, scores);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);
    XliteOpIndexerScores(rt, _q, _kCache, _weight, _scores, _queryStartLoc, _lens, _cachedLens,
                         _blockTables, nHeads, headDim, blockSize, batch);
    rt.Synchronize();
}

void IndexerTopK(XRuntime &rt, at::Tensor &q, at::Tensor &kCache, at::Tensor &weight,
                 at::Tensor &indices, at::Tensor &topkIndices, at::Tensor &queryStartLoc,
                 at::Tensor &lens, at::Tensor &cachedLens, at::Tensor &blockTables, uint32_t nHeads,
                 uint32_t headDim, uint32_t blockSize, uint32_t batch, uint32_t topK)
{
    XTensor _q, _kCache, _weight, _indices, _topkIndices, _queryStartLoc, _lens, _cachedLens,
        _blockTables;

    InitXTensor(_q, q);
    InitXTensor(_kCache, kCache);
    InitXTensor(_weight, weight);
    InitXTensor(_indices, indices);
    InitXTensor(_topkIndices, topkIndices);
    InitXTensor(_queryStartLoc, queryStartLoc);
    InitXTensor(_lens, lens);
    InitXTensor(_cachedLens, cachedLens);
    InitXTensor(_blockTables, blockTables);

    XTensor &scores =
        rt.GetTensor({2 * rt.aicNum * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN}, XDtypeOf(q), DBG_LOC);
    XTensor &lastTopk = rt.GetTensor({_q.shape[0], 2 * topK}, INT32, DBG_LOC);
    XTensor &sync = rt.GetTensor({1, rt.aivNum}, INT32, DBG_LOC);
    sync.Memset(0);

    XliteOpIndexerTopK(rt, _q, _kCache, _weight, scores, lastTopk, _indices, _topkIndices,
                       _queryStartLoc, _lens, _cachedLens, _blockTables, sync, nHeads, headDim,
                       blockSize, batch, topK);
    rt.Synchronize();
    rt.PutTensor(sync);
    rt.PutTensor(lastTopk);
    rt.PutTensor(scores);
}

void Muls(XRuntime &rt, at::Tensor &input, float scale, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input);
    InitXTensor(_output, output);
    XliteOpMuls(rt, _input, scale, _output);
    rt.Synchronize();
}

void ExpertsCountsSum(XRuntime &rt, at::Tensor &expertsCountsInput, at::Tensor &tokensPerEpgroup,
                      at::Tensor &expertsCountsOutput, uint32_t nRoutedExperts)
{
    XTensor _expertsCountsInput, _tokensPerEpgroup, _expertsCountsOutput;

    InitXTensor(_expertsCountsInput, expertsCountsInput);
    InitXTensor(_tokensPerEpgroup, tokensPerEpgroup);
    InitXTensor(_expertsCountsOutput, expertsCountsOutput);
    XliteOpExpertsCountsSum(rt, _expertsCountsInput, _tokensPerEpgroup, _expertsCountsOutput,
                            nRoutedExperts);
    rt.Synchronize();
}

void ReorderMoE(XRuntime &rt, at::Tensor &in, at::Tensor &out, at::Tensor &counts,
                uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward)
{
    XTensor _in, _out, _counts;

    InitXTensor(_in, in);
    InitXTensor(_out, out);
    InitXTensor(_counts, counts);
    XliteOpReorderMoE(rt, _in, _out, _counts, hiddenSize, localStart, localEnd, forward);
    rt.Synchronize();
}

void LinearAttProj(XRuntime &rt, at::Tensor &x, at::Tensor &W_qkv, at::Tensor &W_z, at::Tensor &W_b,
                   at::Tensor &W_a, at::Tensor &mix_qkv, at::Tensor &z, at::Tensor &b,
                   at::Tensor &a, uint32_t m, uint32_t n, uint32_t v, uint32_t h, uint32_t k)
{
    XTensor _x, _W_qkv, _W_z, _W_b, _W_a;
    XTensor _mix_qkv, _z, _b, _a;
    XTensor bias, deqScale;

    InitXTensor(_x, x);
    InitXTensor(_W_qkv, W_qkv);
    InitXTensor(_W_z, W_z);
    InitXTensor(_W_b, W_b);
    InitXTensor(_W_a, W_a);

    InitXTensor(_mix_qkv, mix_qkv);
    InitXTensor(_z, z);
    InitXTensor(_b, b);
    InitXTensor(_a, a);

    std::vector<XTensor> inputs = {_W_qkv, _W_z, _W_b, _W_a};
    XTensor &W = rt.GetTensor({k, n + v + h + h}, XDtypeOf(x), DBG_LOC);
    XliteOpConcatCol(rt, inputs, W);

    XTensor &out = rt.GetTensor({m, n + v + h + h}, XDtypeOf(x), DBG_LOC);
    XliteOpMatmul(rt, _x, W, out, false, bias, deqScale, true);
    rt.PutTensor(W);

    std::vector<XTensor> outputs = {_mix_qkv, _z, _b, _a};
    XliteOpSplitCol(rt, out, outputs);
    rt.PutTensor(out);
    rt.Synchronize();
}

void Transpose_1_2(XRuntime &rt, at::Tensor &input, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input);
    InitXTensor(_output, output);
    XliteOpTranspose_1_2(rt, _input, _output);
    rt.Synchronize();
}

void LinearAttConv1dAndSiLU(XRuntime &rt, at::Tensor &mix_qkv, at::Tensor &conv_state,
                            at::Tensor &weight, at::Tensor &output,
                            std::optional<at::Tensor> query_start_loc = std::nullopt,
                            std::optional<at::Tensor> query_lens = std::nullopt)
{
    XTensor _mix_qkv, _conv_state, _weight, _output, _start, _lens;
    InitXTensor(_mix_qkv, mix_qkv);
    InitXTensor(_conv_state, conv_state);
    InitXTensor(_weight, weight);
    InitXTensor(_output, output);
    XTensor *startPtr = nullptr;
    XTensor *lensPtr = nullptr;
    if (query_start_loc.has_value() && query_lens.has_value() && query_start_loc->defined() &&
        query_lens->defined() && query_start_loc->numel() > 0 && query_lens->numel() > 0) {
        InitXTensor(_start, *query_start_loc);
        InitXTensor(_lens, *query_lens);
        startPtr = &_start;
        lensPtr = &_lens;
    }
    XliteOpConv1dAndSiLU(rt, _conv_state, _mix_qkv, _weight, _output, /*updateState=*/true,
                         startPtr, lensPtr);
    rt.Synchronize();
}

void LinearAttConv1dAndSiLUToken(XRuntime &rt, at::Tensor &mix_qkv, at::Tensor &conv_state,
                                 at::Tensor &weight, at::Tensor &output, int64_t seq_len)
{
    XTensor _mix_qkv, _conv_state, _weight, _output;
    InitXTensor(_mix_qkv, mix_qkv);
    InitXTensor(_conv_state, conv_state);
    InitXTensor(_weight, weight);
    InitXTensor(_output, output);
    XliteOpConv1dAndSiLUToken(rt, _conv_state, _mix_qkv, _weight, _output,
                              static_cast<uint32_t>(seq_len), /*updateState=*/true);
    rt.Synchronize();
}

void SplitCol(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &outputs)
{
    XTensor _in;
    std::vector<XTensor> _outputs;
    InitXTensor(_in, in);
    uint32_t totalWidth = 0;
    for (auto &output : outputs) {
        XTensor x;
        InitXTensor(x, output);
        _outputs.push_back(x);
        totalWidth += x.shape.back();
    }
    if (totalWidth != _in.shape.back()) {
        throw std::runtime_error("input last dim != sum(outputs last dim)");
    }
    XliteOpSplitCol(rt, _in, _outputs);
    rt.Synchronize();
}

void Concat(XRuntime &rt, std::vector<at::Tensor> &inputs, at::Tensor &out)
{
    if (inputs.empty()) {
        return;
    }
    std::vector<XTensor> _inputs;
    for (auto &input : inputs) {
        XTensor x;
        InitXTensor(x, input);
        _inputs.push_back(x);
    }
    XTensor _out;
    InitXTensor(_out, out);

    // Validate byte sizes: the output must hold exactly the concatenated bytes.
    size_t inBytes = 0;
    for (auto &x : _inputs) {
        inBytes += x.bytes;
    }
    size_t outBytes = _out.bytes;
    if (inBytes != outBytes) {
        throw std::runtime_error(std::string(__func__) + ": output bytes (" +
                                 std::to_string(outBytes) + ") != sum(inputs bytes) (" +
                                 std::to_string(inBytes) + ")");
    }
    XliteOpConcat(rt, _inputs, _out);
    rt.Synchronize();
}

void ConcatCol(XRuntime &rt, std::vector<at::Tensor> &inputs, at::Tensor &out)
{
    if (inputs.empty()) {
        return;
    }
    std::vector<XTensor> _inputs;
    size_t totalWidth = 0;
    for (auto &input : inputs) {
        XTensor x;
        InitXTensor(x, input);
        _inputs.push_back(x);
        totalWidth += x.shape.back();
    }
    XTensor _out;
    InitXTensor(_out, out);
    if (totalWidth != _out.shape.back()) {
        throw std::runtime_error(std::string(__func__) + ": out last dim (" +
                                 std::to_string(_out.shape.back()) + ") != sum(inputs last dim) (" +
                                 std::to_string(totalWidth) + ")");
    }
    XliteOpConcatCol(rt, _inputs, _out);
    rt.Synchronize();
}

void Split(XRuntime &rt, at::Tensor &in, std::vector<at::Tensor> &outputs,
           std::vector<uint32_t> &sizes, uint32_t numPackets)
{
    if (outputs.empty()) {
        return;
    }
    if (outputs.size() != sizes.size()) {
        throw std::runtime_error(std::string(__func__) + ": outputs count (" +
                                 std::to_string(outputs.size()) + ") != sizes count (" +
                                 std::to_string(sizes.size()) + ")");
    }
    XTensor _in;
    std::vector<XTensor> _outputs;
    InitXTensor(_in, in);
    for (auto &output : outputs) {
        XTensor x;
        InitXTensor(x, output);
        _outputs.push_back(x);
    }
    std::vector<size_t> _sizes(sizes.begin(), sizes.end());

    size_t totalSize = 0;
    for (size_t s : _sizes) {
        totalSize += s;
    }
    size_t inBytes = _in.bytes;
    if (totalSize * numPackets != inBytes) {
        throw std::runtime_error(std::string(__func__) + ": input bytes (" +
                                 std::to_string(inBytes) + ") != totalSize*numPackets (" +
                                 std::to_string(totalSize * numPackets) + ")");
    }
    // Validate each output holds numPackets * sizes[j] bytes; otherwise the
    // kernel writes out-of-bounds and corrupts adjacent device memory.
    for (size_t j = 0; j < _outputs.size(); j++) {
        size_t need = _sizes[j] * numPackets;
        size_t cap = _outputs[j].bytes;
        if (need > cap) {
            throw std::runtime_error(std::string(__func__) + ": output[" + std::to_string(j) +
                                     "] bytes (" + std::to_string(cap) + ") < numPackets*sizes[" +
                                     std::to_string(j) + "] (" + std::to_string(need) + ")");
        }
    }
    XliteOpSplit(rt, _in, _outputs, _sizes, numPackets);
    rt.Synchronize();
}

void BetaDecay(XRuntime &rt, at::Tensor &b, at::Tensor &a, at::Tensor &A_log, at::Tensor &dt_bias,
               at::Tensor &beta, at::Tensor &g, uint32_t bsz, uint32_t seqlen, uint32_t num_v_heads)
{
    XTensor _b, _a, _A_log, _dt_bias, _beta, _g;
    InitXTensor(_b, b);
    InitXTensor(_a, a);
    InitXTensor(_A_log, A_log);
    InitXTensor(_dt_bias, dt_bias);
    InitXTensor(_beta, beta);
    InitXTensor(_g, g);

    XliteOpBetaDecay(rt, _b, _a, _A_log, _dt_bias, _beta, _g, bsz, seqlen, num_v_heads);
    rt.Synchronize();
}

void RecurrentGatedDeltaRule(XRuntime &rt, at::Tensor &query, at::Tensor &key, at::Tensor &value,
                             at::Tensor &beta, at::Tensor &g, at::Tensor &state, at::Tensor &out,
                             uint32_t batch, uint32_t seqlen, uint32_t num_heads, uint32_t k_dim,
                             uint32_t v_dim,
                             std::optional<at::Tensor> query_start_loc = std::nullopt,
                             std::optional<at::Tensor> query_lens = std::nullopt)
{
    XTensor _query, _key, _value, _beta, _g, _state, _out, _start, _lens;
    InitXTensor(_query, query);
    InitXTensor(_key, key);
    InitXTensor(_value, value);
    InitXTensor(_beta, beta);
    InitXTensor(_g, g);
    InitXTensor(_state, state);
    InitXTensor(_out, out);
    XTensor *startPtr = nullptr;
    XTensor *lensPtr = nullptr;
    if (query_start_loc.has_value() && query_lens.has_value() && query_start_loc->defined() &&
        query_lens->defined() && query_start_loc->numel() > 0 && query_lens->numel() > 0) {
        InitXTensor(_start, *query_start_loc);
        InitXTensor(_lens, *query_lens);
        startPtr = &_start;
        lensPtr = &_lens;
    }
    XliteOpRecurrentGatedDeltaRule(rt, _query, _key, _value, _beta, _g, _state, _out, batch, seqlen,
                                   num_heads, k_dim, v_dim, startPtr, lensPtr);
    rt.Synchronize();
}

void EinsumMhtHdtMhd(XRuntime &rt, at::Tensor &mht, at::Tensor &hdt, at::Tensor &mhd, uint32_t m,
                     uint32_t h, uint32_t t, uint32_t d, bool weightNZ)
{
    XTensor _mht, _hdt, _mhd;
    InitXTensor(_mht, mht);
    InitXTensor(_hdt, hdt);
    InitXTensor(_mhd, mhd);
    XliteOpEinsumMhtHdtMhd(rt, _mht, _hdt, _mhd, m, h, t, d, weightNZ);
    rt.Synchronize();
}

void EinsumMhtHtdMhd(XRuntime &rt, at::Tensor &mht, at::Tensor &htd, at::Tensor &mhd, uint32_t m,
                     uint32_t h, uint32_t t, uint32_t d, bool weightNZ)
{
    XTensor _mht, _htd, _mhd;
    InitXTensor(_mht, mht);
    InitXTensor(_htd, htd);
    InitXTensor(_mhd, mhd);
    XliteOpEinsumMhtHtdMhd(rt, _mht, _htd, _mhd, m, h, t, d, weightNZ);
    rt.Synchronize();
}

void UnpackActivation(XRuntime &rt, at::Tensor &input, at::Tensor &output)
{
    XTensor _input, _output;
    InitXTensor(_input, input);
    InitXTensor(_output, output);

    XliteOpUnpackActivation(rt, _input, _output);
    rt.Synchronize();
}

void HcAct(XRuntime &rt, at::Tensor &mixes, at::Tensor &hcScale, at::Tensor &hcBase,
           at::Tensor &post, at::Tensor &comb, uint32_t hcMult, float eps, uint32_t sinkhornIters,
           at::Tensor &xResid, at::Tensor &output)
{
    XTensor _mixes, _hcScale, _hcBase, _post, _comb, _xResid, _output;

    InitXTensor(_mixes, mixes);
    InitXTensor(_hcScale, hcScale);
    InitXTensor(_hcBase, hcBase);
    InitXTensor(_post, post);
    InitXTensor(_comb, comb);
    if (TensorUsable(xResid)) {
        InitXTensor(_xResid, xResid);
    }
    if (TensorUsable(output)) {
        InitXTensor(_output, output);
    }
    XliteOpHcAct(rt, _mixes, _hcScale, _hcBase, _post, _comb, hcMult, eps, sinkhornIters, false,
                 _xResid, _output);
    rt.Synchronize();
}

void HcPost(XRuntime &rt, at::Tensor &x, at::Tensor &post, at::Tensor &comb, at::Tensor &residual,
            at::Tensor &y, uint32_t m, uint32_t hcMult, uint32_t hidden)
{
    XTensor _x, _post, _comb, _residual, _y;
    InitXTensor(_x, x);
    InitXTensor(_post, post);
    InitXTensor(_comb, comb);
    InitXTensor(_residual, residual);
    InitXTensor(_y, y);
    XliteOpHcPost(rt, _x, _post, _comb, _residual, _y, m, hcMult, hidden);
    rt.Synchronize();
}

PYBIND11_MODULE(_C, m)
{
    m.def("get_build_info", []() {
        py::dict info;
        info["soc"] = XLITE_BUILD_SOC;
        info["kernel_set"] = XLITE_BUILD_KERNEL_SET;
#ifdef XLITE_310P_LLM_FP16_POC
        info["abi"] = 1;
        info["cache_layout"] = "BSHD";
        info["max_batch"] = 20;
        info["max_seq_len"] = 2048;
        info["attention_backend"] = "runtime_selectable";
        info["decode_attention_backends"] =
            py::make_tuple("direct_atb", "native_atb", "batched_aclnn", "legacy");
        // PromptFlashAttentionV2 needs a dense, max-length-padded KV gather for
        // decode on 310P.  Keep it available as a diagnostic backend, but do
        // not select it by default: real ASR batch-20 measurements showed a
        // large regression compared with the per-request legacy path.
        info["default_decode_attention_backend"] = "legacy";
        info["batched_decode_attention"] = true;
        info["batched_decode_attention_api"] = "PromptFlashAttentionV2";
        info["native_decode_attention"] = true;
        info["native_decode_attention_api"] = "atb::_npu_paged_attention";
        info["direct_decode_attention"] = true;
        info["direct_decode_attention_api"] = "atb::Operation::Setup/Execute";
        info["direct_decode_execution"] = "xlite_runtime_stream";
        info["direct_atb_runtime_version"] = 8;
        info["direct_atb_operation_scope"] = "per_layer_batch";
        info["direct_atb_setup_cache"] = true;
        info["direct_atb_fused_rope_staging"] = true;
        info["direct_atb_mixed_batch_decode"] = true;
        info["direct_atb_batched_compact_scatter"] = true;
        info["direct_atb_plan_cache"] = "layer_batch";
        info["direct_atb_metadata_upload"] = "once_per_forward";
        info["direct_atb_pure_decode_direct_output"] = false;
        info["direct_atb_metadata_single_h2d"] = false;
        info["direct_atb_metadata_actual_batch"] = true;
        info["batched_prefill_attention"] = true;
        info["batched_prefill_attention_api"] = "PromptFlashAttentionV1_exact_shape";
        info["batched_prefill_micro_batch"] = 4;
        info["default_batched_prefill_attention"] = false;
        info["batched_prefill_status"] = "experimental_shape_probe";
        info["prefill_shape_telemetry"] = 1;
        info["native_decode_cache_layout"] = "NZ_5D";
        info["direct_atb_task_queue_independent"] = true;
        info["attention_metadata"] = "host_retained_pinned";
        info["attention_execution"] = "async_single_stream";
        info["cross_stream_handoff"] = "split_acl_event_sync";
        info["matmul_backend"] = "runtime_selectable";
        info["matmul_backends"] = py::make_tuple("m200_asr", "aclnn");
        info["default_matmul_backend"] = "m200_asr";
        info["m200_lm_head_max_batch"] = 0;
        info["lm_head_backend"] = "aclnn_batched_submit";
        info["lm_head_synchronizations_per_call"] = 1;
#else
        info["abi"] = 0;
        info["cache_layout"] = "native";
#endif
        return info;
    });

#ifdef XLITE_310P_LLM_FP16_POC
    m.def("get_310p_matmul_stats", [](XRuntime &rt) {
        py::dict stats;
        stats["m200_requests"] = rt.m200MatmulRequests;
        stats["m200_kernel_launches"] = rt.m200MatmulKernelLaunches;
        stats["aclnn_requests"] = rt.aclnnMatmulRequests;
        stats["matmul_backend"] = rt.MatmulBackend310PName();
        py::dict m200ByM;
        py::dict aclnnByM;
        for (size_t batch = 1; batch < rt.m200MatmulRequestsByM.size(); ++batch) {
            if (rt.m200MatmulRequestsByM[batch] != 0) {
                m200ByM[py::int_(batch)] = rt.m200MatmulRequestsByM[batch];
            }
            if (rt.aclnnMatmulRequestsByM[batch] != 0) {
                aclnnByM[py::int_(batch)] = rt.aclnnMatmulRequestsByM[batch];
            }
        }
        stats["m200_requests_by_m"] = m200ByM;
        stats["aclnn_requests_by_m"] = aclnnByM;
        stats["stream_synchronizations"] = rt.StreamSynchronizations();
        stats["prepare_attn_synchronizations"] = rt.PrepareAttnSynchronizations();
        stats["attention_metadata_d2h_bytes"] = rt.AttentionMetadataD2HBytes();
        stats["attention_aclnn_launches"] = rt.AttentionAclnnLaunches();
        stats["attention_forced_synchronizations"] =
            rt.AttentionForcedSynchronizations();
        stats["attention_workspace_reuses"] = rt.AttentionWorkspaceReuses();
        stats["batched_decode_attention_requests"] =
            rt.BatchedDecodeAttentionRequests();
        stats["batched_decode_attention_launches"] =
            rt.BatchedDecodeAttentionLaunches();
        stats["batched_prefill_attention_requests"] =
            rt.BatchedPrefillAttentionRequests();
        stats["batched_prefill_attention_launches"] =
            rt.BatchedPrefillAttentionLaunches();
        stats["prefill_shape_forward_calls"] = rt.PrefillShapeForwardCalls();
        stats["prefill_shape_requests"] = rt.PrefillShapeRequests();
        stats["prefill_exact_groupable_requests"] = rt.PrefillExactGroupableRequests();
        stats["prefill_actual_query_tokens"] = rt.PrefillActualQueryTokens();
        stats["prefill_padded_query_tokens"] = rt.PrefillPaddedQueryTokens();
        py::dict prefillShapes;
        py::dict prefillBatchShapes;
        for (const auto &[shape, count] : rt.PrefillShapeHistogram()) {
            prefillShapes[py::str(shape)] = count;
        }
        for (const auto &[shape, count] : rt.PrefillBatchShapeHistogram()) {
            prefillBatchShapes[py::str(shape)] = count;
        }
        stats["prefill_shape_histogram"] = prefillShapes;
        stats["prefill_batch_shape_histogram"] = prefillBatchShapes;
        stats["aclnn_matmul_synchronizations"] =
            rt.AclnnMatmulSynchronizations();
        stats["lm_head_synchronizations"] = rt.LmHeadSynchronizations();
        stats["legacy_attention_requests"] = rt.LegacyAttentionRequests();
        stats["legacy_decode_attention_requests"] =
            rt.LegacyDecodeAttentionRequests();
        stats["legacy_prefill_attention_requests"] =
            rt.LegacyPrefillAttentionRequests();
        stats["decode_kv_gather_bytes"] = rt.DecodeKvGatherBytes();
        stats["native_atb_decode_requests"] = rt.NativeAtbDecodeRequests();
        stats["native_atb_decode_launches"] = rt.NativeAtbDecodeLaunches();
        stats["native_atb_cache_writes"] = rt.NativeAtbCacheWrites();
        stats["native_atb_staging_copy_bytes"] = rt.NativeAtbStagingBytes();
        stats["direct_atb_setup_count"] = rt.DirectAtbSetupCount();
        stats["direct_atb_execute_count"] = rt.DirectAtbExecuteCount();
        stats["direct_atb_decode_requests"] = rt.DirectAtbDecodeRequests();
        stats["direct_atb_attention_launches"] = rt.DirectAtbAttentionLaunches();
        stats["direct_atb_reshape_launches"] = rt.DirectAtbReshapeLaunches();
        stats["direct_atb_staging_copy_bytes"] = rt.DirectAtbStagingBytes();
        stats["direct_atb_fused_rope_staging_bytes"] =
            rt.DirectAtbFusedRopeStagingBytes();
        stats["direct_atb_plan_reuses"] = rt.DirectAtbPlanReuses();
        stats["direct_atb_plan_rebuilds"] = rt.DirectAtbPlanRebuilds();
        stats["direct_atb_paged_plan_reuses"] = rt.DirectAtbPagedPlanReuses();
        stats["direct_atb_paged_plan_rebuilds"] =
            rt.DirectAtbPagedPlanRebuilds();
        stats["direct_atb_reshape_plan_reuses"] =
            rt.DirectAtbReshapePlanReuses();
        stats["direct_atb_reshape_plan_rebuilds"] =
            rt.DirectAtbReshapePlanRebuilds();
        stats["direct_atb_mixed_decode_requests"] =
            rt.DirectAtbMixedDecodeRequests();
        stats["direct_atb_mixed_decode_launches"] =
            rt.DirectAtbMixedDecodeLaunches();
        stats["direct_atb_compact_launches"] = rt.DirectAtbCompactLaunches();
        stats["direct_atb_compact_bytes"] = rt.DirectAtbCompactBytes();
        stats["direct_atb_scatter_launches"] = rt.DirectAtbScatterLaunches();
        stats["direct_atb_scatter_bytes"] = rt.DirectAtbScatterBytes();
        stats["direct_atb_metadata_h2d_bytes"] =
            rt.DirectAtbMetadataH2DBytes();
        stats["direct_atb_metadata_h2d_launches"] =
            rt.DirectAtbMetadataH2DLaunches();
        stats["forward_input_events"] = rt.ForwardInputEvents();
        stats["forward_output_events"] = rt.ForwardOutputEvents();
        return stats;
    });
#endif

    py::class_<XRuntime>(m, "Runtime")
        .def(py::init<uint32_t, size_t, uint32_t, uint32_t, uint32_t, uint32_t, uint32_t>(),
             py::arg("devid"), py::arg("size") = 0, py::arg("rank") = 0, py::arg("tp_size") = 1,
             py::arg("dp_size") = 1, py::arg("moe_tp_size") = 1, py::arg("moe_ep_size") = 1)
        .def_readwrite("task_id", &XRuntime::taskId)
        .def_readwrite("notify", &XRuntime::notify)
        .def_readwrite("peer_notify", &XRuntime::peerNotify)
        .def_readwrite("multi_task_parallel", &XRuntime::multiTaskParallel)
        .def_readonly("aic_num", &XRuntime::aicNum)
        .def_readonly("aiv_num", &XRuntime::aivNum)
        .def_readonly("reported_aiv_num", &XRuntime::reportedAivNum)
        .def("update_core_num", &XRuntime::UpdateCoreNum, py::arg("util"))
        .def("init_tensor_pool", &XRuntime::InitTensorPool, py::arg("size"))
        .def("set_current_context", &XRuntime::SetCurrentContext)
#ifdef XLITE_ARCH_310P
        .def("set_matmul_backend_310p", &XRuntime::SetMatmulBackend310P,
             py::arg("backend"))
        .def("set_decode_attention_backend", &XRuntime::SetDecodeAttentionBackend310P,
             py::arg("backend"))
        .def("set_batched_prefill_attention_310p",
             &XRuntime::SetBatchedPrefillAttention310P, py::arg("enabled"))
#endif
        .def("get_stats", [](const XRuntime &rt) {
            py::dict stats;
#ifdef XLITE_ARCH_310P
            stats["m200_requests"] = rt.m200MatmulRequests;
            stats["m200_kernel_launches"] = rt.m200MatmulKernelLaunches;
            stats["aclnn_requests"] = rt.aclnnMatmulRequests;
            stats["matmul_backend"] = rt.MatmulBackend310PName();
            py::dict m200ByM;
            py::dict aclnnByM;
            for (size_t batch = 1; batch < rt.m200MatmulRequestsByM.size(); ++batch) {
                if (rt.m200MatmulRequestsByM[batch] != 0) {
                    m200ByM[py::int_(batch)] = rt.m200MatmulRequestsByM[batch];
                }
                if (rt.aclnnMatmulRequestsByM[batch] != 0) {
                    aclnnByM[py::int_(batch)] = rt.aclnnMatmulRequestsByM[batch];
                }
            }
            stats["m200_requests_by_m"] = m200ByM;
            stats["aclnn_requests_by_m"] = aclnnByM;
#endif
            stats["stream_synchronizations"] = rt.StreamSynchronizations();
            stats["prepare_attn_synchronizations"] = rt.PrepareAttnSynchronizations();
            stats["attention_metadata_d2h_bytes"] = rt.AttentionMetadataD2HBytes();
            stats["attention_aclnn_launches"] = rt.AttentionAclnnLaunches();
            stats["attention_forced_synchronizations"] =
                rt.AttentionForcedSynchronizations();
            stats["attention_workspace_reuses"] = rt.AttentionWorkspaceReuses();
#ifdef XLITE_ARCH_310P
            stats["batched_decode_attention_requests"] =
                rt.BatchedDecodeAttentionRequests();
            stats["batched_decode_attention_launches"] =
                rt.BatchedDecodeAttentionLaunches();
            stats["batched_prefill_attention_requests"] =
                rt.BatchedPrefillAttentionRequests();
            stats["batched_prefill_attention_launches"] =
                rt.BatchedPrefillAttentionLaunches();
            stats["prefill_shape_forward_calls"] = rt.PrefillShapeForwardCalls();
            stats["prefill_shape_requests"] = rt.PrefillShapeRequests();
            stats["prefill_exact_groupable_requests"] =
                rt.PrefillExactGroupableRequests();
            stats["prefill_actual_query_tokens"] = rt.PrefillActualQueryTokens();
            stats["prefill_padded_query_tokens"] = rt.PrefillPaddedQueryTokens();
            py::dict prefillShapes;
            py::dict prefillBatchShapes;
            for (const auto &[shape, count] : rt.PrefillShapeHistogram()) {
                prefillShapes[py::str(shape)] = count;
            }
            for (const auto &[shape, count] : rt.PrefillBatchShapeHistogram()) {
                prefillBatchShapes[py::str(shape)] = count;
            }
            stats["prefill_shape_histogram"] = prefillShapes;
            stats["prefill_batch_shape_histogram"] = prefillBatchShapes;
            stats["aclnn_matmul_synchronizations"] =
                rt.AclnnMatmulSynchronizations();
            stats["lm_head_synchronizations"] = rt.LmHeadSynchronizations();
            stats["legacy_attention_requests"] = rt.LegacyAttentionRequests();
            stats["legacy_decode_attention_requests"] =
                rt.LegacyDecodeAttentionRequests();
            stats["legacy_prefill_attention_requests"] =
                rt.LegacyPrefillAttentionRequests();
            stats["decode_kv_gather_bytes"] = rt.DecodeKvGatherBytes();
            stats["native_atb_decode_requests"] = rt.NativeAtbDecodeRequests();
            stats["native_atb_decode_launches"] = rt.NativeAtbDecodeLaunches();
            stats["native_atb_cache_writes"] = rt.NativeAtbCacheWrites();
            stats["native_atb_staging_copy_bytes"] = rt.NativeAtbStagingBytes();
            stats["direct_atb_setup_count"] = rt.DirectAtbSetupCount();
            stats["direct_atb_execute_count"] = rt.DirectAtbExecuteCount();
            stats["direct_atb_decode_requests"] = rt.DirectAtbDecodeRequests();
            stats["direct_atb_attention_launches"] = rt.DirectAtbAttentionLaunches();
            stats["direct_atb_reshape_launches"] = rt.DirectAtbReshapeLaunches();
            stats["direct_atb_staging_copy_bytes"] = rt.DirectAtbStagingBytes();
            stats["direct_atb_fused_rope_staging_bytes"] =
                rt.DirectAtbFusedRopeStagingBytes();
            stats["direct_atb_plan_reuses"] = rt.DirectAtbPlanReuses();
            stats["direct_atb_plan_rebuilds"] = rt.DirectAtbPlanRebuilds();
            stats["direct_atb_paged_plan_reuses"] =
                rt.DirectAtbPagedPlanReuses();
            stats["direct_atb_paged_plan_rebuilds"] =
                rt.DirectAtbPagedPlanRebuilds();
            stats["direct_atb_reshape_plan_reuses"] =
                rt.DirectAtbReshapePlanReuses();
            stats["direct_atb_reshape_plan_rebuilds"] =
                rt.DirectAtbReshapePlanRebuilds();
            stats["direct_atb_mixed_decode_requests"] =
                rt.DirectAtbMixedDecodeRequests();
            stats["direct_atb_mixed_decode_launches"] =
                rt.DirectAtbMixedDecodeLaunches();
            stats["direct_atb_compact_launches"] = rt.DirectAtbCompactLaunches();
            stats["direct_atb_compact_bytes"] = rt.DirectAtbCompactBytes();
            stats["direct_atb_scatter_launches"] = rt.DirectAtbScatterLaunches();
            stats["direct_atb_scatter_bytes"] = rt.DirectAtbScatterBytes();
            stats["direct_atb_metadata_h2d_bytes"] =
                rt.DirectAtbMetadataH2DBytes();
            stats["direct_atb_metadata_h2d_launches"] =
                rt.DirectAtbMetadataH2DLaunches();
#endif
            stats["forward_input_events"] = rt.ForwardInputEvents();
            stats["forward_output_events"] = rt.ForwardOutputEvents();
            return stats;
        })
        .def("set_host_attention_metadata",
             [](XRuntime &rt, const std::vector<uint32_t> &lens,
                const std::vector<uint32_t> &cachedLens,
                const std::vector<uint32_t> &blockTables, uint32_t maxNumBlocks) {
                 if (lens.empty() || lens.size() != cachedLens.size() || maxNumBlocks == 0 ||
                     blockTables.size() != lens.size() * maxNumBlocks) {
                     throw std::invalid_argument("invalid host attention metadata");
                 }
                 rt._lensHost = lens;
                 rt._cachedLensHost = cachedLens;
                 rt._blockTablesHost = blockTables;
             },
             py::arg("lens"), py::arg("cached_lens"), py::arg("block_tables"),
             py::arg("max_num_blocks"))
        .def("configure_swizzle", &XRuntime::ConfigureSwizzle, py::arg("swizzle"),
             py::arg("use_swizzle_table"));

    py::class_<XModelConfig>(m, "ModelConfig")
        .def(py::init<>())
        .def_readwrite("vocab_size", &XModelConfig::vocabSize)
        .def_readwrite("hidden_size", &XModelConfig::hiddenSize)
        .def_readwrite("n_layers", &XModelConfig::nLayers)
        .def_readwrite("attn_type", &XModelConfig::attnType)
        .def_readwrite("rope_type", &XModelConfig::ropeType)
        .def_readwrite("n_heads", &XModelConfig::nHeads)
        .def_readwrite("n_kv_heads", &XModelConfig::nKvHeads)
        .def_readwrite("head_dim", &XModelConfig::headDim)
        .def_readwrite("nope_head_dim", &XModelConfig::nopeHeadDim)
        .def_readwrite("rope_head_dim", &XModelConfig::ropeHeadDim)
        .def_readwrite("v_head_dim", &XModelConfig::vHeadDim)
        .def_readwrite("q_lora_rank", &XModelConfig::qLoraRank)
        .def_readwrite("kv_lora_rank", &XModelConfig::kvLoraRank)
        .def_readwrite("quant_attn_weight_transpose",
                       &XModelConfig::quantAttnWeightTrans)  // only for quantization
        .def_readwrite("quant_attn_weight_nz",
                       &XModelConfig::quantAttnWeightNz)  // only for quantization
        .def_readwrite("norm_eps", &XModelConfig::normEps)
        .def_readwrite("rope_theta", &XModelConfig::ropeTheta)
        .def_readwrite("softmax_scale", &XModelConfig::softmaxScale)
        .def_readwrite("n_dense_layers", &XModelConfig::nDenseLayers)
        .def_readwrite("n_routed_experts", &XModelConfig::nRoutedExperts)
        .def_readwrite("n_shared_experts", &XModelConfig::nSharedExperts)
        .def_readwrite("n_expert_groups", &XModelConfig::nExpertGroups)
        .def_readwrite("n_limited_groups", &XModelConfig::nLimitedGroups)
        .def_readwrite("n_act_experts", &XModelConfig::nActExperts)
        .def_readwrite("intermediate_size", &XModelConfig::intermediateSize)
        .def_readwrite("moe_intermediate_size", &XModelConfig::moeIntermediateSize)
        .def_readwrite("route_scale", &XModelConfig::routeScale)
        .def_readwrite("def_tp_size", &XModelConfig::defTpSize)
        .def_readwrite("def_dp_size", &XModelConfig::defDpSize)
        .def_readwrite("moe_ep_size", &XModelConfig::moeEpSize)
        .def_readwrite("moe_tp_size", &XModelConfig::moeTPSize)
        .def_readwrite("max_seq_len", &XModelConfig::maxSeqLen)
        .def_readwrite("max_batch_size", &XModelConfig::maxBatch)
        .def_readwrite("max_m", &XModelConfig::maxBatchedTokens)
        .def_readwrite("max_num_batched_tokens", &XModelConfig::maxBatchedTokens)
        .def_readwrite("block_size", &XModelConfig::blockSize)
        .def_readwrite("block_sizes", &XModelConfig::blockSizes)
        .def_readwrite("weight_nz", &XModelConfig::weightNZ)
        .def_readwrite("experts_weight_transpose", &XModelConfig::expertsWeightTrans)
        .def_readwrite("experts_weight_nz", &XModelConfig::expertsWeightNZ)
        .def_readwrite("gate_captured", &XModelConfig::gateCaptured)
        .def_readwrite("quant_msd_w4a8", &XModelConfig::quantMsdW4a8)
        .def_readwrite("qkv_bias", &XModelConfig::addBias)
        .def_readwrite("qk_norm", &XModelConfig::qkNorm)
        .def_readwrite("qk_norm_full", &XModelConfig::qkNormFull)
        .def_readwrite("attn_output_gate", &XModelConfig::attnOutputGate)
        .def_readwrite("scoring_func", &XModelConfig::scoringFunc)
        .def_readwrite("norm_topk_prob", &XModelConfig::normTopKProb)
        .def_readwrite("mrope_section", &XModelConfig::mropeSection)
        .def_readwrite("mrope_interleaved", &XModelConfig::mropeInterleaved)
        .def_readwrite("deepstack_num_level", &XModelConfig::deepstackNumLevel)
        .def_readwrite("index_head_dim", &XModelConfig::indexHeadDim)
        .def_readwrite("index_n_heads", &XModelConfig::indexNHeads)
        .def_readwrite("index_topk", &XModelConfig::indexTopK)
        .def_readwrite("index_softmax_scale", &XModelConfig::indexSoftmaxScale)
        .def_readwrite("index_rope_interleaved", &XModelConfig::indexRopeInterleaved)
        .def_readwrite("index_full_mask", &XModelConfig::indexFullMask)
        .def_readwrite("linear_num_k_heads", &XModelConfig::linearNumKHeads)
        .def_readwrite("linear_num_v_heads", &XModelConfig::linearNumVHeads)
        .def_readwrite("linear_key_head_dim", &XModelConfig::linearKeyHeadDim)
        .def_readwrite("linear_value_head_dim", &XModelConfig::linearValueHeadDim)
        .def_readwrite("linear_conv_kernel_dim", &XModelConfig::linearConvKernelDim)
        .def_readwrite("full_attention_interval", &XModelConfig::fullAttentionInterval)
        .def_readwrite("o_groups", &XModelConfig::oGroups)
        .def_readwrite("o_lora_rank", &XModelConfig::oLoraRank)
        .def_readwrite("window_size", &XModelConfig::windowSize)
        .def_readwrite("compress_rope_theta", &XModelConfig::compressRopeTheta)
        .def_readwrite("original_seq_len", &XModelConfig::originalSeqLen)
        .def_readwrite("rope_factor", &XModelConfig::ropeFactor)
        .def_readwrite("beta_fast", &XModelConfig::betaFast)
        .def_readwrite("beta_slow", &XModelConfig::betaSlow)
        .def_readwrite("hc_mult", &XModelConfig::hcMult)
        .def_readwrite("hc_sinkhorn_iters", &XModelConfig::hcSinkhornIters)
        .def_readwrite("hc_eps", &XModelConfig::hcEps)
        .def_readwrite("swiglu_limit", &XModelConfig::swigluLimit)
        .def_readwrite("n_hash_layers", &XModelConfig::nHashLayers)
        .def_readwrite("compress_ratios", &XModelConfig::compressRatios);

    py::class_<CModelAttnMeta>(m, "AttnMeta")
        .def(py::init<>())
        .def_readwrite("lens", &CModelAttnMeta::lens)
        .def_readwrite("cached_lens", &CModelAttnMeta::cachedLens)
        .def_readwrite("block_tables_cpu", &CModelAttnMeta::blockTablesList)
        .def_readwrite("positions", &CModelAttnMeta::positions);

    py::class_<CModelAttnMetaV2>(m, "AttnMetaV2")
        .def(py::init<>())
        .def_readwrite("lens", &CModelAttnMetaV2::lens)
        .def_readwrite("cached_lens", &CModelAttnMetaV2::cachedLens)
        .def_readwrite("positions", &CModelAttnMetaV2::positions)
        .def_readwrite("lens_cpu", &CModelAttnMetaV2::lensCpu)
        .def_readwrite("cached_lens_cpu", &CModelAttnMetaV2::cachedLensCpu)
        .def_readwrite("query_start_loc", &CModelAttnMetaV2::queryStartLoc)
        .def_readwrite("slot_mapping", &CModelAttnMetaV2::slotMapping)
        .def_readwrite("block_tables", &CModelAttnMetaV2::blockTables);

    py::enum_<XModelAttnType>(m, "AttnType")
        .value("AttnMHA", XModelAttnType::XMODEL_ATTN_MHA)
        .value("AttnMLA", XModelAttnType::XMODEL_ATTN_MLA)
        .value("AttnDSA", XModelAttnType::XMODEL_ATTN_DSA)
        .value("AttnHybrid", XModelAttnType::XMODEL_ATTN_HYBRID)
        .value("AttnCxA", XModelAttnType::XMODEL_ATTN_CXA)
        .export_values();
    py::enum_<XModelRopeType>(m, "RopeType")
        .value("RopeNeox", XModelRopeType::XMODEL_ROPE_NEOX)
        .value("RopeGptj", XModelRopeType::XMODEL_ROPE_GPTJ)
        .export_values();

    py::enum_<XModelScoringFuncType>(m, "ScoringFuncType")
        .value("ScoringFuncSoftmax", XModelScoringFuncType::XMODEL_SCORING_FUNC_SOFTMAX)
        .value("ScoringFuncSigmoid", XModelScoringFuncType::XMODEL_SCORING_FUNC_SIGMOID)
        .value("ScoringFuncSqrtsoftplus", XModelScoringFuncType::XMODEL_SCORING_FUNC_SQRTSOFTPLUS)
        .export_values();

    py::class_<_CModel>(m, "Model")
        .def(py::init<>())
        .def_readwrite("embed", &_CModel::embed)
        .def_readwrite("norm", &_CModel::norm)
        .def_readwrite("norm_bias", &_CModel::normBias)
        .def_readwrite("head", &_CModel::head)
        .def_readwrite("attn_norm", &_CModel::attnNorm)
        .def_readwrite("attn_norm_bias", &_CModel::attnNormBias)
        .def_readwrite("attn_out", &_CModel::attnOut)
        .def_readwrite("attn_out_input_scale", &_CModel::attnOutInputScale)
        .def_readwrite("attn_out_input_offset", &_CModel::attnOutInputOffset)
        .def_readwrite("attn_out_quant_bias", &_CModel::attnOutQuantBias)
        .def_readwrite("attn_out_deq_scale", &_CModel::attnOutDeqScale)
        .def_readwrite("mha_qkv", &_CModel::mhaQKV)
        .def_readwrite("mha_qkv_bias", &_CModel::mhaQKVBias)
        .def_readwrite("mha_qkv_input_scale", &_CModel::mhaQKVInputScale)
        .def_readwrite("mha_qkv_input_offset", &_CModel::mhaQKVInputOffset)
        .def_readwrite("mha_qkv_quant_bias", &_CModel::mhaQKVQuantBias)
        .def_readwrite("mha_qkv_deq_scale", &_CModel::mhaQKVDeqScale)
        .def_readwrite("mha_q_norm", &_CModel::mhaQNorm)
        .def_readwrite("mha_q_norm_bias", &_CModel::mhaQNormBias)
        .def_readwrite("mha_k_norm", &_CModel::mhaKNorm)
        .def_readwrite("mha_k_norm_bias", &_CModel::mhaKNormBias)
        .def_readwrite("mla_qkv_a", &_CModel::mlaQKVA)
        .def_readwrite("mla_q_b", &_CModel::mlaQB)
        .def_readwrite("mla_q_norm", &_CModel::mlaQNorm)
        .def_readwrite("mla_q_norm_bias", &_CModel::mlaQNormBias)
        .def_readwrite("mla_wuv", &_CModel::mlaWUV)
        .def_readwrite("mla_wuk_t", &_CModel::mlaWUKT)
        .def_readwrite("mla_kv_norm", &_CModel::mlaKVNorm)
        .def_readwrite("mla_kv_norm_bias", &_CModel::mlaKVNormBias)
        .def_readwrite("mla_qkv_a_input_scale", &_CModel::mlaQKVAInputScale)
        .def_readwrite("mla_qkv_a_input_offset", &_CModel::mlaQKVAInputOffset)
        .def_readwrite("mla_qkv_a_quant_bias", &_CModel::mlaQKVAQuantBias)
        .def_readwrite("mla_qkv_a_deq_scale", &_CModel::mlaQKVADeqScale)
        .def_readwrite("mla_q_b_input_scale", &_CModel::mlaQBInputScale)
        .def_readwrite("mla_q_b_input_offset", &_CModel::mlaQBInputOffset)
        .def_readwrite("mla_q_b_quant_bias", &_CModel::mlaQBQuantBias)
        .def_readwrite("mla_q_b_deq_scale", &_CModel::mlaQBDeqScale)
        .def_readwrite("index_q_b", &_CModel::indexQB)
        .def_readwrite("index_q_b_input_scale", &_CModel::indexQBInputScale)
        .def_readwrite("index_q_b_input_offset", &_CModel::indexQBInputOffset)
        .def_readwrite("index_q_b_quant_bias", &_CModel::indexQBQuantBias)
        .def_readwrite("index_q_b_deq_scale", &_CModel::indexQBDeqScale)
        .def_readwrite("index_k_weights_proj", &_CModel::indexKWeightsProj)
        .def_readwrite("index_k_norm", &_CModel::indexKNorm)
        .def_readwrite("index_k_norm_bias", &_CModel::indexKNormBias)
        .def_readwrite("linear_in_proj_qkv", &_CModel::linearInProjQKV)
        .def_readwrite("linear_in_proj_z", &_CModel::linearInProjZ)
        .def_readwrite("linear_in_proj_b", &_CModel::linearInProjB)
        .def_readwrite("linear_in_proj_a", &_CModel::linearInProjA)
        .def_readwrite("linear_conv1d", &_CModel::linearConv1d)
        .def_readwrite("linear_a_log", &_CModel::linearALog)
        .def_readwrite("linear_dt_bias", &_CModel::linearDtBias)
        .def_readwrite("linear_norm", &_CModel::linearNorm)
        .def_readwrite("linear_out_proj", &_CModel::linearOutProj)
        .def_readwrite("mlp_norm", &_CModel::mlpNorm)
        .def_readwrite("mlp_norm_bias", &_CModel::mlpNormBias)
        .def_readwrite("mlp_up_gate", &_CModel::mlpUpGate)
        .def_readwrite("mlp_up_gate_input_scale", &_CModel::mlpUpGateInputScale)
        .def_readwrite("mlp_up_gate_input_offset", &_CModel::mlpUpGateInputOffset)
        .def_readwrite("mlp_up_gate_quant_bias", &_CModel::mlpUpGateQuantBias)
        .def_readwrite("mlp_up_gate_deq_scale", &_CModel::mlpUpGateDeqScale)
        .def_readwrite("mlp_down", &_CModel::mlpDown)
        .def_readwrite("mlp_down_input_scale", &_CModel::mlpDownInputScale)
        .def_readwrite("mlp_down_input_offset", &_CModel::mlpDownInputOffset)
        .def_readwrite("mlp_down_quant_bias", &_CModel::mlpDownQuantBias)
        .def_readwrite("mlp_down_deq_scale", &_CModel::mlpDownDeqScale)
        .def_readwrite("gate", &_CModel::moeGate)
        .def_readwrite("gate_bias", &_CModel::moeGateBias)
        .def_readwrite("tid2eid", &_CModel::moeTid2Eid)
        .def_readwrite("se_up_gate", &_CModel::moeSEUpGate)
        .def_readwrite("se_up_gate_deq_scale", &_CModel::moeSEUpGateDeqScale)
        .def_readwrite("se_down", &_CModel::moeSEDown)
        .def_readwrite("se_down_deq_scale", &_CModel::moeSEDownDeqScale)
        .def_readwrite("se_gate", &_CModel::moeSEGate)
        .def_readwrite("re_up_gate", &_CModel::moeREUpGate)
        .def_readwrite("re_up_gate_scale", &_CModel::moeREUpGateDeqScale)
        .def_readwrite("re_up_gate_deq_scale", &_CModel::moeREUpGateDeqScale)
        .def_readwrite("re_down", &_CModel::moeREDown)
        .def_readwrite("re_down_scale", &_CModel::moeREDownDeqScale)
        .def_readwrite("re_down_deq_scale", &_CModel::moeREDownDeqScale)
        .def_readwrite("re_up_gate_scale_bias", &_CModel::moeREUpGateScaleBias)
        .def_readwrite("re_down_scale_bias", &_CModel::moeREDownScaleBias)
        // DeepSeek-V4 (CxA)
        .def_readwrite("attn_sink", &_CModel::attnSink)
        .def_readwrite("attn_wq_a", &_CModel::attnWqA)
        .def_readwrite("attn_wq_a_input_scale", &_CModel::attnWqAInputScale)
        .def_readwrite("attn_wq_a_input_offset", &_CModel::attnWqAInputOffset)
        .def_readwrite("attn_wq_a_quant_bias", &_CModel::attnWqAQuantBias)
        .def_readwrite("attn_wq_a_deq_scale", &_CModel::attnWqADeqScale)
        .def_readwrite("attn_wo_a", &_CModel::attnWoA)
        .def_readwrite("attn_wo_b", &_CModel::attnWoB)
        .def_readwrite("attn_wkv", &_CModel::attnWKv)
        .def_readwrite("attn_wkv_input_scale", &_CModel::attnWKvInputScale)
        .def_readwrite("attn_wkv_input_offset", &_CModel::attnWKvInputOffset)
        .def_readwrite("attn_wkv_quant_bias", &_CModel::attnWKvQuantBias)
        .def_readwrite("attn_wkv_deq_scale", &_CModel::attnWKvDeqScale)
        .def_readwrite("comp_ape", &_CModel::compApe)
        .def_readwrite("comp_w_kv", &_CModel::compWKv)
        .def_readwrite("comp_w_gate", &_CModel::compWGate)
        .def_readwrite("comp_norm", &_CModel::compNorm)
        .def_readwrite("idx_wq_b", &_CModel::idxWqB)
        .def_readwrite("idx_wq_b_input_scale", &_CModel::idxWqBInputScale)
        .def_readwrite("idx_wq_b_input_offset", &_CModel::idxWqBInputOffset)
        .def_readwrite("idx_wq_b_quant_bias", &_CModel::idxWqBQuantBias)
        .def_readwrite("idx_wq_b_deq_scale", &_CModel::idxWqBDeqScale)
        .def_readwrite("idx_weights_proj", &_CModel::idxWeightsProj)
        .def_readwrite("idx_comp_ape", &_CModel::idxCompApe)
        .def_readwrite("idx_comp_w_kv", &_CModel::idxCompWKv)
        .def_readwrite("idx_comp_w_gate", &_CModel::idxCompWGate)
        .def_readwrite("idx_comp_norm", &_CModel::idxCompNorm)
        .def_readwrite("hc_attn_fn", &_CModel::hcAttnFn)
        .def_readwrite("hc_ffn_fn", &_CModel::hcFfnFn)
        .def_readwrite("hc_attn_base", &_CModel::hcAttnBase)
        .def_readwrite("hc_ffn_base", &_CModel::hcFfnBase)
        .def_readwrite("hc_attn_scale", &_CModel::hcAttnScale)
        .def_readwrite("hc_ffn_scale", &_CModel::hcFfnScale)
        .def_readwrite("hc_head_fn", &_CModel::hcHeadFn)
        .def_readwrite("hc_head_base", &_CModel::hcHeadBase)
        .def_readwrite("hc_head_scale", &_CModel::hcHeadScale)
        .def("init", &_CModel::Init, "model init", py::arg("config"), py::arg("rank") = 0)
        .def("set_native_kv_cache_310p", &_CModel::SetNativeKvCache310P,
             py::arg("kv_cache"))
        .def("forward", &_CModel::ForwardV1, "forward", py::arg("rt"), py::arg("input"),
             py::arg("attn_meta"), py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward_get_logits", &_CModel::ForwardGetLogits, "forward_get_logits", py::arg("rt"),
             py::arg("input"), py::arg("indices"), py::arg("output"), py::arg("curr_stream") = 0,
             py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits", &_CModel::ForwardAndGetLogitsV1, "forward_and_get_logits",
             py::arg("rt"), py::arg("input"), py::arg("attn_meta"), py::arg("kv_cache"),
             py::arg("freqs_cis"), py::arg("indices"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward_with_inputs_embeds", &_CModel::ForwardWithInputsEmbedsV1,
             "forward_with_inputs_embeds", py::arg("rt"), py::arg("input"), py::arg("attn_meta"),
             py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::arg("deepstack_input") = std::vector<at::Tensor>{},
             py::arg("input_ids") = std::nullopt, py::call_guard<py::gil_scoped_release>())
        .def("forward_v2", &_CModel::ForwardV2, "forward_v2", py::arg("rt"), py::arg("input"),
             py::arg("attn_meta"), py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward_and_get_logits_v2", &_CModel::ForwardAndGetLogitsV2,
             "forward_and_get_logits_v2", py::arg("rt"), py::arg("input"), py::arg("attn_meta"),
             py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("indices"), py::arg("output"),
             py::arg("curr_stream") = 0, py::call_guard<py::gil_scoped_release>())
        .def("forward_with_inputs_embeds_v2", &_CModel::ForwardWithInputsEmbedsV2,
             "forward_with_inputs_embeds_v2", py::arg("rt"), py::arg("input"), py::arg("attn_meta"),
             py::arg("kv_cache"), py::arg("freqs_cis"), py::arg("output"),
             py::arg("curr_stream") = 0, py::arg("deepstack_input") = std::vector<at::Tensor>{},
             py::arg("input_ids") = std::nullopt, py::call_guard<py::gil_scoped_release>())
        .def("get_tensor_pool_size", &_CModel::GetTensorPoolSize, "get_tensor_pool_size",
             py::arg("dbg") = 0);

    py::class_<XCoreAssigner>(m, "CoreAssigner")
        .def(py::init<float>(), py::arg("prefill_ratio"))
        .def("assign_core", &XCoreAssigner::AssignCore, py::arg("is_decode"),
             py::call_guard<py::gil_scoped_release>())
        .def("release_core", &XCoreAssigner::ReleaseCore, py::arg("is_decode"),
             py::call_guard<py::gil_scoped_release>());

    // kernels
    m.def("all_gather", &AllGather, py::arg("rt"), py::arg("out"), py::arg("in_"),
          py::arg("comm_type") = 0);
    m.def("reduce_scatter", &ReduceScatter, py::arg("rt"), py::arg("out"), py::arg("in_"),
          py::arg("comm_type") = 0);
    m.def("all_reduce", &AllReduce, py::arg("rt"), py::arg("out"), py::arg("in_"),
          py::arg("comm_type") = 0);
    m.def("alltoallv", &AlltoAllV, py::arg("rt"), py::arg("out"), py::arg("in_"),
          py::arg("send_counts"), py::arg("recv_counts"), py::arg("sdispls"), py::arg("rdispls"),
          py::arg("comm_type") = 0);
    m.def("add", &Add, py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"));
    m.def("probe_310p", &Probe310P, py::arg("rt"), py::arg("out"),
          py::arg("value") = 0x03102002U);
    m.def("official_add_probe_310p", &OfficialAddProbe310P, py::arg("rt"), py::arg("x"),
          py::arg("y"), py::arg("z"));
    m.def("matmul", &Matmul, "matmul", py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("matmul_bench", &MatmulBench, py::arg("rt"), py::arg("x"), py::arg("y"), py::arg("z"),
          py::arg("x_warmup"), py::arg("y_warmup"), py::arg("z_warmup"), py::arg("iterations"),
          py::arg("warmup_iterations"), py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("matmul_with_bias", &MatmulWithBias, "matmul_with_bias", py::arg("rt"), py::arg("x"),
          py::arg("y"), py::arg("z"), py::arg("bias"), py::arg("weight_nz") = false);
    m.def("embed", &Embed, py::arg("rt"), py::arg("weight"), py::arg("in_"), py::arg("out"),
          py::arg("start"), py::arg("end"));
    m.def("rmsnorm", &RMSNorm, "rmsnorm", py::arg("rt"), py::arg("in_"), py::arg("norm"),
          py::arg("out"), py::arg("norm_eps"), py::arg("norm_dim") = 0,
          py::arg("cnt_per_token") = 1, py::arg("in_start_offset") = 0,
          py::arg("out_start_offset") = 0, py::arg("variance") = std::nullopt);
    m.def("rmsnorm_variance_only", &RMSNormVarianceOnly, "rmsnorm_variance_only", py::arg("rt"),
          py::arg("in_"), py::arg("out"), py::arg("norm_eps"), py::arg("norm_dim") = 0,
          py::arg("cnt_per_token") = 1, py::arg("in_start_offset") = 0,
          py::arg("out_start_offset") = 0);
    m.def("rmsnorm_with_bias", &RMSNormWithBias, "rmsnorm_with_bias", py::arg("rt"), py::arg("in_"),
          py::arg("norm"), py::arg("norm_bias"), py::arg("out"), py::arg("norm_eps"),
          py::arg("norm_dim") = 0, py::arg("cnt_per_token") = 1, py::arg("in_start_offset") = 0,
          py::arg("out_start_offset") = 0);
    m.def("qk_rmsnorm_310p", &QkRmsNorm310P, py::arg("rt"), py::arg("in_"),
          py::arg("q_norm"), py::arg("k_norm"), py::arg("out"), py::arg("norm_eps"),
          py::arg("n_heads") = 16, py::arg("n_kv_heads") = 8,
          py::arg("head_dim") = 128);
    m.def("layernorm", &LayerNorm, py::arg("rt"), py::arg("in_"), py::arg("norm"),
          py::arg("norm_bias"), py::arg("out"), py::arg("norm_eps"), py::arg("norm_dim"));
    m.def("l2norm", &L2Norm, py::arg("rt"), py::arg("in_"), py::arg("out"), py::arg("norm_eps"),
          py::arg("norm_dim") = 0);
    m.def("add_bias", &AddBias, py::arg("rt"), py::arg("in_"), py::arg("weight"), py::arg("out"));
    m.def("silu_and_mul", &SiluAndMul, py::arg("rt"), py::arg("in_"), py::arg("out"),
          py::arg("swiglu_limit") = 0.0f);
    m.def("sigmoid_gate_mul", &SigmoidGateMul, py::arg("rt"), py::arg("attn"), py::arg("gate"),
          py::arg("out"));
    m.def("rope_and_cache", &RopeAndCache, "rope_and_cache", py::arg("rt"), py::arg("inout"),
          py::arg("k_cache"), py::arg("v_cache"), py::arg("position"), py::arg("cosin"),
          py::arg("slot_mapping"), py::arg("n_heads"), py::arg("n_kv_heads"), py::arg("head_dim"),
          py::arg("rot_dim"), py::arg("block_size"), py::arg("is_neox"),
          py::arg("mrope_mask_h") = 0, py::arg("mrope_mask_w") = 0);
    m.def("attention", &Attention, py::arg("rt"), py::arg("qkv"), py::arg("k_cache"),
          py::arg("v_cache"), py::arg("output"), py::arg("query_start_loc"), py::arg("lens"),
          py::arg("cached_lens"), py::arg("block_tables"), py::arg("n_heads"),
          py::arg("n_kv_heads"), py::arg("head_dim"), py::arg("block_size"), py::arg("batch"),
          py::arg("enable_flash_attention") = false, py::arg("tile_size_of_cached_kv") = 8192);
    m.def("add_and_rmsnorm", &AddAndRMSNorm, py::arg("rt"), py::arg("in_"), py::arg("add_in_out"),
          py::arg("norm"), py::arg("out"), py::arg("norm_eps"));
    m.def("softmax_topk", &SoftmaxTopK, py::arg("rt"), py::arg("scores"), py::arg("indices"),
          py::arg("out_weights"), py::arg("out_routing"), py::arg("top_k"),
          py::arg("norm_top_k_prob"));
    m.def("sigmoid_topk", &SigmoidTopK, py::arg("rt"), py::arg("scores"), py::arg("indices"),
          py::arg("bias"), py::arg("scale"), py::arg("out_weights"), py::arg("out_routing"),
          py::arg("n_group"), py::arg("n_topk_group"), py::arg("top_k"),
          py::arg("norm_top_k_prob"));
    m.def("sqrtsoftplus_hash_topk", &SqrtsoftplusHashTopK, py::arg("rt"), py::arg("scores"),
          py::arg("indices"), py::arg("bias"), py::arg("input_ids"), py::arg("tid2eid"),
          py::arg("out_weights"), py::arg("routing_map"), py::arg("scale"), py::arg("top_k"),
          py::arg("hash"));
    m.def("topk", &TopK, py::arg("rt"), py::arg("scores"), py::arg("indices"),
          py::arg("outIndices"), py::arg("query_lens"), py::arg("cached_lens"), py::arg("k"));
    m.def("cast_up", &CastUp, py::arg("rt"), py::arg("in_"), py::arg("out"));
    m.def("permutation", &Permutation, py::arg("rt"), py::arg("in_"), py::arg("routing"),
          py::arg("start"), py::arg("end"), py::arg("out"), py::arg("unp_idx"), py::arg("counts"));
    m.def("unpermutation", &UnPermutation, py::arg("rt"), py::arg("in_"), py::arg("routing"),
          py::arg("weights"), py::arg("start"), py::arg("end"), py::arg("out"), py::arg("unp_idx"));
    m.def("group_matmul", &GroupMatmul, py::arg("rt"), py::arg("in_"), py::arg("weights"),
          py::arg("scales"), py::arg("counts"), py::arg("start"), py::arg("end"),
          py::arg("out_dim"), py::arg("in_dim"), py::arg("output"), py::arg("weight_nz"),
          py::arg("transpose"));
    m.def("softmax", &Softmax, py::arg("rt"), py::arg("x"), py::arg("calc_len"),
          py::arg("is_long"));
    m.def("rope_complex", &RopeComplex, "rope_complex", py::arg("rt"), py::arg("n_local_heads"),
          py::arg("step_dim"), py::arg("rope_dim"), py::arg("input_with_r"), py::arg("freqs"),
          py::arg("position"), py::arg("output"), py::arg("inverse") = false,
          py::arg("out_interleaved") = false);
    m.def("rope_complex_and_cache", &RopeComplexAndCache, "rope_complex_and_cache", py::arg("rt"),
          py::arg("n_local_heads"), py::arg("step_dim"), py::arg("rope_dim"), py::arg("offset"),
          py::arg("vdim"), py::arg("input_with_r"), py::arg("freqs"), py::arg("position"),
          py::arg("block_size"), py::arg("v_cache"), py::arg("slot_mapping"),
          py::arg("out_interleaved") = false);
    m.def("mla_prepare", &MlaPrepare, py::arg("rt"), py::arg("attn_qkvc"), py::arg("q_norm"),
          py::arg("q_norm_bias"), py::arg("attn_norm_qc"), py::arg("kv_norm"),
          py::arg("kv_norm_bias"), py::arg("attn_norm_kvc"), py::arg("freqs"), py::arg("position"),
          py::arg("q_lora_rank"), py::arg("kv_lora_rank"), py::arg("rope_head_dim"),
          py::arg("block_size"), py::arg("k_cache"), py::arg("pe_cache"), py::arg("slot_mapping"),
          py::arg("norm_eps"));
    m.def("indexer_prepare", &IndexerPrepare, py::arg("rt"), py::arg("kw"), py::arg("k_norm"),
          py::arg("k_norm_bias"), py::arg("freqs"), py::arg("position"), py::arg("index_head_dim"),
          py::arg("index_n_heads"), py::arg("rope_head_dim"), py::arg("block_size"),
          py::arg("index_k_cache"), py::arg("slot_mapping"), py::arg("norm_eps"), py::arg("q"),
          py::arg("scale"), py::arg("top_k"), py::arg("is_long"));
    m.def("quant", &Quant, py::arg("rt"), py::arg("x"), py::arg("scale_reciprocal"),
          py::arg("offset"), py::arg("out"));
    m.def("quant_dynamic", &QuantDyn, py::arg("rt"), py::arg("x"), py::arg("scale"),
          py::arg("out"));
    m.def("msd_merge_dequant", &MSDMergeDequant, "msd_merge_dequant", py::arg("rt"),
          py::arg("y_merged"), py::arg("scale_biases"), py::arg("counts"),
          py::arg("per_token_scale"), py::arg("out"));
    m.def("matmul_dequant", &MatmulDeQuant, "matmul_dequant", py::arg("rt"), py::arg("x"),
          py::arg("y"), py::arg("bias"), py::arg("deq_scale"), py::arg("z"),
          py::arg("weight_nz") = false, py::arg("transpose") = false);
    m.def("dequant", &DeQuant, py::arg("rt"), py::arg("in_"), py::arg("scale"), py::arg("out"),
          py::arg("has_scale"));
    m.def("mla_v2", &MLAV2, py::arg("rt"), py::arg("q_with_qr"), py::arg("qr"), py::arg("k_cache"),
          py::arg("pe_cache"), py::arg("wuk_t"), py::arg("wuv"), py::arg("output"),
          py::arg("query_start_loc"), py::arg("lens"), py::arg("cached_lens"),
          py::arg("block_tables"), py::arg("n_heads"), py::arg("rope_head_dim"),
          py::arg("nope_head_dim"), py::arg("v_head_dim"), py::arg("kv_lora_rank"),
          py::arg("block_size"), py::arg("batch"), py::arg("scale"), py::arg("topk_indices"),
          py::arg("top_k") = 0, py::arg("nz") = false, py::arg("enable_flash_attention") = false,
          py::arg("tile_size_of_cached_kv") = 8192);
    m.def("gather_sparse_kv_cache", &GatherSparseKVCache, py::arg("rt"), py::arg("k_cache"),
          py::arg("pe_cache"), py::arg("block_tables"), py::arg("topk_indices"),
          py::arg("query_lens"), py::arg("cached_lens"), py::arg("k_dense_cache"),
          py::arg("pe_dense_cache"), py::arg("batch"), py::arg("index_topk"), py::arg("block_size"),
          py::arg("kv_lora_rank"), py::arg("rope_head_dim"), py::arg("kv_heads") = 1);
    m.def("mla_v3", &MLAV3, py::arg("rt"), py::arg("q_absorb"), py::arg("qr"),
          py::arg("k_dense_cache"), py::arg("pe_dense_cache"), py::arg("o_absorb"),
          py::arg("query_start_loc"), py::arg("lens"), py::arg("cached_lens"), py::arg("n_heads"),
          py::arg("rope_head_dim"), py::arg("kv_lora_rank"), py::arg("batch"),
          py::arg("index_topk"), py::arg("scale"));
    m.def("indexer_scores", &IndexerScores, py::arg("rt"), py::arg("q"), py::arg("k_cache"),
          py::arg("weight"), py::arg("scores"), py::arg("query_start_loc"), py::arg("lens"),
          py::arg("cached_lens"), py::arg("block_tables"), py::arg("n_heads"), py::arg("head_dim"),
          py::arg("block_size"), py::arg("batch"));
    m.def("indexer_topk", &IndexerTopK, py::arg("rt"), py::arg("q"), py::arg("k_cache"),
          py::arg("weight"), py::arg("indices"), py::arg("topk_indices"),
          py::arg("query_start_loc"), py::arg("lens"), py::arg("cached_lens"),
          py::arg("block_tables"), py::arg("n_heads"), py::arg("head_dim"), py::arg("block_size"),
          py::arg("batch"), py::arg("top_k"));
    m.def("muls", &Muls, py::arg("rt"), py::arg("input"), py::arg("scale"), py::arg("output"));
    m.def("experts_counts_sum", &ExpertsCountsSum, py::arg("rt"), py::arg("experts_counts_input"),
          py::arg("tokens_per_epgroup"), py::arg("experts_counts_output"),
          py::arg("n_routed_experts"));
    m.def("reorder_moe", &ReorderMoE, py::arg("rt"), py::arg("in_"), py::arg("out"),
          py::arg("counts"), py::arg("hidden_size"), py::arg("local_start"), py::arg("local_end"),
          py::arg("forward"));
    m.def("linear_att_proj", &LinearAttProj, py::arg("rt"), py::arg("x"), py::arg("W_qkv"),
          py::arg("W_z"), py::arg("W_b"), py::arg("W_a"), py::arg("mix_qkv"), py::arg("z"),
          py::arg("b"), py::arg("a"), py::arg("m"), py::arg("n"), py::arg("v"), py::arg("h"),
          py::arg("k"));
    m.def("transpose_1_2", &Transpose_1_2, py::arg("rt"), py::arg("input"), py::arg("output"));
    m.def("linear_att_conv_and_silu", &LinearAttConv1dAndSiLU, py::arg("rt"), py::arg("mix_qkv"),
          py::arg("conv_state"), py::arg("weight"), py::arg("output"),
          py::arg("query_start_loc") = py::none(), py::arg("query_lens") = py::none());
    m.def("linear_att_conv_and_silu_token", &LinearAttConv1dAndSiLUToken, py::arg("rt"),
          py::arg("mix_qkv"), py::arg("conv_state"), py::arg("weight"), py::arg("output"),
          py::arg("seq_len"));
    m.def("split_col", &SplitCol, py::arg("rt"), py::arg("in"), py::arg("outputs"));
    m.def("concat", &Concat, py::arg("rt"), py::arg("inputs"), py::arg("out"));
    m.def("concat_col", &ConcatCol, py::arg("rt"), py::arg("inputs"), py::arg("out"));
    m.def("split", &Split, py::arg("rt"), py::arg("in"), py::arg("outputs"), py::arg("sizes"),
          py::arg("num_packets"));
    m.def("beta_decay", &BetaDecay, py::arg("rt"), py::arg("b"), py::arg("a"), py::arg("A_log"),
          py::arg("dt_bias"), py::arg("beta"), py::arg("g"), py::arg("bsz"), py::arg("seqlen"),
          py::arg("num_v_heads"));
    m.def("recurrent_gated_delta_rule", &RecurrentGatedDeltaRule, py::arg("rt"), py::arg("query"),
          py::arg("key"), py::arg("value"), py::arg("beta"), py::arg("g"), py::arg("state"),
          py::arg("out"), py::arg("batch"), py::arg("seqlen"), py::arg("num_heads"),
          py::arg("k_dim"), py::arg("v_dim"), py::arg("query_start_loc") = py::none(),
          py::arg("query_lens") = py::none());
    m.def("einsum_mht_hdt_mhd", &EinsumMhtHdtMhd, "einsum_mht_hdt_mhd", py::arg("rt"),
          py::arg("mht"), py::arg("hdt"), py::arg("mhd"), py::arg("m"), py::arg("h"), py::arg("t"),
          py::arg("d"), py::arg("weight_nz") = false);
    m.def("einsum_mht_htd_mhd", &EinsumMhtHtdMhd, "einsum_mht_htd_mhd", py::arg("rt"),
          py::arg("mht"), py::arg("htd"), py::arg("mhd"), py::arg("m"), py::arg("h"), py::arg("t"),
          py::arg("d"), py::arg("weight_nz") = false);
    m.def("unpack_activation", &UnpackActivation, py::arg("rt"), py::arg("input"),
          py::arg("output"));
    m.def("hc_act", &HcAct, py::arg("rt"), py::arg("mixes"), py::arg("hc_scale"),
          py::arg("hc_base"), py::arg("post"), py::arg("comb"), py::arg("hc_mult"), py::arg("eps"),
          py::arg("sinkhorn_iters"), py::arg("x_resid"), py::arg("output"));
    m.def("hc_post", &HcPost, py::arg("rt"), py::arg("x"), py::arg("post"), py::arg("comb"),
          py::arg("residual"), py::arg("y"), py::arg("m"), py::arg("hc_mult"), py::arg("hidden"));
    // funcs
    m.def("print", &Print, "print", py::arg("x"), py::arg("name") = "", py::arg("row") = 6,
          py::arg("col") = 6);
    m.def("get_tile_size_of_cached_kv", &GetTileSizeOfCachedKV,
          "Get optimal tile size for cached KV based on workload", py::arg("cached_lens"),
          py::arg("query_lens"), py::arg("head_num_in_group"), py::arg("n_kv_heads"),
          py::arg("block_size"), py::arg("aic_num"));
}
