/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <algorithm>
#include <sstream>
#include <string>
#include <tuple>
#include "ascend.h"
#include "base.h"
#include "runtime.h"
#include "op.h"
#include "model.h"
#include "debug.h"

#define XLITE_MLA_V3_THRESHOLD 280

XModel::XModel(struct XModelConfig &c, uint32_t rankId) : _c(c), _rankId(rankId)
{
#ifdef XLITE_310P_LLM_FP16_POC
    if (rankId != 0 || c.defTpSize != 1 || c.defDpSize != 1 || c.moeEpSize != 1 ||
        c.moeTPSize != 1) {
        throw std::runtime_error(
            "Ascend310P llm_fp16 POC only supports rank=0, TP=1, DP=1 and EP=1");
    }
    if (c.attnType != XMODEL_ATTN_MHA || c.nRoutedExperts != 0 || c.nSharedExperts != 0) {
        throw std::runtime_error(
            "Ascend310P llm_fp16 POC only supports dense MHA/GQA decoder models");
    }
    if (c.quantMsdW4a8 || c.quantAttnWeightTrans || c.quantAttnWeightNz) {
        throw std::runtime_error("Ascend310P llm_fp16 POC does not support quantization");
    }
    if (c.nHeads != 16 || c.nKvHeads != 8 || c.headDim != 128 || c.blockSize != 128 ||
        c.intermediateSize != 6144 || c.addBias) {
        throw std::runtime_error(
            "Ascend310P llm_fp16 POC requires Q16/KV8, head_dim=128, block_size=128, "
            "intermediate_size=6144 and bias-free QKV");
    }
#endif
    attnNorm.resize(c.nLayers);
    attnOut.resize(c.nLayers);
    mhaQKV.resize(c.nLayers);
    mhaQKVBias.resize(c.nLayers);
    mhaQNorm.resize(c.nLayers);
    mhaKNorm.resize(c.nLayers);
    mlaQKVA.resize(c.nLayers);
    mlaQB.resize(c.nLayers);
    mlaQNorm.resize(c.nLayers);
    mlaQNormBias.resize(c.nLayers);
    mlaWUV.resize(c.nLayers);
    mlaWUKT.resize(c.nLayers);
    mlaKVNorm.resize(c.nLayers);
    mlaKVNormBias.resize(c.nLayers);
    mlpNorm.resize(c.nLayers);
    mlpUpGate.resize(c.nDenseLayers);
    mlpDown.resize(c.nDenseLayers);
    indexQB.resize(c.nLayers);
    indexKWeightsProj.resize(c.nLayers);
    indexKNorm.resize(c.nLayers);
    indexKNormBias.resize(c.nLayers);
    linearInProjQKV.resize(c.nLayers);
    linearInProjZ.resize(c.nLayers);
    linearInProjB.resize(c.nLayers);
    linearInProjA.resize(c.nLayers);
    linearConv1d.resize(c.nLayers);
    linearALog.resize(c.nLayers);
    linearDtBias.resize(c.nLayers);
    linearNorm.resize(c.nLayers);
    linearOutProj.resize(c.nLayers);
    moeGate.resize(c.nLayers);
    moeGateBias.resize(c.nLayers);
    moeSEUpGate.resize(c.nLayers);
    moeSEDown.resize(c.nLayers);
    moeREUpGate.resize(c.nLayers);
    moeREUpGateDeqScale.resize(c.nLayers);
    moeREDown.resize(c.nLayers);
    moeREDownDeqScale.resize(c.nLayers);
    _moeREUpGate.resize(c.nLayers);
    _moeREUpGateDeqScale.resize(c.nLayers);
    _moeREDown.resize(c.nLayers);
    _moeREDownDeqScale.resize(c.nLayers);
    _moeREUpGateScaleBias.resize(c.nLayers);
    _moeREDownScaleBias.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGate[i].resize(c.nRoutedExperts);
        moeREDown[i].resize(c.nRoutedExperts);
    }

    attnSink.resize(c.nLayers);
    attnWqA.resize(c.nLayers);
    attnWoA.resize(c.nLayers);
    attnWoB.resize(c.nLayers);
    attnWKv.resize(c.nLayers);
    compApe.resize(c.nLayers);
    compWKv.resize(c.nLayers);
    compWGate.resize(c.nLayers);
    compNorm.resize(c.nLayers);
    idxWqB.resize(c.nLayers);
    idxWeightsProj.resize(c.nLayers);
    idxCompApe.resize(c.nLayers);
    idxCompWKv.resize(c.nLayers);
    idxCompWGate.resize(c.nLayers);
    idxCompNorm.resize(c.nLayers);
    hcAttnFn.resize(c.nLayers);
    hcFfnFn.resize(c.nLayers);
    hcAttnBase.resize(c.nLayers);
    hcFfnBase.resize(c.nLayers);
    hcAttnScale.resize(c.nLayers);
    hcFfnScale.resize(c.nLayers);

    attnNormBias.resize(c.nLayers);
    mhaQNormBias.resize(c.nLayers);
    mhaKNormBias.resize(c.nLayers);
    mlpNormBias.resize(c.nLayers);
    for (uint32_t i = 0; i < c.nLayers; i++) {
        moeREUpGateDeqScale[i].resize(c.nRoutedExperts);
        moeREDownDeqScale[i].resize(c.nRoutedExperts);
    }
    if (c.quantMsdW4a8) {
        moeREUpGateScaleBias.resize(c.nLayers);
        moeREDownScaleBias.resize(c.nLayers);
        for (uint32_t i = 0; i < c.nLayers; i++) {
            moeREUpGateScaleBias[i].resize(c.nRoutedExperts);
            moeREDownScaleBias[i].resize(c.nRoutedExperts);
        }
    }

    if (c.attnType == XMODEL_ATTN_HYBRID) {
        if (c.fullAttentionInterval == 0) {
            throw std::invalid_argument("fullAttentionInterval must be > 0 for hybrid attention");
        }
        _layerTypes.resize(c.nLayers);
        for (uint32_t i = 0; i < c.nLayers; i++) {
            _layerTypes[i] = ((i + 1) % c.fullAttentionInterval == 0) ? XMODEL_LAYER_ATTN_FULL
                                                                      : XMODEL_LAYER_ATTN_LINEAR;
        }
    }
}

void XModel::Init(void)
{
    std::vector<uint32_t> vbitsortIndices;
    std::vector<uint64_t> weights;
    size_t size;
    bool isWeightEmpty = true;
    void *ptr;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;

    if (_c.nDenseLayers != _c.nLayers || _c.attnType == XMODEL_ATTN_DSA) {
        size_t maxIndices = _c.maxSeqLen > _c.nRoutedExperts ? _c.maxSeqLen : _c.nRoutedExperts;
        vbitsortIndices.resize(maxIndices);
        size = maxIndices * XDtypeBit(INT32) / 8;
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t i = 0; i < maxIndices; i++) {
            vbitsortIndices[i] = i;
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, vbitsortIndices.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        if (_c.nDenseLayers != _c.nLayers) {
            _gateIndices.Init({_c.nRoutedExperts}, INT32, ptr);
        }
        if (_c.attnType == XMODEL_ATTN_DSA) {
            _dsaTopkIndices.Init({_c.maxSeqLen}, INT32, ptr);
        }
    }

    if (_c.attnType == XMODEL_ATTN_DSA) {
        _dsaIndexerScale = 1.0f / std::sqrt(static_cast<float>(_c.indexNHeads));
        _dsaIndexerScale *= 1.0f / std::sqrt(static_cast<float>(_c.indexHeadDim));
    }

    if (_c.attnType == XMODEL_ATTN_CXA) {
        if (_c.oGroups == 0 || _c.oLoraRank == 0) {
            throw std::invalid_argument("CxA: o_groups and o_lora_rank must be set");
        }
        if (_c.windowSize == 0) {
            throw std::invalid_argument("CxA: window_size must be set");
        }
        if (!_c.compressRatios.empty() &&
            _c.compressRatios.size() < static_cast<size_t>(_c.nLayers)) {
            throw std::invalid_argument(
                "CxA: compress_ratios size must be 0 or greater than n_layers");
        }
        if (_c.hcMult == 0) {
            throw std::invalid_argument("CxA: hc_mult must be set");
        }
    }

    size = _c.nRoutedExperts * XDtypeBit(INT64) / 8;
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        weights.resize(_c.nRoutedExperts);
        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREUpGate[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREUpGate[i].Init({_c.nRoutedExperts}, INT64, ptr);

        CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            weights[j] = reinterpret_cast<uint64_t>(moeREDown[i][j].ptr);
        }
        CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
        _moeREDown[i].Init({_c.nRoutedExperts}, INT64, ptr);

        isWeightEmpty = true;
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            if (j >= start && j < end) {
                weights[j] = reinterpret_cast<uint64_t>(moeREUpGateDeqScale[i][j].ptr);
                if (weights[j] != 0) {
                    isWeightEmpty = false;
                }
            } else {
                weights[j] = 0;
            }
        }
        if (!isWeightEmpty) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREUpGateDeqScale[i].Init({_c.nRoutedExperts}, INT64, ptr);
        }

        isWeightEmpty = true;
        for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
            if (j >= start && j < end) {
                weights[j] = reinterpret_cast<uint64_t>(moeREDownDeqScale[i][j].ptr);
                if (weights[j] != 0) {
                    isWeightEmpty = false;
                }
            } else {
                weights[j] = 0;
            }
        }

        if (!isWeightEmpty) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREDownDeqScale[i].Init({_c.nRoutedExperts}, INT64, ptr);
        }

        if (_c.quantMsdW4a8) {
            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
                weights[j] = reinterpret_cast<uint64_t>(moeREUpGateScaleBias[i][j].ptr);
            }
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREUpGateScaleBias[i].Init({_c.nRoutedExperts}, INT64, ptr);

            CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
            for (uint32_t j = 0; j < _c.nRoutedExperts; j++) {
                weights[j] = reinterpret_cast<uint64_t>(moeREDownScaleBias[i][j].ptr);
            }
            CHECK_ACL(aclrtMemcpy(ptr, size, weights.data(), size, ACL_MEMCPY_HOST_TO_DEVICE));
            _moeREDownScaleBias[i].Init({_c.nRoutedExperts}, INT64, ptr);
        }
    }

    size = AIC_MAX_NUM;
    CHECK_ACL(aclrtMalloc(&ptr, size, ACL_MEM_MALLOC_NORMAL_ONLY));
    CHECK_ACL(aclrtMemset(ptr, size, 0, size));
    _sync.Init({AIC_MAX_NUM}, INT32, ptr);

    _mropeMaskH = 0;
    _mropeMaskW = 0;
    uint32_t mropeSectionH = _c.mropeSection.size() > 1 ? _c.mropeSection[1] : 0;
    uint32_t mropeSectionW = _c.mropeSection.size() > 2 ? _c.mropeSection[2] : 0;
    if (_c.mropeInterleaved) {
        for (uint32_t i = 1; i < mropeSectionH * 3; i += 3) {
            _mropeMaskH |= 1ULL << i;
        }
        for (uint32_t i = 2; i < mropeSectionW * 3; i += 3) {
            _mropeMaskW |= 1ULL << i;
        }
    } else {
        uint32_t sectionStartH = !_c.mropeSection.empty() ? _c.mropeSection[0] : 0;
        uint32_t sectionStartW = sectionStartH + mropeSectionH;
        for (uint32_t i = sectionStartH; i < sectionStartW; i++) {
            _mropeMaskH |= 1ULL << i;
        }
        for (uint32_t i = sectionStartW; i < sectionStartW + mropeSectionW; i++) {
            _mropeMaskW |= 1ULL << i;
        }
    }

    if (_c.nSharedExperts != 0 && !moeSEUpGate[_c.nDenseLayers].weight.shape.empty() &&
        !moeSEDown[_c.nDenseLayers].weight.shape.empty()) {
        if (_c.quantAttnWeightTrans) {
            _isSharedExpertWeightFull =
                moeSEUpGate[_c.nDenseLayers].weight.shape.size() >= 2 &&
                moeSEUpGate[_c.nDenseLayers].weight.shape[1] == _c.moeIntermediateSize * 2 &&
                moeSEDown[_c.nDenseLayers].weight.shape[0] == _c.moeIntermediateSize;
        } else {
            _isSharedExpertWeightFull =
                moeSEDown[_c.nDenseLayers].weight.shape.size() >= 2 &&
                moeSEUpGate[_c.nDenseLayers].weight.shape[0] == _c.moeIntermediateSize * 2 &&
                moeSEDown[_c.nDenseLayers].weight.shape[1] == _c.moeIntermediateSize;
        }
    }
}

XModel::~XModel(void)
{
    if (_gateIndices.ptr != nullptr) {
        (void)aclrtFree(_gateIndices.ptr);
        _dsaTopkIndices.ptr = nullptr;
    }
    if (_dsaTopkIndices.ptr != nullptr) {
        (void)aclrtFree(_dsaTopkIndices.ptr);
    }
    for (uint32_t i = _c.nDenseLayers; i < _c.nLayers; i++) {
        (void)aclrtFree(_moeREUpGate[i].ptr);
        (void)aclrtFree(_moeREUpGateDeqScale[i].ptr);
        (void)aclrtFree(_moeREDown[i].ptr);
        (void)aclrtFree(_moeREDownDeqScale[i].ptr);
        (void)aclrtFree(_moeREUpGateScaleBias[i].ptr);
        (void)aclrtFree(_moeREDownScaleBias[i].ptr);
    }
    (void)aclrtFree(_sync.ptr);
}

void XModel::ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output)
{
    uint32_t vocabPerTp = DIV_ROUND_UP(_c.vocabSize, _c.defTpSize);
    uint32_t id = _rankId % _c.defTpSize;
    uint32_t start = id * vocabPerTp;
    uint32_t end = start + vocabPerTp;

    XliteOpEmbed(rt, input, embed, start, end, output);
    if (_c.defTpSize > 1) {
        XliteOpAllReduceSum(rt, output, output, TP, false, DBG_LOC);
    }
}

// TODO: optimize XTensor allocation and release order
void XModel::ForwardLinear(XRuntime &rt, uint32_t layer, XTensor &x,
                           std::vector<MatmulWeight> &weights, XTensor &out,
                           const std::vector<XTensor> &weightBias)
{
    MatmulWeight &w = weights[layer];
    enum QuantType quantType = w.GetQuantType();
    bool isNz = _c.weightNZ;
    bool isTransposed = false;

    if (quantType == NO_QUANT) {
        if (layer < weightBias.size()) {
            XliteOpMatmul(rt, x, w.weight, out, isNz, weightBias[layer]);
        } else {
            XliteOpMatmul(rt, x, w.weight, out, isNz);
        }
        return;
    }

    isNz = _c.weightNZ || _c.quantAttnWeightNz;
    isTransposed = _c.quantAttnWeightTrans;

    XTensor &xQuanted = rt.GetTensor(x.shape, INT8, DBG_LOC);
    if (quantType == DYNAMIC_QUANT) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &scale = rt.GetTensor({x.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, x, scale, xQuanted);
        // matmulDequant(xQuanted, weight, quantBias, deqScale, perChannelScale) -> out
        XliteOpMatmulDeQuant(rt, xQuanted, w.weight, out, w.quantBias, w.deqScale, isNz,
                             isTransposed, scale);
        rt.PutTensor(scale);
    } else {
        // quant(x) -> xQuanted
        XliteOpQuant(rt, x, w.inputScale, w.inputOffset, xQuanted);
        // matmul(xQuanted, weight, quantBias, deqScale) -> outQuanted
        XliteOpMatmulDeQuant(rt, xQuanted, w.weight, out, w.quantBias, w.deqScale, isNz,
                             isTransposed);
    }
    rt.PutTensor(xQuanted);
}

XTensor *XModel::ForwardAttnIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                                    XTensor &attnNormQc, XTensor &indexKCache, XTensor &freqsCis)
{
    // TODO not interleaved case
    if (!_c.indexRopeInterleaved) {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }

    XTensor &kw = rt.GetTensor({hiddenState.shape[0], _c.indexHeadDim + _c.indexNHeads},
                               hiddenState.dtype, DBG_LOC);
    XliteOpMatmul(rt, hiddenState, indexKWeightsProj[layer], kw, _c.weightNZ);

    // only use sparse attention when the sequence length is long enough
    bool isLong = rt._maxNumBlocks * _c.blockSize > _c.indexTopK;
    XTensor *qPtr = nullptr;
    if (isLong) {
        qPtr = &rt.GetTensor({hiddenState.shape[0], _c.indexNHeads * _c.indexHeadDim},
                             hiddenState.dtype, DBG_LOC);
        ForwardLinear(rt, layer, attnNormQc, indexQB, *qPtr);
    }
    XliteOpIndexerPrepare(rt, kw, indexKNorm[layer], indexKNormBias[layer], freqsCis,
                          rt._attnPosition, _c.indexHeadDim, _c.indexNHeads, _c.ropeHeadDim,
                          _c.blockSize, indexKCache, rt._attnSlotMapping, _c.normEps,
                          qPtr == nullptr ? XTensor() : *qPtr, _dsaIndexerScale, _c.indexTopK,
                          isLong);

    if (!isLong) {
        rt.PutTensor(kw);
        return nullptr;
    }

    XTensor &scores = rt.GetTensor({2 * rt.aicNum * XLITE_MAX_M0, MAX_INDEXER_KV_TILE_LEN},
                                   hiddenState.dtype, DBG_LOC);
    XTensor &lastTopk = rt.GetTensor({hiddenState.shape[0], 2 * _c.indexTopK}, INT32, DBG_LOC);
    rt._dsaTopkBuffer.View({hiddenState.shape[0], _c.indexTopK});
    XliteOpIndexerTopK(rt, *qPtr, indexKCache, kw, scores, lastTopk, _dsaTopkIndices,
                       rt._dsaTopkBuffer, rt._queryStartLoc, rt._lens, rt._cachedLens,
                       rt._attnBlockTables, _sync, _c.indexNHeads, _c.indexHeadDim, _c.blockSize,
                       rt._batch, rt._maxNumBlocks, _c.indexTopK);
    rt.PutTensor(kw);
    rt.PutTensor(*qPtr);
    rt.PutTensor(lastTopk);
    rt.PutTensor(scores);
    rt._dsaTopkValid = true;
    return &rt._dsaTopkBuffer;
}

std::tuple<XTensor &, XTensor &, XTensor &> XModel::ForwardAttnMLACommonV2(
    XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
    XTensor &hiddenState)
{
    if (_c.defTpSize == 0 || _c.nHeads % _c.defTpSize != 0) {
        throw std::invalid_argument("nHeads must be divisible by defTpSize and defTpSize > 0");
    }
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;

    XTensor &kCache = kvCache[layer][0];
    XTensor &peCache = kvCache[layer][1];
    XTensor &attnQkvc =
        rt.GetTensor({hiddenState.shape[0], _c.qLoraRank + _c.kvLoraRank + _c.ropeHeadDim},
                     hiddenState.dtype, DBG_LOC);
    XTensor &attnNormQc =
        rt.GetTensor({hiddenState.shape[0], _c.qLoraRank}, hiddenState.dtype, DBG_LOC);
    XTensor &attnQWithQr =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads * (_c.nopeHeadDim + _c.ropeHeadDim)},
                     hiddenState.dtype, DBG_LOC);

    ForwardLinear(rt, layer, hiddenState, mlaQKVA, attnQkvc);

    XliteOpMlaPrepare(rt, attnQkvc, mlaQNorm[layer], mlaQNormBias[layer], attnNormQc,
                      mlaKVNorm[layer], mlaKVNormBias[layer], freqsCis, rt._attnPosition,
                      _c.qLoraRank, _c.kvLoraRank, _c.ropeHeadDim, _c.blockSize, kCache, peCache,
                      rt._attnSlotMapping, _c.normEps);
    rt.PutTensor(attnQkvc);

    ForwardLinear(rt, layer, attnNormQc, mlaQB, attnQWithQr);

    XTensor &qPe = rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.ropeHeadDim},
                                hiddenState.dtype, DBG_LOC);
    XliteOpRopeComplex(rt, nLocalHeads, _c.nopeHeadDim + _c.ropeHeadDim, _c.ropeHeadDim,
                       _c.ropeHeadDim, _c.nopeHeadDim, 0, attnQWithQr, freqsCis, rt._attnPosition,
                       qPe);

    return {attnQWithQr, qPe, attnNormQc};
}

void XModel::ForwardAttnMLAV2(XRuntime &rt, uint32_t layer,
                              std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                              XTensor &hiddenState)
{
    XTensor &kCache = kvCache[layer][0];
    XTensor &peCache = kvCache[layer][1];
    uint32_t nLocalHeads = _c.nHeads / _c.defTpSize;

    auto [attnQWithQr, qPe, attnNormQc] =
        ForwardAttnMLACommonV2(rt, layer, kvCache, freqsCis, hiddenState);

    XTensor *topkIndices = nullptr;
    if (_c.attnType == XMODEL_ATTN_DSA) {
        // DSA top-k sharing: shared layers reuse prev full layer's topkIndices.
        if (layer < _c.indexerSkipLayers.size() && _c.indexerSkipLayers[layer]) {
            topkIndices = rt._dsaTopkValid ? &rt._dsaTopkBuffer : nullptr;
        } else {
            topkIndices =
                ForwardAttnIndexer(rt, layer, hiddenState, attnNormQc, kvCache[layer][2], freqsCis);
        }
    }
    rt.PutTensor(attnNormQc);

    XTensor &qAbsorb = rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.kvLoraRank},
                                    hiddenState.dtype, DBG_LOC);
    XliteOpEinsumMhtHtdMhd(rt, attnQWithQr, mlaWUKT[layer], qAbsorb, hiddenState.shape[0],
                           nLocalHeads, _c.nopeHeadDim, _c.kvLoraRank, _c.weightNZ,
                           static_cast<int>(_c.nopeHeadDim + _c.ropeHeadDim));
    rt.PutTensor(attnQWithQr);

    XTensor &oAbsorb = rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.kvLoraRank},
                                    hiddenState.dtype, DBG_LOC);
    // Route decode+DSA to the mla_v3 (sparse gather) path only when it pays off.
    // Below ~280 tokens/seq the gather overhead dominates and v3 regresses by up
    // to 9% on short-prompt + large-batch cases (see mla_v2 vs mla_v3 perf sweep).
    // Cond 1 (cached KV > tile): long-seq fallback where v3 is consistently faster.
    // Cond 2 (per-seq cached KV > 280 * batch): perf-derived threshold that steers
    //   degenerate short-prompt + large-batch cases back to v2 to avoid regression.
    if (rt._decodeStep && _c.attnType == XMODEL_ATTN_DSA && topkIndices != nullptr &&
        (rt._maxNumBlocks * _c.blockSize > rt._tileSizeOfCachedKV ||
         rt._maxNumBlocks * _c.blockSize > XLITE_MLA_V3_THRESHOLD * rt._batch)) {
        // Decode + DSA long-sequence path: gather sparse top-k tokens into a
        // contiguous dense cache, then run mla_v3 on the dense cache.
        XTensor &kDense =
            rt.GetTensor({static_cast<size_t>(rt._batch), _c.indexTopK, _c.kvLoraRank},
                         hiddenState.dtype, DBG_LOC);
        XTensor &peDense =
            rt.GetTensor({static_cast<size_t>(rt._batch), _c.indexTopK, _c.ropeHeadDim},
                         hiddenState.dtype, DBG_LOC);
        XliteOpGatherSparseKVCache(rt, kCache, peCache, rt._attnBlockTables, *topkIndices, rt._lens,
                                   rt._cachedLens, kDense, peDense, rt._batch, _c.indexTopK,
                                   _c.blockSize, rt._maxNumBlocks, _c.kvLoraRank, _c.ropeHeadDim,
                                   _c.nKvHeads);
        XTensor &qkDense =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, _c.indexTopK}, hiddenState.dtype, DBG_LOC);
        XliteOpMLAV3(rt, qAbsorb, qPe, kDense, peDense, qkDense, oAbsorb, rt._queryStartLoc,
                     rt._lens, rt._cachedLens, nLocalHeads, _c.ropeHeadDim, _c.kvLoraRank,
                     rt._batch, _c.indexTopK, _c.softmaxScale);
        rt.PutTensor(qkDense);
        rt.PutTensor(peDense);
        rt.PutTensor(kDense);
    } else if (rt._maxNumBlocks * _c.blockSize <= rt._tileSizeOfCachedKV) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, rt._maxNumBlocks * _c.blockSize},
                                   hiddenState.dtype, DBG_LOC);
        XliteOpMLAV2(rt, qAbsorb, qPe, kCache, peCache, qk, oAbsorb, rt._queryStartLoc, rt._lens,
                     rt._cachedLens, rt._attnBlockTables, nLocalHeads, _c.ropeHeadDim,
                     _c.kvLoraRank, _c.blockSize, rt._batch, rt._maxNumBlocks, _c.softmaxScale,
                     _c.indexTopK, topkIndices == nullptr ? XTensor() : *topkIndices);
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, rt._tileSizeOfCachedKV},
                                   hiddenState.dtype, DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, _c.kvLoraRank}, hiddenState.dtype, DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({hiddenState.shape[0], nLocalHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({hiddenState.shape[0], nLocalHeads}, FP32, DBG_LOC);
        XliteOpFlashMLAV2(rt, qAbsorb, qPe, kCache, peCache, qk, sv, max, sum, lastMax, lastSum,
                          _sync, oAbsorb, rt._queryStartLoc, rt._lens, rt._cachedLens,
                          rt._attnBlockTables, nLocalHeads, _c.ropeHeadDim, _c.kvLoraRank,
                          _c.blockSize, rt._batch, rt._maxNumBlocks, _c.softmaxScale,
                          rt._tileSizeOfCachedKV, _c.indexTopK,
                          topkIndices == nullptr ? XTensor() : *topkIndices);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }
    rt.PutTensor(qPe);
    rt.PutTensor(qAbsorb);

    XTensor &attnOutput =
        rt.GetTensor({hiddenState.shape[0], nLocalHeads * _c.vHeadDim}, hiddenState.dtype, DBG_LOC);
    XliteOpEinsumMhtHtdMhd(rt, oAbsorb, mlaWUV[layer], attnOutput, hiddenState.shape[0],
                           nLocalHeads, _c.kvLoraRank, _c.vHeadDim, _c.weightNZ);
    rt.PutTensor(oAbsorb);

    ForwardLinear(rt, layer, attnOutput, attnOut, hiddenState);
    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize || rt.enableMoEAllToAll) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false, DBG_LOC);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP, false, DBG_LOC);
        }
    }
    rt.PutTensor(attnOutput);
}

void XModel::ForwardAttnMHA(XRuntime &rt, uint32_t layer,
                            std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                            XTensor &hiddenState)
{
    XTensor &kCache = kvCache[layer][0];
    XTensor &vCache = kvCache[layer][1];
    uint32_t qHeads = _c.nHeads / _c.defTpSize;
    uint32_t kHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
    uint32_t qDim = qHeads * _c.headDim;
    // With output gate: fused mhaQKV layout is [Q | K | V | Gate] so a single
    // SplitCol yields contiguous [Q|K|V] for Rope/Attention (no ConcatCol).
    uint32_t qkvDim = (qHeads + 2 * kHeads) * _c.headDim;
    uint32_t projOutDim = _c.attnOutputGate ? (qkvDim + qDim) : qkvDim;

    XTensor &projOut = rt.GetTensor({hiddenState.shape[0], projOutDim}, hiddenState.dtype, DBG_LOC);
    ForwardLinear(rt, layer, hiddenState, mhaQKV, projOut, mhaQKVBias);

    XTensor *gatePtr = nullptr;
    XTensor *qkvPtr = nullptr;
    if (_c.attnOutputGate) {
        XTensor &qkv = rt.GetTensor({hiddenState.shape[0], qkvDim}, hiddenState.dtype, DBG_LOC);
        XTensor &gate = rt.GetTensor({hiddenState.shape[0], qDim}, hiddenState.dtype, DBG_LOC);
        std::vector<XTensor> splitOut = {qkv, gate};
        XliteOpSplitCol(rt, projOut, splitOut);
        rt.PutTensor(projOut);
        gatePtr = &gate;
        qkvPtr = &qkv;
    } else {
        qkvPtr = &projOut;
    }
    XTensor &qkv = *qkvPtr;

    if (_c.qkNorm && !_c.qkNormFull) {
        XliteOpQkRmsNorm(rt, qkv, mhaQNorm[layer], mhaQNormBias[layer], mhaKNorm[layer],
                         mhaKNormBias[layer], qkv, _c.normEps, _c.headDim, qHeads, _c.headDim,
                         kHeads, qHeads * _c.headDim, true);
    }
    if (_c.qkNormFull) {
        size_t rows = qkv.shape[0];
        size_t bytesPerVar = rows * XDtypeBit(FP32) / 8;
        XTensor &packedVar = rt.GetTensor({rows * 2, 1}, FP32, DBG_LOC);
        XTensor qLocalVariance;
        XTensor kLocalVariance;
        qLocalVariance.Init({rows, 1}, FP32, packedVar.ptr);
        kLocalVariance.Init(
            {rows, 1}, FP32,
            reinterpret_cast<void *>(reinterpret_cast<uint64_t>(packedVar.ptr) + bytesPerVar));
        XliteOpRmsNorm(rt, qkv, XTensor(), qLocalVariance, _c.normEps, _c.headDim * qHeads, false,
                       XTensor());
        XliteOpRmsNorm(rt, qkv, XTensor(), kLocalVariance, _c.normEps, _c.headDim * kHeads, false,
                       XTensor(), 1, qHeads * _c.headDim);
        if (_c.defTpSize > 1) {
            // Single AllReduceSum over the packed buffer reduces both variances in place.
            XliteOpAllReduceSum(rt, packedVar, packedVar, TP, false, DBG_LOC);
        }
        XliteOpQkRmsNorm(rt, qkv, mhaQNorm[layer], XTensor(), mhaKNorm[layer], XTensor(), qkv,
                         _c.normEps, _c.headDim * qHeads, 1, _c.headDim * kHeads, 1,
                         qHeads * _c.headDim, true, qLocalVariance, kLocalVariance);
        rt.PutTensor(packedVar);
    }
    XliteOpRopeCache(rt, qkv, kCache, vCache, rt._attnPosition, freqsCis, rt._attnSlotMapping,
                     _c.nHeads, _c.nKvHeads, _c.headDim, _c.ropeHeadDim, _c.blockSize,
                     _c.ropeType == XMODEL_ROPE_NEOX, _mropeMaskH, _mropeMaskW);

    XTensor &attn =
        rt.GetTensor({hiddenState.shape[0], qHeads * _c.headDim}, hiddenState.dtype, DBG_LOC);
    if (rt._maxNumBlocks * _c.blockSize <= rt._tileSizeOfCachedKV) {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, rt._maxNumBlocks * _c.blockSize},
                                   hiddenState.dtype, DBG_LOC);
        XliteOpAttention(rt, qkv, kCache, vCache, qk, attn, rt._queryStartLoc, rt._lens,
                         rt._cachedLens, rt._attnBlockTables, qHeads, kHeads, _c.headDim,
                         _c.blockSize, rt._batch, rt._maxNumBlocks);
        rt.PutTensor(qk);
    } else {
        XTensor &qk = rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, rt._tileSizeOfCachedKV},
                                   hiddenState.dtype, DBG_LOC);
        XTensor &sv =
            rt.GetTensor({rt.aicNum * XLITE_MAX_M0 * 2, _c.headDim}, hiddenState.dtype, DBG_LOC);
        XTensor &max = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &sum = rt.GetTensor({rt.aivNum * XLITE_MAX_M0 * 2}, FP32, DBG_LOC);
        XTensor &lastMax = rt.GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XTensor &lastSum = rt.GetTensor({qkv.shape[0], qHeads}, FP32, DBG_LOC);
        XliteOpFlashAttention(rt, qkv, kCache, vCache, qk, sv, max, sum, lastMax, lastSum, _sync,
                              attn, rt._queryStartLoc, rt._lens, rt._cachedLens,
                              rt._attnBlockTables, qHeads, kHeads, _c.headDim, _c.blockSize,
                              rt._batch, rt._maxNumBlocks, rt._tileSizeOfCachedKV);
        rt.PutTensor(lastSum);
        rt.PutTensor(lastMax);
        rt.PutTensor(sum);
        rt.PutTensor(max);
        rt.PutTensor(sv);
        rt.PutTensor(qk);
    }

    if (gatePtr != nullptr) {
        XliteOpSigmoidGateMul(rt, attn, *gatePtr, attn);
        rt.PutTensor(*gatePtr);
    }

    ForwardLinear(rt, layer, attn, attnOut, hiddenState);

    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (rt.enableCommOptimize || rt.enableMoEAllToAll) {
            XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false, DBG_LOC);
        } else {
            XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP, false, DBG_LOC);
        }
    }
    rt.PutTensor(qkv);
    rt.PutTensor(attn);
}

namespace
{

void ExpandLinearHeads(XRuntime &rt, XTensor &in, XTensor &out, uint32_t numTokens,
                       uint32_t nKHeads, uint32_t nVHeads, uint32_t headDim)
{
    if (rt.IsDummyRuntime()) {
        return;
    }
    if (nKHeads == nVHeads) {
        if (in.ptr != out.ptr) {
            size_t bytes = in.numel * XDtypeBit(in.dtype) / 8;
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, bytes, in.ptr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                       rt.stream));
        }
        return;
    }
    if (nVHeads % nKHeads != 0) {
        throw std::runtime_error("ForwardAttnLinear: num_v_heads must be divisible by num_k_heads");
    }
    uint32_t expand = nVHeads / nKHeads;
    size_t headBytes = headDim * XDtypeBit(in.dtype) / 8;
    size_t inRowBytes = nKHeads * headDim * XDtypeBit(in.dtype) / 8;
    size_t outRowBytes = nVHeads * headDim * XDtypeBit(in.dtype) / 8;
    for (uint32_t t = 0; t < numTokens; ++t) {
        auto *srcRow = static_cast<uint8_t *>(in.ptr) + t * inRowBytes;
        auto *dstRow = static_cast<uint8_t *>(out.ptr) + t * outRowBytes;
        for (uint32_t h = 0; h < nKHeads; ++h) {
            auto *srcHead = srcRow + h * headBytes;
            for (uint32_t e = 0; e < expand; ++e) {
                auto *dstHead = dstRow + (h * expand + e) * headBytes;
                CHECK_ACL(aclrtMemcpyAsync(dstHead, headBytes, srcHead, headBytes,
                                           ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            }
        }
    }
}

}  // namespace

void XModel::ForwardAttnLinear(XRuntime &rt, uint32_t layer,
                               std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                               XTensor &hiddenState)
{
    (void)freqsCis;
    if (kvCache.size() <= layer || kvCache[layer].size() < 2) {
        throw std::runtime_error(std::string(__func__) +
                                 ": linear-attention state cache missing at layer " +
                                 std::to_string(layer));
    }
    XTensor &convState = kvCache[layer][0];
    XTensor &ssmState = kvCache[layer][1];

    uint32_t m = hiddenState.shape[0];
    uint32_t batch = rt._batch;
    if (batch == 0 || m % batch != 0) {
        throw std::runtime_error(
            "ForwardAttnLinear: token layout requires uniform seqlen per batch item");
    }
    uint32_t seqlen = m / batch;

    uint32_t nLocalKHeads = _c.linearNumKHeads / _c.defTpSize;
    uint32_t nLocalVHeads = _c.linearNumVHeads / _c.defTpSize;
    uint32_t localKeyDim = nLocalKHeads * _c.linearKeyHeadDim;
    uint32_t localValueDim = nLocalVHeads * _c.linearValueHeadDim;
    uint32_t convDim = localKeyDim * 2 + localValueDim;
    uint32_t qkvDim = localKeyDim * 2 + localValueDim;
    uint32_t zDim = localValueDim;
    uint32_t bDim = nLocalVHeads;
    uint32_t aDim = nLocalVHeads;
    uint32_t totalOutDim = qkvDim + zDim + bDim + aDim;

    // Step 1: Fused ND projection [W_qkv; W_z; W_b; W_a].
    // Must stay ND: NZ weights cannot be Concat'd; tiny NZ matmul tiles over-read.
    XTensor &mixQkv = rt.GetTensor({m, qkvDim}, hiddenState.dtype, DBG_LOC);
    XTensor &z = rt.GetTensor({m, zDim}, hiddenState.dtype, DBG_LOC);
    XTensor &b = rt.GetTensor({m, bDim}, hiddenState.dtype, DBG_LOC);
    XTensor &a = rt.GetTensor({m, aDim}, hiddenState.dtype, DBG_LOC);
    {
        std::vector<XTensor> weightInputs = {
            linearInProjQKV[layer].weight, linearInProjZ[layer].weight, linearInProjB[layer].weight,
            linearInProjA[layer].weight};
        XTensor &W = rt.GetTensor({totalOutDim, hiddenState.shape[1]}, hiddenState.dtype, DBG_LOC);
        XliteOpConcat(rt, weightInputs, W);

        XTensor &projOut = rt.GetTensor({m, totalOutDim}, hiddenState.dtype, DBG_LOC);
        XliteOpMatmul(rt, hiddenState, W, projOut, false);
        rt.PutTensor(W);

        std::vector<XTensor> projSplit = {mixQkv, z, b, a};
        XliteOpSplitCol(rt, projOut, projSplit);
        rt.PutTensor(projOut);
    }

    // Step 2: beta = sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
    XTensor &beta = rt.GetTensor({m, nLocalVHeads}, hiddenState.dtype, DBG_LOC);
    XTensor &g = rt.GetTensor({m, nLocalVHeads}, hiddenState.dtype, DBG_LOC);
    XliteOpBetaDecay(rt, b, a, linearALog[layer], linearDtBias[layer], beta, g, batch, seqlen,
                     nLocalVHeads);
    rt.PutTensor(b);
    rt.PutTensor(a);

    // Step 3: causal conv1d + SiLU on mixed qkv
    XTensor &mixTrans = rt.GetTensor({batch, qkvDim, seqlen}, hiddenState.dtype, DBG_LOC);
    XTensor &convOut = rt.GetTensor({batch, convDim, seqlen}, hiddenState.dtype, DBG_LOC);
    XTensor &convSeq = rt.GetTensor({batch, seqlen, convDim}, hiddenState.dtype, DBG_LOC);
    // Clear conv/ssm only for fresh prefill (cached_lens==0). Decode and
    // chunked-prefill continuation (cached>0) must reuse state: xlite has no
    // chunk_gated_delta_rule, so recurrent resumes from the previous step.
    auto linearCachedLen = [&](uint32_t req) -> uint32_t {
        if (req < rt._cachedLensHost.size()) {
            return rt._cachedLensHost[req];
        }
        return 0;
    };
    {
        XTensor mix3d;
        mix3d.Init({batch, seqlen, qkvDim}, mixQkv.dtype, mixQkv.ptr);
        XliteOpTranspose_1_2(rt, mix3d, mixTrans);

        if (!rt.IsDummyRuntime()) {
            size_t convRowBytes = static_cast<size_t>(convDim) * _c.linearConvKernelDim *
                                  XDtypeBit(convState.dtype) / 8;
            for (uint32_t i = 0; i < batch; ++i) {
                if (linearCachedLen(i) != 0) {
                    continue;
                }
                auto *row = static_cast<char *>(convState.ptr) + i * convRowBytes;
                CHECK_ACL(aclrtMemsetAsync(row, convRowBytes, 0, convRowBytes, rt.stream));
            }
        }

        XTensor convStateBatch;
        convStateBatch.Init({batch, convDim, _c.linearConvKernelDim}, convState.dtype,
                            convState.ptr);
        XliteOpConv1dAndSiLU(rt, convStateBatch, mixTrans, linearConv1d[layer], convOut,
                             /*updateState=*/true);

        rt.PutTensor(mixTrans);
        XliteOpTranspose_1_2(rt, convOut, convSeq);
        rt.PutTensor(convOut);
        rt.PutTensor(mixQkv);
    }

    XTensor convFlat;
    convFlat.Init({m, convDim}, convSeq.dtype, convSeq.ptr);

    // Step 4-7: split / L2 / expand / gated delta
    XTensor &query = rt.GetTensor({m, localKeyDim}, hiddenState.dtype, DBG_LOC);
    XTensor &key = rt.GetTensor({m, localKeyDim}, hiddenState.dtype, DBG_LOC);
    XTensor &value = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
    XTensor &queryExp =
        rt.GetTensor({m, nLocalVHeads * _c.linearKeyHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &keyExp =
        rt.GetTensor({m, nLocalVHeads * _c.linearKeyHeadDim}, hiddenState.dtype, DBG_LOC);
    XTensor &coreAttn = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
    {
        std::vector<XTensor> qkvSplit = {query, key, value};
        XliteOpSplitCol(rt, convFlat, qkvSplit);
        rt.PutTensor(convSeq);

        XliteOpL2Norm(rt, query, query, _c.normEps, _c.linearKeyHeadDim, nLocalKHeads);
        XliteOpL2Norm(rt, key, key, _c.normEps, _c.linearKeyHeadDim, nLocalKHeads);

        ExpandLinearHeads(rt, query, queryExp, m, nLocalKHeads, nLocalVHeads, _c.linearKeyHeadDim);
        ExpandLinearHeads(rt, key, keyExp, m, nLocalKHeads, nLocalVHeads, _c.linearKeyHeadDim);
        rt.PutTensor(query);
        rt.PutTensor(key);

        if (!rt.IsDummyRuntime()) {
            if (ssmState.shape.empty() || ssmState.shape[0] == 0) {
                throw std::runtime_error("ForwardAttnLinear: invalid ssmState batch dim");
            }
            size_t ssmRowBytes =
                (ssmState.numel / ssmState.shape[0]) * XDtypeBit(ssmState.dtype) / 8;
            for (uint32_t i = 0; i < batch; ++i) {
                if (linearCachedLen(i) != 0) {
                    continue;
                }
                auto *row = static_cast<char *>(ssmState.ptr) + i * ssmRowBytes;
                CHECK_ACL(aclrtMemsetAsync(row, ssmRowBytes, 0, ssmRowBytes, rt.stream));
            }
        }

        XliteOpRecurrentGatedDeltaRule(rt, queryExp, keyExp, value, beta, g, ssmState, coreAttn,
                                       batch, seqlen, nLocalVHeads, _c.linearKeyHeadDim,
                                       _c.linearValueHeadDim);
        rt.PutTensor(queryExp);
        rt.PutTensor(keyExp);
        rt.PutTensor(beta);
        rt.PutTensor(g);
        rt.PutTensor(value);
    }

    // Step 8: RMSNormGated(core, z) = rmsnorm(core) * silu(z)
    XTensor &gated = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
    {
        XTensor &normed = rt.GetTensor({m, localValueDim}, hiddenState.dtype, DBG_LOC);
        XliteOpRmsNorm(rt, coreAttn, linearNorm[layer], normed, _c.normEps, _c.linearValueHeadDim,
                       true, XTensor(), nLocalVHeads);
        XTensor &siluIn = rt.GetTensor({m, localValueDim * 2}, hiddenState.dtype, DBG_LOC);
        std::vector<XTensor> normGateInputs = {z, normed};
        XliteOpConcatCol(rt, normGateInputs, siluIn);
        rt.PutTensor(z);
        rt.PutTensor(normed);

        XliteOpSiluAndMul(rt, siluIn, gated);
        rt.PutTensor(siluIn);
        rt.PutTensor(coreAttn);
    }

    // Step 9: output projection + TP all-reduce
    {
        ForwardLinear(rt, layer, gated, linearOutProj, hiddenState);
        rt.PutTensor(gated);

        if (_c.defTpSize > 1) {
            if (rt.multiTaskParallel) {
                rt.NotifyRecordPeerStream();
            }
            if (rt.enableCommOptimize || rt.enableMoEAllToAll) {
                XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false,
                                     DBG_LOC);
            } else {
                XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP, false, DBG_LOC);
            }
        }
    }
}

void XModel::ForwardAttnCXA(XRuntime &rt, uint32_t layer,
                            std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
                            XTensor &hiddenState)
{
    // TODO
}

void XModel::ForwardAttn(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                         XTensor &freqsCis, XTensor &hiddenState)
{
    if (_c.attnType == XMODEL_ATTN_HYBRID) {
        if (_layerTypes[layer] == XMODEL_LAYER_ATTN_FULL) {
            ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
        } else {
            ForwardAttnLinear(rt, layer, kvCache, freqsCis, hiddenState);
        }
    } else if (_c.attnType == XMODEL_ATTN_MLA || _c.attnType == XMODEL_ATTN_DSA) {
        ForwardAttnMLAV2(rt, layer, kvCache, freqsCis, hiddenState);
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        ForwardAttnMHA(rt, layer, kvCache, freqsCis, hiddenState);
    } else {
        throw std::runtime_error(std::string(__func__) + ": TODO");
    }
}

void XModel::ForwardMLP(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                        std::vector<MatmulWeight> &upGate, std::vector<MatmulWeight> &down,
                        bool withAllReduce)
{
    size_t m = hiddenState.shape[0], k = hiddenState.shape[1];
    std::vector<size_t> &downShape = down[layer].weight.shape;
    size_t localIntermediateSize = (downShape[0] == k) ? downShape[1] : downShape[0];
    XTensor &h13 = rt.GetTensor({m, localIntermediateSize * 2}, hiddenState.dtype, DBG_LOC);
    XTensor &h2 = rt.GetTensor({m, localIntermediateSize}, hiddenState.dtype, DBG_LOC);

    ForwardLinear(rt, layer, hiddenState, upGate, h13);
    XliteOpSiluAndMul(rt, h13, h2);
    ForwardLinear(rt, layer, h2, down, hiddenState);

    if (withAllReduce && _c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (!rt.enableMoEAllToAll) {
            if (rt.enableCommOptimize) {
                XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false,
                                     DBG_LOC);
            } else {
                XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP, false, DBG_LOC);
            }
        }
    }
    rt.PutTensor(h2);
    rt.PutTensor(h13);
}

std::tuple<XTensor &, XTensor &> XModel::ForwardMoEGate(XRuntime &rt, uint32_t layer,
                                                        XTensor &input)
{
    uint32_t m = input.shape[0], M = rt.maxTokensDp;  // input may or may not be padded to match DPs
    XTensor &weights = rt.GetTensor({M, _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC);
    XTensor &routing = rt.GetTensor({M, _c.nRoutedExperts}, BIT1, DBG_LOC);
    XTensor &scores = rt.GetTensor({M, _c.nRoutedExperts}, moeGate[layer].dtype, DBG_LOC).View(m);

    XliteOpMatmul(rt, input, moeGate[layer], scores, _c.gateCaptured && _c.weightNZ);

    if (_c.scoringFunc == XMODEL_SCORING_FUNC_SIGMOID) {
        XliteOpSigmoidTopK(rt, scores.View(M), _gateIndices, moeGateBias[layer], _c.routeScale,
                           weights, routing, _c.nExpertGroups, _c.nLimitedGroups, _c.nActExperts,
                           _c.normTopKProb);
    } else {
        XliteOpSoftmaxTopK(rt, scores.View(M), _gateIndices, weights, routing, _c.nActExperts,
                           _c.normTopKProb);
    }

    rt.PutTensor(scores);
    return {weights, routing};
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
    XModel::ForwardMoEDispatch(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                               XTensor &routing)
{
    // NOTE: the input XTensor's must already be viewed/padded to match DPs, if needed
    uint32_t m = rt.maxTokensDp;  // should equal to `tokenSorted.shape[0]`
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;

    if (_c.defDpSize > 1) {
        XTensor &inputPerDp = tokenSorted, &weightsPerDp = weights, &routingPerDp = routing;
        // 计算字节大小
        size_t totalBytes = inputPerDp.bytes + weightsPerDp.bytes + routingPerDp.bytes;
        size_t allTotalBytes = totalBytes * _c.defDpSize;

        // 获取最终输出用的tensor
        XTensor &inputAllDp = rt.GetTensor({mAllDp, _c.hiddenSize}, inputPerDp.dtype, DBG_LOC);
        XTensor &weightsAllDp =
            rt.GetTensor({mAllDp, _c.nRoutedExperts}, weightsPerDp.dtype, DBG_LOC);
        XTensor &routingAllDp =
            rt.GetTensor({mAllDp, _c.nRoutedExperts}, routingPerDp.dtype, DBG_LOC);

        bool inGraph = rt.AllGatherInGraphActive(DP) && (m <= _c.maxBatch);

        // packedSend: graph → bind fixed _agSendBuf; eager → pool GetTensor (returned below).
        XTensor sendFixed;
        XTensor *sendOut = nullptr;
        if (inGraph) {
            sendFixed.Init({totalBytes}, INT8, rt._agSendBuf.ptr);
            sendOut = &sendFixed;
        } else {
            sendOut = &rt.GetTensor({totalBytes}, INT8, DBG_LOC);
        }

        // 打包tensor
        std::vector<XTensor> inputs = {inputPerDp, weightsPerDp, routingPerDp};
        XliteOpConcat(rt, inputs, *sendOut);
        rt.PutTensor(routingPerDp);
        rt.PutTensor(weightsPerDp);

        // packedRecv: graph → bind fixed _agRecvBuf; eager → pool GetTensor.
        XTensor recvFixed;
        XTensor *recvOut = nullptr;
        if (inGraph) {
            recvFixed.Init({allTotalBytes}, INT8, rt._agRecvBuf.ptr);
            recvOut = &recvFixed;
        } else {
            recvOut = &rt.GetTensor({allTotalBytes}, INT8, DBG_LOC);
        }
        if (inGraph) {
            rt.AllGatherInGraph(sendOut->ptr, recvOut->ptr, m,
                                static_cast<int>(XDtype2HcclDtype(INT8)), DP);
        } else {
            XliteOpAllGather(rt, *sendOut, *recvOut, DP, true, DBG_LOC);
        }

        std::vector<XTensor> outputs = {inputAllDp, weightsAllDp, routingAllDp};
        std::vector<size_t> sizes = {inputPerDp.bytes, weightsPerDp.bytes, routingPerDp.bytes};
        XliteOpSplit(rt, *recvOut, outputs, sizes, _c.defDpSize);

        if (!inGraph) {
            rt.PutTensor(*sendOut);  // eager: return pool; graph: fixed buffer, no Put
            rt.PutTensor(*recvOut);
        }

        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        size_t shape0 = mAllDp * _c.nActExperts;
        if (mAllDp >= XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD) {
            shape0 = static_cast<size_t>(mAllDp * _c.nActExperts * rt.activeTokensRatioPerEp);
        }
        XTensor &expertsSorted = rt.GetTensor({shape0, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XliteOpPermutation(rt, inputAllDp, routingAllDp, start, end, expertsSorted, unpIdx,
                           expertsCounts);
        rt.PutTensor(inputAllDp);
        return {weightsAllDp, routingAllDp, unpIdx, expertsSorted, expertsCounts, {}};
    } else {
        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        size_t shape0 = mAllDp * _c.nActExperts;
        if (mAllDp >= XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD) {
            shape0 = static_cast<size_t>(mAllDp * _c.nActExperts * rt.activeTokensRatioPerEp);
        }
        XTensor &expertsSorted = rt.GetTensor({shape0, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx,
                           expertsCounts);
        return {weights, routing, unpIdx, expertsSorted, expertsCounts, {}};
    }
}

void XModel::ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                               XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                               XTensor &expertsCounts)
{
    // NOTE: the input XTensor's must already be viewed/padded to match DPs, if needed
    uint32_t m = rt.maxTokensDp, mAllDp = rt.maxTokensDp * _c.defDpSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;

    if (_c.defDpSize > 1) {
        bool inGraph = rt.ReduceScatterInGraphActive(DP) && (m <= _c.maxBatch);
        XTensor rsSendFixed;  // graph: bound to _rsSendBuf; eager: unused
        XTensor *sendOut = nullptr;
        if (inGraph) {
            rsSendFixed.Init({mAllDp, _c.hiddenSize}, tokenSorted.dtype, rt._rsSendBuf.ptr);
            sendOut = &rsSendFixed;
        } else {
            sendOut = &rt.GetTensor({mAllDp, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        }
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, *sendOut);
        if (inGraph) {
            rt.ReduceScatterInGraph(rt._rsSendBuf.ptr, rt._rsRecvBuf.ptr, m,
                                    static_cast<int>(XDtype2HcclDtype(tokenSorted.dtype)), DP);
            size_t recvBytes =
                static_cast<size_t>(m) * _c.hiddenSize * XDtypeBit(tokenSorted.dtype) / 8;
            CHECK_ACL(aclrtMemcpyAsync(tokenSorted.ptr, recvBytes, rt._rsRecvBuf.ptr, recvBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        } else {
            XliteOpReduceScatter(rt, *sendOut, tokenSorted, DP, true, DBG_LOC);
        }
        if (!inGraph) {
            rt.PutTensor(*sendOut);  // eager: return pool tensor; graph: fixed buffer, skip
        }
    } else {
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSorted);
    }

    rt.PutTensor(expertsCounts);
    rt.PutTensor(expertsSorted);
    rt.PutTensor(unpIdx);
    rt.PutTensor(routing);
    rt.PutTensor(weights);
}

std::vector<int64_t> ComputePrefixSum(const std::vector<int64_t> &counts)
{
    std::vector<int64_t> displs(counts.size());
    displs[0] = 0;
    for (size_t i = 1; i < counts.size(); i++) {
        displs[i] = displs[i - 1] + counts[i - 1];
    }
    return displs;
}

MoEAlltoAllMeta XModel::MoeComputeAlltoAllVMeta(const int32_t *tokensPerEpGroupAllEpHost,
                                                uint32_t moeEpSize, uint32_t moeTpSize,
                                                uint32_t hiddenSize, uint32_t rankId,
                                                uint32_t nRoutedExperts)
{
    uint32_t currentEp = rankId / moeTpSize;

    std::vector<int64_t> sendCountsData(moeEpSize);
    std::vector<int64_t> recvCountsData(moeEpSize);
    for (uint32_t ep = 0; ep < moeEpSize; ep++) {
        sendCountsData[ep] =
            static_cast<int64_t>(tokensPerEpGroupAllEpHost[currentEp * moeEpSize + ep]) *
            hiddenSize;
        recvCountsData[ep] =
            static_cast<int64_t>(tokensPerEpGroupAllEpHost[ep * moeEpSize + currentEp]) *
            hiddenSize;
    }

    std::vector<int64_t> sdisplsData = ComputePrefixSum(sendCountsData);
    std::vector<int64_t> rdisplsData = ComputePrefixSum(recvCountsData);

    uint64_t totalRecvElements = 0;
    for (auto v : recvCountsData) {
        totalRecvElements += v;
    }

    MoEAlltoAllMeta meta;
    meta.sendCountsData = sendCountsData;
    meta.recvCountsData = recvCountsData;
    meta.sdisplsData = sdisplsData;
    meta.rdisplsData = rdisplsData;
    meta.sendCounts.Init({moeEpSize}, INT64, meta.sendCountsData.data());
    meta.recvCounts.Init({moeEpSize}, INT64, meta.recvCountsData.data());
    meta.sdispls.Init({moeEpSize}, INT64, meta.sdisplsData.data());
    meta.rdispls.Init({moeEpSize}, INT64, meta.rdisplsData.data());
    meta.totalRecvElements = totalRecvElements;
    meta.nRoutedExperts = nRoutedExperts;
    return meta;
}

MoEAlltoAllMeta XModel::MoeComputeReverseAlltoAllVMeta(const MoEAlltoAllMeta &sendMeta,
                                                       uint32_t moeEpSize)
{
    uint64_t totalSendElements = 0;

    for (auto v : sendMeta.sendCountsData) {
        totalSendElements += v;
    }

    MoEAlltoAllMeta meta;
    meta.sendCountsData = sendMeta.recvCountsData;
    meta.recvCountsData = sendMeta.sendCountsData;
    meta.sdisplsData = sendMeta.rdisplsData;
    meta.rdisplsData = sendMeta.sdisplsData;
    meta.sendCounts.Init({moeEpSize}, INT64, meta.sendCountsData.data());
    meta.recvCounts.Init({moeEpSize}, INT64, meta.recvCountsData.data());
    meta.sdispls.Init({moeEpSize}, INT64, meta.sdisplsData.data());
    meta.rdispls.Init({moeEpSize}, INT64, meta.rdisplsData.data());
    meta.totalRecvElements = totalSendElements;
    meta.nRoutedExperts = sendMeta.nRoutedExperts;
    meta.expertsCountsAllEpDevice = sendMeta.expertsCountsAllEpDevice;
    return meta;
}

std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
    XModel::ForwardMoEDispatchAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                       XTensor &routing)
{
    uint32_t m = tokenSorted.shape[0];
    uint32_t mAllDp = m * _c.defDpSize;
    uint32_t start = 0;
    uint32_t end = start + _c.nRoutedExperts;

    if (_c.defDpSize > 1) {
        // step 1: permutation — route tokens to experts, output expertsCounts per rank
        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, m + 1}, INT32, DBG_LOC);
        XTensor &expertsSorted =
            rt.GetTensor({m * _c.nActExperts, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx,
                           expertsCounts);

        // step 2: allgather expertsCounts across EP ranks — each rank knows all ranks' token counts
        XTensor &expertsCountsAllEp =
            rt.GetTensor({_c.moeEpSize, _c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpAllGather(rt, expertsCounts, expertsCountsAllEp, EP, false, DBG_LOC);
        rt.PutTensor(expertsCounts);

        // step 3: compute tokensPerEpGroupAllEp[ep][ep] (tokens each EP rank sends to each EP rank)
        //         and tokensPerExperts[expert] (total tokens per expert across all EP ranks)
        XTensor &tokensPerEpGroupAllEp = rt.GetTensor({_c.moeEpSize, _c.moeEpSize}, INT32, DBG_LOC);
        XTensor &tokensPerExperts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpExpertsCountsSum(rt, expertsCountsAllEp, tokensPerEpGroupAllEp, tokensPerExperts,
                                _c.nRoutedExperts);

        // step 4: copy tokensPerEpGroupAllEp to host, compute meta
        rt.MemcpyD2HAsync(rt._tokensPerEpGroupAllEpHost.ptr, tokensPerEpGroupAllEp.ptr,
                          tokensPerEpGroupAllEp.numel * sizeof(int32_t));
        rt.Synchronize();

        MoEAlltoAllMeta meta = MoeComputeAlltoAllVMeta(
            static_cast<int32_t *>(rt._tokensPerEpGroupAllEpHost.ptr), _c.moeEpSize, _c.moeTPSize,
            _c.hiddenSize, rt.rankId(), _c.nRoutedExperts);
        meta.expertsCountsAllEpDevice = &expertsCountsAllEp;

        uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
        uint32_t localStart = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
        uint32_t localEnd = localStart + nLocalRoutedExperts;

        // step 5: alltoallv — dispatch token data across EP ranks according to meta
        XTensor &recvBuffer = rt.GetTensor({meta.totalRecvElements / _c.hiddenSize, _c.hiddenSize},
                                           tokenSorted.dtype, DBG_LOC);
        XliteOpAlltoAllV(rt, expertsSorted, recvBuffer, meta.sendCounts, meta.recvCounts,
                         meta.sdispls, meta.rdispls, EP, DBG_LOC);

        // step 6: reorder source-grouped → expert-grouped for GroupMatmul
        XTensor &expertGrouped = rt.GetTensor(
            {meta.totalRecvElements / _c.hiddenSize, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XliteOpReorderMoE(rt, recvBuffer, expertGrouped, expertsCountsAllEp, _c.hiddenSize,
                          localStart, localEnd, true);
        rt.PutTensor(recvBuffer);
        rt.PutTensor(tokensPerEpGroupAllEp);
        rt.PutTensor(expertsSorted);

        return {weights, routing, unpIdx, expertGrouped, tokensPerExperts, meta};
    } else {
        XTensor &unpIdx = rt.GetTensor({_c.nRoutedExperts, mAllDp + 1}, INT32, DBG_LOC);
        size_t shape0 = mAllDp * _c.nActExperts;
        if (mAllDp >= XLITE_ACTIVE_TOKENS_RATIO_PER_EP_THRESHOLD) {
            shape0 = static_cast<size_t>(mAllDp * _c.nActExperts * rt.activeTokensRatioPerEp);
        }
        XTensor &expertsSorted = rt.GetTensor({shape0, _c.hiddenSize}, tokenSorted.dtype, DBG_LOC);
        XTensor &expertsCounts = rt.GetTensor({_c.nRoutedExperts}, INT32, DBG_LOC);
        XliteOpPermutation(rt, tokenSorted, routing, start, end, expertsSorted, unpIdx,
                           expertsCounts);

        MoEAlltoAllMeta meta;
        return {weights, routing, unpIdx, expertsSorted, expertsCounts, meta};
    }
}

void XModel::ForwardMoECombineAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                       XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                                       XTensor &expertsCounts, const MoEAlltoAllMeta &meta)
{
    if (_c.defDpSize > 1) {
        if (!meta.expertsCountsAllEpDevice) {
            throw std::runtime_error(std::string(__func__) +
                                     ": meta.expertsCountsAllEpDevice is nullptr");
        }
        uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
        uint32_t localStart = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
        uint32_t localEnd = localStart + nLocalRoutedExperts;

        // reorder expert-grouped → source-grouped for reverse AlltoAllV
        XTensor &sourceGrouped = rt.GetTensor({expertsSorted.shape[0], expertsSorted.shape[1]},
                                              expertsSorted.dtype, DBG_LOC);
        XliteOpReorderMoE(rt, expertsSorted, sourceGrouped, *meta.expertsCountsAllEpDevice,
                          _c.hiddenSize, localStart, localEnd, false);
        MoEAlltoAllMeta revMeta = MoeComputeReverseAlltoAllVMeta(meta, _c.moeEpSize);
        XTensor &backBuffer =
            rt.GetTensor({revMeta.totalRecvElements / _c.hiddenSize, _c.hiddenSize},
                         expertsSorted.dtype, DBG_LOC);
        XliteOpAlltoAllV(rt, sourceGrouped, backBuffer, revMeta.sendCounts, revMeta.recvCounts,
                         revMeta.sdispls, revMeta.rdispls, EP, DBG_LOC);
        XliteOpUnpermutation(rt, backBuffer, unpIdx, routing, weights, 0, _c.nRoutedExperts,
                             tokenSorted);

        rt.PutTensor(sourceGrouped);
        rt.PutTensor(backBuffer);
    } else {
        uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
        uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
        uint32_t end = start + nLocalRoutedExperts;
        XliteOpUnpermutation(rt, expertsSorted, unpIdx, routing, weights, start, end, tokenSorted);
    }

    rt.PutTensor(expertsCounts);
    rt.PutTensor(expertsSorted);
    rt.PutTensor(unpIdx);
    rt.PutTensor(routing);
    rt.PutTensor(weights);
}

void XModel::ForwardMoEMSD(XRuntime &rt, uint32_t layer, XTensor &expertsSorted, XTensor &counts,
                           XTensor &num, uint32_t start, uint32_t end, uint32_t outDim,
                           std::vector<XTensor> &weights, std::vector<XTensor> &deqScales,
                           std::vector<XTensor> &scaleBias, XTensor &output)
{
    // MSD W4A8 routed-expert three-stage pipeline, group form:
    //   1. QuantDyn  : bf16 activation → xINT8 + perTokenScale[m] (FP32)
    //   2. Unpack    : xINT8 [m, k] → xINT8 [2m, k/2] interleaved (token r → row 2r low, 2r+1 high)
    //   3. GroupMatmul xInt4 [2m, k/2] × w_int4 [k/2, n] (single pass, grouped by 2×counts)
    //      → yMerged [2m, n] FP16, where rows (2r, 2r+1) hold the low/high result pair for token r
    //   4. MergeDequant: (yMerged [2m, n] + scaleBias[m, n]) * perTokenScale[m] → [m, n] BF16
    uint32_t m = expertsSorted.shape[0];
    uint32_t k = expertsSorted.shape[1];

    // Step 1: dynamic per-token quantization bf16 → int8
    XTensor &xInt8 = rt.GetTensor(expertsSorted.shape, INT8, DBG_LOC);
    XTensor &perTokenScale = rt.GetTensor({m}, FP32, DBG_LOC);
    XliteOpQuantDyn(rt, expertsSorted, perTokenScale, xInt8, num);

    // Step 2: unpack INT8 → interleaved double-row INT4-packed layout [2m, k/2]
    // For input token r, row 2r = low nibbles, row 2r+1 = high nibbles
    XTensor &xUnpacked = rt.GetTensor({2 * m, k}, INT4, DBG_LOC);
    XliteOpUnpackActivation(rt, xInt8, xUnpacked);
    rt.PutTensor(xInt8);

    // Step 3: INT4 group_matmul → FP16 mid-stage [2m, outDim] (single pass, interleaved rows)
    XTensor &yMerged = rt.GetTensor({2 * m, outDim}, FP16, DBG_LOC);
    XliteOpGroupMatmul(rt, xUnpacked, weights[layer], deqScales[layer], counts, start, end, INT4,
                       outDim, k, yMerged, _c.weightNZ || _c.expertsWeightNZ,
                       _c.expertsWeightTrans);
    rt.PutTensor(xUnpacked);

    // Step 4: (yMerged [2m, n] + scaleBias[m, n]) * perTokenScale[m] for target experts
    XliteOpMSDMergeDequant(rt, yMerged, scaleBias[layer], counts, start, end, perTokenScale,
                           output);
    rt.PutTensor(yMerged);
    rt.PutTensor(perTokenScale);
}

void XModel::ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    // the original shape of hiddenState must not be smaller than (maxTokensDp, hiddenSize)
    size_t m = rt.currTokens, M = rt.maxTokensDp;
    uint32_t intermediateSize = _c.moeIntermediateSize / _c.moeTPSize;
    uint32_t nLocalRoutedExperts = _c.nRoutedExperts / _c.moeEpSize;
    uint32_t start = _c.moeEpSize == 1 ? 0 : _rankId / _c.moeTPSize * nLocalRoutedExperts;
    uint32_t end = start + nLocalRoutedExperts;
    enum XDtype moeReDtype = moeREUpGate[layer][start].dtype;

    auto [w, r] = ForwardMoEGate(rt, layer, hiddenState);
    auto [weights, routing, unpIdx, expertsSorted, expertsCounts, meta] =
        rt.enableMoEAllToAll ? ForwardMoEDispatchAllToAll(rt, hiddenState, w, r)
                             : ForwardMoEDispatch(rt, hiddenState.View(M), w, r);
    // actual token num for current rank
    XTensor num = rt.enableMoEAllToAll ? XTensor() : XTensor({1}, INT32, unpIdx.ptr);

    // routed experts
    XTensor *h13Ptr;
    if (moeReDtype == INT4) {
        // MSD W4A8: QuantDyn → Unpack → INT4 group_matmul → merge_dequant
        XTensor &h13 = rt.GetTensor({expertsSorted.shape[0], intermediateSize * 2},
                                    hiddenState.dtype, DBG_LOC);
        ForwardMoEMSD(rt, layer, expertsSorted, expertsCounts, num, start, end,
                      intermediateSize * 2, _moeREUpGate, _moeREUpGateDeqScale,
                      _moeREUpGateScaleBias, h13);
        rt.PutTensor(expertsSorted);
        h13Ptr = &h13;
    } else if (moeReDtype == INT8) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &xQuanted = rt.GetTensor(expertsSorted.shape, moeReDtype, DBG_LOC);
        XTensor &scale = rt.GetTensor({expertsSorted.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, expertsSorted, scale, xQuanted, num);
        rt.PutTensor(expertsSorted);

        // group_matmul(xQuanted * w13 * w13Scale) * perChannelScale -> h13
        XTensor &h13 = rt.GetTensor({expertsSorted.shape[0], intermediateSize * 2},
                                    hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmulDeQuant(rt, xQuanted, _moeREUpGate[layer], _moeREUpGateDeqScale[layer],
                                  expertsCounts, start, end, moeREUpGate[layer][start].dtype,
                                  intermediateSize * 2, _c.hiddenSize, h13, scale, num,
                                  _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(xQuanted);
        rt.PutTensor(scale);
        h13Ptr = &h13;
    } else {
        XTensor &h13 = rt.GetTensor({expertsSorted.shape[0], intermediateSize * 2},
                                    hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmul(rt, expertsSorted, _moeREUpGate[layer], _moeREUpGateDeqScale[layer],
                           expertsCounts, start, end, moeREUpGate[layer][start].dtype,
                           intermediateSize * 2, _c.hiddenSize, h13,
                           _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(expertsSorted);
        h13Ptr = &h13;
    }

    XTensor &h2 =
        rt.GetTensor({expertsSorted.shape[0], intermediateSize}, hiddenState.dtype, DBG_LOC);
    XliteOpSiluAndMul(rt, *h13Ptr, h2, num);
    rt.PutTensor(*h13Ptr);

    XTensor *outPtr;
    if (moeReDtype == INT4) {
        // MSD W4A8: QuantDyn → Unpack → INT4 group_matmul → merge_dequant (w2/down)
        XTensor &out =
            rt.GetTensor({expertsSorted.shape[0], _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        ForwardMoEMSD(rt, layer, h2, expertsCounts, num, start, end, _c.hiddenSize, _moeREDown,
                      _moeREDownDeqScale, _moeREDownScaleBias, out);
        rt.PutTensor(h2);
        outPtr = &out;
    } else if (moeReDtype == INT8) {
        // quant(x) -> xQuanted, perChannelScale
        XTensor &xQuanted = rt.GetTensor(h2.shape, moeReDtype, DBG_LOC);
        XTensor &scale = rt.GetTensor({h2.shape[0]}, FP32, DBG_LOC);
        XliteOpQuantDyn(rt, h2, scale, xQuanted, num);
        rt.PutTensor(h2);

        // group_matmul(xQuanted * w2 * w2Scale) * perChannelScale -> h2
        XTensor &out =
            rt.GetTensor({expertsSorted.shape[0], _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmulDeQuant(rt, xQuanted, _moeREDown[layer], _moeREDownDeqScale[layer],
                                  expertsCounts, start, end, moeREDown[layer][start].dtype,
                                  _c.hiddenSize, intermediateSize, out, scale, num,
                                  _c.weightNZ || _c.expertsWeightNZ, _c.expertsWeightTrans);
        rt.PutTensor(xQuanted);
        rt.PutTensor(scale);
        outPtr = &out;
    } else {
        XTensor &out =
            rt.GetTensor({expertsSorted.shape[0], _c.hiddenSize}, hiddenState.dtype, DBG_LOC);
        XliteOpGroupMatmul(rt, h2, _moeREDown[layer], _moeREDownDeqScale[layer], expertsCounts,
                           start, end, moeREDown[layer][start].dtype, _c.hiddenSize,
                           intermediateSize, out, _c.weightNZ || _c.expertsWeightNZ,
                           _c.expertsWeightTrans);
        rt.PutTensor(h2);
        outPtr = &out;
    }

    // Check if shared experts should be processed on this rank:
    // 1. Shared experts are enabled (nSharedExperts != 0)
    // 2. Either the weight is not full (all ranks process), or only rank 0 processes when weight is
    // full
    if (_c.nSharedExperts != 0 &&
        ((_isSharedExpertWeightFull && ((rt.rankId() % rt.tpSize()) == 0)) ||
         !_isSharedExpertWeightFull)) {
        XTensor &h = rt.GetTensor(hiddenState.shape, hiddenState.dtype, DBG_LOC);
        if (rt.enableMoEAllToAll) {  // hiddenState of shape (currTokens = m, hiddenSize)
            ForwardMoECombineAllToAll(rt, h, weights, routing, unpIdx, *outPtr, expertsCounts,
                                      meta);
        } else {  // hiddenState of shape (maxTokensDp, hiddenSize)
            ForwardMOECombine(rt, h, weights, routing, unpIdx, *outPtr, expertsCounts);
        }
        // share experts
        ForwardMLP(rt, layer, hiddenState.View(m), moeSEUpGate, moeSEDown, false);
        XliteOpAdd(rt, hiddenState, h.View(m), hiddenState);
        rt.PutTensor(h);
    } else {
        if (rt.enableMoEAllToAll) {  // hiddenState of shape (currTokens = m, hiddenSize)
            ForwardMoECombineAllToAll(rt, hiddenState, weights, routing, unpIdx, *outPtr,
                                      expertsCounts, meta);
        } else {  // hiddenState of shape (maxTokensDp, hiddenSize)
            ForwardMOECombine(rt, hiddenState, weights, routing, unpIdx, *outPtr, expertsCounts);
            hiddenState.View(m);
        }
    }

    if (rt.enableMoEAllToAll && meta.expertsCountsAllEpDevice != nullptr) {
        rt.PutTensor(*meta.expertsCountsAllEpDevice);
    }

    if (_c.defTpSize > 1) {
        if (rt.multiTaskParallel) {
            rt.NotifyRecordPeerStream();
        }
        if (!rt.enableMoEAllToAll) {
            if (rt.enableCommOptimize) {
                XliteOpReduceScatter(rt, rt.hiddenStatePad, rt.hiddenStateSlice, TP, false,
                                     DBG_LOC);
            } else {
                XliteOpAllReduceSum(rt, hiddenState, hiddenState, TP, false, DBG_LOC);
            }
        }
    }
}

void XModel::ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState)
{
    if (layer < _c.nDenseLayers) {
        ForwardMLP(rt, layer, hiddenState, mlpUpGate, mlpDown, true);
    } else {
        ForwardMoE(rt, layer, hiddenState);
    }
}

void XModel::ForwardLayersCommOptimize(XRuntime &rt, XTensor &xPad,
                                       std::vector<std::vector<XTensor>> &kvCache,
                                       std::vector<XTensor> &deepstackInputEmbeds,
                                       XTensor &freqsCis, XTensor &output)
{
    // m: actual token num (x, h, output)
    // mPad: token num padded to match TP size (xPad, hiddenStatePad)
    // mPadPerTp: mPad / TP size (xSlice, hiddenStateSlice)
    // M: max possible token num considering both TP and DP (outputPad, original h/hiddenStatePad)
    XTensor xSlice;
    size_t m = rt.currTokens, mPad = xPad.shape[0], M = std::max(mPad, rt.maxTokensDp);
    size_t mPadPerTp = mPad / _c.defTpSize;
    size_t sizePerTp = mPad * _c.hiddenSize / _c.defTpSize * XDtypeBit(embed.dtype) / 8;
    XTensor x, h;
    x.Init({m, _c.hiddenSize}, embed.dtype, xPad.ptr);

    XTensor &outputPad = rt.GetTensor({M, _c.hiddenSize}, embed.dtype, DBG_LOC);
    h.Init({M, _c.hiddenSize}, embed.dtype, outputPad.ptr).View(m);
    rt.hiddenStatePad.Init({M, _c.hiddenSize}, embed.dtype, outputPad.ptr).View(mPad);

    void *slicePtr = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(rt.hiddenStatePad.ptr) +
                                              (rt.rankId() % _c.defTpSize) * sizePerTp);
    rt.hiddenStateSlice.Init({mPadPerTp, _c.hiddenSize}, rt.hiddenStatePad.dtype, slicePtr);
    slicePtr = reinterpret_cast<void *>(reinterpret_cast<uint64_t>(xPad.ptr) +
                                        (rt.rankId() % _c.defTpSize) * sizePerTp);
    xSlice.Init({mPadPerTp, _c.hiddenSize}, xPad.dtype, slicePtr);

    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1], true, attnNormBias[i]);
        }
        XDEBUG_SET_STATE(_rankId == 0 && (i == 0 || i == _c.nDenseLayers));
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " in").c_str(), 1e6f);
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after attn").c_str(), 1e6f);
        XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, mlpNorm[i], _c.normEps,
                             rt.hiddenStateSlice, mlpNormBias[i]);
        if (!rt.enableMoEAllToAll) {
            XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP, false, DBG_LOC);
        }
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
        ForwardFFN(rt, i, rt.enableMoEAllToAll ? rt.hiddenStateSlice : h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after ffn").c_str(), 1e6f);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, attnNorm[i + 1], _c.normEps,
                                 rt.hiddenStateSlice, attnNormBias[i + 1]);
            XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP, false, DBG_LOC);
        }
        if (rt.multiTaskParallel) {
            rt.NotifyWaitPeerStream();
        }
    }
    XliteOpAddAndRmsNorm(rt, rt.hiddenStateSlice, xSlice, norm, _c.normEps, rt.hiddenStateSlice,
                         normBias);
    XliteOpAllGather(rt, rt.hiddenStateSlice, rt.hiddenStatePad, TP, false, DBG_LOC);
    aclrtMemcpyAsync(output.ptr, output.bytes, rt.hiddenStatePad.ptr, output.bytes,
                     ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
    rt.PutTensor(outputPad);
}

void XModel::ForwardLayersNaive(XRuntime &rt, XTensor &x,
                                std::vector<std::vector<XTensor>> &kvCache,
                                std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                XTensor &output)
{
    XTensor &h =
        rt.GetTensor({rt.maxTokensDp, _c.hiddenSize}, embed.dtype, DBG_LOC).View(rt.currTokens);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        if (i == 0) {
            XliteOpRmsNorm(rt, x, attnNorm[i], h, _c.normEps, x.shape[1], true, attnNormBias[i]);
        }
        XDEBUG_SET_STATE(_rankId == 0 && (i == 0 || i == _c.nDenseLayers));
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " in").c_str(), 1e6f);
        ForwardAttn(rt, i, kvCache, freqsCis, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after attn").c_str(), 1e6f);
        XliteOpAddAndRmsNorm(rt, h, x, mlpNorm[i], _c.normEps, h, mlpNormBias[i]);
        ForwardFFN(rt, i, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after ffn").c_str(), 1e6f);
        if (i < _c.deepstackNumLevel) {
            XliteOpAdd(rt, h, deepstackInputEmbeds[i], h);
        }
        if (i < (_c.nLayers - 1)) {
            XliteOpAddAndRmsNorm(rt, h, x, attnNorm[i + 1], _c.normEps, h, attnNormBias[i + 1]);
        }
    }
    XliteOpAddAndRmsNorm(rt, h, x, norm, _c.normEps, output, normBias);
    rt.PutTensor(h);
}

void XModel::ForwardHcPre(XRuntime &rt, XTensor &input, XTensor &hcFn, XTensor &hcScale,
                          XTensor &hcBase, XTensor &output, const XTensor &post,
                          const XTensor &comb)
{
    // TODO
}

void XModel::ForwardHcPost(XRuntime &rt, XTensor &input, XTensor &post, XTensor &comb,
                           XTensor &residual, XTensor &output)
{
    // TODO
}

void XModel::ForwardLayersMhc(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                              std::vector<XTensor> &freqsCis, XTensor &output)
{
    XTensor &residual = rt.GetTensor({x.shape[0], _c.hcMult, _c.hiddenSize}, embed.dtype, DBG_LOC);
    XTensor &h =
        rt.GetTensor({rt.maxTokensDp, _c.hiddenSize}, embed.dtype, DBG_LOC).View(rt.currTokens);
    XTensor &post = rt.GetTensor({x.shape[0], _c.hcMult}, embed.dtype, DBG_LOC);
    XTensor &comb = rt.GetTensor({x.shape[0], _c.hcMult * _c.hcMult}, embed.dtype, DBG_LOC);
    for (uint32_t i = 0; i < _c.nLayers; i++) {
        XDEBUG_SET_STATE(_rankId == 0 && (i == 0 || i == _c.nDenseLayers));
        XDEBUG_PRINT_X(rt, i == 0 ? x : residual, ("L" + std::to_string(i) + " in").c_str(), 1e6f);
        ForwardHcPre(rt, i == 0 ? x : residual, hcAttnFn[i], hcAttnScale[i], hcAttnBase[i], h, post,
                     comb);
        XliteOpRmsNorm(rt, h, attnNorm[i], h, _c.normEps, _c.hiddenSize, true, attnNormBias[i]);
        ForwardAttnCXA(rt, i, kvCache, freqsCis[i], h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after attn").c_str(), 1e6f);
        ForwardHcPost(rt, h, post, comb, i == 0 ? x : residual, residual);

        ForwardHcPre(rt, residual, hcFfnFn[i], hcFfnScale[i], hcFfnBase[i], h, post, comb);
        XliteOpRmsNorm(rt, h, mlpNorm[i], h, _c.normEps, _c.hiddenSize, true, mlpNormBias[i]);
        ForwardFFN(rt, i, h);
        XDEBUG_PRINT_X(rt, h, ("L" + std::to_string(i) + " after ffn").c_str(), 1e6f);
        ForwardHcPost(rt, h, post, comb, residual, residual);
    }
    ForwardHcPre(rt, residual, hcHeadFn, hcHeadScale, hcHeadBase, h);
    XliteOpRmsNorm(rt, h, norm, output, _c.normEps, _c.hiddenSize, true, normBias);
    rt.PutTensor(comb);
    rt.PutTensor(post);
    rt.PutTensor(h);
    rt.PutTensor(residual);
}

void XModel::ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                                   std::vector<std::vector<XTensor>> &kvCache,
                                   std::vector<XTensor> &deepstackInputEmbeds,
                                   std::vector<XTensor> &freqsCis, XTensor &h)
{
    if (_c.hcMult == 0) {
        // under DP, different ranks may routed differently with `enableCommOptimize`
        // enabled/disabled;
        // thus, `ForwardFNN` from both paths must be guarded for correct DP communications
        if (rt.enableCommOptimize) {
            size_t m = rt.currTokens, mPad = ROUND_UP(rt.currTokens, _c.defTpSize);
            XTensor &xPad = rt.GetTensor({mPad, _c.hiddenSize}, embed.dtype, DBG_LOC);
            ForwardParallelEmbed(rt, input, embed, xPad.View(m));
            ForwardLayersCommOptimize(rt, xPad.View(mPad), kvCache, deepstackInputEmbeds,
                                      freqsCis[0], h);
            rt.PutTensor(xPad);
        } else {
            XTensor &x = rt.GetTensor({rt.currTokens, _c.hiddenSize}, embed.dtype, DBG_LOC);
            ForwardParallelEmbed(rt, input, embed, x);
            ForwardLayersNaive(rt, x, kvCache, deepstackInputEmbeds, freqsCis[0], h);
            rt.PutTensor(x);
        }
    } else {
        XTensor &x = rt.GetTensor({rt.currTokens, _c.hiddenSize}, embed.dtype, DBG_LOC);
        ForwardParallelEmbed(rt, input, embed, x);
        ForwardLayersMhc(rt, x, kvCache, freqsCis, h);
        rt.PutTensor(x);
    }
}

void XModel::ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &indices, XTensor &output)
{
    uint32_t batch = rt._batch;
    XTensor localOutput({output.shape[1], output.shape[2]}, output.dtype, output.ptr);

    if (batch < input.shape[0]) {
        XTensor &x = rt.GetTensor({batch, _c.hiddenSize}, input.dtype, DBG_LOC);
        XliteOpEmbed(rt, indices, input, 0, input.shape[0], x);
        XliteOpMatmul(rt, x, head, localOutput, _c.weightNZ);
        rt.PutTensor(x);
    } else {
        XliteOpMatmul(rt, input, head, localOutput, _c.weightNZ);
    }

    if (_c.defTpSize > 1) {
        XliteOpAllGather(rt, localOutput, output, TP, false, DBG_LOC);
    }
}

void XModel::ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                     std::vector<std::vector<XTensor>> &kvCache,
                                     std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                     XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads, _c.nKvHeads,
                   _c.blockSize, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                   static_cast<int>(embed.dtype),
                   (_c.nDenseLayers < _c.nLayers) ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                                                  : static_cast<int>(embed.dtype),
                   _c.indexTopK);
    if (rt.batchedTokens < input.shape[0]) {
        input.View(rt.batchedTokens);
    }
    rt.currTokens = input.shape[0];
    rt.maxTokensDp = PadForDp(rt) ? output.OrigShape()[0] : rt.currTokens;
    output.View(rt.currTokens);

    ConfigRtCommOptimize(rt, rt.currTokens);
    if (rt.enableCommOptimize) {
        size_t mPad = ROUND_UP(input.shape[0], _c.defTpSize);
        XTensor *xPadPtr, xPad;
        if (input.OrigShape()[0] >= mPad) {
            xPad.Init({mPad, _c.hiddenSize}, embed.dtype, input.ptr);
            xPadPtr = &xPad;
        } else {
            xPadPtr = &rt.GetTensor({mPad, _c.hiddenSize}, embed.dtype, DBG_LOC);
            aclrtMemcpyAsync(xPadPtr->ptr, input.bytes, input.ptr, input.bytes,
                             ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream);
        }
        ForwardLayersCommOptimize(rt, *xPadPtr, kvCache, deepstackInputEmbeds, freqsCis, output);
        rt.PutTensor(*xPadPtr);
    } else {
        ForwardLayersNaive(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
    }
}

void XModel::Forward(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                     std::vector<std::vector<XTensor>> &kvCache,
                     std::vector<XTensor> &deepstackInputEmbeds, std::vector<XTensor> &freqsCis,
                     XTensor &output)
{
    CheckForwardParam(rt, kvCache);
    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads, _c.nKvHeads,
                   _c.blockSize, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                   static_cast<int>(embed.dtype),
                   (_c.nDenseLayers < _c.nLayers) ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                                                  : static_cast<int>(embed.dtype),
                   _c.indexTopK);
    if (rt.batchedTokens < input.shape[0]) {
        input.View(rt.batchedTokens);
    }
    rt.currTokens = input.shape[0];
    rt.maxTokensDp = PadForDp(rt) ? output.OrigShape()[0] : rt.currTokens;
    output.View(rt.currTokens);

    ConfigRtCommOptimize(rt, rt.currTokens);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, output);
}

void XModel::ForwardAndGetLogits(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<XTensor>> &kvCache,
                                 std::vector<XTensor> &deepstackInputEmbeds,
                                 std::vector<XTensor> &freqsCis, XTensor &indices, XTensor &output)
{
    CheckForwardParam(rt, kvCache);

    rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads, _c.nKvHeads,
                   _c.blockSize, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                   static_cast<int>(embed.dtype),
                   (_c.nDenseLayers < _c.nLayers) ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                                                  : static_cast<int>(embed.dtype),
                   _c.indexTopK);
    if (rt.batchedTokens < input.shape[0]) {
        input.View(rt.batchedTokens);
    }
    rt.currTokens = input.shape[0];
    // shape of output: world_size, numTokens, vocab_size // world_size (numTokens = maxTokensDp)
    rt.maxTokensDp = PadForDp(rt) ? output.OrigShape()[output.shape.size() - 2] : rt.currTokens;

    ConfigRtCommOptimize(rt, rt.currTokens);
    XTensor &h =
        rt.GetTensor({rt.maxTokensDp, _c.hiddenSize}, embed.dtype, DBG_LOC).View(rt.currTokens);
    ForwardEmbedAndLayers(rt, input, kvCache, deepstackInputEmbeds, freqsCis, h);
    // here, output is not contiguous by `numTokens` dimension, thus we must view `h` instead
    // TODO: change output's shape to be of `numTokens, world_size, vocab_size // world_size`
    ForwardGetLogits(rt, h.View(rt.maxTokensDp), indices, output);
    rt.PutTensor(h);
}

void XModel::CheckForwardParam(XRuntime &rt, std::vector<std::vector<XTensor>> &kvCache)
{
#ifdef XLITE_310P_LLM_FP16_POC
    if (rt.multiTaskParallel) {
        throw std::runtime_error(
            "Ascend310P llm_fp16 POC does not support CoreAssigner/multi-task concurrency");
    }
    if (embed.dtype != FP16 || norm.dtype != FP16 || head.dtype != FP16) {
        throw std::runtime_error("Ascend310P llm_fp16 POC model tensors must all be FP16");
    }
#endif
    if (rt.rankId() != _rankId || rt.tpSize() != _c.defTpSize || rt.dpSize() != _c.defDpSize) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": check runtime communication setting failed");
    }

    if (!rt.Inited()) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": xlite runtime not inited");
    }

    if (kvCache.size() != _c.nLayers) {
        throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                 ": state cache size must equal nLayers");
    }

#ifdef XLITE_310P_LLM_FP16_POC
    for (size_t layer = 0; layer < kvCache.size(); ++layer) {
        if (kvCache[layer].size() != 2 || kvCache[layer][0].dtype != FP16 ||
            kvCache[layer][1].dtype != FP16) {
            throw std::runtime_error(
                "Ascend310P llm_fp16 POC requires exactly two FP16 4D KV tensors per layer");
        }
    }
#endif

    if (_c.attnType == XMODEL_ATTN_HYBRID) {
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        uint32_t nLocalKHeads = _c.linearNumKHeads / _c.defTpSize;
        uint32_t nLocalVHeads = _c.linearNumVHeads / _c.defTpSize;
        uint32_t convDim =
            nLocalKHeads * _c.linearKeyHeadDim * 2 + nLocalVHeads * _c.linearValueHeadDim;
        for (uint32_t i = 0; i < _c.nLayers; i++) {
            if (kvCache[i].size() != 2) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": each layer must provide two state tensors");
            }
            const XTensor &c0 = kvCache[i][0];
            const XTensor &c1 = kvCache[i][1];
            if (_layerTypes[i] == XMODEL_LAYER_ATTN_FULL) {
                if (c0.shape.size() != 4 || c1.shape.size() != 4 || c0.shape[1] != _c.blockSize ||
                    c1.shape[1] != _c.blockSize || c0.shape[2] != expectedKvHeads ||
                    c1.shape[2] != expectedKvHeads || c0.shape[3] != _c.headDim ||
                    c1.shape[3] != _c.headDim) {
                    throw std::runtime_error(
                        std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                        ": full-attention cache shape mismatch at layer " + std::to_string(i));
                }
            } else {
                if (c0.shape.size() != 3 || c0.shape[0] != _c.maxBatch || c0.shape[1] != convDim ||
                    c0.shape[2] != _c.linearConvKernelDim || c1.shape.size() != 4 ||
                    c1.shape[0] != _c.maxBatch || c1.shape[1] != nLocalVHeads ||
                    c1.shape[2] != _c.linearKeyHeadDim || c1.shape[3] != _c.linearValueHeadDim) {
                    throw std::runtime_error(
                        std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                        ": linear-attention state shape mismatch at layer " + std::to_string(i));
                }
            }
        }
    } else if (_c.attnType == XMODEL_ATTN_MHA) {
        XTensor &kCache = kvCache[0][0];
        XTensor &vCache = kvCache[0][1];
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        if (kCache.shape[1] != _c.blockSize || vCache.shape[1] != _c.blockSize ||
            kCache.shape[2] != expectedKvHeads || vCache.shape[2] != expectedKvHeads ||
            kCache.shape[3] != _c.headDim || vCache.shape[3] != _c.headDim) {
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": kv cache's shape not match [block_num, block_size, kv_head_num, head_size]");
        }
    } else if (_c.attnType == XMODEL_ATTN_MLA || _c.attnType == XMODEL_ATTN_DSA) {
        XTensor &kCache = kvCache[0][0];
        XTensor &vCache = kvCache[0][1];
        uint32_t expectedKvHeads = std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
        if (kCache.shape[1] != _c.blockSize || kCache.shape[2] != expectedKvHeads ||
            kCache.shape[3] != _c.kvLoraRank) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": k nope cache's shape not match [block_num, block_size, "
                                     "kv_head_num, kv_lora_rank]");
        }
        if (vCache.shape[1] != _c.blockSize || vCache.shape[2] != expectedKvHeads ||
            vCache.shape[3] != _c.ropeHeadDim) {
            throw std::runtime_error(
                std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                ": pe cache's shape not match [block_num, block_size, kv_head_num, rope_head_dim]");
        }
        if (_c.attnType == XMODEL_ATTN_DSA) {
            XTensor &indexKCache = kvCache[0][2];
            if (indexKCache.shape[1] != _c.blockSize || indexKCache.shape[2] != 1 ||
                indexKCache.shape[3] != _c.indexHeadDim) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": DSA index k cache's shape not match [block_num, "
                                         "block_size, 1, index_head_size]");
            }
        }
    } else if (_c.attnType == XMODEL_ATTN_CXA) {
        // Per-layer 5-tuple: (indexer_state, indexer_k, compress_kv, state, swa_kv).
        // Each cache has its own block_num sized to the slots it actually holds.
        // indexer_* exist only on compress_ratio==4 layers; compress_kv/state exist
        // only on compress_ratio!=0 layers; swa_kv exists on every layer.
        if (kvCache[0].size() != 5) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": CXA kv cache must be a 5-tuple per layer");
        }
        // check layer 2
        uint32_t checkLayer = _c.nLayers <= 2 ? 0 : 2;
        uint32_t ratio = _c.compressRatios[checkLayer];
        bool hasIndexer = (ratio == 4);
        bool hasCompress = (ratio != 0);
        uint32_t coff = 1 + (hasIndexer ? 1 : 0);  // overlap == (ratio == 4)

        auto isEmpty = [](XTensor &t) { return t.numel == 0 || t.shape.empty(); };
        auto checkShape = [](XTensor &t, uint32_t blockSize, uint32_t headNum, size_t dim,
                             const char *what) {
            if (t.shape.size() != 4 || t.shape[1] != blockSize || t.shape[2] != headNum ||
                t.shape[3] != dim) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA " + what +
                                         " shape not match [block_num, "
                                         "block_size, head_num, dim]");
            }
        };

        XTensor &indexerState = kvCache[checkLayer][0];
        XTensor &indexerK = kvCache[checkLayer][1];
        if (hasIndexer) {
            if (isEmpty(indexerState) || isEmpty(indexerK)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA indexer cache must be non-empty on ratio==4 layer");
            }
            checkShape(indexerState, _c.blockSize, 1, 2 * coff * _c.indexHeadDim,
                       "indexer state cache");
            checkShape(indexerK, _c.blockSize, 1, _c.indexHeadDim, "indexer k cache");
        } else {
            if (!isEmpty(indexerState) || !isEmpty(indexerK)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA indexer cache must be empty on non-ratio==4 layer");
            }
        }

        XTensor &compressKv = kvCache[checkLayer][2];
        XTensor &state = kvCache[checkLayer][3];
        if (hasCompress) {
            if (isEmpty(compressKv) || isEmpty(state)) {
                throw std::runtime_error(
                    std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                    ": CXA compress cache must be non-empty on ratio!=0 layer");
            }
            checkShape(compressKv, _c.blockSize, 1, _c.headDim, "compress kv cache");
            checkShape(state, _c.blockSize, 1, 2 * coff * _c.headDim, "state cache");
        } else {
            if (!isEmpty(compressKv) || !isEmpty(state)) {
                throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                         ": CXA compress cache must be empty on ratio==0 layer");
            }
        }

        XTensor &swaKv = kvCache[checkLayer][4];
        if (isEmpty(swaKv)) {
            throw std::runtime_error(std::string(__FILE__) + ":" + std::to_string(__LINE__) +
                                     ": CXA swa kv cache must be non-empty");
        }
        checkShape(swaKv, _c.blockSize, 1, _c.headDim, "swa kv cache");
    }
}

size_t XModel::DummyRun()
{
    XDummyRuntime rt(0, 0, _rankId, _c.defTpSize, _c.defDpSize, _c.moeTPSize, _c.moeEpSize);
    rt.InitDummyRuntime(1ull << 40);

    _c.maxBatchedTokens = ROUND_UP(_c.maxBatchedTokens, _c.defTpSize);

    auto buildKvCache = [&](uint32_t maxNumBlocks) {
        std::vector<std::vector<XTensor>> kvCache(_c.nLayers);
        for (uint32_t i = 0; i < _c.nLayers; i++) {
            uint32_t expectedKvHeads =
                std::max(_c.nKvHeads / _c.defTpSize, static_cast<uint32_t>(1));
            if (_c.attnType == XMODEL_ATTN_HYBRID) {
                if (_layerTypes[i] == XMODEL_LAYER_ATTN_FULL) {
                    XTensor kCache(
                        {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.headDim},
                        embed.dtype, nullptr);
                    XTensor vCache(
                        {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.headDim},
                        embed.dtype, nullptr);
                    kvCache[i] = {kCache, vCache};
                } else {
                    uint32_t nLocalKHeads = _c.linearNumKHeads / _c.defTpSize;
                    uint32_t nLocalVHeads = _c.linearNumVHeads / _c.defTpSize;
                    uint32_t convDim = nLocalKHeads * _c.linearKeyHeadDim * 2 +
                                       nLocalVHeads * _c.linearValueHeadDim;
                    XTensor convState({_c.maxBatch, convDim, _c.linearConvKernelDim}, embed.dtype,
                                      nullptr);
                    XTensor ssmState(
                        {_c.maxBatch, nLocalVHeads, _c.linearKeyHeadDim, _c.linearValueHeadDim},
                        embed.dtype, nullptr);
                    kvCache[i] = {convState, ssmState};
                }
            } else if (_c.attnType == XMODEL_ATTN_MHA) {
                XTensor kCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.headDim},
                    embed.dtype, nullptr);
                XTensor vCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.headDim},
                    embed.dtype, nullptr);
                kvCache[i] = {kCache, vCache};
            } else if (_c.attnType == XMODEL_ATTN_MLA) {
                XTensor kCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.kvLoraRank},
                    embed.dtype, nullptr);
                XTensor vCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.ropeHeadDim},
                    embed.dtype, nullptr);
                kvCache[i] = {kCache, vCache};
            } else if (_c.attnType == XMODEL_ATTN_DSA) {
                XTensor kCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.kvLoraRank},
                    embed.dtype, nullptr);
                XTensor vCache(
                    {_c.maxBatch * maxNumBlocks, _c.blockSize, expectedKvHeads, _c.ropeHeadDim},
                    embed.dtype, nullptr);
                XTensor indexKCache({_c.maxBatch * maxNumBlocks, _c.blockSize, 1, _c.indexHeadDim},
                                    embed.dtype, nullptr);
                kvCache[i] = {kCache, vCache, indexKCache};
            } else if (_c.attnType == XMODEL_ATTN_CXA) {
                uint32_t ratio = _c.compressRatios.empty() ? 0 : _c.compressRatios[i];
                bool hasIndexer = (ratio == 4);
                bool hasCompress = (ratio != 0);
                uint32_t coff = 1 + (hasIndexer ? 1 : 0);
                uint32_t swaBlocks = _c.maxBatch * DIV_ROUND_UP(_c.windowSize, _c.blockSize);
                uint32_t compKvBlocks =
                    hasCompress ? _c.maxBatch * DIV_ROUND_UP(_c.maxSeqLen / ratio, _c.blockSize)
                                : 0;
                uint32_t compStateBlocks =
                    hasCompress ? _c.maxBatch * DIV_ROUND_UP(coff * ratio, _c.blockSize) : 0;
                uint32_t idxStateBlocks =
                    hasIndexer ? _c.maxBatch * DIV_ROUND_UP(coff * ratio, _c.blockSize) : 0;

                auto emptyTensor = [&]() -> XTensor { return XTensor({0}, embed.dtype, nullptr); };

                XTensor indexerState =
                    hasIndexer
                        ? XTensor({idxStateBlocks, _c.blockSize, 1, 2 * coff * _c.indexHeadDim},
                                  embed.dtype, nullptr)
                        : emptyTensor();
                XTensor indexerK = hasIndexer
                                       ? XTensor({compKvBlocks, _c.blockSize, 1, _c.indexHeadDim},
                                                 embed.dtype, nullptr)
                                       : emptyTensor();
                XTensor compressKv =
                    hasCompress
                        ? XTensor({compKvBlocks, _c.blockSize, 1, _c.headDim}, embed.dtype, nullptr)
                        : emptyTensor();
                XTensor state =
                    hasCompress ? XTensor({compStateBlocks, _c.blockSize, 1, 2 * coff * _c.headDim},
                                          embed.dtype, nullptr)
                                : emptyTensor();
                XTensor swaKv({swaBlocks, _c.blockSize, 1, _c.headDim}, embed.dtype, nullptr);
                kvCache[i] = {indexerState, indexerK, compressKv, state, swaKv};
            }
        }
        return kvCache;
    };

    auto runPrefill = [&]() {
        XModelAttnMeta attnMeta;
        attnMeta.version = 0;
        uint32_t batchSize = 1;
        uint32_t seqLen = _c.maxBatchedTokens;
        uint32_t maxNumBlocks = 0;
        for (uint32_t i = 0; i < batchSize; i++) {
            attnMeta.lens.push_back(seqLen);
            attnMeta.cachedLens.push_back(_c.maxSeqLen > seqLen ? _c.maxSeqLen - seqLen : 0);
            uint32_t blocks = DIV_ROUND_UP(attnMeta.lens[i] + attnMeta.cachedLens[i], _c.blockSize);
            std::vector<uint32_t> blockTable(blocks);
            for (uint32_t j = 0; j < blocks; j++) {
                blockTable[j] = i * blocks + j;
            }
            attnMeta.blockTables.push_back(blockTable);
            if (blocks > maxNumBlocks) {
                maxNumBlocks = blocks;
            }
        }
        auto kvCache = buildKvCache(maxNumBlocks);
        std::vector<XTensor> deepstackInputEmbeds(_c.deepstackNumLevel);
        for (uint32_t i = 0; i < _c.deepstackNumLevel; i++) {
            XTensor deepstackEmbed({_c.maxBatchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
            deepstackInputEmbeds[i] = deepstackEmbed;
        }
        XTensor freqsCis({_c.maxBatchedTokens, _c.ropeHeadDim}, embed.dtype, nullptr);
        std::vector<XTensor> freqsCisVec = {freqsCis};
        XTensor input({_c.maxBatchedTokens}, INT32, nullptr);
        XTensor output({_c.maxBatchedTokens, _c.hiddenSize}, embed.dtype, nullptr);
        XTensor logits({_c.defTpSize, _c.maxBatch, _c.vocabSize / _c.defTpSize}, embed.dtype,
                       nullptr);
        rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads,
                       _c.nKvHeads, _c.blockSize, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                       static_cast<int>(embed.dtype),
                       (_c.nDenseLayers < _c.nLayers)
                           ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                           : static_cast<int>(embed.dtype),
                       _c.indexTopK);
        Forward(rt, input, attnMeta, kvCache, deepstackInputEmbeds, freqsCisVec, output);
        XTensor indices;
        indices.Init({batchSize}, INT32, nullptr);
        ForwardGetLogits(rt, output, indices, logits);
    };

    auto runDecode = [&]() -> size_t {
        if (_c.maxBatch == 0) {
            return 0;
        }
        XModelAttnMeta attnMeta;
        attnMeta.version = 0;
        uint32_t batchSize = static_cast<uint32_t>(_c.maxBatch);
        uint32_t cachedLen = _c.maxSeqLen > 0 ? static_cast<uint32_t>(_c.maxSeqLen - 1) : 0;
        uint32_t maxNumBlocks = 0;
        for (uint32_t i = 0; i < batchSize; i++) {
            attnMeta.lens.push_back(1);
            attnMeta.cachedLens.push_back(cachedLen);
            uint32_t blocks = DIV_ROUND_UP(1 + cachedLen, _c.blockSize);
            std::vector<uint32_t> blockTable(blocks);
            for (uint32_t j = 0; j < blocks; j++) {
                blockTable[j] = i * blocks + j;
            }
            attnMeta.blockTables.push_back(blockTable);
            if (blocks > maxNumBlocks) {
                maxNumBlocks = blocks;
            }
        }
        auto kvCache = buildKvCache(maxNumBlocks);
        std::vector<XTensor> deepstackInputEmbeds(_c.deepstackNumLevel);
        for (uint32_t i = 0; i < _c.deepstackNumLevel; i++) {
            XTensor deepstackEmbed({_c.maxBatch, _c.hiddenSize}, embed.dtype, nullptr);
            deepstackInputEmbeds[i] = deepstackEmbed;
        }
        XTensor freqsCis({_c.maxBatch, _c.ropeHeadDim}, embed.dtype, nullptr);
        std::vector<XTensor> freqsCisVec = {freqsCis};
        XTensor input({_c.maxBatch}, INT32, nullptr);
        XTensor output({_c.maxBatch, _c.hiddenSize}, embed.dtype, nullptr);
        XTensor logits({_c.defTpSize, _c.maxBatch, _c.vocabSize / _c.defTpSize}, embed.dtype,
                       nullptr);
        rt.PrepareAttn(attnMeta, _c.maxBatchedTokens, _c.maxBatch, _c.maxSeqLen, _c.nHeads,
                       _c.nKvHeads, _c.blockSize, _c.hiddenSize, _c.nRoutedExperts, _c.defDpSize,
                       static_cast<int>(embed.dtype),
                       (_c.nDenseLayers < _c.nLayers)
                           ? static_cast<int>(moeGate[_c.nDenseLayers].dtype)
                           : static_cast<int>(embed.dtype),
                       _c.indexTopK);
        Forward(rt, input, attnMeta, kvCache, deepstackInputEmbeds, freqsCisVec, output);
        XTensor indices;
        indices.Init({batchSize}, INT32, nullptr);
        ForwardGetLogits(rt, output, indices, logits);
        return rt.maxUsedSize();
    };

    size_t prefillSize = 0;
    runPrefill();
    prefillSize = rt.maxUsedSize();
    size_t decodeSize = runDecode();
    return std::max(prefillSize, decodeSize);
}

size_t XModel::GetTensorPoolSize(int dbg)
{
    size_t size = DummyRun();

    if (_rankId == 0 && dbg) {
        XDebugStream s(_rankId, __func__);
        s << "calculated size: " << size << " bytes (" << DIV_ROUND_UP(size, 1ull << MB_BIT)
          << " MB)" << std::endl;
    }
    return DIV_ROUND_UP(size, 1ull << MB_BIT);
}
