/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_MODEL_H_
#define _XLITE_MODEL_H_

#include "runtime.h"
#include "base.h"

enum XModelRopeType {
    XMODEL_ROPE_NEOX,
    XMODEL_ROPE_GPTJ,
    XMODEL_ROPE_MAX_TYPE,
};

enum XModelScoringFuncType {
    XMODEL_SCORING_FUNC_SOFTMAX,
    XMODEL_SCORING_FUNC_SIGMOID,
    XMODEL_SCORING_FUNC_SQRTSOFTPLUS,
    XMODEL_SCORING_FUNC_MAX_TYPE,
};

struct XModelConfig {
    // global config
    uint32_t vocabSize;
    uint32_t hiddenSize;
    uint32_t nLayers = 0;

    // attention config
    enum XModelAttnType attnType = XMODEL_ATTN_MHA;
    enum XModelRopeType ropeType = XMODEL_ROPE_NEOX;
    bool addBias = false;
    bool qkNorm = false;
    bool qkNormFull = false;
    // Qwen3.5 full-attention output gate: attn *= sigmoid(gate).
    // When true, fused mhaQKV layout is [Q | K | V | Gate].
    bool attnOutputGate = false;
    uint32_t nHeads = 0;
    uint32_t nKvHeads = 1;
    uint32_t headDim;
    uint32_t ropeHeadDim;
    uint32_t nopeHeadDim;
    uint32_t vHeadDim;
    uint32_t qLoraRank;
    uint32_t kvLoraRank;
    uint32_t blockSize;  // drop after next version, use blockSizes instead
    std::vector<uint32_t> blockSizes;
    uint32_t deepstackNumLevel = 0;
    uint64_t maxBatchedTokens;
    uint64_t maxBatch;
    uint64_t maxSeqLen;
    bool quantAttnWeightTrans = false;
    bool quantAttnWeightNz = false;
    float normEps;
    float ropeTheta;
    float softmaxScale;
    std::vector<uint32_t> mropeSection;
    bool mropeInterleaved = false;
    uint32_t indexHeadDim;
    uint32_t indexNHeads;
    uint32_t indexTopK;
    float indexSoftmaxScale;
    bool indexRopeInterleaved = false;
    std::vector<bool> indexFullMask;  // per-layer indexer mask - true: full indexer, false: shared

    // linear attention config
    uint32_t linearNumKHeads = 0;
    uint32_t linearNumVHeads = 0;
    uint32_t linearKeyHeadDim = 0;
    uint32_t linearValueHeadDim = 0;
    uint32_t linearConvKernelDim = 0;
    // Hybrid (Qwen3.5): full-attention every fullAttentionInterval layers.
    uint32_t fullAttentionInterval = 4;

    // mlp
    uint32_t nDenseLayers = 0;
    uint32_t nRoutedExperts = 0;
    uint32_t nSharedExperts = 0;
    uint32_t nExpertGroups = 1;
    uint32_t nLimitedGroups = 1;
    uint32_t nActExperts = 0;
    uint32_t intermediateSize;
    uint32_t moeIntermediateSize;
    enum XModelScoringFuncType scoringFunc = XMODEL_SCORING_FUNC_SOFTMAX;
    float routeScale = 1.0f;
    bool normTopKProb;
    bool expertsWeightTrans = false;
    bool expertsWeightNZ = false;
    // For GLM4/GLM5, vllm-ascend doesn't capture the Gate layer, so its gate won't use NZ format
    bool gateCaptured = true;
    // MSD W4A8 flag
    bool quantMsdW4a8 = false;

    // parallel config
    uint32_t defTpSize;
    uint32_t defDpSize;
    uint32_t moeEpSize;
    uint32_t moeTPSize;

    bool weightNZ = false;

    // ===== DeepSeek-V4 (CxA) =====
    uint32_t oGroups = 0;
    uint32_t oLoraRank = 0;
    uint32_t windowSize = 0;
    float compressRopeTheta = 10000.0f;
    uint64_t originalSeqLen = 0;
    float ropeFactor = 1.0f;
    uint32_t betaFast = 32;
    uint32_t betaSlow = 1;
    uint32_t hcMult = 0;
    uint32_t hcSinkhornIters = 0;
    float hcEps = 1e-6f;
    float swigluLimit = 0.0f;
    uint32_t nHashLayers = 0;
    std::vector<uint32_t> compressRatios;
};

struct MoEAlltoAllMeta {
    std::vector<int64_t> sendCountsData;
    std::vector<int64_t> recvCountsData;
    std::vector<int64_t> sdisplsData;
    std::vector<int64_t> rdisplsData;
    XTensor sendCounts;
    XTensor recvCounts;
    XTensor sdispls;
    XTensor rdispls;
    uint64_t totalRecvElements = 0;
    // per-source per-expert counts for reorder, pointer to device tensor (not value copy)
    XTensor *expertsCountsAllEpDevice = nullptr;
    uint32_t nRoutedExperts = 0;
};

#define AIC_MAX_NUM 25
#define AIV_MAX_NUM 50

class XModel
{
public:
    XModel(struct XModelConfig &c, uint32_t rankId);
    void Init(void);
    ~XModel(void);
    void Forward(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                 std::vector<std::vector<XTensor>> &kvCache,
                 std::vector<XTensor> &deepstackInputEmbeds, std::vector<XTensor> &freqsCis,
                 XTensor &output);
    void ForwardGetLogits(XRuntime &rt, XTensor &input, XTensor &indices, XTensor &output);
    void ForwardAndGetLogits(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                             std::vector<std::vector<XTensor>> &kvCache,
                             std::vector<XTensor> &deepstackInputEmbeds,
                             std::vector<XTensor> &freqsCis, XTensor &indices, XTensor &output);
    void ForwardWithInputsEmbeds(XRuntime &rt, XTensor &input, XModelAttnMeta &attnMeta,
                                 std::vector<std::vector<XTensor>> &kvCache,
                                 std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                 XTensor &inputIds, XTensor &output);
    void PrepareForwardWithInputsEmbeds(XRuntime &rt, XTensor &input,
                                        XModelAttnMeta &attnMeta,
                                        std::vector<std::vector<XTensor>> &kvCache,
                                        XTensor &inputIds, XTensor &output);
    void ForwardWithInputsEmbedsPrepared(XRuntime &rt, XTensor &input,
                                         std::vector<std::vector<XTensor>> &kvCache,
                                         std::vector<XTensor> &deepstackInputEmbeds,
                                         XTensor &freqsCis, XTensor &output);
    // Ascend310P ASR decode graph segments. Attention is deliberately executed
    // by the caller between these two methods because Direct ATB operations
    // cannot be nested reliably in a Model-RI capture on CANN 9.1.
    void ForwardDecodeLayerPre310P(XRuntime &rt, uint32_t layer,
                                   std::vector<std::vector<XTensor>> &kvCache,
                                   XTensor &freqsCis, XTensor &residual,
                                   XTensor &hidden, XTensor &qkv);
    void ForwardDecodeLayerPost310P(XRuntime &rt, uint32_t layer,
                                    XTensor &residual, XTensor &hidden,
                                    XTensor &attn, XTensor &output);
    uint32_t NumLayers() const { return _c.nLayers; }
    size_t GetTensorPoolSize(int dbg);
    // whether to use communication optimization
    void ConfigRtCommOptimize(XRuntime &rt, size_t tokenNum)
    {
        rt.enableCommOptimize =
            (_c.defTpSize > 1 && (tokenNum >= rt.commOptimizeLen || rt.multiTaskParallel)) ||
            (_c.defDpSize > 1 && _c.defTpSize > 1 && _c.moeEpSize > 1 && rt.enableMoEAllToAll);
    }
    // whether to pad the token number to match the max tokens across DP ranks
    bool PadForDp(XRuntime &rt)
    {
        return (!rt.IsDummyRuntime() && rt.dpSize() > 1 && _c.nRoutedExperts > 0 &&
                !rt.enableMoEAllToAll);
    }

    // weights
    XTensor embed;
    XTensor norm;
    XTensor normBias;
    XTensor head;

    std::vector<XTensor> attnNorm;
    std::vector<XTensor> attnNormBias;
    std::vector<MatmulWeight> attnOut;
    std::vector<MatmulWeight> mhaQKV;
    std::vector<XTensor> mhaQKVBias;
    std::vector<XTensor> mhaQNorm;
    std::vector<XTensor> mhaQNormBias;
    std::vector<XTensor> mhaKNorm;
    std::vector<XTensor> mhaKNormBias;

    std::vector<MatmulWeight> mlaQKVA;
    std::vector<MatmulWeight> mlaQB;
    std::vector<XTensor> mlaQNorm;
    std::vector<XTensor> mlaQNormBias;
    std::vector<XTensor> mlaWUV;
    std::vector<XTensor> mlaWUKT;
    std::vector<XTensor> mlaKVNorm;
    std::vector<XTensor> mlaKVNormBias;

    std::vector<MatmulWeight> indexQB;
    std::vector<XTensor> indexKWeightsProj;
    std::vector<XTensor> indexKNorm;
    std::vector<XTensor> indexKNormBias;

    std::vector<MatmulWeight> linearInProjQKV;
    std::vector<MatmulWeight> linearInProjZ;
    std::vector<MatmulWeight> linearInProjB;
    std::vector<MatmulWeight> linearInProjA;
    std::vector<XTensor> linearConv1d;
    std::vector<XTensor> linearALog;
    std::vector<XTensor> linearDtBias;
    std::vector<XTensor> linearNorm;
    std::vector<MatmulWeight> linearOutProj;

    std::vector<XTensor> mlpNorm;
    std::vector<XTensor> mlpNormBias;
    std::vector<MatmulWeight> mlpUpGate;
    std::vector<MatmulWeight> mlpDown;

    std::vector<XTensor> moeGate;
    std::vector<XTensor> moeGateBias;
    std::vector<XTensor> moeTid2Eid;
    std::vector<MatmulWeight> moeSEUpGate;
    std::vector<MatmulWeight> moeSEDown;
    std::vector<XTensor> moeSEGate;
    std::vector<std::vector<XTensor>> moeREUpGate;
    std::vector<std::vector<XTensor>> moeREUpGateDeqScale;
    std::vector<std::vector<XTensor>> moeREDown;
    std::vector<std::vector<XTensor>> moeREDownDeqScale;
    std::vector<std::vector<XTensor>> moeREUpGateScaleBias;
    std::vector<std::vector<XTensor>> moeREDownScaleBias;

    // ===== DeepSeek-V4 (CxA) attention weights =====
    std::vector<XTensor> attnSink;
    std::vector<MatmulWeight> attnWqA;
    std::vector<XTensor> attnWoA;
    std::vector<XTensor> attnWoB;
    std::vector<MatmulWeight> attnWKv;

    // Compressor (per-layer; empty tensor for non-compress layers)
    std::vector<XTensor> compApe;
    std::vector<XTensor> compWKv;
    std::vector<XTensor> compWGate;
    std::vector<XTensor> compNorm;

    // Indexer (per-layer; empty tensor for non-indexer layers)
    std::vector<MatmulWeight> idxWqB;
    std::vector<XTensor> idxWeightsProj;

    // Indexer's internal Compressor (per-layer; only on compress_ratio==4 layers)
    std::vector<XTensor> idxCompApe;
    std::vector<XTensor> idxCompWKv;
    std::vector<XTensor> idxCompWGate;
    std::vector<XTensor> idxCompNorm;

    // Multi-stage Hyper-Connections (per-layer)
    std::vector<XTensor> hcAttnFn;
    std::vector<XTensor> hcFfnFn;
    std::vector<XTensor> hcAttnBase;
    std::vector<XTensor> hcFfnBase;
    std::vector<XTensor> hcAttnScale;
    std::vector<XTensor> hcFfnScale;
    // MHC head (one for the whole Transformer)
    XTensor hcHeadFn;
    XTensor hcHeadBase;
    XTensor hcHeadScale;

private:
    void ForwardParallelEmbed(XRuntime &rt, XTensor &input, XTensor &embed, XTensor &output);
    std::tuple<XTensor &, XTensor &, XTensor &> ForwardAttnMLACommonV2(
        XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache, XTensor &freqsCis,
        XTensor &hiddenState);
    void ForwardAttnIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState, XTensor &attnNormQc,
                            XTensor &indexKCache, XTensor &freqsCis);
    void ForwardAttnMLAV2(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                          XTensor &freqsCis, XTensor &hiddenState);
    void XliteOpQKNorm(XRuntime &rt, uint32_t layer, XTensor &qkv);
    void ForwardAttnMHA(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void ForwardAttnLinear(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                           XTensor &freqsCis, XTensor &hiddenState);
    std::tuple<XTensor &, XTensor &> ForwardAttnCXACommon(XRuntime &rt, uint32_t layer,
                                                          std::vector<XTensor> &kvCache,
                                                          XTensor &freqsCis, XTensor &hiddenState);
    XTensor *ForwardAttnCXAIndexer(XRuntime &rt, uint32_t layer, XTensor &hiddenState, XTensor &qr,
                                   std::vector<XTensor> &kvCache, XTensor &freqsCis);
    void ForwardAttnCXA(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                        XTensor &freqsCis, XTensor &hiddenState);
    void ForwardAttn(XRuntime &rt, uint32_t layer, std::vector<std::vector<XTensor>> &kvCache,
                     XTensor &freqsCis, XTensor &hiddenState);
    void ForwardMLP(XRuntime &rt, uint32_t layer, XTensor &hiddenState,
                    std::vector<MatmulWeight> &upGate, std::vector<MatmulWeight> &down,
                    bool withAllReduce);
    std::tuple<XTensor &, XTensor &> ForwardMoEGate(XRuntime &rt, uint32_t layer, XTensor &input);
    std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
        ForwardMoEDispatch(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing);
    void ForwardMOECombine(XRuntime &rt, XTensor &tokenSorted, XTensor &weights, XTensor &routing,
                           XTensor &unpIdx, XTensor &expertsSorted, XTensor &expertsCounts);
    MoEAlltoAllMeta MoeComputeAlltoAllVMeta(const int32_t *tokensPerEpGroupAllEpHost,
                                            uint32_t moeEpSize, uint32_t moeTpSize,
                                            uint32_t hiddenSize, uint32_t rankId,
                                            uint32_t nRoutedExperts);
    MoEAlltoAllMeta MoeComputeReverseAlltoAllVMeta(const MoEAlltoAllMeta &meta, uint32_t moeEpSize);
    std::tuple<XTensor &, XTensor &, XTensor &, XTensor &, XTensor &, MoEAlltoAllMeta>
        ForwardMoEDispatchAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                   XTensor &routing);
    void ForwardMoECombineAllToAll(XRuntime &rt, XTensor &tokenSorted, XTensor &weights,
                                   XTensor &routing, XTensor &unpIdx, XTensor &expertsSorted,
                                   XTensor &expertsCounts, const MoEAlltoAllMeta &meta);
    void ForwardMoE(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardMoEMSD(XRuntime &rt, uint32_t layer, XTensor &expertsSorted, XTensor &counts,
                       XTensor &num, uint32_t start, uint32_t end, uint32_t outDim,
                       std::vector<XTensor> &weights, std::vector<XTensor> &deqScales,
                       std::vector<XTensor> &scaleBias, XTensor &output);
    void ForwardFFN(XRuntime &rt, uint32_t layer, XTensor &hiddenState);
    void ForwardEmbedAndLayers(XRuntime &rt, XTensor &input,
                               std::vector<std::vector<XTensor>> &kvCache,
                               std::vector<XTensor> &deepstackInputEmbeds,
                               std::vector<XTensor> &freqsCis, XTensor &h);
    void ForwardLayers(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                       std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis, XTensor &h);
    void ForwardLayersNaive(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                            std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                            XTensor &output);
    void ForwardHcPre(XRuntime &rt, XTensor &input, XTensor &hcFn, XTensor &hcScale,
                      XTensor &hcBase, XTensor &output, XTensor &post, XTensor &comb);
    void ForwardHcPost(XRuntime &rt, XTensor &input, XTensor &post, XTensor &comb,
                       XTensor &residual, XTensor &output);
    void ForwardLayersMhc(XRuntime &rt, XTensor &x, std::vector<std::vector<XTensor>> &kvCache,
                          std::vector<XTensor> &freqsCis, XTensor &output);
    void ForwardLayersCommOptimize(XRuntime &rt, XTensor &x,
                                   std::vector<std::vector<XTensor>> &kvCache,
                                   std::vector<XTensor> &deepstackInputEmbeds, XTensor &freqsCis,
                                   XTensor &output);
    void CheckForwardParam(XRuntime &rt, std::vector<std::vector<XTensor>> &kvCache);
    size_t DummyRun();
    void ForwardLinear(XRuntime &rt, uint32_t layer, XTensor &x, std::vector<MatmulWeight> &weights,
                       XTensor &out, const std::vector<XTensor> &weightBias = {});

    struct XModelConfig _c;
    uint32_t _rankId;
    // Hybrid (Qwen3.5): per-layer attn type, 0=full 1=linear. Empty => non-hybrid.
    std::vector<uint32_t> _layerTypes;

    // FFN
    XTensor _gateIndices;
    XTensor _inputIds;
    std::vector<XTensor> _moeREUpGate;
    std::vector<XTensor> _moeREUpGateDeqScale;
    std::vector<XTensor> _moeREDown;
    std::vector<XTensor> _moeREDownDeqScale;
    std::vector<XTensor> _moeREUpGateScaleBias;
    std::vector<XTensor> _moeREDownScaleBias;
    bool _isSharedExpertWeightFull = false;

    // ATTN
    uint64_t _mropeMaskH;
    uint64_t _mropeMaskW;
    XTensor _sync;
    XTensor _dsaTopkIndices;
    float _dsaIndexerScale = 1.0f;
};

#endif
