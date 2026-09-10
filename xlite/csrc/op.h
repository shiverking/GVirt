/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_OP_H_
#define _XLITE_OP_H_

#include "base.h"
#include "runtime.h"
#include "ascend.h"
#include "auto_tuner.h"
#include "kernels/kernel_param.h"

#define MATMUL_M0_N0_K0_DEFAULT_VALUE ((uint64_t)(-1))
static_assert(MAX_KV_TILE_SIZE <= MAX_SOFTMAX_PINGPONG_LEN);

// Derive maxNumBlocks from blockTables: 2-D [batch, maxNumBlocks] -> shape[1];
// 1-D [batch * maxNumBlocks] -> len / batch. Avoids shape[1] UB on 1-D input.
static inline uint32_t DeriveMaxNumBlocks(const XTensor &blockTables, uint32_t batch)
{
    const auto &shape = blockTables.shape;
    if (shape.size() == 2) {
        return static_cast<uint32_t>(shape[1]);
    }
    if (shape.size() == 1) {
        if (batch == 0) {
            throw std::runtime_error(
                std::string(__func__) +
                ": batch == 0 when deriving maxNumBlocks from 1-D blockTables");
        }
        if (shape[0] % batch != 0) {
            throw std::runtime_error(std::string(__func__) + ": 1-D blockTables length (" +
                                     std::to_string(shape[0]) + ") is not divisible by batch (" +
                                     std::to_string(batch) + ")");
        }
        return static_cast<uint32_t>(shape[0] / batch);
    }
    throw std::runtime_error(std::string(__func__) + ": blockTables must be 1-D or 2-D, got rank " +
                             std::to_string(shape.size()));
}

HcclDataType XDtype2HcclDtype(enum XDtype dtype);
void XliteOpProbe310P(XRuntime &rt, XTensor &out, uint32_t value);
void XliteOpOfficialAddProbe310P(XRuntime &rt, XTensor &x, XTensor &y, XTensor &z);
void XliteOpMixedDecodeCopy310P(XRuntime &rt, void *source, void *destination,
                                XTensor &queryOffsets, uint32_t rows, bool scatter);
void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                      bool fetchOffset = false, DebugSrcLoc loc = UNKNOWN_DBG_LOC,
                      uint32_t copySize = COPY_SIZE);
void XliteOpReduceScatter(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                          bool fetchOffset = false, DebugSrcLoc loc = UNKNOWN_DBG_LOC,
                          uint32_t copySize = COPY_SIZE);
void XliteOpAllReduceSum(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                         bool fetchOffset = false, DebugSrcLoc loc = UNKNOWN_DBG_LOC,
                         uint32_t copySize = COPY_SIZE);
void XliteOpAlltoAllV(XRuntime &rt, XTensor &in, XTensor &out, XTensor &sendCounts,
                      XTensor &recvCounts, XTensor &sdispls, XTensor &rdispls, enum commType type,
                      DebugSrcLoc loc = UNKNOWN_DBG_LOC);

void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end,
                  XTensor &out);
void XliteOpRmsNorm(XRuntime &rt, XTensor &in, const XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, bool useNorm = true, const XTensor &normBias = XTensor(),
                    uint32_t cntPerToken = 1, uint32_t inStartOffset = 0,
                    uint32_t outStartOffset = 0, const XTensor &variance = XTensor());
void XliteOpLayerNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &normBias, XTensor &out,
                      float normEps, uint32_t normDim, uint32_t cntPerToken = 1,
                      uint32_t inStartOffset = 0, uint32_t outStartOffset = 0);
void XliteOpL2Norm(XRuntime &rt, XTensor &in, XTensor &out, float normEps, uint32_t normDim,
                   uint32_t cntPerToken = 1, uint32_t inStartOffset = 0,
                   uint32_t outStartOffset = 0);
void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out);
void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ = false,
                   const XTensor &bias = XTensor(), const XTensor &deqScale = XTensor(),
                   bool transpose = false, uint64_t m0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t n0 = MATMUL_M0_N0_K0_DEFAULT_VALUE,
                   uint64_t k0 = MATMUL_M0_N0_K0_DEFAULT_VALUE);

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &num = XTensor(),
                       float swigluLimit = 0.0f);
void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale);
void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out);
void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts);
void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing,
                          XTensor &weights, uint32_t start, uint32_t end, XTensor &out);
void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                        XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                        long outDim, long inDim, XTensor &output, bool weightNZ = false,
                        bool transpose = false);
void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping, uint32_t nHeads,
                      uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim, uint32_t blockSize,
                      bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW);
void XliteOpAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &qk,
                      XTensor &output, XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                      XTensor &blockTables, uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t blockSize, uint32_t batch);
void XliteOpFlashAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                           XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum, XTensor &lastMax,
                           XTensor &lastSum, XTensor &sync, XTensor &output, XTensor &queryStartLoc,
                           XTensor &lens, XTensor &cachedLens, XTensor &blockTables,
                           uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t blockSize,
                           uint32_t batch, uint32_t tileSizeOfCachedKV = MAX_KV_TILE_SIZE);
void XliteOpMLAV2(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kCache, XTensor &peCache,
                  XTensor &qk, XTensor &oAbsorb, XTensor &queryStartLoc, XTensor &lens,
                  XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                  uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, float scale,
                  uint32_t topK = 0, const XTensor &topkIndices = XTensor());
void XliteOpFlashMLAV2(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kCache,
                       XTensor &peCache, XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum,
                       XTensor &lastMax, XTensor &lastSum, XTensor &sync, XTensor &oAbsorb,
                       XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                       XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                       uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, float scale,
                       uint32_t tileSizeOfCachedKV = MAX_KV_TILE_SIZE, uint32_t topK = 0,
                       const XTensor &topkIndices = XTensor());
void XliteOpGatherSparseKVCache(XRuntime &rt, XTensor &kCache, XTensor &peCache,
                                XTensor &blockTables, XTensor &topkIndices, XTensor &queryLens,
                                XTensor &cachedLens, XTensor &kDenseCache, XTensor &peDenseCache,
                                uint32_t batch, uint32_t indexTopK, uint32_t blockSize,
                                uint32_t kvLoraRank, uint32_t ropeHeadDim, uint32_t kvHeads);
void XliteOpMLAV3(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kDenseCache,
                  XTensor &peDenseCache, XTensor &qk, XTensor &oAbsorb, XTensor &queryStartLoc,
                  XTensor &lens, XTensor &cachedLens, uint32_t nHeads, uint32_t ropeHeadDim,
                  uint32_t kvLoraRank, uint32_t batch, uint32_t indexTopK, float scale);
void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output);
void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in, XTensor &addInOut, XTensor &norm,
                          float normEps, XTensor &out, const XTensor &normBias = XTensor());
void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb);
void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb);
void XliteOpSqrtsoftplusHashTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias,
                                 XTensor &inputIds, const XTensor &tid2eid, XTensor &outWeights,
                                 XTensor &routingMap, float scale, uint32_t topK, bool hash);
void XliteOpTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outIndices,
                 XTensor &queryLens, XTensor &cachedLens, uint32_t batch, size_t k);
void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x);
void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf);
void XliteOpRopeComplex(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t outStepDim,
                        uint32_t ropeDim, uint32_t offset, uint32_t outOffset, XTensor &inputWithR,
                        XTensor &freqs, XTensor &position, XTensor &output, bool inverse = false,
                        bool outInterleaved = false);
void XliteOpRopeComplexAndCache(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim,
                                uint32_t ropeDim, uint32_t offset, uint32_t vdim,
                                XTensor &inputWithR, XTensor &freqs, XTensor &position,
                                uint32_t blockSize, XTensor &vCache, XTensor &slotMapping,
                                bool outInterleaved = false);
void XliteOpMlaPrepare(XRuntime &rt, XTensor &attnQkvc, const XTensor &qNorm,
                       const XTensor &qNormBias, XTensor &attnNormQc, const XTensor &kvNorm,
                       const XTensor &kvNormBias, const XTensor &freqs, const XTensor &position,
                       uint32_t qLoraRank, uint32_t kvLoraRank, uint32_t ropeHeadDim,
                       uint32_t blockSize, XTensor &kCache, XTensor &peCache,
                       const XTensor &slotMapping, float normEps,
                       const XTensor &attnNormKvc = XTensor());
void XliteOpQkRmsNorm(XRuntime &rt, XTensor &in, const XTensor &qNorm, const XTensor &qNormBias,
                      const XTensor &kNorm, const XTensor &kNormBias, XTensor &out, float normEps,
                      uint32_t qNormDim, uint32_t qCntPerToken, uint32_t kNormDim,
                      uint32_t kCntPerToken, uint32_t kStartOffset, bool useNorm,
                      const XTensor &qVariance = XTensor(), const XTensor &kVariance = XTensor());
void XliteOpIndexerPrepare(XRuntime &rt, XTensor &kw, const XTensor &kNorm,
                           const XTensor &kNormBias, const XTensor &freqs, const XTensor &position,
                           uint32_t indexHeadDim, uint32_t indexNHeads, uint32_t ropeHeadDim,
                           uint32_t blockSize, XTensor &indexKCache, const XTensor &slotMapping,
                           float normEps, const XTensor &q = XTensor(), float scale = 1.0f,
                           uint32_t topK = 2048, bool isLong = false, uint32_t tpSize = 1);

void XliteOpQuant(XRuntime &rt, XTensor &x, XTensor &scale_reciprocal, XTensor &offset,
                  XTensor &out);
void XliteOpQuantDyn(XRuntime &rt, XTensor &x, XTensor &scale, XTensor &out,
                     const XTensor &num = XTensor());
void XliteOpDeQuant(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &scale = XTensor(),
                    const XTensor &num = XTensor());
void XliteOpMSDMergeDequant(XRuntime &rt, XTensor &yMerged, XTensor &scaleBiasPtrs, XTensor &counts,
                            uint32_t start, uint32_t end, XTensor &perTokenScale, XTensor &out);
void XliteOpMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out,
                          const XTensor &quantBias = XTensor(),
                          const XTensor &weightScale = XTensor(), bool weightNZ = false,
                          bool transpose = false, const XTensor &outScale = XTensor(),
                          const XTensor &num = XTensor());
void XliteOpGroupMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                               XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                               long outDim, long inDim, XTensor &output, XTensor &outScale,
                               XTensor &num, bool weightNZ = false, bool transpose = false);
void XliteOpConcat(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out);
void XliteOpConcatCol(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out);
void XliteOpSplitCol(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs);
void XliteOpRepeatInterleave(XRuntime &rt, XTensor &in, XTensor &out, uint32_t numTokens,
                             uint32_t nKHeads, uint32_t nVHeads, uint32_t headBytes);
void XliteOpSplit(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs,
                  const std::vector<size_t> &sizes, uint32_t numPackets);
void XliteOpSigmoidGateMul(XRuntime &rt, XTensor &attn, XTensor &gate, XTensor &out);
void XliteOpIndexerScores(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight,
                          XTensor &scores, XTensor &queryStartLoc, XTensor &lens,
                          XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads,
                          uint32_t headDim, uint32_t blockSize, uint32_t batch);
void XliteOpIndexerTopK(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight, XTensor &scores,
                        XTensor &lastTopk, XTensor &indices, XTensor &topkIndices,
                        XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                        XTensor &blockTables, XTensor &sync, uint32_t nHeads, uint32_t headDim,
                        uint32_t blockSize, uint32_t batch, uint32_t topK);
void XliteOpMuls(XRuntime &rt, XTensor &input, float scale, XTensor &output,
                 uint32_t calcOffset = 0, uint32_t calcNum = UINT32_MAX);
void XliteOpExpertsCountsSum(XRuntime &rt, XTensor &expertsCountsInput, XTensor &tokensPerEpgroup,
                             XTensor &expertsCountsOutput, uint32_t nRoutedExperts);
void XliteOpReorderMoE(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &counts,
                       uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward);
void XliteOpTranspose_1_2(XRuntime &rt, XTensor &input, XTensor &output);
void XliteOpConv1dAndSiLU(XRuntime &rt, XTensor &state, XTensor &input, XTensor &weight,
                          XTensor &output, bool updateState = true,
                          XTensor *queryStartLoc = nullptr, XTensor *queryLens = nullptr);
// Token-row-parallel conv1d+SiLU for uniform prefill (P3-A v2). input/output are
// [T,C] token-major; parallelizes over contiguous token-row segments with a
// sliding window (chunked strided DMA) and a vgather-based weight tap
// transpose. When updateState is true the conv state update runs as a separate
// kernel launch right after the conv kernel (the conv cores read state GM for
// the window context, so an in-kernel update would race them; the launch
// boundary is the inter-core barrier). Constraints: kernelDim in {1,2,4},
// seqLen >= kernelDim, channels % 1024 == 0.
void XliteOpConv1dAndSiLUToken(XRuntime &rt, XTensor &state, XTensor &input, XTensor &weight,
                               XTensor &output, uint32_t seqLen, bool updateState = true);
void XliteOpBetaDecay(XRuntime &rt, XTensor &b, XTensor &a, XTensor &A_log, XTensor &dt_bias,
                      XTensor &beta, XTensor &g, uint32_t bsz, uint32_t seqlen,
                      uint32_t num_v_heads);
// Recurrent gated delta rule. q/k: [T, H*kDim], v/out: [T, H*vDim],
// beta/g: [T, H], state: [B, H, kDim, vDim] (updated in-place).
// g is log-space (from BetaDecay); kernel applies exp(g). Q/K should already be L2-normalized.
// queryStartLoc/queryLens: optional packed mixed-length layout ([B] int32).
void XliteOpRecurrentGatedDeltaRule(XRuntime &rt, XTensor &query, XTensor &key, XTensor &value,
                                    XTensor &beta, XTensor &g, XTensor &state, XTensor &out,
                                    uint32_t batch, uint32_t seqlen, uint32_t numHeads,
                                    uint32_t kDim, uint32_t vDim, XTensor *queryStartLoc = nullptr,
                                    XTensor *queryLens = nullptr);
void XliteOpEinsumMhtHdtMhd(XRuntime &rt, XTensor &mht, XTensor &hdt, XTensor &mhd, uint32_t m,
                            uint32_t h, uint32_t t, uint32_t d, bool weightNZ, int T = -1,
                            int D = -1);
void XliteOpEinsumMhtHtdMhd(XRuntime &rt, XTensor &mht, XTensor &htd, XTensor &mhd, uint32_t m,
                            uint32_t h, uint32_t t, uint32_t d, bool weightNZ, int T = -1,
                            int D = -1);
void XliteOpUnpackActivation(XRuntime &rt, XTensor &input, XTensor &output);
void XliteOpHcAct(XRuntime &rt, XTensor &mixes, const XTensor &hcScale, const XTensor &hcBase,
                  XTensor &post, XTensor &comb, uint32_t hcMult, float eps, uint32_t sinkhornIters,
                  bool headOnly, XTensor &xResid, XTensor &output);
void XliteOpHcPost(XRuntime &rt, XTensor &x, XTensor &post, XTensor &comb, XTensor &residual,
                   XTensor &y, uint32_t m, uint32_t hcMult, uint32_t hidden);
#endif
