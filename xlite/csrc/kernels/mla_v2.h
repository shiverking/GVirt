/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "kernel_operator.h"
// #define XLITE_KERNEL_DEBUG
#include "debug.h"
#include "softmax_attn_aiv.h"
#include "mla_aic_helper.h"

template <typename Dtype>
class MLAV2
{
public:
    __aicore__ inline MLAV2()
    {
    }

    __aicore__ inline void Init(GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kCache, GM_ADDR peCache,
                                GM_ADDR topkIndices, GM_ADDR qk, GM_ADDR oAbsorb,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                                uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch,
                                uint32_t maxNumBlocks, float scale, uint32_t topK)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->qAbsorb.SetGlobalBuffer((__gm__ Dtype *)qAbsorb);
        this->qr.SetGlobalBuffer((__gm__ Dtype *)qr);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->peCache.SetGlobalBuffer((__gm__ Dtype *)peCache);
        this->topkIndices = (__gm__ int32_t *)topkIndices;
        this->oAbsorb.SetGlobalBuffer((__gm__ Dtype *)oAbsorb);

        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;

        this->nHeads = nHeads;
        this->ropeHeadDim = ropeHeadDim;
        this->kvLoraRank = kvLoraRank;
        this->batch = batch;
        this->maxNumBlocks = maxNumBlocks;
        this->maxSeqLen = maxNumBlocks * blockSize;
        this->scale = scale;
        this->topK = (topkIndices == nullptr) ? 0 : topK;
        this->qkStride = this->maxSeqLen;

        this->qk[0].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride);
        this->qk[1].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride +
                                    block_num * XLITE_MAX_M0 * qkStride);

        svk0 = aicHelper.Init(nHeads, ropeHeadDim, kvLoraRank, blockSize, qkStride, false);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskOffset, lastQueryTaskLen, last, lastAbsorbOffset,
            lastCalcLen;
        __gm__ uint32_t *lastBlockTable;

        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int coreOffset = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryTileSize = m0 / nHeads;
            if (queryTileSize == 0) {
                queryTileSize = m0;
            }
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum;
            int firstCore = (GetBlockIdx() + GetBlockNum() - coreOffset) % GetBlockNum();
            for (int idx = firstCore; idx < taskNum; idx += GetBlockNum()) {
                int queryIdx = idx;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                // do queryIdx & (0，nHeads)'s QK
                uint32_t mhOffset = queryTaskOffset * nHeads;
                uint32_t absorbOffset = mhOffset * kvLoraRank;
                uint32_t qrOffset = mhOffset * ropeHeadDim;

                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u)}"
                           " use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           nHeads, curr);
                aicHelper.RunAicQK(qAbsorb[absorbOffset], qr[qrOffset], kCache, peCache,
                                   queryTaskLen, blockTable, 0, calcLen, qk[curr]);
#if !defined(XLITE_DEVICE_310P)
                ffts_cross_core_sync(PIPE_FIX, config);
#endif

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(1);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u)}"
                               " use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, nHeads, last);
                    aicHelper.RunAicSV(qk[last], kCache, lastQueryTaskLen, lastBlockTable, 0,
                                       lastCalcLen, oAbsorb[lastAbsorbOffset]);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastAbsorbOffset = absorbOffset;
                lastQueryTaskLen = queryTaskLen;
                lastBlockTable = blockTable;
                lastCalcLen = calcLen;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            coreOffset = (coreOffset + taskNum) % GetBlockNum();
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(1);
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u)}"
                       " use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, nHeads, last);
            aicHelper.RunAicSV(qk[last], kCache, lastQueryTaskLen, lastBlockTable, 0, lastCalcLen,
                               oAbsorb[lastAbsorbOffset]);
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t flagIdx = 1;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int coreOffset = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryTileSize = m0 / nHeads;
            if (queryTileSize == 0) {
                queryTileSize = m0;
            }
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum;
            int firstCore = (block_idx + block_num - coreOffset) % block_num;
            for (int idx = firstCore; idx < taskNum; idx += block_num) {
                int queryIdx = idx;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                int nWork = queryTaskLen * nHeads;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                uint32_t subIdx = get_subblockid();
                int nWorkStart = subIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t qkOffset = nWorkStart * qkStride;
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + 1;
                // Since RunAicSVAbsorb reads (queryTokens, nHeads, 4 * svk0) from QK
                // each time, softmax must be padded to 4 * svk0 to prevent residual values in
                // QK from being included in SV computation along the cached token dimension and
                // degrading accuracy.
                uint32_t outN = ROUND_UP(calcLen, 4 * svk0);
                if (outN > maxSeqLen) {
                    outN = maxSeqLen;
                }

                // wait aic qk done
                wait_flag_dev(0);

                // do softmax
                int dbgBlockIdx = block_idx;
                dbg_printf("block%d subblock%u: {batch %d, query [%u - %u) "
                           "query x head group [%u - "
                           "%u)} calcSoftmaxLen %u, off %u, stride %u, outN %u, use %d temp buf: "
                           "SOFTMAX\n",
                           dbgBlockIdx, subIdx, batchIdx, queryTaskOffset,
                           queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore,
                           calcSoftmaxLen, nWorkStart, nHeads, outN, curr);
                if (topK == 0) {
                    RunAivSoftmax(
                        (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(),
                        m0 > (XLITE_MAX_M0 - 4)
                            ? 0
                            : (__gm__ float *)qk[curr][(m0 + subIdx * 2) * qkStride].GetPhyAddr(),
                        nWorkCurCore, qkStride, calcSoftmaxLen, outN, true, nWorkStart, nHeads,
                        true, scale);
                } else {
                    RunAivSoftmaxPingPong(
                        (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(), nWorkCurCore, qkStride,
                        calcSoftmaxLen, outN, true, nWorkStart, nHeads, nullptr, nullptr, true,
                        scale, 0, calcLen > topK ? topK : 0,
                        (calcLen > topK && topK > 0) ? topkIndices + topK * queryTaskOffset
                                                     : nullptr);
                }

                ffts_cross_core_sync(PIPE_MTE3, config);
                curr = 1 - curr;
            }
            coreOffset = (coreOffset + taskNum) % block_num;
            queryStart = -1;
            cachedLen = -1;
        }
    }

    __aicore__ inline void Run()
    {
#ifdef __DAV_C220_CUBE__
        RunAic();
#elif __DAV_C220_VEC__
        RunAiv();
#endif
    }

private:
    GlobalTensor<Dtype> qAbsorb;
    GlobalTensor<Dtype> qr;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> peCache;
    __gm__ int32_t *topkIndices;
    GlobalTensor<Dtype> oAbsorb;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];

    MlaAicHelper<Dtype> aicHelper;

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;

    uint32_t nHeads;
    uint32_t ropeHeadDim;
    uint32_t kvLoraRank;
    uint32_t batch;
    uint32_t maxNumBlocks;
    uint32_t maxSeqLen;
    float scale;
    uint32_t topK;
    uint32_t qkStride;
    int svk0;
};

#if defined(XLITE_DEVICE_310P)
#define MLA_V2_FUNC_DEFINE(dtype)                                                                 \
    extern "C" __global__ __aicore__ void mla_v2_##dtype(                                        \
        GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kCache, GM_ADDR peCache, GM_ADDR topkIndices,        \
        GM_ADDR qk, GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, \
        GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim, uint32_t kvLoraRank,          \
        uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks, float scale, uint32_t topK)    \
    {                                                                                             \
    }
#else
#define MLA_V2_FUNC_DEFINE(dtype)                                                                  \
    extern "C" __global__ __aicore__ void mla_v2_##dtype(                                          \
        GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kCache, GM_ADDR peCache, GM_ADDR topkIndices,         \
        GM_ADDR qk, GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, \
        GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim, uint32_t kvLoraRank,           \
        uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks, float scale, uint32_t topK)     \
    {                                                                                              \
        MLAV2<dtype> op;                                                                           \
        op.Init(qAbsorb, qr, kCache, peCache, topkIndices, qk, oAbsorb, queryStartLoc, queryLens,  \
                cachedLens, blockTables, nHeads, ropeHeadDim, kvLoraRank, blockSize, batch,        \
                maxNumBlocks, scale, topK);                                                        \
        op.Run();                                                                                  \
    }
#endif
