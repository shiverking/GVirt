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
#include "ring_sync.h"
#include "mla_aic_helper.h"

template <typename Dtype>
class FLASHMLAV2
{
public:
    __aicore__ inline FLASHMLAV2()
    {
    }

    __aicore__ inline void Init(GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kCache, GM_ADDR peCache,
                                GM_ADDR topkIndices, GM_ADDR qk, GM_ADDR sv, GM_ADDR max,
                                GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync,
                                GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens,
                                GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads,
                                uint32_t ropeHeadDim, uint32_t kvLoraRank, uint32_t blockSize,
                                uint32_t batch, uint32_t maxNumBlocks, float scale,
                                uint32_t tileSizeOfCachedKV, uint32_t topK)
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
        this->scale = scale;
        this->topK = (topkIndices == nullptr) ? 0 : topK;
        this->tileSizeOfCachedKV = tileSizeOfCachedKV;
        this->qkStride = tileSizeOfCachedKV;
        ringSync.Init(sync);

        this->qk[0].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride);
        this->qk[1].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride +
                                    block_num * XLITE_MAX_M0 * qkStride);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * kvLoraRank);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * kvLoraRank +
                                    block_num * XLITE_MAX_M0 * kvLoraRank);
        this->max[0].SetGlobalBuffer(((__gm__ float *)max) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0);
        this->max[1].SetGlobalBuffer(((__gm__ float *)max) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0 +
                                     block_num * XLITE_MAX_M0 * 2);
        this->sum[0].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0);
        this->sum[1].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * XLITE_MAX_M0 * 2 +
                                     get_subblockid() * XLITE_MAX_M0 +
                                     block_num * XLITE_MAX_M0 * 2);
        this->lastMax.SetGlobalBuffer((__gm__ float *)lastMax);
        this->lastSum.SetGlobalBuffer((__gm__ float *)lastSum);

        svk0 = aicHelper.Init(nHeads, ropeHeadDim, kvLoraRank, blockSize, qkStride, false);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t softmaxConfig = 1 | (mode << 4) | (flagIdx << 8);
        flagIdx = 1;
        uint64_t updateConfig = 1 | (mode << 4) | (flagIdx << 8);

        int lastBatchIdx, lastQueryTaskOffset, lastQueryTaskLen, last, lastKvOffset, lastKvLen;
        __gm__ uint32_t *lastBlockTable;

        int queryTileSize = XLITE_MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = XLITE_MAX_M0;
        }
        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV);
            int taskNum = queryNum * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int queryIdx = idx / kvNum;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * tileSizeOfCachedKV;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = tileSizeOfCachedKV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                // do queryIdx & (0，nHeads)'s QK
                uint32_t mhOffset = queryTaskOffset * nHeads;
                uint32_t absorbOffset = mhOffset * kvLoraRank;
                uint32_t qrOffset = mhOffset * ropeHeadDim;

                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), "
                           "kv [%u - %u)} use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           nHeads, kvOffset, kvOffset + kvLen, curr);
                aicHelper.RunAicQK(qAbsorb[absorbOffset], qr[qrOffset], kCache, peCache,
                                   queryTaskLen, blockTable, kvOffset, kvLen, qk[curr]);
#if !defined(XLITE_DEVICE_310P)
                ffts_cross_core_sync(PIPE_FIX, softmaxConfig);
#endif

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(2);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), "
                               "kv [%u - %u)} use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, nHeads, kvOffset,
                               kvOffset + kvLen, last);
                    aicHelper.RunAicSV(qk[last], kCache, lastQueryTaskLen, lastBlockTable,
                                       lastKvOffset, lastKvLen, sv[last]);
#if !defined(XLITE_DEVICE_310P)
                    ffts_cross_core_sync(PIPE_FIX, updateConfig);
#endif
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastBlockTable = blockTable;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                last = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(2);
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), "
                       "kv [%u - %u)} use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, nHeads, lastKvOffset,
                       lastKvOffset + lastKvLen, last);
            aicHelper.RunAicSV(qk[last], kCache, lastQueryTaskLen, lastBlockTable, lastKvOffset,
                               lastKvLen, sv[last]);
#if !defined(XLITE_DEVICE_310P)
            ffts_cross_core_sync(PIPE_FIX, updateConfig);
#endif
        }
    }

    __aicore__ inline void RunAiv()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t flagIdx = 2;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        int dbgBlockIdx = block_idx;

        int lastBatchIdx, lastQueryTaskLen, last, lastKvOffset, lastKvLen, lastQueryTaskOffset,
            lastWorkStart, lastWorkCurCore, lastActualCalcSoftmaxLen;
        int lastIsLastKvTile;
        uint32_t lastOutOffset;

        int queryTileSize = XLITE_MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = XLITE_MAX_M0;
        }
        int needDoUpdate = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int resetPrevCore = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV);
            int taskNum = queryNum * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int queryIdx = idx / kvNum;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                int kvOffset = kvIdx * tileSizeOfCachedKV;
                if (calcLen <= kvOffset) {
                    continue;
                }
                if (totalIdx % block_num != block_idx) {
                    totalIdx++;
                    continue;
                }
                totalIdx++;

                int kvLen = tileSizeOfCachedKV;
                if (kvOffset + kvLen > calcLen) {
                    kvLen = calcLen - kvOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                uint32_t outOffset = queryTaskOffset * nHeads;

                int isLastKvTile = (kvOffset + kvLen == calcLen) ? 1 : 0;

                int nWork = queryTaskLen * nHeads;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = get_subblockid() * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t qkOffset = nWorkStart * qkStride;
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + 1;
                int actualCalcSoftmaxLen = calcSoftmaxLen - kvOffset;
                if (actualCalcSoftmaxLen > kvLen) {
                    actualCalcSoftmaxLen = kvLen;
                }
                // Since RunAicSVAbsorb reads (queryTokens, nHeads, 4 * svk0) from QK
                // each time, softmax must be padded to 4 * svk0 to prevent residual values in
                // QK from being included in SV computation along the cached token dimension and
                // degrading accuracy.
                uint32_t outN = ROUND_UP(kvLen, 4 * svk0);
                if (outN > tileSizeOfCachedKV) {
                    outN = tileSizeOfCachedKV;
                }

                // wait aic qk done
                wait_flag_dev(0);

                // do softmax
                dbg_printf(
                    "block%d subblock%u: {batch %d, query [%u - %u) "
                    "query x head group [%u - "
                    "%u) "
                    "kv [%u - %u)} calcSoftmaxLen %u, off %u, stride %u, outN %u, use %d temp buf: "
                    "SOFTMAX\n",
                    dbgBlockIdx, get_subblockid(), batchIdx, queryTaskOffset,
                    queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore, kvOffset,
                    kvOffset + kvLen, actualCalcSoftmaxLen, nWorkStart, nHeads, outN, curr);
                RunAivSoftmaxPingPong(
                    (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(), nWorkCurCore, qkStride,
                    actualCalcSoftmaxLen, outN, true, nWorkStart, nHeads,
                    (__gm__ float *)max[curr][nWorkStart].GetPhyAddr(),
                    (__gm__ float *)sum[curr][nWorkStart].GetPhyAddr(), true, scale, kvOffset,
                    calcLen > topK ? topK : 0,
                    (calcLen > topK && topK > 0) ? topkIndices + topK * queryTaskOffset : nullptr);
                ffts_cross_core_sync(PIPE_MTE3, config);

                if (needDoUpdate != 0) {
                    // wait aic sv done
                    wait_flag_dev(1);
                    if (lastKvOffset != 0) {
                        ringSync.WaitPrevCore();
                        resetPrevCore = 1;
                    }
                    dbg_printf("block%d subblock%u: {batch %d, query [%u - %u)"
                               "query x head group [%u "
                               "- %u) "
                               "kv [%u - %u)} use %d temp buf: UPDATE\n",
                               dbgBlockIdx, get_subblockid(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                               lastWorkStart + lastWorkCurCore, lastKvOffset,
                               lastKvOffset + lastKvLen, last);
                    // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] &
                    // max[last]
                    RunAivSoftmaxUpdate(
                        (__gm__ Dtype *)sv[last][lastWorkStart * kvLoraRank].GetPhyAddr(),
                        (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ Dtype *)oAbsorb[lastOutOffset * kvLoraRank].GetPhyAddr(),
                        (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                        (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore,
                        nHeads, kvLoraRank, lastKvOffset == 0, lastActualCalcSoftmaxLen, true,
                        lastWorkStart, nHeads);
                    if (!lastIsLastKvTile) {
                        set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                        ringSync.SetNextCore();
                    }
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastWorkStart = nWorkStart;
                lastWorkCurCore = nWorkCurCore;
                lastOutOffset = outOffset;
                lastKvOffset = kvOffset;
                lastKvLen = kvLen;
                lastIsLastKvTile = isLastKvTile;
                lastActualCalcSoftmaxLen = actualCalcSoftmaxLen;
                last = curr;
                needDoUpdate = 1;
                curr = 1 - curr;
            }
            queryStart = -1;
        }

        // do last update
        if (needDoUpdate != 0) {
            // wait aic sv done
            wait_flag_dev(1);
            if (lastKvOffset != 0) {
                ringSync.WaitPrevCore();
                resetPrevCore = 1;
            }
            dbg_printf("block%d subblock%u: {batch %d, query [%u - %u)"
                       "query x head group [%u "
                       "- %u) "
                       "kv [%u - %u)} use %d temp buf: UPDATE\n",
                       dbgBlockIdx, get_subblockid(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                       lastWorkStart + lastWorkCurCore, lastKvOffset, lastKvOffset + lastKvLen,
                       last);
            // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] &
            // max[last]
            RunAivSoftmaxUpdate((__gm__ Dtype *)sv[last][lastWorkStart * kvLoraRank].GetPhyAddr(),
                                (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                                (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                                (__gm__ Dtype *)oAbsorb[lastOutOffset * kvLoraRank].GetPhyAddr(),
                                (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                                (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(),
                                lastWorkCurCore, nHeads, kvLoraRank, lastKvOffset == 0,
                                lastActualCalcSoftmaxLen, true, lastWorkStart, nHeads);
            if (!lastIsLastKvTile) {
                set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                ringSync.SetNextCore();
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ringSync.ResetPrevCore();
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
    RingSync<Dtype> ringSync;
    GlobalTensor<Dtype> qAbsorb;
    GlobalTensor<Dtype> qr;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> peCache;
    __gm__ int32_t *topkIndices;
    GlobalTensor<Dtype> oAbsorb;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<float> lastMax;
    GlobalTensor<float> lastSum;

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
    float scale;
    uint32_t topK;
    uint32_t qkStride;
    uint32_t tileSizeOfCachedKV;
    int svk0;
};

#if defined(XLITE_DEVICE_310P)
#define FLASH_MLA_V2_FUNC_DEFINE(dtype)                                                       \
    extern "C" __global__ __aicore__ void flash_mla_v2_##dtype(                              \
        GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kCache, GM_ADDR peCache, GM_ADDR topkIndices,    \
        GM_ADDR qk, GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum,   \
        GM_ADDR sync, GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens,              \
        GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim,       \
        uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks,       \
        float scale, uint32_t tileSizeOfCachedKV, uint32_t topK)                              \
    {                                                                                         \
    }
#else
#define FLASH_MLA_V2_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void flash_mla_v2_##dtype(                                \
        GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kCache, GM_ADDR peCache, GM_ADDR topkIndices,     \
        GM_ADDR qk, GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum,    \
        GM_ADDR sync, GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens,               \
        GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads, uint32_t ropeHeadDim,        \
        uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks,        \
        float scale, uint32_t tileSizeOfCachedKV, uint32_t topK)                               \
    {                                                                                          \
        FLASHMLAV2<dtype> op;                                                                  \
        op.Init(qAbsorb, qr, kCache, peCache, topkIndices, qk, sv, max, sum, lastMax, lastSum, \
                sync, oAbsorb, queryStartLoc, queryLens, cachedLens, blockTables, nHeads,      \
                ropeHeadDim, kvLoraRank, blockSize, batch, maxNumBlocks, scale,                \
                tileSizeOfCachedKV, topK);                                                     \
        op.Run();                                                                              \
    }
#endif
