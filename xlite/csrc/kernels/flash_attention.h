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
#include "attention_aic_helper.h"

template <typename Dtype>
class FlashAttention
{
public:
    __aicore__ inline FlashAttention()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk,
                                GM_ADDR sv, GM_ADDR max, GM_ADDR sum, GM_ADDR lastMax,
                                GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                GM_ADDR blockTables, uint32_t nHeads, uint32_t nKVHeads,
                                uint32_t headSize, uint32_t blockSize, uint32_t batch,
                                uint32_t maxNumBlocks, uint32_t tileSizeOfCachedKV)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->input.SetGlobalBuffer((__gm__ Dtype *)input);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->vCache.SetGlobalBuffer((__gm__ Dtype *)vCache);
        this->output.SetGlobalBuffer((__gm__ Dtype *)output);

        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;

        this->nHeads = nHeads;
        this->nKVHeads = nKVHeads;
        this->nQKVHeads = nHeads + 2 * nKVHeads;
        this->headNumInGroup = nHeads / nKVHeads;
        this->headSize = headSize;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlocks = maxNumBlocks;
        this->tileSizeOfCachedKV = tileSizeOfCachedKV;
        this->groupMemSize = headNumInGroup * headSize;
        ringSync.Init(sync);

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) +
                                    block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) +
                                    block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV +
                                    block_num * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * headSize);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * headSize +
                                    block_num * XLITE_MAX_M0 * headSize);
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

        aicHelper.Init(nHeads, nKVHeads, headNumInGroup, headSize, blockSize, tileSizeOfCachedKV);
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

        int lastBatchIdx, lastQueryTaskLen, lastkvHeadIdx, last, lastKvOffset, lastKvLen,
            lastQueryTaskOffset;
        __gm__ uint32_t *lastBlockTable;

        int queryTileSize = XLITE_MAX_M0 / headNumInGroup;
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
            int taskNum = queryNum * nKVHeads * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int kvHeadIdx = (idx / kvNum) % nKVHeads;
                int queryIdx = idx / (kvNum * nKVHeads);
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
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                // do queryIdx & kvHeadIdx & kvIdx's QK
                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;

                dbg_printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                           "kv [%u - %u)} use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           kvHeadIdx, kvOffset, kvOffset + kvLen, curr);
                aicHelper.RunAicQK(input[qOffset], this->kCache, queryTaskLen, kvHeadIdx,
                                   blockTable, kvOffset, kvLen, qk[curr]);
#if !defined(XLITE_DEVICE_310P)
                ffts_cross_core_sync(PIPE_FIX, softmaxConfig);
#endif

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(2);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                               "kv [%u - %u)} use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, lastkvHeadIdx, lastKvOffset,
                               lastKvOffset + lastKvLen, last);
                    aicHelper.RunAicSV(qk[last], this->vCache, lastQueryTaskLen, lastkvHeadIdx,
                                       lastBlockTable, lastKvOffset, lastKvLen, sv[last], false);
#if !defined(XLITE_DEVICE_310P)
                    ffts_cross_core_sync(PIPE_FIX, updateConfig);
#endif
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastQueryTaskLen = queryTaskLen;
                lastkvHeadIdx = kvHeadIdx;
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
            dbg_printf("block%d: {batch %d, query [%u - %u), kvHeadIdx %u, "
                       "kv [%u - %u)} use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, lastkvHeadIdx, lastKvOffset,
                       lastKvOffset + lastKvLen, last);
            aicHelper.RunAicSV(qk[last], this->vCache, lastQueryTaskLen, lastkvHeadIdx,
                               lastBlockTable, lastKvOffset, lastKvLen, sv[last], false);
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

        int lastBatchIdx, lastQueryTaskLen, lastkvHeadIdx, last, lastKvOffset, lastKvLen,
            lastQueryTaskOffset, lastWorkStart, lastWorkCurCore, lastActualCalcSoftmaxLen;
        int lastIsLastKvTile;
        uint32_t lastOutOffset;

        int queryTileSize = XLITE_MAX_M0 / headNumInGroup;
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
            int taskNum = queryNum * nKVHeads * kvNum;
            for (int idx = 0; idx < taskNum; idx++) {
                int kvIdx = idx % kvNum;
                int kvHeadIdx = (idx / kvNum) % nKVHeads;
                int queryIdx = idx / (kvNum * nKVHeads);
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
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                uint32_t outOffset = queryTaskOffset * nHeads + kvHeadIdx * headNumInGroup;
                uint32_t qOffset =
                    (queryTaskOffset * nQKVHeads + kvHeadIdx * headNumInGroup) * headSize;

                int isLastKvTile = (kvOffset + kvLen == calcLen) ? 1 : 0;

                int nWork = queryTaskLen * headNumInGroup;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = get_subblockid() * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                int subQOffset = nWorkStart / headNumInGroup;
                int subHeadOffset = nWorkStart % headNumInGroup;
                uint32_t calcSoftmaxLen = cachedLen + queryTaskStart + 1;
                int actualCalcSoftmaxLen = calcSoftmaxLen - kvOffset;
                if (actualCalcSoftmaxLen > kvLen) {
                    actualCalcSoftmaxLen = kvLen;
                }
                uint32_t outN = ROUND_UP(kvLen, blockSize);
                // wait aic qk done
                wait_flag_dev(0);

                dbg_printf(
                    "block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - "
                    "%u), kvHeadIdx %u "
                    "kv [%u - %u)} use %d temp buf: SOFTMAX\n",
                    dbgBlockIdx, get_subblockid(), batchIdx, queryTaskOffset,
                    queryTaskOffset + queryTaskLen, nWorkStart, nWorkStart + nWorkCurCore,
                    kvHeadIdx, kvOffset, kvOffset + kvLen, curr);
                RunAivSoftmaxPingPong(
                    (__gm__ Dtype *)qk[curr][nWorkStart * tileSizeOfCachedKV].GetPhyAddr(),
                    nWorkCurCore, tileSizeOfCachedKV, actualCalcSoftmaxLen, outN, true, nWorkStart,
                    headNumInGroup, (__gm__ float *)max[curr][nWorkStart].GetPhyAddr(),
                    (__gm__ float *)sum[curr][nWorkStart].GetPhyAddr());
                ffts_cross_core_sync(PIPE_MTE3, config);

                if (needDoUpdate != 0) {
                    // wait aic sv done
                    wait_flag_dev(1);
                    if (lastKvOffset != 0) {
                        ringSync.WaitPrevCore();
                        resetPrevCore = 1;
                    }
                    dbg_printf(
                        "block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u "
                        "- %u), kvHeadIdx %u "
                        "kv [%u - %u)} use %d temp buf: UPDATE\n",
                        dbgBlockIdx, get_subblockid(), lastBatchIdx, lastQueryTaskOffset,
                        lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                        lastWorkStart + lastWorkCurCore, lastkvHeadIdx, lastKvOffset,
                        lastKvOffset + lastKvLen, last);
                    // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] &
                    // max[last]
                    RunAivSoftmaxUpdate(
                        (__gm__ Dtype *)sv[last][lastWorkStart * headSize].GetPhyAddr(),
                        (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                        (__gm__ Dtype *)output[lastOutOffset * headSize].GetPhyAddr(),
                        (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                        (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(), lastWorkCurCore,
                        nHeads, headSize, lastKvOffset == 0, lastActualCalcSoftmaxLen, true,
                        lastWorkStart, headNumInGroup);
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
                lastkvHeadIdx = kvHeadIdx;
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
            wait_flag_dev(1);
            if (lastKvOffset != 0) {
                ringSync.WaitPrevCore();
                resetPrevCore = 1;
            }
            dbg_printf(
                "block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - %u), "
                "kvHeadIdx %u "
                "kv [%u - %u)} use %d temp buf: UPDATE\n",
                dbgBlockIdx, get_subblockid(), lastBatchIdx, lastQueryTaskOffset,
                lastQueryTaskOffset + lastQueryTaskLen, lastWorkStart,
                lastWorkStart + lastWorkCurCore, lastkvHeadIdx, lastKvOffset,
                lastKvOffset + lastKvLen, last);
            // do update with sv[last] & sum[last] & max[last] & prevcore's sum[last] & max[last]
            RunAivSoftmaxUpdate((__gm__ Dtype *)sv[last][lastWorkStart * headSize].GetPhyAddr(),
                                (__gm__ float *)max[last][lastWorkStart].GetPhyAddr(),
                                (__gm__ float *)sum[last][lastWorkStart].GetPhyAddr(),
                                (__gm__ Dtype *)output[lastOutOffset * headSize].GetPhyAddr(),
                                (__gm__ float *)lastMax[lastOutOffset].GetPhyAddr(),
                                (__gm__ float *)lastSum[lastOutOffset].GetPhyAddr(),
                                lastWorkCurCore, nHeads, headSize, lastKvOffset == 0,
                                lastActualCalcSoftmaxLen, true, lastWorkStart, headNumInGroup);
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
    GlobalTensor<Dtype> input;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> vCache;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<float> lastMax;
    GlobalTensor<float> lastSum;
    GlobalTensor<Dtype> output;

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;

    uint32_t nHeads;
    uint32_t nKVHeads;
    uint32_t nQKVHeads;
    uint32_t headNumInGroup;
    uint32_t headSize;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlocks;
    uint32_t tileSizeOfCachedKV;
    uint32_t groupMemSize;

    AicHelper<Dtype> aicHelper;
};

#if defined(XLITE_DEVICE_310P)
#define FLASH_ATTN_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void flash_attention_##dtype(                               \
        GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR sv, GM_ADDR max,       \
        GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,              \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,        \
        uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize, uint32_t batch, \
        uint32_t maxNumBlocks, uint32_t tileSizeOfCachedKV)                                       \
    {                                                                                             \
    }
#else
#define FLASH_ATTN_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void flash_attention_##dtype(                                 \
        GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR sv, GM_ADDR max,        \
        GM_ADDR sum, GM_ADDR lastMax, GM_ADDR lastSum, GM_ADDR sync, GM_ADDR output,               \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,         \
        uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize, uint32_t batch, \
        uint32_t maxNumBlocks, uint32_t tileSizeOfCachedKV)                                        \
    {                                                                                              \
        FlashAttention<dtype> op;                                                                  \
        op.Init(input, kCache, vCache, qk, sv, max, sum, lastMax, lastSum, sync, output,           \
                queryStartLoc, queryLens, cachedLens, blockTables, nHeads, nKVHeads, headSize,     \
                blockSize, batch, maxNumBlocks, tileSizeOfCachedKV);                               \
        op.Run();                                                                                  \
    }
#endif
