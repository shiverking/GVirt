/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_param.h"
#include "kernel_operator.h"
// #define XLITE_KERNEL_DEBUG
#include "debug.h"
#include "softmax_attn_aiv.h"
#include "attention_aic_helper.h"

template <typename Dtype>
class Attention
{
public:
    __aicore__ inline Attention()
    {
    }

    __aicore__ inline void Init(GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk,
                                GM_ADDR output, GM_ADDR queryStartLoc, GM_ADDR queryLens,
                                GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads,
                                uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize,
                                uint32_t batch, uint32_t maxNumBlocks)
    {
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
        this->maxSeqLen = maxNumBlocks * blockSize;
        this->groupMemSize = headNumInGroup * headSize;

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * XLITE_MAX_M0 * maxSeqLen);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) + block_idx * XLITE_MAX_M0 * maxSeqLen +
                                    block_num * XLITE_MAX_M0 * maxSeqLen);

        aicHelper.Init(nHeads, nKVHeads, headNumInGroup, headSize, blockSize, maxSeqLen);
    }

    __aicore__ inline void RunAic()
    {
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);

        uint64_t flagIdx = 0;
        uint64_t mode = 2;  // inner-group aic/aiv sync
        uint64_t config = 1 | (mode << 4) | (flagIdx << 8);

        uint32_t lastOutOffset, lastCalcLen;
        int lastQueryTaskLen, lastkvHeadIdx, lastQkIdx;
        __gm__ uint32_t *lastBlockTable;

        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = m0;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum * nKVHeads;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num !=
                    ((totalIdx / block_num) % 2 == 0 ? block_idx : block_num - 1 - block_idx)) {
                    continue;
                }
                int kvHeadIdx = idx % nKVHeads;
                int queryIdx = idx / nKVHeads;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;
                int kvHeadOffset = kvHeadIdx * groupMemSize;

                // do queryIdx & kvHeadIdx's QK
                uint32_t qOffset = queryTaskOffset * headSize * nQKVHeads + kvHeadOffset;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + queryTaskLen;
                dbg_printf("block%d: batch %d query start %u query [%u - %u) do QK kvHeadIdx %u "
                           "calcLen %u, use %d qk buf\n",
                           GetBlockIdx(), batchIdx, queryStart, queryTaskOffset,
                           queryTaskOffset + queryTaskLen, kvHeadIdx, calcLen, curr);
                aicHelper.RunAicQK(input[qOffset], this->kCache, queryTaskLen, kvHeadIdx,
                                   blockTable, 0, calcLen, qk[curr]);
#if !defined(XLITE_DEVICE_310P)
                ffts_cross_core_sync(PIPE_FIX, config);
#endif

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(1);
                    // do softmax * V
                    dbg_printf("block%d: do SV kvHeadIdx %u calcLen %u, use %d qk buf\n",
                               GetBlockIdx(), lastkvHeadIdx, lastCalcLen, lastQkIdx);
                    aicHelper.RunAicSV(qk[lastQkIdx], this->vCache, lastQueryTaskLen, lastkvHeadIdx,
                                       lastBlockTable, 0, lastCalcLen, output[lastOutOffset], true);
                }

                lastOutOffset = queryTaskOffset * headSize * nHeads + kvHeadOffset;
                lastQueryTaskLen = queryTaskLen;
                lastkvHeadIdx = kvHeadIdx;
                lastBlockTable = blockTable;
                lastCalcLen = calcLen;
                lastQkIdx = curr;
                needDoSV = 1;

                curr = 1 - curr;
            }
            queryStart = -1;
            cachedLen = -1;
        }

        // do last softmax * V
        if (needDoSV != 0) {
            wait_flag_dev(1);
            dbg_printf("block%d: do SV kvHeadIdx %u calcLen %u, use %d qk buf\n", GetBlockIdx(),
                       lastkvHeadIdx, lastCalcLen, lastQkIdx);
            aicHelper.RunAicSV(qk[lastQkIdx], this->vCache, lastQueryTaskLen, lastkvHeadIdx,
                               lastBlockTable, 0, lastCalcLen, output[lastOutOffset], true);
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
        int cachedLen = -1;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlocks * sizeof(uint32_t));

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            uint32_t m0 = GetOptimalM0(queryLen, cachedLen);
            int queryTileSize = m0 / headNumInGroup;
            if (queryTileSize == 0) {
                queryTileSize = m0;
            }

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int taskNum = queryNum * nKVHeads;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num !=
                    ((totalIdx / block_num) % 2 == 0 ? block_idx : block_num - 1 - block_idx)) {
                    continue;
                }
                int kvHeadIdx = idx % nKVHeads;
                int queryIdx = idx / nKVHeads;
                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }

                int nWork = queryTaskLen * headNumInGroup;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                uint32_t subIdx = get_subblockid();
                int nWorkStart = subIdx * nWorkPerCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t qkOffset = nWorkStart * maxSeqLen;
                if (cachedLen < 0) {
                    cachedLen = cachedLens[batchIdx];
                }
                uint32_t calcLen = cachedLen + queryTaskStart + 1;
                uint32_t outN = ROUND_UP(cachedLen + queryTaskStart + queryTaskLen, blockSize);

                // wait aic qk done
                wait_flag_dev(0);

                // do softmax
                int dbgBlockIdx = block_idx;
                dbg_printf(
                    "block%d subblock%u: batch %d do softmax kvHeadIdx %u m %d calcLen %u outN "
                    "%u mask off %u, use %d qk buf\n",
                    dbgBlockIdx, subIdx, batchIdx, kvHeadIdx, nWorkCurCore, calcLen, outN,
                    nWorkStart, curr);
                RunAivSoftmax(
                    (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(),
                    m0 > (XLITE_MAX_M0 - 4)
                        ? 0
                        : (__gm__ float *)qk[curr][(m0 + subIdx * 2) * maxSeqLen].GetPhyAddr(),
                    nWorkCurCore, maxSeqLen, calcLen, outN, true, nWorkStart, headNumInGroup);

                ffts_cross_core_sync(PIPE_MTE3, config);
                curr = 1 - curr;
            }
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
    GlobalTensor<Dtype> input;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> vCache;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];
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
    uint32_t maxSeqLen;
    uint32_t groupMemSize;

    AicHelper<Dtype> aicHelper;
};

#if defined(XLITE_DEVICE_310P)
#define ATTN_FUNC_DEFINE(dtype)                                                                   \
    extern "C" __global__ __aicore__ void attention_##dtype(                                     \
        GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR output,                \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,        \
        uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize, uint32_t batch, \
        uint32_t maxNumBlocks)                                                                    \
    {                                                                                             \
    }
#else
#define ATTN_FUNC_DEFINE(dtype)                                                                    \
    extern "C" __global__ __aicore__ void attention_##dtype(                                       \
        GM_ADDR input, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR qk, GM_ADDR output,                 \
        GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,         \
        uint32_t nHeads, uint32_t nKVHeads, uint32_t headSize, uint32_t blockSize, uint32_t batch, \
        uint32_t maxNumBlocks)                                                                     \
    {                                                                                              \
        Attention<dtype> op;                                                                       \
        op.Init(input, kCache, vCache, qk, output, queryStartLoc, queryLens, cachedLens,           \
                blockTables, nHeads, nKVHeads, headSize, blockSize, batch, maxNumBlocks);          \
        op.Run();                                                                                  \
    }
#endif
