/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_macro.h"
#include "kernel_operator.h"
#include "kernel_param.h"
#include "softmax_attn_aiv.h"

// #define XLITE_KERNEL_DEBUG
#include "debug.h"

#define MBLOCKSIZE 16
#define NBLOCKSIZE 16
// Ascend cube L0A/L0B are 64KB each. Pingpong needs 2 tiles to fit.
#define ASCEND_L0_BYTES (64 * 1024)

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
        this->maxSeqLen = maxNumBlocks * blockSize;
        this->qMemSize = nHeads * headSize;
        this->kvMemSize = nKVHeads * headSize;
        this->qkvMemSize = nQKVHeads * headSize;
        this->groupMemSize = headNumInGroup * headSize;
        this->blockMemSize = blockSize * kvMemSize;
        this->blockIdx = block_idx;
        this->subBlockIdx = get_subblockid();
        this->nextBlockIdx = (blockIdx + 1) % block_num;
        this->prevBlockIdx = blockIdx == 0 ? (block_num - 1) : (blockIdx - 1);
        this->setNextGeneration = 1;
        this->waitPrevGeneration = 1;

        this->qk[0].SetGlobalBuffer(((__gm__ Dtype *)qk) +
                                    block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->qk[1].SetGlobalBuffer(((__gm__ Dtype *)qk) +
                                    block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV +
                                    block_num * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->sv[0].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * headSize);
        this->sv[1].SetGlobalBuffer(((__gm__ Dtype *)sv) + block_idx * XLITE_MAX_M0 * headSize +
                                    block_num * XLITE_MAX_M0 * headSize);
        this->max[0].SetGlobalBuffer(((__gm__ float *)max) + block_idx * XLITE_MAX_M0 * 2 +
                                     subBlockIdx * XLITE_MAX_M0);
        this->max[1].SetGlobalBuffer(((__gm__ float *)max) + block_idx * XLITE_MAX_M0 * 2 +
                                     subBlockIdx * XLITE_MAX_M0 + block_num * XLITE_MAX_M0 * 2);
        this->sum[0].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * XLITE_MAX_M0 * 2 +
                                     subBlockIdx * XLITE_MAX_M0);
        this->sum[1].SetGlobalBuffer(((__gm__ float *)sum) + block_idx * XLITE_MAX_M0 * 2 +
                                     subBlockIdx * XLITE_MAX_M0 + block_num * XLITE_MAX_M0 * 2);
        this->lastMax.SetGlobalBuffer((__gm__ float *)lastMax);
        this->lastSum.SetGlobalBuffer((__gm__ float *)lastSum);
        this->setNextSync = (__gm__ int32_t *)sync + blockIdx * 2 + subBlockIdx;
        this->waitPrevSync = (__gm__ int32_t *)sync + prevBlockIdx * 2 + subBlockIdx;

        // 分配L1/L0：保留 pingpong；Cube KV tile 与 blockSize 解耦，
        // 保证 2 * tile * headSize * sizeof 落在 64KB L0 内（headDim=256 时
        // 原先 blockSize=128 单 tile 已满 64KB，pingpong 会 CCU 越界）。
        const uint64_t dtypeBytes = sizeof(Dtype);
        uint32_t tile = blockSize;
        while (tile > static_cast<uint32_t>(NBLOCKSIZE) &&
               (2ull * tile * headSize * dtypeBytes > ASCEND_L0_BYTES ||
                2ull * XLITE_MAX_M0 * tile * dtypeBytes > ASCEND_L0_BYTES)) {
            tile >>= 1;
        }
        this->cubeKvTile = tile;

        uint64_t aTileBytes = static_cast<uint64_t>(XLITE_MAX_M0) *
                              (headSize > cubeKvTile ? headSize : cubeKvTile) * dtypeBytes;
        uint64_t bTileBytes = static_cast<uint64_t>(cubeKvTile) * headSize * dtypeBytes;
        uint64_t l0aSvBytes = static_cast<uint64_t>(XLITE_MAX_M0) * cubeKvTile * dtypeBytes;
        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            l1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += aTileBytes;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += bTileBytes;
        }
        off = 0;
        l0aBuf[0].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf[0].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        l0aBuf[1].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0aBuf[1].address_.bufferAddr = reinterpret_cast<uint64_t>(off + l0aSvBytes);
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += bTileBytes;
        }
        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void SetNextCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
        dbg_printf("block%d subblock%u set block%d subblock%u %u\n", blockIdx, subBlockIdx,
                   nextBlockIdx, subBlockIdx, setNextGeneration);
        *val = setNextGeneration;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b16(setNextSync, val, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
        PipeBarrier<PIPE_ALL>();
        setNextGeneration++;
    }

    __aicore__ inline void WaitPrevCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
        dbg_printf("block%d subblock%u wait block%d subblock%u %u\n", blockIdx, subBlockIdx,
                   prevBlockIdx, subBlockIdx, waitPrevGeneration);
        do {
            copy_gm_to_ubuf_align_b16(val, waitPrevSync, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
            set_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
            wait_flag(PIPE_MTE2, PIPE_S, EVENT_ID0);
        } while (*val < waitPrevGeneration);
        waitPrevGeneration++;
    }

    __aicore__ inline void ResetPrevCore()
    {
        __ubuf__ int32_t *val = (__ubuf__ int32_t *)(0ull);
        dbg_printf("block%d subblock%u reset block%d subblock%u\n", blockIdx, subBlockIdx,
                   prevBlockIdx, subBlockIdx);
        *val = 0;
        set_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        wait_flag(PIPE_S, PIPE_MTE3, EVENT_ID0);
        copy_ubuf_to_gm_align_b16(waitPrevSync, val, 0, 1, sizeof(int32_t), 0, 0, 0, 0);
    }

    /*
     * m: tokens
     * n: cachedTokens
     * k: headSize
     * Cube loads KV in cubeKvTile-sized panels (may be < blockSize).
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> query, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> qk)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int kBlockNum = DIV_ROUND_UP(headSize, kBlockSize);
        int kvHeadOffset = kvHeadIdx * headSize;
        int tile = static_cast<int>(cubeKvTile);

        Nd2NzParams nd2nzParams(1 /* NdNum */, queryLen /* nValue */, headSize /* dValue */,
                                0 /* srcNdMatrixStride */, qkvMemSize /* srcDValue */,
                                mBlockPad /* dstNzC0Stride */, headNumInGroup /* dstNzNStride */,
                                0 /* dstNzMatrixStride */);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        for (int h = 0; h < headNumInGroup; h++) {
            DataCopy(l1aBuf[0][MBLOCKSIZE * h], query[headSize * h], nd2nzParams);
        }
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        CopyToL0ACol(l0aBuf[0], l1aBuf[0], mBlockNum, 0, kBlockNum);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

        int curIdx = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int local = 0; local < kvLen; local += tile) {
            int absPos = kvOffset + local;
            int rowInBlock = absPos % static_cast<int>(blockSize);
            int nSize = tile;
            if (rowInBlock + nSize > static_cast<int>(blockSize)) {
                nSize = static_cast<int>(blockSize) - rowInBlock;
            }
            if (local + nSize > kvLen) {
                nSize = kvLen - local;
            }
            int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
            int nBlockNum = nBlockPad / NBLOCKSIZE;
            uint32_t block = blockTable[absPos / static_cast<int>(blockSize)];

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx],
                            kCache[block * blockMemSize + rowInBlock * kvMemSize + kvHeadOffset],
                            nSize, headSize, kvMemSize, nBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            CopyToL0BCol(l0bBuf[curIdx], l1bBuf[curIdx], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            CalMmad(l0cBuf, l0aBuf[0], l0bBuf[curIdx], mBlockPad, nBlockPad, headSize, true);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);

            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            CopyToGm(qk[local], l0cBuf, mActual, nSize, mBlockPad, tileSizeOfCachedKV);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            PipeBarrier<PIPE_M>();
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /*
     * m: tokens
     * n: headSize
     * k: cachedTokens
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, int queryLen, int kvHeadIdx,
                                    __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                    GlobalTensor<Dtype> sv)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mActual = queryLen * headNumInGroup;
        int mBlockPad = ROUND_UP(mActual, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nBlockNum = DIV_ROUND_UP(headSize, NBLOCKSIZE);
        int kvHeadOffset = kvHeadIdx * headSize;
        int tile = static_cast<int>(cubeKvTile);

        int curIdx = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID2);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID3);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        int first = 1;
        for (int local = 0; local < kvLen; local += tile) {
            int absPos = kvOffset + local;
            int rowInBlock = absPos % static_cast<int>(blockSize);
            int kSize = tile;
            if (rowInBlock + kSize > static_cast<int>(blockSize)) {
                kSize = static_cast<int>(blockSize) - rowInBlock;
            }
            if (local + kSize > kvLen) {
                kSize = kvLen - local;
            }
            int kBlockPad = ROUND_UP(kSize, kBlockSize);
            int kBlockNum = kBlockPad / kBlockSize;
            uint32_t block = blockTable[absPos / static_cast<int>(blockSize)];

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            CopyGmToL1Nd2Nz(l1aBuf[curIdx], qk[local], mActual, kBlockPad, tileSizeOfCachedKV,
                            mBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            CopyGmToL1Nd2Nz(l1bBuf[curIdx],
                            vCache[block * blockMemSize + rowInBlock * kvMemSize + kvHeadOffset],
                            kBlockPad, headSize, kvMemSize, kBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curIdx);
            CopyToL0ACol(l0aBuf[curIdx], l1aBuf[curIdx], mBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curIdx);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curIdx);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            CopyToL0BTCol(l0bBuf[curIdx], l1bBuf[curIdx], nBlockNum, 0, kBlockNum, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curIdx);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);

            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curIdx);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID2 + curIdx);
            CalMmad(l0cBuf, l0aBuf[curIdx], l0bBuf[curIdx], mBlockPad, headSize, kBlockPad,
                    first != 0);
            first = 0;
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curIdx);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID2 + curIdx);
            PipeBarrier<PIPE_M>();
            curIdx = 1 - curIdx;
        }
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID3);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID2);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        CopyToGm(sv, l0cBuf, mActual, headSize, mBlockPad, headSize);
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
                RunAicQK(input[qOffset], queryTaskLen, kvHeadIdx, blockTable, kvOffset, kvLen,
                         qk[curr]);
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
                    RunAicSV(qk[last], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable,
                             lastKvOffset, lastKvLen, sv[last]);
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
            RunAicSV(qk[last], lastQueryTaskLen, lastkvHeadIdx, lastBlockTable, lastKvOffset,
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
                int nWorkStart = subBlockIdx * nWorkPerCore;
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
                    blockIdx, subBlockIdx, batchIdx, queryTaskOffset,
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
                        WaitPrevCore();
                        resetPrevCore = 1;
                    }
                    dbg_printf(
                        "block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u "
                        "- %u), kvHeadIdx %u "
                        "kv [%u - %u)} use %d temp buf: UPDATE\n",
                        blockIdx, subBlockIdx, lastBatchIdx, lastQueryTaskOffset,
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
                        SetNextCore();
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
                WaitPrevCore();
                resetPrevCore = 1;
            }
            dbg_printf(
                "block%d subblock%u: {batch %d, query [%u - %u) query x head group [%u - %u), "
                "kvHeadIdx %u "
                "kv [%u - %u)} use %d temp buf: UPDATE\n",
                blockIdx, subBlockIdx, lastBatchIdx, lastQueryTaskOffset,
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
                SetNextCore();
            }
        }
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ResetPrevCore();
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
    GlobalTensor<Dtype> sv[PINGPONG_BUF_NUM];
    GlobalTensor<float> max[PINGPONG_BUF_NUM];
    GlobalTensor<float> sum[PINGPONG_BUF_NUM];
    GlobalTensor<float> lastMax;
    GlobalTensor<float> lastSum;
    GlobalTensor<Dtype> output;
    __gm__ int32_t *setNextSync;
    __gm__ int32_t *waitPrevSync;

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
    uint32_t maxSeqLen;
    uint32_t qMemSize;
    uint32_t kvMemSize;
    uint32_t qkvMemSize;
    uint32_t groupMemSize;
    uint32_t blockMemSize;
    // Cube KV panel length (<= blockSize) so L0 pingpong fits for large headDim.
    uint32_t cubeKvTile;
    int blockIdx;
    int subBlockIdx;
    int nextBlockIdx;
    int prevBlockIdx;
    uint32_t setNextGeneration;
    uint32_t waitPrevGeneration;

    LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;
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
