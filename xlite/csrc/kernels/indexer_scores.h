/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once
#include "kernel_operator.h"
#include "kernel_macro.h"

#define MAX_M0 128
#define MAX_N0 MAX_M0
#define MBLOCKSIZE 16
#define NBLOCKSIZE 16

template <typename Dtype>
class IndexerScores
{
public:
    __aicore__ inline IndexerScores()
    {
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR scores,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                GM_ADDR blockTables, uint32_t nHeads, uint32_t headDim,
                                uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
        this->q.SetGlobalBuffer((__gm__ Dtype *)q);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->weight.SetGlobalBuffer((__gm__ Dtype *)weight);
        this->scores.SetGlobalBuffer((__gm__ Dtype *)scores);
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;
        this->nHeads = nHeads;
        this->headDim = headDim;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlock = maxNumBlock;

        /*
         * scores = k * q
         *     m: cachedTokens, n: queryLen * nHeads, k: headDim
         *     m0: blockSize, n0: MAX_N0, k0: MAX_K0
         * index_scores = weights * scores
         *     m: queryLen, n: cachedTokens, k: nHeads
         *     m0: 16, n0: blockSize, k0: MAX_K0
         */
        constexpr uint32_t k0 = 256 / sizeof(Dtype);
        uint64_t off = 0;
        assert(blockSize <= MAX_M0);
        assert(headDim <= k0);
        assert(nHeads <= k0);
        uint64_t kl1Size = blockSize * headDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kl1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kl1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kl1Size;
        }

        uint64_t ql1Size = MAX_N0 * headDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            ql1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            ql1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ql1Size;
        }

        uint64_t wl1Size = MAX_N0 * nHeads * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            wl1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            wl1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += wl1Size;
        }

        uint64_t kql1Size = blockSize * MAX_N0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kql1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kql1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kql1Size;
        }

        off = 0;
        uint64_t l0aSize = MAX_M0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            l0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0aSize;
        }

        off = 0;
        uint64_t l0bSize = MAX_N0 * k0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bSize;
        }

        off = 0;
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void Run()
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
#ifdef __DAV_C220_CUBE__
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);
#endif
        int queryTileSize = MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = 1;
        }

        int kBlockNum = headDim / kBlockSize;
        int wkBlockNum = nHeads / kBlockSize;

        int totalIdx = 0;
        int curr = 0;

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlock * sizeof(uint32_t));
            int cachedLen = cachedLens[batchIdx];
            int queryStart = queryStartLoc[batchIdx];
            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int totalLen = cachedLen + queryLen;
            int kvNum = DIV_ROUND_UP(totalLen, blockSize);
            int taskNum = queryNum * kvNum;

            // do queryTileSize's QK^T * weight
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int kvIdx = idx % kvNum;
                int queryIdx = idx / kvNum;
                int kvLen = blockSize;
                int kvStart = kvIdx * blockSize;
                if (kvStart + kvLen > totalLen) {
                    kvLen = totalLen - kvStart;
                }
                int mBlockPad = ROUND_UP(kvLen, MBLOCKSIZE);
                int mBlockNum = mBlockPad / MBLOCKSIZE;
                int wnBlockPad = ROUND_UP(kvLen, NBLOCKSIZE);
                int wnBlockNum = wnBlockPad / NBLOCKSIZE;

                int queryTaskLen = queryTileSize;
                int queryTaskStart = queryIdx * queryTileSize;
                if (queryTaskStart + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryTaskStart;
                }
                int qOffset = queryStart + queryTaskStart;

                int nSize = queryTileSize * nHeads;
                int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                int nBlockNum = nBlockPad / NBLOCKSIZE;

                // copy k (kvLen, headDim) to L1
                uint32_t block = blockTable[kvIdx];
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);
                CopyGmToL1Nd2Nz(kl1Buf[curr], kCache[block * blockSize * headDim], kvLen, headDim,
                                headDim, mBlockPad);
                // copy q (queryTaskLen, nHeads, headDim) to L1
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
                CopyGmToL1Nd2Nz(ql1Buf[curr], q[qOffset * nHeads * headDim], nSize, headDim,
                                headDim, nBlockPad);

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], kl1Buf[curr], mBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);

                CopyToL0BCol(l0bBuf[curr], ql1Buf[curr], nBlockNum, 0, kBlockNum);
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad scores (kvLen, queryTaskLen, nHeads)
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, headDim, true);
                PipeBarrier<PIPE_M>();
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);

                int wmSize = queryTileSize;
                int wmBlockPad = ROUND_UP(wmSize, MBLOCKSIZE);
                int wmBlockNum = wmBlockPad / MBLOCKSIZE;

                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

                // copy scores (kvLen, queryTaskLen, nHeads) from L0C to L1
                WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
                CopyL0CToL1(kql1Buf[curr], l0cBuf, mBlockPad, nBlockPad, mBlockPad,
                            mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);

                SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);

                // copy weight (queryTaskLen, nHeads) to L1
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                CopyGmToL1Nd2Nz(wl1Buf[curr], weight[qOffset * (headDim + nHeads) + headDim],
                                wmSize, nHeads, (headDim + nHeads), wmBlockPad);

                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + curr);

                for (int q = 0; q < queryTaskLen; q++) {
                    // copy weight (1, nHeads) to L0A
                    WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    CopyToL0ACol(l0aBuf[curr], wl1Buf[curr][q * kBlockSize], 1, 0, wkBlockNum);
                    // copy scores (kvLen, nHeads) to L0B
                    CopyToL0BCol(l0bBuf[curr], kql1Buf[curr][q * mBlockPad * nHeads], wnBlockNum, 0,
                                 wkBlockNum);

                    SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                    // mmad index_scores (1, kvLen)
                    WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                    CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], MBLOCKSIZE, wnBlockPad, nHeads,
                            true);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                    PipeBarrier<PIPE_M>();

                    SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                    WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

                    // copy index_scores (1, kvLen) from L0C to GM
                    CopyToGm(scores[(qOffset + q) * maxNumBlock * blockSize + kvStart], l0cBuf, 1,
                             kvLen, MBLOCKSIZE, maxNumBlock * blockSize);
                    SetFlag<HardEvent::FIX_M>(EVENT_ID0);
                }
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + curr);
                SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
                curr = 1 - curr;
            }
        }
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

private:
    GlobalTensor<Dtype> q;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> weight;
    GlobalTensor<Dtype> scores;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlock;

    LocalTensor<Dtype> kl1Buf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> ql1Buf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> wl1Buf[PINGPONG_BUF_NUM];   // event 4/5
    LocalTensor<Dtype> kql1Buf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;  // event 0
};

#if defined(XLITE_DEVICE_310P)
#define INDEXER_SCORES_FUNC_DEFINE(dtype)                                                    \
    extern "C" __global__ __aicore__ void indexer_scores_##dtype(                           \
        GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR scores, GM_ADDR queryStartLoc,   \
        GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads,        \
        uint32_t headDim, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)         \
    {                                                                                        \
    }
#else
#define INDEXER_SCORES_FUNC_DEFINE(dtype)                                                     \
    extern "C" __global__ __aicore__ void indexer_scores_##dtype(                             \
        GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR scores, GM_ADDR queryStartLoc,     \
        GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables, uint32_t nHeads,          \
        uint32_t headDim, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)           \
    {                                                                                         \
        IndexerScores<dtype> op;                                                              \
        op.Init(q, kCache, weight, scores, queryStartLoc, queryLens, cachedLens, blockTables, \
                nHeads, headDim, blockSize, batch, maxNumBlock);                              \
        op.Run();                                                                             \
    }
#endif
