/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "kernel_operator.h"
#include "kernel_macro.h"
#include "kernel_param.h"
#include "debug.h"

#pragma once

#define MAX_N0 XLITE_MAX_M0

template <typename Dtype>
class IndexerTopK
{
public:
    __aicore__ inline IndexerTopK()
    {
    }

    __aicore__ inline void Init(GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc,
                                GM_ADDR queryLens, GM_ADDR cachedLens, GM_ADDR blockTables,
                                GM_ADDR scores, GM_ADDR lastTopk, GM_ADDR indices,
                                GM_ADDR topkIndices, GM_ADDR sync, uint32_t nHeads,
                                uint32_t headDim, uint32_t blockSize, uint32_t batch,
                                uint32_t maxNumBlock, uint32_t topK)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->q.SetGlobalBuffer((__gm__ Dtype *)q);
        this->kCache.SetGlobalBuffer((__gm__ Dtype *)kCache);
        this->weight.SetGlobalBuffer((__gm__ Dtype *)weight);
        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;
        this->blockTables = (__gm__ int32_t *)blockTables;
        this->lastTopk = (__gm__ uint32_t *)lastTopk;
        this->indices = (__gm__ uint32_t *)indices;
        this->topkIndices = (__gm__ uint32_t *)topkIndices;
        this->nHeads = nHeads;
        this->headDim = headDim;
        this->blockSize = blockSize;
        this->batch = batch;
        this->maxNumBlock = maxNumBlock;
        this->topK = topK;
        this->tileSizeOfCachedKV = MAX_INDEXER_KV_TILE_LEN;
        this->blockIdx = block_idx;
        this->subBlockIdx = get_subblockid();
        this->nextBlockIdx = (blockIdx + 1) % block_num;
        this->prevBlockIdx = blockIdx == 0 ? (block_num - 1) : (blockIdx - 1);
        this->setNextGeneration = 1;
        this->waitPrevGeneration = 1;
        this->resetPrevCore = 0;

        this->scores[0].SetGlobalBuffer(((__gm__ Dtype *)scores) +
                                        block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->scores[1].SetGlobalBuffer(((__gm__ Dtype *)scores) +
                                        block_idx * XLITE_MAX_M0 * tileSizeOfCachedKV +
                                        block_num * XLITE_MAX_M0 * tileSizeOfCachedKV);
        this->setNextSync = (__gm__ int32_t *)sync + blockIdx * 2 + subBlockIdx;
        this->waitPrevSync = (__gm__ int32_t *)sync + prevBlockIdx * 2 + subBlockIdx;

#ifdef __DAV_C220_CUBE__
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

        uint64_t wl1Size = XLITE_MAX_M0 * nHeads * sizeof(Dtype);
        wl1Buf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        wl1Buf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += wl1Size;

        uint64_t kql1Size = blockSize * MAX_N0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            kql1Buf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            kql1Buf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kql1Size;
        }

        off = 0;
        uint64_t l0aSize = XLITE_MAX_M0 * k0 * sizeof(Dtype);
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
#endif
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

    __aicore__ inline void RunAicIndexerScores(GlobalTensor<Dtype> query,
                                               GlobalTensor<Dtype> weight, int queryLen,
                                               __gm__ uint32_t *blockTable, int kvOffset, int kvLen,
                                               GlobalTensor<Dtype> scores)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mIdxStart = kvOffset / blockSize;
        int mLoop = DIV_ROUND_UP(kvLen, blockSize);
        int mSize = blockSize;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = queryLen * nHeads;
        int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int kBlockNum = headDim / kBlockSize;

        int wmSize = queryLen;
        int wmBlockPad = ROUND_UP(wmSize, MBLOCKSIZE);
        int wmBlockNum = wmBlockPad / MBLOCKSIZE;
        int wnSize = blockSize;
        int wnBlockPad = ROUND_UP(wnSize, NBLOCKSIZE);
        int wnBlockNum = wnBlockPad / NBLOCKSIZE;
        int wkBlockNum = nHeads / kBlockSize;

        // copy weight (queryTaskLen, nHeads) to L1
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        CopyGmToL1Nd2Nz(wl1Buf, weight, wmSize, nHeads, (headDim + nHeads), wmBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);

        int curr = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        // do queryTileSize's QK^T * weight
        for (int mIdx = 0; mIdx < mLoop; mIdx++) {  // kvLen
            int mOffset = mIdx * blockSize;
            if (mOffset + mSize > kvLen) {
                mSize = kvLen - mOffset;
                mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
                mBlockNum = mBlockPad / MBLOCKSIZE;
                wnSize = mSize;
                wnBlockPad = ROUND_UP(wnSize, NBLOCKSIZE);
                wnBlockNum = wnBlockPad / NBLOCKSIZE;
            }

            // copy k (blockSize, headDim) to L1
            uint32_t block = blockTable[mIdx + mIdxStart];
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);
            CopyGmToL1Nd2Nz(kl1Buf[curr], kCache[block * blockSize * headDim], mSize, headDim,
                            headDim, mBlockPad);
            // copy q (queryTaskLen, nHeads, headDim) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            CopyGmToL1Nd2Nz(ql1Buf[curr], query, nSize, headDim, headDim, nBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(l0aBuf[curr], kl1Buf[curr], mBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + curr);

            CopyToL0BCol(l0bBuf[curr], ql1Buf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // mmad scores (blockSize, queryTaskLen, nHeads)
            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], mBlockPad, nBlockPad, headDim, true);
            PipeBarrier<PIPE_M>();
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);

            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy scores (blockSize, queryTaskLen, nHeads) from L0C to L1
            WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
            CopyL0CToL1(kql1Buf[curr], l0cBuf, mBlockPad, nBlockPad, mBlockPad,
                        mBlockPad * sizeof(Dtype) * kBlockSize / BLOCK_SIZE);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);

            SetFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::FIX_MTE1>(EVENT_ID0 + curr);

            for (int q = 0; q < queryLen; q++) {
                // copy weight (1, nHeads) to L0A
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(l0aBuf[curr], wl1Buf[0][q * kBlockSize], 1, 0, wkBlockNum);
                // copy scores (blockSize, nHeads) to L0B
                CopyToL0BCol(l0bBuf[curr], kql1Buf[curr][q * mBlockPad * nHeads], wnBlockNum, 0,
                             wkBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad index_scores (1, blockSize)
                WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
                CalMmad(l0cBuf, l0aBuf[curr], l0bBuf[curr], MBLOCKSIZE, wnBlockPad, nHeads, true);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();

                SetFlag<HardEvent::M_FIX>(EVENT_ID0);
                WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

                // copy index_scores (1, blockSize) from L0C to GM
                CopyToGm(scores[q * tileSizeOfCachedKV + mOffset], l0cBuf, 1, mSize, MBLOCKSIZE,
                         tileSizeOfCachedKV);
                SetFlag<HardEvent::FIX_M>(EVENT_ID0);
            }
            SetFlag<HardEvent::MTE1_FIX>(EVENT_ID0 + curr);
            curr = 1 - curr;
        }
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

#ifdef __DAV_C220_VEC__
    __aicore__ inline void RunAivTopk(__gm__ Dtype *scores, __gm__ uint32_t *lastTopk,
                                      __gm__ uint32_t *indices, int queryLen, int kvOffset,
                                      int kvLen, int totalLen, uint32_t topK,
                                      __gm__ uint32_t *topkIndices)
    {
        constexpr float min = FLOAT_MIN;

        assert(kvLen <= MAX_INDEXER_KV_TILE_LEN);
        assert(topK <= MAX_TOPK_NUM);

        // total sort & WaitPrevCore & SetNextCore use
        uint64_t off = 0;
        __ubuf__ float *totalSort = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 4 * sizeof(float), VECTOR_MAX_BYTESIZE);

        // in
        __ubuf__ Dtype *in0 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
        __ubuf__ Dtype *in1 = reinterpret_cast<__ubuf__ Dtype *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(Dtype), VECTOR_MAX_BYTESIZE);
        __ubuf__ Dtype *in[PINGPONG_BUF_NUM] = {in0, in1};
        __ubuf__ uint32_t *sortIndices0 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *sortIndices1 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *sortIndices[PINGPONG_BUF_NUM] = {sortIndices0, sortIndices1};
        __ubuf__ float *lastSort0 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        __ubuf__ float *lastSort1 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        __ubuf__ float *lastSort[PINGPONG_BUF_NUM] = {lastSort0, lastSort1};

        // out
        __ubuf__ uint32_t *out0 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *out1 = reinterpret_cast<__ubuf__ uint32_t *>(off);
        off += ROUND_UP(MAX_TOPK_NUM * sizeof(uint32_t), VECTOR_MAX_BYTESIZE);
        __ubuf__ uint32_t *out[PINGPONG_BUF_NUM] = {out0, out1};

        // calc
        __ubuf__ float *mrgSortBuf0 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        __ubuf__ float *mrgSortBuf1 = reinterpret_cast<__ubuf__ float *>(off);
        off += ROUND_UP(MAX_INDEXER_KV_TILE_LEN * 2 * sizeof(float), VECTOR_MAX_BYTESIZE);
        assert(off <= UB_SIZE);

        constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
        constexpr int calcPad = VECTOR_MAX_BYTESIZE / sizeof(float);
        int repeat = DIV_ROUND_UP(kvLen, calcPad);
        int sortRepeat = DIV_ROUND_UP(kvLen, SORT_BLOCK_SIZE);
        int topKSortRepeat = DIV_ROUND_UP(topK, SORT_BLOCK_SIZE);
        uint64_t lens = (kvLen > topK ? topK : kvLen) | (uint64_t)topK << 16;
        uint64_t config = 1 | 0x3ull << MGR_SORT_VALID_BITS_OFFSET;

        int curr = 0;
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        set_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        for (int idx = 0; idx < queryLen; idx++) {
            // copy scores to in
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            CopyGmToUbufAligned(in[curr], scores + idx * tileSizeOfCachedKV, kvLen * sizeof(Dtype));
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);

            // copy indices to sortIndices
            wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);
            CopyGmToUbufAligned(sortIndices[curr], indices + kvOffset, kvLen * sizeof(uint32_t));
            set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);

            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID0 + curr);
            // convert in to float
            if constexpr (std::is_same<Dtype, half>::value) {
                vconv_f162f32(mrgSortBuf0, in[curr], repeat, 1, 1, 8, 4);
            } else {
                vconv_bf162f32(mrgSortBuf0, in[curr], repeat, 1, 1, 8, 4);
            }
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID0 + curr);
            pipe_barrier(PIPE_V);

            // pad
            int remain = kvLen % calcPad;
            if (remain != 0) {
                SetMaskFromHighBit(calcPad, calcPad - remain);
                vector_dup(mrgSortBuf0 + ROUND_DOWN(kvLen, calcPad), float(min), 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
                set_vector_mask((uint64_t)-1, (uint64_t)-1);
            }

            wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
            // sort local
            vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices[curr], sortRepeat);
            pipe_barrier(PIPE_V);
            set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);

            int dstBufIdx = 0;
            MrgSort(mrgSortBuf1, mrgSortBuf0, sortRepeat, &dstBufIdx);
            __ubuf__ float *localSort = dstBufIdx == 0 ? mrgSortBuf1 : mrgSortBuf0;

            wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            // sort local & last
            int outNum = topK;
            if (kvOffset != 0) {
                if (idx == 0) {
                    WaitPrevCore();
                    resetPrevCore = 1;
                }
                // copy last topk to last sort
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4 + curr);
                CopyGmToUbufAligned(lastSort[curr], lastTopk + idx * 2 * topK,
                                    topK * 2 * sizeof(uint32_t));

                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID4 + curr);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID4 + curr);

                __ubuf__ float *addrs[4] = {localSort, lastSort[curr]};
                vmrgsort4(totalSort, addrs, lens, config);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID4 + curr);
                pipe_barrier(PIPE_V);
            } else {
                // copy localSort to totalSort
                copy_ubuf_to_ubuf(totalSort, localSort, 0, 1,
                                  DIV_ROUND_UP(topK * 2 * sizeof(uint32_t), BLOCK_SIZE), 1, 1);
                pipe_barrier(PIPE_V);
                outNum = kvLen > topK ? topK : kvLen;
            }

            if (kvOffset + kvLen == totalLen) {
                // copy indices to sortIndices
                wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);
                CopyGmToUbufAligned(sortIndices[curr], indices, outNum * sizeof(uint32_t));
                set_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);

                __ubuf__ uint32_t *index0 = (__ubuf__ uint32_t *)mrgSortBuf0;
                vreducev2(index0, (__ubuf__ uint32_t *)totalSort, (__ubuf__ uint32_t *)totalSort,
                          DIV_ROUND_UP(outNum, 32), 1, 2, 8, 0);
                pipe_barrier(PIPE_V);

                // pad
                int remain = outNum % calcPad;
                if (remain != 0) {
                    SetMaskFromHighBit(calcPad, calcPad - remain);
                    vector_dup(index0 + ROUND_DOWN(outNum, calcPad), 0, 1, 1, 1, 8, 0);
                    pipe_barrier(PIPE_V);
                    set_vector_mask((uint64_t)-1, (uint64_t)-1);
                }

                vconv_s322f32(mrgSortBuf0, (__ubuf__ int *)index0, DIV_ROUND_UP(topK, calcPad), 1,
                              1, 8, 8);
                pipe_barrier(PIPE_V);

                topKSortRepeat = DIV_ROUND_UP(outNum, SORT_BLOCK_SIZE);
                wait_flag(PIPE_MTE2, PIPE_V, EVENT_ID2 + curr);
                vbitsort(mrgSortBuf1, mrgSortBuf0, sortIndices[curr], topKSortRepeat);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE2, EVENT_ID2 + curr);

                MrgSort(mrgSortBuf1, mrgSortBuf0, topKSortRepeat, &dstBufIdx);
                __ubuf__ float *index = dstBufIdx == 0 ? mrgSortBuf1 : mrgSortBuf0;

                __ubuf__ float *indexSorted = dstBufIdx == 0 ? mrgSortBuf0 : mrgSortBuf1;
                vreducev2((__ubuf__ uint32_t *)indexSorted, (__ubuf__ uint32_t *)index,
                          (__ubuf__ uint32_t *)index, DIV_ROUND_UP(outNum, 32), 1, 1, 8, 0);
                pipe_barrier(PIPE_V);

                vconv_f322s32r((__ubuf__ int *)index, indexSorted, DIV_ROUND_UP(outNum, calcPad), 1,
                               1, 8, 8);
                pipe_barrier(PIPE_V);

                // get total topk indices
                wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
                copy_ubuf_to_ubuf(out[curr], index, 0, 1,
                                  DIV_ROUND_UP(outNum * sizeof(uint32_t), BLOCK_SIZE), 1, 1);
                pipe_barrier(PIPE_V);
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID0 + curr);
                // copy totalSort to topkIndices
                CopyUbufToGmAligned(topkIndices + idx * topK, out[curr], outNum * sizeof(uint32_t));
                set_flag(PIPE_MTE3, PIPE_V, EVENT_ID0 + curr);
            } else {
                set_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                wait_flag(PIPE_V, PIPE_MTE3, EVENT_ID2);
                // copy totalSort to last
                CopyUbufToGmAligned(lastTopk + idx * topK * 2, totalSort,
                                    outNum * 2 * sizeof(uint32_t));
                if (idx == queryLen - 1) {
                    set_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                    wait_flag(PIPE_MTE3, PIPE_S, EVENT_ID0);
                    SetNextCore();
                }
            }
            set_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
            curr = 1 - curr;
        }
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID2);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID1);
        wait_flag(PIPE_MTE3, PIPE_V, EVENT_ID0);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID5);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID4);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID3);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID2);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID1);
        wait_flag(PIPE_V, PIPE_MTE2, EVENT_ID0);
    }
#endif

    __aicore__ inline void Run()
    {
        uint64_t flagIdx0 = 0;
        uint64_t flagIdx1 = 1;
        uint64_t flagIdx2 = 2;
        uint64_t flagIdx3 = 3;
        uint64_t mode = 2;  // inner-group aic/aiv sync
#ifdef __DAV_C220_CUBE__
        set_padding(0);
        set_atomic_none();
        set_nd_para((uint64_t)1);
        uint64_t sync0 = 1 | (mode << 4) | (flagIdx0 << 8);
        uint64_t sync1 = 1 | (mode << 4) | (flagIdx1 << 8);
        uint64_t a2vSyncFlag[PINGPONG_BUF_NUM] = {sync0, sync1};
        uint64_t v2aSyncFlag[PINGPONG_BUF_NUM] = {flagIdx2, flagIdx3};
#elif __DAV_C220_VEC__
        set_atomic_none();
        set_mask_norm();
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        uint64_t sync2 = 1 | (mode << 4) | (flagIdx2 << 8);
        uint64_t sync3 = 1 | (mode << 4) | (flagIdx3 << 8);
        uint64_t a2vSyncFlag[PINGPONG_BUF_NUM] = {flagIdx0, flagIdx1};
        uint64_t v2aSyncFlag[PINGPONG_BUF_NUM] = {sync2, sync3};
#endif

        int queryTileSize = XLITE_MAX_M0 / nHeads;
        if (queryTileSize == 0) {
            queryTileSize = XLITE_MAX_M0;
        }

        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
#ifdef __DAV_C220_VEC__
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[0]);
        ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[1]);
#endif
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];
            int cachedLen = cachedLens[batchIdx];
            int totalLen = cachedLen + queryLen;
            __gm__ uint32_t *blockTable =
                (__gm__ uint32_t *)((uint64_t)blockTables +
                                    batchIdx * maxNumBlock * sizeof(uint32_t));

            int queryNum = DIV_ROUND_UP(queryLen, queryTileSize);
            int kvNum = DIV_ROUND_UP(cachedLen + queryLen, tileSizeOfCachedKV);
            int taskNum = queryNum * kvNum;
            for (int idx = 0; idx < taskNum; idx++, totalIdx++) {
                if (totalIdx % block_num != block_idx) {
                    continue;
                }
                int kvIdx = idx % kvNum;
                int kvLen = tileSizeOfCachedKV;
                int kvOffset = kvIdx * tileSizeOfCachedKV;
                if (kvOffset + kvLen > totalLen) {
                    kvLen = totalLen - kvOffset;
                }
                int queryIdx = idx / kvNum;
                int queryTaskLen = queryTileSize;
                int queryOffset = queryIdx * queryTileSize;
                if (queryOffset + queryTaskLen > queryLen) {
                    queryTaskLen = queryLen - queryOffset;
                }

                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }

                int queryTaskOffset = queryStart + queryOffset;
#ifdef __DAV_C220_CUBE__
                uint32_t qOffset = queryTaskOffset * nHeads * headDim;
                uint32_t wOffset = queryTaskOffset * (headDim + nHeads) + headDim;
                dbg_printf("block%d: {batch %d, query start loc %u, query [%u - %u), index k [%u - "
                           "%u)} use %d buf\n",
                           blockIdx, batchIdx, queryStart, queryOffset, queryOffset + queryTaskLen,
                           kvOffset, kvOffset + kvLen, curr);
                wait_flag_dev(v2aSyncFlag[curr]);
                RunAicIndexerScores(q[qOffset], weight[wOffset], queryTaskLen, blockTable, kvOffset,
                                    kvLen, scores[curr]);
                ffts_cross_core_sync(PIPE_FIX, a2vSyncFlag[curr]);
#elif defined(__DAV_C220_VEC__)
                int nWork = queryTaskLen;
                int nWorkPerCore = DIV_ROUND_UP(nWork, 2);
                int nWorkCurCore = nWorkPerCore;
                int nWorkStart = subBlockIdx * nWorkCurCore;
                if (nWorkStart + nWorkCurCore > nWork) {
                    nWorkCurCore = nWork - nWorkStart;
                }
                uint32_t outOffset = (queryTaskOffset + nWorkStart) * topK;
                wait_flag_dev(a2vSyncFlag[curr]);
                dbg_printf("block%d subblock%u: {batch %d, query start loc %u, query [%u - %u), "
                           "index k [%u - %u)} use %d buf\n",
                           blockIdx, subBlockIdx, batchIdx, queryStart, queryOffset + nWorkStart,
                           queryOffset + nWorkStart + nWorkCurCore, kvOffset, kvOffset + kvLen,
                           curr);
                RunAivTopk(
                    (__gm__ Dtype *)scores[curr][nWorkStart * tileSizeOfCachedKV].GetPhyAddr(),
                    lastTopk + outOffset * 2, indices, nWorkCurCore, kvOffset, kvLen, totalLen,
                    topK, topkIndices + outOffset);
                ffts_cross_core_sync(PIPE_MTE2, v2aSyncFlag[curr]);
#endif
                curr = 1 - curr;
            }
            queryStart = -1;
        }
#ifdef __DAV_C220_CUBE__
        wait_flag_dev(v2aSyncFlag[0]);
        wait_flag_dev(v2aSyncFlag[1]);
#elif defined(__DAV_C220_VEC__)
        PipeBarrier<PIPE_ALL>();
        if (resetPrevCore) {
            ResetPrevCore();
        }
#endif
    }

private:
    GlobalTensor<Dtype> q;
    GlobalTensor<Dtype> kCache;
    GlobalTensor<Dtype> weight;
    GlobalTensor<Dtype> scores[PINGPONG_BUF_NUM];
    __gm__ int32_t *setNextSync;
    __gm__ int32_t *waitPrevSync;
    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;
    __gm__ int32_t *blockTables;
    __gm__ uint32_t *lastTopk;
    __gm__ uint32_t *indices;
    __gm__ uint32_t *topkIndices;
    uint32_t nHeads;
    uint32_t headDim;
    uint32_t blockSize;
    uint32_t batch;
    uint32_t maxNumBlock;
    uint32_t topK;
    uint32_t tileSizeOfCachedKV;
    int blockIdx;
    int subBlockIdx;
    int nextBlockIdx;
    int prevBlockIdx;
    uint32_t setNextGeneration;
    uint32_t waitPrevGeneration;
    int resetPrevCore;

    LocalTensor<Dtype> kl1Buf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> ql1Buf[PINGPONG_BUF_NUM];   // event 2/3
    LocalTensor<Dtype> wl1Buf;                     // event 4
    LocalTensor<Dtype> kql1Buf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<float> l0cBuf;  // event 0
};

#if defined(XLITE_DEVICE_310P)
#define INDEXER_TOPK_FUNC_DEFINE(dtype)                                                       \
    extern "C" __global__ __aicore__ void indexer_topk_##dtype(                              \
        GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc, GM_ADDR queryLens, \
        GM_ADDR cachedLens, GM_ADDR blockTables, GM_ADDR scores, GM_ADDR lastTopk,           \
        GM_ADDR indices, GM_ADDR topkIndices, GM_ADDR sync, uint32_t nHeads, uint32_t headDim, \
        uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock, uint32_t topK)             \
    {                                                                                         \
    }
#else
#define INDEXER_TOPK_FUNC_DEFINE(dtype)                                                        \
    extern "C" __global__ __aicore__ void indexer_topk_##dtype(                                \
        GM_ADDR q, GM_ADDR kCache, GM_ADDR weight, GM_ADDR queryStartLoc, GM_ADDR queryLens,   \
        GM_ADDR cachedLens, GM_ADDR blockTables, GM_ADDR scores, GM_ADDR lastTopk,             \
        GM_ADDR indices, GM_ADDR topkIndices, GM_ADDR sync, uint32_t nHeads, uint32_t headDim, \
        uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock, uint32_t topK)               \
    {                                                                                          \
        IndexerTopK<dtype> op;                                                                 \
        op.Init(q, kCache, weight, queryStartLoc, queryLens, cachedLens, blockTables, scores,  \
                lastTopk, indices, topkIndices, sync, nHeads, headDim, blockSize, batch,       \
                maxNumBlock, topK);                                                            \
        op.Run();                                                                              \
    }
#endif
