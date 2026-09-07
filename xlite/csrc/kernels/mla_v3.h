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

#define MBLOCKSIZE 16
#define NBLOCKSIZE 16
#define SEQLEN_64 64
#define SEQLEN_12K 12288
#define SEQLEN_20K 20480
#define SEQLEN_24K 24576
#define SEQLEN_30K 30720
#define SEQLEN_48K 49152
#define SEQLEN_60K 61440
#define SEQLEN_96K 98304

template <typename Dtype>
class MLAV3
{
public:
    __aicore__ inline MLAV3()
    {
    }

    __aicore__ inline void Init(GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kDenseCache,
                                GM_ADDR peDenseCache, GM_ADDR qk, GM_ADDR oAbsorb,
                                GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,
                                uint32_t nHeads, uint32_t ropeHeadDim, uint32_t kvLoraRank,
                                uint32_t batch, uint32_t indexTopK, float scale)
    {
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_MIX_AIC_1_2);
        this->qAbsorb.SetGlobalBuffer((__gm__ Dtype *)qAbsorb);
        this->qr.SetGlobalBuffer((__gm__ Dtype *)qr);
        this->kDenseCache.SetGlobalBuffer((__gm__ Dtype *)kDenseCache);
        this->peDenseCache.SetGlobalBuffer((__gm__ Dtype *)peDenseCache);
        this->oAbsorb.SetGlobalBuffer((__gm__ Dtype *)oAbsorb);

        this->queryStartLoc = (__gm__ int32_t *)queryStartLoc;
        this->queryLens = (__gm__ int32_t *)queryLens;
        this->cachedLens = (__gm__ int32_t *)cachedLens;

        this->nHeads = nHeads;
        this->ropeHeadDim = ropeHeadDim;
        this->kvLoraRank = kvLoraRank;
        this->batch = batch;
        this->indexTopK = indexTopK;
        this->maxSeqLen = indexTopK;
        this->scale = scale;
        this->topK = 0;
        this->qkStride = this->maxSeqLen;

        this->qk[0].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride);
        this->qk[1].SetGlobalBuffer((__gm__ Dtype *)qk + block_idx * XLITE_MAX_M0 * qkStride +
                                    block_num * XLITE_MAX_M0 * qkStride);

        qkn0 = 128;
        qkk0 = 128;
        svn0 = 256;
        svk0 = 64;
        uint64_t off = 0;

        // QK
        off = 0;
        uint64_t qnSize = XLITE_MAX_M0 * kvLoraRank * sizeof(Dtype);
        aqnl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqnl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qnSize;

        uint64_t qrSize = XLITE_MAX_M0 * ropeHeadDim * sizeof(Dtype);
        aqrl1aBuf.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        aqrl1aBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
        off += qrSize;

        uint64_t kSize = qkn0 * 4 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += kSize;
        }

        uint64_t krSize = qkn0 * ropeHeadDim * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            akrl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            akrl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += krSize;
        }
        uint64_t total_qk = off;
        dbg_printf("QK buf: qnSize %lu, qrSize %lu, kSize %lu x 2, krSize %lu x 2, total %lu\n",
                   qnSize, qrSize, kSize, krSize, total_qk);

        // l0
        off = 0;
        uint64_t l0aSize = XLITE_MAX_M0 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            qkl0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0aSize;
        }

        off = 0;
        uint64_t l0bSize = qkn0 * qkk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            qkl0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bSize;
        }

        off = 0;
        uint64_t l0cSize = XLITE_MAX_M0 * qkn0 * sizeof(float);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            qkl0cBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
            qkl0cBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0cSize;
        }

        // SV
        off = 0;
        uint64_t qkSize = XLITE_MAX_M0 * 4 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aqkl1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aqkl1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += qkSize;
        }

        uint64_t ktSize = svn0 * 2 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            aktl1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            aktl1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += ktSize;
        }
        uint64_t total_sv = off;
        dbg_printf("SV buf: ktSize %lu x 2, qkSize %lu x 2, total %lu\n", ktSize, qkSize, total_sv);

        // l0
        off = 0;
        l0aSize = XLITE_MAX_M0 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            svl0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            svl0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0aSize;
        }

        off = 0;
        l0bSize = svn0 * svk0 * sizeof(Dtype);
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            svl0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            svl0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0bSize;
        }

        off = 0;
        l0cSize = XLITE_MAX_M0 * svn0 * sizeof(float);
        svl0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        svl0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    /*
     * C = Absorb * K
     *     Absorb: (queryTokens * nHeads, kvLoraRank)
     *     K: (cachedTokens, kvLoraRank)
     *     C: (queryTokens * nHeads, cachedTokens)
     *     m0: XLITE_MAX_M0, n0: qkn0, k0: qkk0
     * R = QR * KR
     *     QR: (queryTokens * nHeads, ropeHeadDim)
     *     K: (cachedTokens, ropeHeadDim)
     *     R: (queryTokens * nHeads, cachedTokens)
     *     m0: XLITE_MAX_M0, n0: qkn0, k0: qkk0
     * QK = C + R
     */
    __aicore__ inline void RunAicQK(GlobalTensor<Dtype> qAbsorb, GlobalTensor<Dtype> qr,
                                    uint32_t queryTaskLen, uint32_t calcLen, GlobalTensor<Dtype> qk,
                                    GlobalTensor<Dtype> kCache, GlobalTensor<Dtype> peCache)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mSize = queryTaskLen * nHeads;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = qkn0;
        int nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
        int nBlockNum = nBlockPad / NBLOCKSIZE;
        int nLoop = DIV_ROUND_UP(calcLen, qkn0);
        int kSize = qkk0;
        int kBlockPad = qkk0;
        int kBlockNum = qkk0 / kBlockSize;
        int kLoop = DIV_ROUND_UP(kvLoraRank, qkk0);

        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        // copy Absorb (queryTokens * nHeads, kvLoraRank) to L1
        CopyGmToL1Nd2Nz(aqnl1aBuf, qAbsorb, mSize, kvLoraRank, kvLoraRank, mBlockPad);
        // copy QR (queryTokens * nHeads, ropeHeadDim) to L1
        CopyGmToL1Nd2Nz(aqrl1aBuf, qr, mSize, ropeHeadDim, ropeHeadDim, mBlockPad);
        SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

        int curr = 0;
        int pingpongL1B = 0;
        int pingpongL0C = 0;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        SetFlag<HardEvent::FIX_M>(EVENT_ID1);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // calcLen
            // dense cache: physical block id == nIdx (contiguous layout)
            uint32_t block = nIdx;
            int nOffset = nIdx * qkn0;
            if (nOffset + nSize > calcLen) {
                nSize = calcLen - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0 + pingpongL0C);

            kSize = qkk0;
            kBlockPad = qkk0;
            kBlockNum = qkk0 / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // kvLoraRank
                int kIdx4 = kIdx % 4;
                int kOffset = kIdx * qkk0;
                if (kOffset + kSize > kvLoraRank) {
                    kSize = kvLoraRank - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }
                // copy K (qkn0, 4 * qkk0) to L1
                if (kIdx4 == 0) {
                    int kRemSize = 4 * qkk0;
                    if (kOffset + kRemSize > kvLoraRank) {
                        kRemSize = kvLoraRank - kOffset;
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    CopyGmToL1Nd2Nz(akl1bBuf[pingpongL1B],
                                    kCache[block * qkn0 * kvLoraRank + kOffset], nSize, kRemSize,
                                    kvLoraRank, nBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                }

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0BCol(qkl0bBuf[curr], akl1bBuf[pingpongL1B], nBlockNum,
                             kIdx4 * qkk0 / kBlockSize, kBlockNum);
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    pingpongL1B ^= 1;
                }
                CopyToL0ACol(qkl0aBuf[curr], aqnl1aBuf[kOffset * mBlockPad], mBlockNum, 0,
                             kBlockNum);

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad C (queryTokens * nHeads, qkn0) = Absorb * K
                CalMmad(qkl0cBuf[pingpongL0C], qkl0aBuf[curr], qkl0bBuf[curr], mBlockPad, nBlockPad,
                        kBlockPad, kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            if (kLoop % 4 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                pingpongL1B ^= 1;
            }

            kSize = ropeHeadDim;
            kBlockPad = ROUND_UP(kSize, kBlockSize);
            kBlockNum = kBlockPad / kBlockSize;
            // copy KR (qkn0, ropeHeadDim) to L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);
            CopyGmToL1Nd2Nz(akrl1bBuf[curr], peCache[block * qkn0 * ropeHeadDim], nSize,
                            ropeHeadDim, ropeHeadDim, nBlockPad);

            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + curr);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            CopyToL0ACol(qkl0aBuf[curr], aqrl1aBuf, mBlockNum, 0, kBlockNum);
            CopyToL0BCol(qkl0bBuf[curr], akrl1bBuf[curr], nBlockNum, 0, kBlockNum);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + curr);

            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

            // mmad R (queryTokens * nHeads, qkn0) = QR * KR
            CalMmad(qkl0cBuf[pingpongL0C], qkl0aBuf[curr], qkl0bBuf[curr], mBlockPad, nBlockPad,
                    kBlockPad, false);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
            PipeBarrier<PIPE_M>();

            SetFlag<HardEvent::M_FIX>(EVENT_ID0 + pingpongL0C);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0 + pingpongL0C);

            // copy final QK(queryTokens * nHeads, nSize) from L0C to GM
            CopyToGm(qk[nIdx * qkn0], qkl0cBuf[pingpongL0C], mSize, nSize, mBlockPad, qkStride);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0 + pingpongL0C);
            curr = 1 - curr;
            pingpongL0C ^= 1;
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID1);
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
    }

    /*
     * Absorb = QK * K(T)
     *     QK: (queryTokens * nHeads, cachedTokens)
     *     K: (cachedTokens, kvLoraRank)
     *     Absorb: (queryTokens * nHeads, kvLoraRank)
     *     m0: XLITE_MAX_M0, n0: svn0, k0: svk0
     */
    __aicore__ inline void RunAicSV(GlobalTensor<Dtype> qk, uint32_t queryTaskLen, uint32_t calcLen,
                                    GlobalTensor<Dtype> oAbsorb, GlobalTensor<Dtype> kCache)
    {
        constexpr int kBlockSize = 32 / sizeof(Dtype);
        int mSize = queryTaskLen * nHeads;
        int mBlockPad = ROUND_UP(mSize, MBLOCKSIZE);
        int mBlockNum = mBlockPad / MBLOCKSIZE;
        int nSize = svn0;
        int nBlockPad = svn0;
        int nBlockNum = svn0 / NBLOCKSIZE;
        int nLoop = DIV_ROUND_UP(kvLoraRank, svn0);
        int kSize = svk0;
        int kBlockPad = svk0;
        int kBlockNum = svk0 / kBlockSize;
        int kLoop = DIV_ROUND_UP(calcLen, svk0);

        int curr = 0;
        int pingpongL1A = 0;
        int pingpongL1B = 0;
        int L1BkRemBlockPad = ROUND_UP(2 * svk0, kBlockSize);
        int L1BkRemBlockNum = L1BkRemBlockPad / kBlockSize;
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        for (int nIdx = 0; nIdx < nLoop; nIdx++) {  // kvLoraRank
            int nOffset = nIdx * svn0;
            if (nOffset + nSize > kvLoraRank) {
                nSize = kvLoraRank - nOffset;
                nBlockPad = ROUND_UP(nSize, NBLOCKSIZE);
                nBlockNum = nBlockPad / NBLOCKSIZE;
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);

            kSize = svk0;
            kBlockPad = svk0;
            kBlockNum = svk0 / kBlockSize;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {  // calcLen
                int kIdx4 = kIdx % 4;
                int kIdx2 = kIdx % 2;
                int kOffset = kIdx * svk0;
                if (kOffset + kSize > calcLen) {
                    kSize = calcLen - kOffset;
                    kBlockPad = ROUND_UP(kSize, kBlockSize);
                    kBlockNum = kBlockPad / kBlockSize;
                }

                if (kIdx4 == 0) {
                    int kRemSize = 4 * svk0;
                    int kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    if (kOffset + kRemSize > calcLen) {
                        kRemSize = calcLen - kOffset;
                        kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    // copy QK (queryTokens * nHeads, 4 * svk0) to L1
                    CopyGmToL1Nd2Nz(aqkl1aBuf[pingpongL1A], qk[kOffset], mSize, kRemBlockPad,
                                    qkStride, mBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }

                if (kIdx2 == 0) {
                    int kRemSize = 2 * svk0;
                    int kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    if (kOffset + kRemSize > calcLen) {
                        kRemSize = calcLen - kOffset;
                        kRemBlockPad = ROUND_UP(kRemSize, kBlockSize);
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    // copy K(T) (kRemSize, nSize) to L1 — dense cache is contiguous
                    CopyGmToL1Nd2Nz(aktl1bBuf[pingpongL1B], kCache[kOffset * kvLoraRank + nOffset],
                                    kRemBlockPad, nSize, kvLoraRank, L1BkRemBlockPad);

                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4 + pingpongL1B);
                }

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                CopyToL0ACol(svl0aBuf[curr], aqkl1aBuf[pingpongL1A], mBlockNum,
                             kIdx4 * svk0 / kBlockSize, kBlockNum);
                if (kIdx4 == 3) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A ^= 1;
                }

                CopyToL0BTCol(svl0bBuf[curr], aktl1bBuf[pingpongL1B], nBlockNum,
                              kIdx2 * svk0 / kBlockSize, kBlockNum, L1BkRemBlockNum);
                if (kIdx2 == 1) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                    pingpongL1B ^= 1;
                }

                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + curr);

                // mmad Absorb (queryTokens, nSize) = QK * K（T）
                CalMmad(svl0cBuf, svl0aBuf[curr], svl0bBuf[curr], mBlockPad, nBlockPad, kBlockPad,
                        kIdx == 0);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + curr);
                PipeBarrier<PIPE_M>();
                curr = 1 - curr;
            }
            if (kLoop % 4 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                pingpongL1A ^= 1;
            }
            if (kLoop % 2 != 0) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4 + pingpongL1B);
                pingpongL1B ^= 1;
            }
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);

            // copy Absorb (queryTokens * nHeads, nSize) to GM
            CopyToGm(oAbsorb[nOffset], svl0cBuf, mSize, nSize, mBlockPad, kvLoraRank);
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
    }

    /*
     * When the totalLen exceeds MAX_SUB_CONTEXT_SIZE in the RunAivSoftmaxLong function,
     * 4 rows must be reserved for the exp buffer (float type) used in softmax.
     * Therefore, m0 must be less than or equal to XLITE_MAX_M0 - 4.
     */
    __aicore__ inline uint32_t GetOptimalM0(int queryLen, int cachedLen)
    {
        if (queryLen <= SEQLEN_64) {
            return 16;
        } else {
            int totalLen = queryLen + cachedLen;
            if (totalLen <= SEQLEN_12K) {
                return 128;
            } else if (totalLen <= SEQLEN_20K) {
                return 112;
            } else if (totalLen <= SEQLEN_24K) {
                return 96;
            } else if (totalLen <= SEQLEN_30K) {
                return 80;
            } else if (totalLen <= SEQLEN_48K) {
                return 64;
            } else if (totalLen <= SEQLEN_60K) {
                return 48;
            } else if (totalLen <= SEQLEN_96K) {
                return 32;
            } else {
                return 16;
            }
        }
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
        GlobalTensor<Dtype> lastKCache;

        int needDoSV = 0;
        int totalIdx = 0;
        int curr = 0;
        int queryStart = -1;
        int cachedLen = -1;
        int coreOffset = 0;
        for (int batchIdx = 0; batchIdx < batch; batchIdx++) {
            int queryLen = queryLens[batchIdx];

            if (cachedLen < 0) {
                cachedLen = cachedLens[batchIdx];
            }

            // per-batch dense cache subview (batch b starts at b * indexTopK tokens)
            GlobalTensor<Dtype> kCache = kDenseCache[batchIdx * indexTopK * kvLoraRank];
            GlobalTensor<Dtype> peCache = peDenseCache[batchIdx * indexTopK * ropeHeadDim];

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
                if (calcLen > indexTopK) {
                    calcLen = indexTopK;
                }
                if (queryStart < 0) {
                    queryStart = queryStartLoc[batchIdx];
                }
                int queryTaskOffset = queryStart + queryTaskStart;

                // do queryIdx & (0，nHeads)'s QK
                uint32_t mhOffset = queryTaskOffset * nHeads;
                uint32_t absorbOffset = mhOffset * kvLoraRank;
                uint32_t qrOffset = mhOffset * ropeHeadDim;

                dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), calcLen %d}"
                           " use %d temp buf: QK\n",
                           GetBlockIdx(), batchIdx, queryTaskOffset, queryTaskOffset + queryTaskLen,
                           nHeads, calcLen, curr);
                RunAicQK(qAbsorb[absorbOffset], qr[qrOffset], queryTaskLen, calcLen, qk[curr],
                         kCache, peCache);
#if !defined(XLITE_DEVICE_310P)
                ffts_cross_core_sync(PIPE_FIX, config);
#endif

                if (needDoSV != 0) {
                    // wait vector softmax done
                    wait_flag_dev(1);
                    // do softmax * V
                    dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), calcLen %d}"
                               " use %d temp buf: SV\n",
                               GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                               lastQueryTaskOffset + lastQueryTaskLen, nHeads, lastCalcLen, last);
                    RunAicSV(qk[last], lastQueryTaskLen, lastCalcLen, oAbsorb[lastAbsorbOffset],
                             lastKCache);
                }

                lastBatchIdx = batchIdx;
                lastQueryTaskOffset = queryTaskOffset;
                lastAbsorbOffset = absorbOffset;
                lastQueryTaskLen = queryTaskLen;
                lastCalcLen = calcLen;
                lastKCache = kCache;
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
            dbg_printf("block%d: {batch %d, query [%u - %u), headIdx [0 - %u), calcLen %d}"
                       " use %d temp buf: SV\n",
                       GetBlockIdx(), lastBatchIdx, lastQueryTaskOffset,
                       lastQueryTaskOffset + lastQueryTaskLen, nHeads, lastCalcLen, last);
            RunAicSV(qk[last], lastQueryTaskLen, lastCalcLen, oAbsorb[lastAbsorbOffset],
                     lastKCache);
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
                if (calcLen > indexTopK) {
                    calcLen = indexTopK;
                }
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
                RunAivSoftmax(
                    (__gm__ Dtype *)qk[curr][qkOffset].GetPhyAddr(),
                    m0 > (XLITE_MAX_M0 - 4)
                        ? 0
                        : (__gm__ float *)qk[curr][(m0 + subIdx * 2) * qkStride].GetPhyAddr(),
                    nWorkCurCore, qkStride, calcSoftmaxLen, outN, true, nWorkStart, nHeads, true,
                    scale);

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
    GlobalTensor<Dtype> kDenseCache;
    GlobalTensor<Dtype> peDenseCache;
    GlobalTensor<Dtype> oAbsorb;
    GlobalTensor<Dtype> qk[PINGPONG_BUF_NUM];

    LocalTensor<Dtype> aqnl1aBuf;                    // event 0
    LocalTensor<Dtype> aqrl1aBuf;                    // event 0
    LocalTensor<Dtype> akl1bBuf[PINGPONG_BUF_NUM];   // event 4/5
    LocalTensor<Dtype> akrl1bBuf[PINGPONG_BUF_NUM];  // event 2/3
    LocalTensor<Dtype> aqkl1aBuf[PINGPONG_BUF_NUM];  // event 0/1
    LocalTensor<Dtype> aktl1bBuf[PINGPONG_BUF_NUM];  // event 4/5
    LocalTensor<Dtype> qkl0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> qkl0bBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<float> qkl0cBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> svl0aBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<Dtype> svl0bBuf[PINGPONG_BUF_NUM];   // event 0/1
    LocalTensor<float> svl0cBuf;                     // event 0

    __gm__ int32_t *queryStartLoc;
    __gm__ int32_t *queryLens;
    __gm__ int32_t *cachedLens;

    uint32_t nHeads;
    uint32_t ropeHeadDim;
    uint32_t kvLoraRank;
    uint32_t batch;
    uint32_t indexTopK;
    uint32_t maxSeqLen;
    float scale;
    uint32_t topK;
    uint32_t qkStride;
    int qkn0;
    int qkk0;
    int svn0;
    int svk0;
};

#if defined(XLITE_DEVICE_310P)
#define MLA_V3_FUNC_DEFINE(dtype)                                                             \
    extern "C" __global__ __aicore__ void mla_v3_##dtype(                                    \
        GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kDenseCache, GM_ADDR peDenseCache, GM_ADDR qk,  \
        GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,       \
        uint32_t nHeads, uint32_t ropeHeadDim, uint32_t kvLoraRank, uint32_t batch,          \
        uint32_t indexTopK, float scale)                                                     \
    {                                                                                        \
    }
#else
#define MLA_V3_FUNC_DEFINE(dtype)                                                              \
    extern "C" __global__ __aicore__ void mla_v3_##dtype(                                      \
        GM_ADDR qAbsorb, GM_ADDR qr, GM_ADDR kDenseCache, GM_ADDR peDenseCache, GM_ADDR qk,    \
        GM_ADDR oAbsorb, GM_ADDR queryStartLoc, GM_ADDR queryLens, GM_ADDR cachedLens,         \
        uint32_t nHeads, uint32_t ropeHeadDim, uint32_t kvLoraRank, uint32_t batch,            \
        uint32_t indexTopK, float scale)                                                       \
    {                                                                                          \
        MLAV3<dtype> op;                                                                       \
        op.Init(qAbsorb, qr, kDenseCache, peDenseCache, qk, oAbsorb, queryStartLoc, queryLens, \
                cachedLens, nHeads, ropeHeadDim, kvLoraRank, batch, indexTopK, scale);         \
        op.Run();                                                                              \
    }
#endif
