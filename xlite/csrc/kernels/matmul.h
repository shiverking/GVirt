/*
 * @file matmul.h
 *
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef _XLITE_MATMUL_H_
#define _XLITE_MATMUL_H_

#include "kernel_operator.h"
#include "kernel_macro.h"
using namespace AscendC;

// out = ((A * B) + bias) * deqScale
// bias: int32 or float
// deqScale: TF32 格式, 1 符号位, 8 指数位, 10 尾数位, 后 13 位不参与计算
// The entry kernel function must set KERNEL_TASK_TYPE_DEFAULT itself
template <typename Dtype, typename MatDtype, typename OutDtype>
class Matmul
{
public:
    __aicore__ inline Matmul()
    {
    }

    // Compute tile-shape-derived L1/L0 buffer layout, then set the HF32/mask/atomic
    // modes and the initial flags.  Must be called before TaskTilesInit / RunTileByIdx.
    __aicore__ inline void Init(uint64_t m0, uint64_t n0, uint64_t k0, bool hasBias,
                                bool hasDeqScale, uint64_t transpose, uint64_t nz, uint64_t swizzle)
    {
        SetHF32Mode(false);
        SetMaskNorm();
        SetAtomicNone();

        int dtypeBits = GetDTypeBits(Dtype);

        if (m0 == (uint64_t)-1) {
            m0 = 128;
            // The L1 buffer was fully used. When the deqscale need space,
            // the L1 buffer will overflow, so shrink n0.
            n0 = (hasBias || hasDeqScale) ? 128 : 256;
            k0 = 512 * BYTE_BITS / dtypeBits;
        }

        this->m0 = m0;
        this->n0 = n0;
        this->k0 = k0;
        this->hasBias = hasBias;
        this->hasDeqScale = hasDeqScale;
        this->transpose = transpose;
        this->nz = nz;
        this->swizzlCount = static_cast<int32_t>(swizzle >> 8);
        this->swizzleDirection = static_cast<int32_t>(swizzle & 0xFF);

        this->mBlockSize = 16;
        if (transpose) {
            // When the B matrix is the transpose matrix of the int8 data type, the
            // unit of the fractal during the L1 to L0 data conversion is 32*32*1B.
            this->nBlockSize = 32 * BYTE_BITS / dtypeBits;
        } else {
            this->nBlockSize = 16;
        }
        this->kBlockSize = 32 * BYTE_BITS / dtypeBits;

        kDtileSize = k0 << 1;
        kQtileSize = k0 >> 2;

        int l1ATileBytes = m0 * kDtileSize * dtypeBits / BYTE_BITS;
        int l1BTileBytes = n0 * k0 * dtypeBits / BYTE_BITS;
        int l1BiasTileBytes = n0 * sizeof(MatDtype);
        int l1DeqScaleTileBytes = n0 * sizeof(uint64_t);
        int l0ATileBytes = m0 * kQtileSize * dtypeBits / BYTE_BITS;
        int l0BTileBytes = n0 * kQtileSize * dtypeBits / BYTE_BITS;
        uint64_t off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A1);
            l1aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1ATileBytes;
        }
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l1bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B1);
            l1bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BTileBytes;
        }
        if (hasBias) {
            l1BiasBuf.address_.logicPos = static_cast<uint8_t>(TPosition::C1);
            l1BiasBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1BiasTileBytes;
        }
        if (hasDeqScale) {
            l1DeqScaleBuf.address_.logicPos = static_cast<uint8_t>(TPosition::C1);
            l1DeqScaleBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l1DeqScaleTileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0aBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::A2);
            l0aBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0ATileBytes;
        }
        off = 0;
        for (int i = 0; i < PINGPONG_BUF_NUM; i++) {
            l0bBuf[i].address_.logicPos = static_cast<uint8_t>(TPosition::B2);
            l0bBuf[i].address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off += l0BTileBytes;
        }
        off = 0;
        if (hasBias) {
            l0BiasBuf.address_.logicPos = static_cast<uint8_t>(TPosition::C2);
            l0BiasBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off = 0;
        }
        if (hasDeqScale) {
            fixpipeBuf.address_.logicPos = static_cast<uint8_t>(TPosition::C2PIPE2GM);
            fixpipeBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
            off = 0;
        }
        l0cBuf.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0cBuf.address_.bufferAddr = reinterpret_cast<uint64_t>(off);
    }

    __aicore__ inline void SetFlags()
    {
        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    // Bind GM addresses + shape for one GEMM batch and return the number of
    // (m0 x n0) output tiles to dispatch.  Call between the (once) Init and
    // the per-tile RunTileByIdx calls.
    __aicore__ inline int64_t TaskTilesInit(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR bias,
                                            GM_ADDR deqScale, uint64_t m, uint64_t n, uint64_t k,
                                            int xSrcDValue = -1, int zDstDValue = -1)
    {
        this->aGmBuf.SetGlobalBuffer((__gm__ Dtype *)x);
        this->bGmBuf.SetGlobalBuffer((__gm__ Dtype *)y);
        this->biasGmBuf.SetGlobalBuffer((__gm__ MatDtype *)bias);
        this->deqScaleGmBuf.SetGlobalBuffer((__gm__ uint64_t *)deqScale);
        this->cGmBuf.SetGlobalBuffer((__gm__ OutDtype *)z);
        this->m = m;
        this->n = n;
        this->k = k;
        this->srcDValue = xSrcDValue == -1 ? static_cast<int>(k) : xSrcDValue;
        this->dstDValue = zDstDValue == -1 ? static_cast<int>(n) : zDstDValue;
        this->nLoop = DIV_ROUND_UP(n, n0);
        this->mLoop = DIV_ROUND_UP(m, m0);
        return static_cast<int64_t>(nLoop) * mLoop;
    }

    __aicore__ inline void WaitFlags()
    {
        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

    // Execute one output tile.  `idx` is the linear tile index in [0, tileCount)
    // returned by the preceding TaskTilesInit.
    __aicore__ inline void RunTileByIdx(int64_t idx)
    {
        int64_t midx = idx / nLoop;
        int64_t nidx = idx % nLoop;
        GetMNBlockIdx(static_cast<int32_t>(idx), mLoop, nLoop, swizzleDirection, swizzlCount, midx,
                      nidx);
        RunTileBody(midx, nidx);
    }

    // Convenience: TaskTilesInit + tile dispatch + WaitFlags
    __aicore__ inline void Run(GM_ADDR x, GM_ADDR y, GM_ADDR z, GM_ADDR bias, GM_ADDR deqScale,
                               uint64_t m, uint64_t n, uint64_t k, int xSrcDValue = -1,
                               int zDstDValue = -1)
    {
        SetFlags();
        int64_t tiles = TaskTilesInit(x, y, z, bias, deqScale, m, n, k, xSrcDValue, zDstDValue);
        for (int64_t idx = GetBlockIdx(); idx < tiles; idx += GetBlockNum()) {
            RunTileByIdx(idx);
        }
        WaitFlags();
    }

private:
    __aicore__ inline void RunTileBody(int64_t midx, int64_t nidx)
    {
        int kQtileBlockNum = kQtileSize / kBlockSize;
        int kLoop = DIV_ROUND_UP(k, kQtileSize);
        int nStride = ROUND_UP(n, nBlockSize);
        int kStride = ROUND_UP(k, kBlockSize);

        // Ping-pong selectors are tile-local: they start at 0 each tile because the
        // tile exit releases both L1A/L1B buffers (see the class-level invariant).
        int pingpongL1A = 0;
        int pingpongL1B = 0;

        int nOffset = nidx * n0;
        int mOffset = midx * m0;

        int mActual = m0;
        if (mOffset + m0 > m) {
            mActual = m - mOffset;
        }
        int mActualBlockPad = ROUND_UP(mActual, mBlockSize);
        int mActualBlockNum = DIV_ROUND_UP(mActual, mBlockSize);

        int nActual = n0;
        if (nOffset + n0 > n) {
            nActual = n - nOffset;
        }
        int nActualBlockPad = ROUND_UP(nActual, nBlockSize);
        int nActualBlockNum = DIV_ROUND_UP(nActual, nBlockSize);

        GlobalTensor<OutDtype> outGm = cGmBuf[mOffset * dstDValue + nOffset];

        if (hasBias) {
            // Bias GM -> L1
            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
            DataCopy(l1BiasBuf, biasGmBuf[nOffset], nActualBlockPad);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);

            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
            // Bias L1 -> L0
            // C2(Bias Table Buffer) Size is 1KB
            // If dst is in C2(Bias Table Buffer), the size of DataBlock is 64B
            DataCopy(
                l0BiasBuf, l1BiasBuf,
                {1, (uint16_t)(DIV_ROUND_UP((nActualBlockPad * sizeof(MatDtype)), C2_DATABLOCK)), 0,
                 0});

            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID4);
        }
        if (hasDeqScale) {
            // DeqScale GM -> L1
            WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
            DataCopy(l1DeqScaleBuf, deqScaleGmBuf[nOffset], nActualBlockPad);
            SetFlag<HardEvent::MTE2_FIX>(EVENT_ID5);

            // DeqScale L1 -> fbuf
            WaitFlag<HardEvent::MTE2_FIX>(EVENT_ID5);
            // If dst is in C2PIPE2GM(fixpipe Buffer), the size of DataBlock is 128B
            // fixpipe硬件要求：以uint64_t存储fp32，高位为0，低位为fp32格式的二进制值
            // Notice: DataCopy from L1 to fbuf is belong to fixpipe barrier
            DataCopy(
                fixpipeBuf, l1DeqScaleBuf,
                {1,
                 (uint16_t)(DIV_ROUND_UP((nActualBlockPad * sizeof(uint64_t)), FIXPIPE_DATABLOCK)),
                 0, 0});
            PipeBarrier<PIPE_FIX>();
        }

        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        int kOffset = 0;
        for (int kIdx = 0; kIdx < kLoop; kIdx++) {
            int kIdx8 = kIdx % 8;
            int kIdx4 = kIdx % 4;
            int kIdx2 = kIdx % 2;

            /* A GM -> L1 */
            if (kIdx8 == 0) {
                int kRemSize = kDtileSize;
                if (kOffset + kRemSize > k) {
                    kRemSize = k - kOffset;
                }
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                CopyGmToL1Nd2Nz(l1aBuf[pingpongL1A], aGmBuf[mOffset * srcDValue + kOffset], mActual,
                                kRemSize, srcDValue, mActualBlockPad);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
            }

            /* B GM -> L1 */
            int k0ActualBlockNum;
            if (kIdx4 == 0) {
                int kRemSize = k0;
                if (kOffset + kRemSize > k) {
                    kRemSize = k - kOffset;
                }
                k0ActualBlockNum = DIV_ROUND_UP(kRemSize, kBlockSize);
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                if (transpose == 0 && nz == 0) {
                    CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[nOffset * k + kOffset], nActual,
                                    kRemSize, k, nActualBlockPad);
                } else if (transpose == 0 && nz == 1) {
                    CopyGmToL1(l1bBuf[pingpongL1B],
                               bGmBuf[kOffset * nStride + nOffset * kBlockSize], nActual,
                               k0ActualBlockNum, nStride);
                } else if (transpose == 1 && nz == 0) {
                    CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[kOffset * n + nOffset], kRemSize,
                                    nActual, n, ROUND_UP(kRemSize, kBlockSize));
                } else if (transpose == 1 && nz == 1) {
                    CopyGmToL1(l1bBuf[pingpongL1B],
                               bGmBuf[nOffset * kStride + kOffset * nBlockSize], kRemSize,
                               DIV_ROUND_UP(nActual, nBlockSize), kStride);
                }
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
            }

            int kActual = kQtileSize;
            if (kOffset + kActual > k) {
                kActual = k - kOffset;
            }
            int kActualBlockPad = ROUND_UP(kActual, kBlockSize);
            int kActualBlockNum = DIV_ROUND_UP(kActual, kBlockSize);

            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + kIdx2);

            /* A L1 -> L0A */
            if (kIdx8 == 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
            }
            CopyToL0ACol(l0aBuf[kIdx2], l1aBuf[pingpongL1A], mActualBlockNum,
                         kIdx8 * kQtileBlockNum, kActualBlockNum);
            if (kIdx8 == 7 || kIdx == (kLoop - 1)) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                pingpongL1A = 1 - pingpongL1A;
            }

            /* B L1 -> L0B */
            if (kIdx4 == 0) {
                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
            }
            if (transpose) {
                CopyToL0BTCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum,
                              kIdx4 * kQtileBlockNum, kActualBlockNum, k0ActualBlockNum);
            } else {
                CopyToL0BCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum,
                             kIdx4 * kQtileBlockNum, kActualBlockNum);
            }
            if (kIdx4 == 3 || kIdx == (kLoop - 1)) {
                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                pingpongL1B = 1 - pingpongL1B;
            }

            /* Mmad L0A L0B -> L0C */
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + kIdx2);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + kIdx2);

            PipeBarrier<PIPE_M>();
            if (hasBias && kIdx == 0) {
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID4);
                CalMmadWithBias(l0cBuf, l0aBuf[kIdx2], l0bBuf[kIdx2], l0BiasBuf, mActualBlockPad,
                                nActualBlockPad, kActualBlockPad);
                SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
            } else {
                CalMmad(l0cBuf, l0aBuf[kIdx2], l0bBuf[kIdx2], mActualBlockPad, nActualBlockPad,
                        kActualBlockPad, kIdx == 0);
            }
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + kIdx2);
            kOffset += kActual;
        }

        /* C L0C -> GM */
        SetFlag<HardEvent::M_FIX>(EVENT_ID0);
        WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
        CopyToGmWithDequant(outGm, l0cBuf, mActual, nActual, mActualBlockPad, dstDValue,
                            hasDeqScale, fixpipeBuf);
        if (hasDeqScale) {
            SetFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
        }
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);
    }

    __aicore__ inline void GetMNBlockIdx(int32_t loopIdx, int32_t mLoop, int32_t nLoop,
                                         int32_t swizzlDirection, int32_t swizzlCount,
                                         int64_t &mIdx, int64_t &nIdx)
    {
        // Adjust the traversal order of matmul block by setting swizzlCount(upper 8 bits) and
        // swizzleDirection(lower 8 bits). Adjusting the swizzle strategy helps improve cache hit
        // rate and reduce data read overhead, thereby enhancing the overall computational
        // efficiency of matmul.
        uint32_t tileBlockLoop, tileBlockIdx, inTileBlockIdx;
        uint32_t inBatchIdx = loopIdx % (mLoop * nLoop);
        if (swizzlDirection == 0) {  // Zn
            tileBlockLoop = (mLoop + swizzlCount - 1) / swizzlCount;
            tileBlockIdx = inBatchIdx / (swizzlCount * nLoop);
            inTileBlockIdx = inBatchIdx % (swizzlCount * nLoop);

            uint32_t nRow = swizzlCount;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nRow = mLoop - swizzlCount * tileBlockIdx;
            }
            mIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % nRow;
            nIdx = inTileBlockIdx / nRow;
            if (tileBlockIdx % 2 != 0) {
                nIdx = nLoop - nIdx - 1;
            }
        } else if (swizzlDirection == 1) {  // Nz
            tileBlockLoop = (nLoop + swizzlCount - 1) / swizzlCount;
            tileBlockIdx = inBatchIdx / (swizzlCount * mLoop);
            inTileBlockIdx = inBatchIdx % (swizzlCount * mLoop);

            uint32_t nCol = swizzlCount;
            if (tileBlockIdx == tileBlockLoop - 1) {
                nCol = nLoop - swizzlCount * tileBlockIdx;
            }
            mIdx = inTileBlockIdx / nCol;
            nIdx = tileBlockIdx * swizzlCount + inTileBlockIdx % nCol;
            if (tileBlockIdx % 2 != 0) {
                mIdx = mLoop - mIdx - 1;
            }
        }
    }

#if 0  // Superseded by the upstream TaskTilesInit/RunTileByIdx implementation.
    __aicore__ inline void Run()
    {
        int kQtileBlockNum = kQtileSize / kBlockSize;
        int kLoop = DIV_ROUND_UP(k, kQtileSize);
        int nStride = ROUND_UP(n, nBlockSize);
        int kStride = ROUND_UP(k, kBlockSize);

        int pingpongL1A = 0;
        int pingpongL1B = 0;

        SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID1);
        SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        SetFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
        SetFlag<HardEvent::FIX_M>(EVENT_ID0);

        for (int32_t loopIdx = firstCore; loopIdx < coreLoop; loopIdx += GetBlockNum()) {
            int64_t midx = loopIdx / nLoop;
            int64_t nidx = loopIdx % nLoop;
            GetMNBlockIdx(loopIdx, mLoop, nLoop, swizzleDirection, swizzlCount, midx, nidx);
            int nOffset = nidx * n0;
            int mOffset = midx * m0;

            int mActual = m0;
            if (mOffset + m0 > m) {
                mActual = m - mOffset;
            }
            int mActualBlockPad = ROUND_UP(mActual, mBlockSize);
            int mActualBlockNum = DIV_ROUND_UP(mActual, mBlockSize);

            int nActual = n0;
            if (nOffset + n0 > n) {
                nActual = n - nOffset;
            }
            int nActualBlockPad = ROUND_UP(nActual, nBlockSize);
            int nActualBlockNum = DIV_ROUND_UP(nActual, nBlockSize);

            GlobalTensor<OutDtype> outGm = cGmBuf[mOffset * dstDValue + nOffset];

            if (hasBias) {
                // Bias GM -> L1
                WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
                DataCopy(l1BiasBuf, biasGmBuf[nOffset], nActualBlockPad);
                SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);

                WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID4);
                WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
                // Bias L1 -> L0
                // C2(Bias Table Buffer) Size is 1KB
                // If dst is in C2(Bias Table Buffer), the size of DataBlock is 64B
                DataCopy(
                    l0BiasBuf, l1BiasBuf,
                    {1,
                     (uint16_t)(DIV_ROUND_UP((nActualBlockPad * sizeof(MatDtype)), C2_DATABLOCK)),
                     0, 0});

                SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
                SetFlag<HardEvent::MTE1_M>(EVENT_ID4);
            }
            if (hasDeqScale) {
                // DeqScale GM -> L1
                WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
                DataCopy(l1DeqScaleBuf, deqScaleGmBuf[nOffset], nActualBlockPad);
                SetFlag<HardEvent::MTE2_FIX>(EVENT_ID5);

                // DeqScale L1 -> fbuf
                WaitFlag<HardEvent::MTE2_FIX>(EVENT_ID5);
                // If dst is in C2PIPE2GM(fixpipe Buffer), the size of DataBlock is 128B
                // fixpipe硬件要求：以uint64_t存储fp32，高位为0，低位为fp32格式的二进制值
                // Notice: DataCopy from L1 to fbuf is belong to fixpipe barrier
                DataCopy(fixpipeBuf, l1DeqScaleBuf,
                         {1,
                          (uint16_t)(DIV_ROUND_UP((nActualBlockPad * sizeof(uint64_t)),
                                                  FIXPIPE_DATABLOCK)),
                          0, 0});
#if !defined(XLITE_DEVICE_310P)
                PipeBarrier<PIPE_FIX>();
#endif
            }

            WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
            int kOffset = 0;
            for (int kIdx = 0; kIdx < kLoop; kIdx++) {
                int kIdx8 = kIdx % 8;
                int kIdx4 = kIdx % 4;
                int kIdx2 = kIdx % 2;

                /* A GM -> L1 */
                if (kIdx8 == 0) {
                    int kRemSize = kDtileSize;
                    if (kOffset + kRemSize > k) {
                        kRemSize = k - kOffset;
                    }
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    CopyGmToL1Nd2Nz(l1aBuf[pingpongL1A], aGmBuf[mOffset * srcDValue + kOffset],
                                    mActual, kRemSize, srcDValue, mActualBlockPad);
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }

                /* B GM -> L1 */
                int k0ActualBlockNum;
                if (kIdx4 == 0) {
                    int kRemSize = k0;
                    if (kOffset + kRemSize > k) {
                        kRemSize = k - kOffset;
                    }
                    k0ActualBlockNum = DIV_ROUND_UP(kRemSize, kBlockSize);
                    WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    if (transpose == 0 && nz == 0) {
                        CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[nOffset * k + kOffset], nActual,
                                        kRemSize, k, nActualBlockPad);
                    } else if (transpose == 0 && nz == 1) {
                        CopyGmToL1(l1bBuf[pingpongL1B],
                                   bGmBuf[kOffset * nStride + nOffset * kBlockSize], nActual,
                                   k0ActualBlockNum, nStride);
                    } else if (transpose == 1 && nz == 0) {
                        CopyGmToL1Nd2Nz(l1bBuf[pingpongL1B], bGmBuf[kOffset * n + nOffset],
                                        kRemSize, nActual, n, ROUND_UP(kRemSize, kBlockSize));
                    } else if (transpose == 1 && nz == 1) {
                        CopyGmToL1(l1bBuf[pingpongL1B],
                                   bGmBuf[nOffset * kStride + kOffset * nBlockSize], kRemSize,
                                   DIV_ROUND_UP(nActual, nBlockSize), kStride);
                    }
                    SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                }

                int kActual = kQtileSize;
                if (kOffset + kActual > k) {
                    kActual = k - kOffset;
                }
                int kActualBlockPad = ROUND_UP(kActual, kBlockSize);
                int kActualBlockNum = DIV_ROUND_UP(kActual, kBlockSize);

                WaitFlag<HardEvent::M_MTE1>(EVENT_ID0 + kIdx2);

                /* A L1 -> L0A */
                if (kIdx8 == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0 + pingpongL1A);
                }
                CopyToL0ACol(l0aBuf[kIdx2], l1aBuf[pingpongL1A], mActualBlockNum,
                             kIdx8 * kQtileBlockNum, kActualBlockNum);
                if (kIdx8 == 7 || kIdx == (kLoop - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0 + pingpongL1A);
                    pingpongL1A = 1 - pingpongL1A;
                }

                /* B L1 -> L0B */
                if (kIdx4 == 0) {
                    WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID2 + pingpongL1B);
                }
                if (transpose) {
                    CopyToL0BTCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum,
                                  kIdx4 * kQtileBlockNum, kActualBlockNum, k0ActualBlockNum);
                } else {
                    CopyToL0BCol(l0bBuf[kIdx2], l1bBuf[pingpongL1B], nActualBlockNum,
                                 kIdx4 * kQtileBlockNum, kActualBlockNum);
                }
                if (kIdx4 == 3 || kIdx == (kLoop - 1)) {
                    SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID2 + pingpongL1B);
                    pingpongL1B = 1 - pingpongL1B;
                }

                /* Mmad L0A L0B -> L0C */
                SetFlag<HardEvent::MTE1_M>(EVENT_ID0 + kIdx2);
                WaitFlag<HardEvent::MTE1_M>(EVENT_ID0 + kIdx2);

                PipeBarrier<PIPE_M>();
                if (hasBias && kIdx == 0) {
                    WaitFlag<HardEvent::MTE1_M>(EVENT_ID4);
                    CalMmadWithBias(l0cBuf, l0aBuf[kIdx2], l0bBuf[kIdx2], l0BiasBuf,
                                    mActualBlockPad, nActualBlockPad, kActualBlockPad);
                    SetFlag<HardEvent::M_MTE1>(EVENT_ID4);
                } else {
                    CalMmad(l0cBuf, l0aBuf[kIdx2], l0bBuf[kIdx2], mActualBlockPad, nActualBlockPad,
                            kActualBlockPad, kIdx == 0);
                }
                SetFlag<HardEvent::M_MTE1>(EVENT_ID0 + kIdx2);
                kOffset += kActual;
            }

            /* C L0C -> GM */
            SetFlag<HardEvent::M_FIX>(EVENT_ID0);
            WaitFlag<HardEvent::M_FIX>(EVENT_ID0);
            CopyToGmWithDequant(outGm, l0cBuf, mActual, nActual, mActualBlockPad, dstDValue,
                                hasDeqScale, fixpipeBuf);
            if (hasDeqScale) {
                SetFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
            }
            SetFlag<HardEvent::FIX_M>(EVENT_ID0);
        }  // M * N

        WaitFlag<HardEvent::FIX_M>(EVENT_ID0);
        WaitFlag<HardEvent::FIX_MTE2>(EVENT_ID5);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID4);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID3);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID2);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID1);
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID4);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID1);
        WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
    }

#endif
private:
    GlobalTensor<Dtype> aGmBuf;
    GlobalTensor<Dtype> bGmBuf;
    GlobalTensor<OutDtype> cGmBuf;
    GlobalTensor<MatDtype> biasGmBuf;
    GlobalTensor<uint64_t> deqScaleGmBuf;
    LocalTensor<Dtype> l1aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l1bBuf[PINGPONG_BUF_NUM];
    LocalTensor<MatDtype> l1BiasBuf;
    LocalTensor<uint64_t> l1DeqScaleBuf;
    LocalTensor<Dtype> l0aBuf[PINGPONG_BUF_NUM];
    LocalTensor<Dtype> l0bBuf[PINGPONG_BUF_NUM];
    LocalTensor<MatDtype> l0BiasBuf;
    LocalTensor<uint64_t> fixpipeBuf;
    LocalTensor<MatDtype> l0cBuf;

    uint64_t m;
    uint64_t n;
    uint64_t k;
    uint64_t nz;
    uint64_t transpose;
    uint64_t m0;
    uint64_t n0;
    uint64_t k0;
    uint64_t mBlockSize;
    uint64_t nBlockSize;
    uint64_t kBlockSize;
    int srcDValue;
    int dstDValue;
    int kDtileSize;
    int kQtileSize;
    int32_t swizzlCount;
    int32_t swizzleDirection;
    int mLoop;
    int nLoop;
    bool hasBias = false;
    bool hasDeqScale = false;
};

#if defined(XLITE_DEVICE_310P)
#define MATMUL_FUNC_DEFINE(dtype)                                                                 \
    extern "C" __global__ __aicore__ void matmul_##dtype(                                        \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t m, uint64_t n, uint64_t k, uint64_t nz,         \
        uint64_t transpose, uint64_t m0, uint64_t n0, uint64_t k0, uint64_t swizzl, GM_ADDR bias, \
        GM_ADDR deqScale)                                                                         \
    {                                                                                             \
    }
#else
#define MATMUL_FUNC_DEFINE(dtype)                                                                  \
    extern "C" __global__ __aicore__ void matmul_##dtype(                                          \
        GM_ADDR x, GM_ADDR y, GM_ADDR z, uint64_t m, uint64_t n, uint64_t k, uint64_t nz,          \
        uint64_t transpose, uint64_t m0, uint64_t n0, uint64_t k0, uint64_t swizzl, GM_ADDR bias,  \
        GM_ADDR deqScale)                                                                          \
    {                                                                                              \
        KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);                                            \
        if constexpr (std::is_same<dtype, int8_t>::value || std::is_same<dtype, int4b_t>::value) { \
            Matmul<dtype, int32_t, half> op;                                                       \
            op.Init(m0, n0, k0, bias != nullptr, deqScale != nullptr, transpose, nz, swizzl);      \
            op.Run(x, y, z, bias, deqScale, m, n, k);                                              \
        } else {                                                                                   \
            Matmul<dtype, float, dtype> op;                                                        \
            op.Init(m0, n0, k0, bias != nullptr, deqScale != nullptr, transpose, nz, swizzl);      \
            op.Run(x, y, z, bias, deqScale, m, n, k);                                              \
        }                                                                                          \
    }
#endif

#endif
