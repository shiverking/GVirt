/*
 * @file kernel_macro.h
 *
 * Copyright (C) 2025-2026. Huawei Technologies Co., Ltd. All rights reserved.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
 */
#ifndef _XLITE_KERNEL_MACRO_H_
#define _XLITE_KERNEL_MACRO_H_

#include "kernel_operator.h"
#ifdef CCE_STUB
#include "cce_stub.h"
#endif

// CANN's 310P compiler defines __NPU_ARCH__ for every AscendC target, including
// compatibility targets that intentionally do not receive XLITE_ARCH_310P.
// Keep device-header feature checks tied to the real compiler architecture so
// common/BF16 stub translation units do not parse unsupported BF16/FixPipe APIs.
#if defined(XLITE_ARCH_310P) || (defined(__NPU_ARCH__) && (__NPU_ARCH__ == 2002))
#define XLITE_DEVICE_310P 1
#endif
using namespace AscendC;

#define ROUND_DOWN(x, y) (((x) / (y)) * (y))
#define ROUND_UP(x, y) ((((x) + ((y) - 1)) / (y)) * (y))
#define DIV_ROUND_UP(x, y) (((x) + ((y) - 1)) / (y))
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

#define BLOCK_SIZE 32
#define VECTOR_MAX_REPEAT 255
#define VECTOR_MAX_BYTESIZE 256      // The maximum byte size of one repeat in vector
#define VECTOR_REPEAT_BYTESIZE 256   // Size per Vector repeat, in bytes; i.e., VECTOR_MAX_BYTESIZE
#define VECTOR_MAX_NUM_OF_FP32 64    // The maximum num of float32 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_FP16 128   // The maximum num of float16 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_BF16 128   // The maximum num of bfloat16 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_INT16 128  // The maximum num of int16 dtype in one vector repeat
#define VECTOR_MAX_NUM_OF_INT8 256   // The maximum num of int8 dtype in one vector repeat
#define AIC_CACHE_LINE_SIZE 512
#define MATMUL_M0_N0_K0_DEFAULT_VALUE ((uint64_t)(-1))
#define GM_UB_BURST_SIZE 16384  // 16KB, recommended burst size for GM <--> UB copy
#define UB_BUF_ALIGN_SIZE 32    // The align size of UB buffer address; i.e., BLOCK_SIZE
#if defined(XLITE_ARCH_310P)
#if !defined(__NPU_ARCH__)
#error "XLITE_ARCH_310P requires the AscendC compiler to define __NPU_ARCH__"
#elif __NPU_ARCH__ != 2002
#error "XLITE_ARCH_310P expected __NPU_ARCH__=2002; refusing a mismatched device target"
#endif
// Conservative 310P POC profile.  Do not reuse the 910B3 bank-conflict
// offset until it has been tuned on silicon.  The working-set limit stays at
// 192 KiB so kernels never depend on capacity above the existing baseline.
#define UB_SIZE 196608
#define UB_BANK_SIZE 4096
#define UB_BANK_NUM 48
#define UB_BANKGROUP_UNUM 16
#define UB_BANKGROUP_BATCH 3
#define UB_BANKGROUP_ROW_SIZE 512
#define UB_BANKGROUP_SIZE 65536
#define UB_BANK_CONFLICT_OFFSET 0
#elif __NPU_ARCH__ != 3510
// the following is architecture specific (NpuArch 2201)
#define UB_SIZE 196608     // 192KB, total size of unified buffer (UB), = UB_BANK_SIZE * UB_BANK_NUM
#define UB_BANK_SIZE 4096  // 4KB, size per UB bank
#define UB_BANK_NUM 48     // The number of UB banks, = UB_BANKGROUP_UNUM * UB_BANKGROUP_BATCH
#define UB_BANKGROUP_UNUM 16         // The number of unique UB bank groups
#define UB_BANKGROUP_BATCH 3         // The number of UB bank group batches
#define UB_BANKGROUP_ROW_SIZE 512    // 512B, UB_BUF_ALIGN_SIZE * UB_BANK_GROUP_NUM
#define UB_BANKGROUP_SIZE 65536      // 64KB, UB_BANK_SIZE * UB_BANK_GROUP_NUM
#define UB_BANK_CONFLICT_OFFSET 256  // recommended offset trick in bytes to avoid UB bank conflict
#else                                // NpuArch 3510
#define UB_SIZE 262144               // 256 KB
#define UB_BANK_SIZE 16384           // 16 KB
#define UB_BANK_NUM 16
#define UB_BANKGROUP_UNUM 8
#define UB_BANKGROUP_BATCH 2
#define UB_BANKGROUP_ROW_SIZE 256
#define UB_BANKGROUP_SIZE 131072   // 128 KB
#define UB_BANK_CONFLICT_OFFSET 0  // UB_BANKGROUP_ROW_SIZE == VECTOR_REPEAT_BYTESIZE
#endif
#define PINGPONG_BUF_NUM 2
#define C2_DATABLOCK 64        // The data block size of C2
#define FIXPIPE_DATABLOCK 128  // The data block size of fixpipe
#define FLOAT_MIN -3.4028235e+38
#define SORT_BLOCK_SIZE 32ull
#define MGR_SORT_VALID_BITS_OFFSET 8
#define MGR_SORT_IF_EXHAUSTED_SUSPENSION_OFFSET 12
#define BYTE_BITS 8

// 设置拷贝数据的config
inline __aicore__ uint64_t __set_dmi_config(uint8_t sid, uint16_t nBurst, uint16_t lenBurst,
                                            uint16_t srcGap, uint16_t dstGap)
{
    uint64_t config = 0;
    const uint64_t _DST_GAP = 48;
    const uint64_t _SRC_GAP = 32;
    const uint64_t _LEN_BURST = 16;
    const uint64_t _N_BURST = 4;

    config |= (uint64_t)(sid & 0xfULL);
    config |= (uint64_t)dstGap << _DST_GAP;
    config |= (uint64_t)srcGap << _SRC_GAP;
    config |= (uint64_t)lenBurst << _LEN_BURST;
    config |= (uint64_t)(nBurst & 0xfffULL) << _N_BURST;

    return config;
}

// 设置vector运算xt（双操作数）
inline __aicore__ uint64_t set_vector_xt(uint64_t repeatDestStride, uint64_t repeatSrc0Stride,
                                         uint64_t repeatSrc1Stride, uint64_t destStride,
                                         uint64_t src0Stride, uint64_t src1Stride, uint64_t repeat)
{
    uint64_t config = 0;
    const uint64_t _DST_STRIDE = 0;
    const uint64_t _SRC0_STRIDE = 8;
    const uint64_t _SRC1_STRIDE = 16;
    const uint64_t _REP_DST_STRIDE = 24;
    const uint64_t _REP_SRC0_STRIDE = 32;
    const uint64_t _REP_SRC1_STRIDE = 40;
    const uint64_t _REPEAT = 56;

    config |= (uint64_t)destStride << _DST_STRIDE;
    config |= (uint64_t)src0Stride << _SRC0_STRIDE;
    config |= (uint64_t)src1Stride << _SRC1_STRIDE;
    config |= (uint64_t)repeatDestStride << _REP_DST_STRIDE;
    config |= (uint64_t)repeatSrc0Stride << _REP_SRC0_STRIDE;
    config |= (uint64_t)repeatSrc1Stride << _REP_SRC1_STRIDE;
    config |= (uint64_t)repeat << _REPEAT;

    return config;
}

// 设置vector运算xt（单操作数）
inline __aicore__ uint64_t set_vector_1src_xt(uint64_t repeatDestStride, uint64_t repeatSrcStride,
                                              uint64_t destStride, uint64_t srcStride,
                                              uint64_t repeat)
{
    uint64_t config = 0;
    const uint64_t _DST_STRIDE = 0;
    const uint64_t _SRC_STRIDE = 16;
    const uint64_t _REP_DST_STRIDE = 32;
    const uint64_t _REP_SRC_STRIDE = 40;
    const uint64_t _REPEAT = 56;

    config |= (uint64_t)destStride << _DST_STRIDE;
    config |= (uint64_t)srcStride << _SRC_STRIDE;
    config |= (uint64_t)repeatDestStride << _REP_DST_STRIDE;
    config |= (uint64_t)repeatSrcStride << _REP_SRC_STRIDE;
    config |= (uint64_t)repeat << _REPEAT;

    return config;
}

template <typename Dtype>
__aicore__ inline void CopyGmToL1Nd2Nz(const LocalTensor<Dtype> &dst,
                                       const GlobalTensor<Dtype> &src, int nValue, int dValue,
                                       int srcDValue, int dstNzC0Stride)
{
    int dtypeBits = std::is_same<Dtype, int4b_t>::value ? 4 : (sizeof(Dtype) * BYTE_BITS);
    srcDValue = std::is_same<Dtype, int4b_t>::value ? srcDValue / 2 : srcDValue;
    dValue = std::is_same<Dtype, int4b_t>::value ? dValue / 2 : dValue;
    if (srcDValue <= 65535) {
        Nd2NzParams nd2nzParams(1 /* NdNum */, nValue, dValue, 0 /* srcNdMatrixStride */, srcDValue,
                                dstNzC0Stride, 1 /* dstNzNStride */, 0 /* dstNzMatrixStride */);
        DataCopy(dst, src, nd2nzParams);
    } else {
        int kBlockSize = 32 * 8 / dtypeBits;
        Nd2NzParams nd2nzParams(1 /* NdNum */, 1, dValue, 0 /* srcNdMatrixStride */, srcDValue,
                                dstNzC0Stride, 1 /* dstNzNStride */, 0 /* dstNzMatrixStride */);
        for (int i = 0; i < nValue; i++) {
            DataCopy(dst[i * kBlockSize], src[i * srcDValue], nd2nzParams);
        }
    }
}

template <typename Dtype>
__aicore__ inline void CopyGmToL1(const LocalTensor<Dtype> &dst, const GlobalTensor<Dtype> &src,
                                  int nValue, int kBlockNum, int nStride)
{
    DataCopyParams repeatParams(kBlockNum, nValue, nStride - nValue, 0);
    DataCopy(dst, src, repeatParams);
}

template <typename Dtype>
__aicore__ inline void CopyToL0ACol(const LocalTensor<Dtype> &dst, const LocalTensor<Dtype> &src,
                                    int mBlockNum, int kBlockStart, int kBlockNum)
{
    int dtypeBits = std::is_same<Dtype, int4b_t>::value ? 4 : (sizeof(Dtype) * BYTE_BITS);
    int cubeSize = 512 * 8 / dtypeBits;
    LoadData2dParams params(0 /* startIndex */, mBlockNum /* repeatTimes */, 1 /* srcStride */,
                            0 /* sid */, kBlockNum - 1 /* dstGap */, 0, inc);
    for (int k = kBlockStart; k < kBlockStart + kBlockNum; k++) {
        LoadData(dst[(k - kBlockStart) * cubeSize], src[k * mBlockNum * cubeSize], params);
    }
}

template <typename Dtype>
__aicore__ inline void CopyToL0BCol(const LocalTensor<Dtype> &dst, const LocalTensor<Dtype> &src,
                                    int nBlockNum, int kBlockStart, int kBlockNum)
{
    int dtypeBits = std::is_same<Dtype, int4b_t>::value ? 4 : (sizeof(Dtype) * BYTE_BITS);
    int cubeSize = 512 * 8 / dtypeBits;
    LoadData2dParams params(0, kBlockNum * nBlockNum, 1, 0, 0, 0, inc);
    LoadData(dst, src[kBlockStart * nBlockNum * cubeSize], params);
}

template <typename Dtype>
__aicore__ inline void CopyToL0BTCol(const LocalTensor<Dtype> &dst, const LocalTensor<Dtype> &src,
                                     int nBlockNum, int kBlockStart, int kBlockNum, int srcStride)
{
    int dtypeBits = std::is_same<Dtype, int4b_t>::value ? 4 : (sizeof(Dtype) * BYTE_BITS);
    if constexpr (std::is_same<Dtype, int8_t>::value || std::is_same<Dtype, int4b_t>::value) {
        int cubeSize = 32 * 8 / dtypeBits;
        cubeSize = cubeSize * cubeSize;
        int dstGap = std::is_same<Dtype, int4b_t>::value ? 3 : 1;
        LoadData2dTransposeParams params(0, nBlockNum, srcStride, dstGap, 0);
        for (int k = kBlockStart; k < kBlockStart + kBlockNum; k++) {
            LoadDataWithTranspose(dst[(k - kBlockStart) * nBlockNum * cubeSize], src[k * cubeSize],
                                  params);
        }
    } else {
        int cubeSize = 512 * 8 / dtypeBits;
        LoadData2dParams params(0, nBlockNum, srcStride, 0, 0, 1, inc);
        for (int k = kBlockStart; k < kBlockStart + kBlockNum; k++) {
            LoadData(dst[(k - kBlockStart) * nBlockNum * cubeSize], src[k * cubeSize], params);
        }
    }
}

template <typename Dtype, typename MatDtype>
__aicore__ inline void CalMmad(const LocalTensor<MatDtype> &c, const LocalTensor<Dtype> &a,
                               const LocalTensor<Dtype> &b, int m, int n, int k, bool init,
                               int unit = 0)
{
    MmadParams params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.cmatrixInitVal = init;
    params.unitFlag = unit;
    Mmad(c, a, b, params);
}

template <typename Dtype, typename MatDtype>
__aicore__ inline void CalMmadWithBias(const LocalTensor<MatDtype> &c, const LocalTensor<Dtype> &a,
                                       const LocalTensor<Dtype> &b,
                                       const LocalTensor<MatDtype> &bias, int m, int n, int k)
{
    MmadParams params;
    params.m = m;
    params.n = n;
    params.k = k;
    params.cmatrixInitVal = false;
    Mmad(c, a, b, bias, params);
}

template <typename Dtype>
__aicore__ inline void CopyL0CToL1(const LocalTensor<Dtype> &dst, const LocalTensor<float> &src,
                                   int mSize, int nSize, int srcStride, int dstStride)
{
    QuantMode_t mode;
    if constexpr (std::is_same<Dtype, float>::value) {
        mode = NoQuant;
    } else if constexpr (std::is_same<Dtype, float16_t>::value) {
        mode = F322F16;
#if !defined(XLITE_DEVICE_310P)
    } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
        mode = F322BF16;
#endif
    }
    DataCopyCO12DstParams param(nSize, mSize, dstStride, srcStride, mode, 0, 0, 0);
    DataCopy(dst, src, param);
}

template <typename Dtype>
inline __aicore__ void CopyToGm(const GlobalTensor<Dtype> &dst, const LocalTensor<float> &src,
                                int mSize, int nSize, int srcStride, int dstStride,
                                uint8_t unitFlag)
{
    QuantMode_t mode;
    if constexpr (std::is_same<Dtype, float>::value) {
        mode = NoQuant;
    } else if constexpr (std::is_same<Dtype, float16_t>::value) {
        mode = F322F16;
#if !defined(XLITE_DEVICE_310P)
    } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
        mode = F322BF16;
#endif
    }
#ifdef __DAV_C220_CUBE__
    set_nd_para(0x1);
    copy_matrix_cc_to_gm((__gm__ Dtype *)dst.GetPhyAddr(), (__cc__ float *)src.GetPhyAddr(),
                         0 /* fixpipeInfo.sid */, nSize, mSize, dstStride, srcStride, unitFlag,
                         mode, 0 /* static_cast<uint8_t>(fixpipeInfo.reluEn) */,
                         0 /* fixpipeInfo.channelSplit */, 1 /* fixpipeInfo.nz2ndEn */);
#endif
}

template <typename Dtype>
__aicore__ inline void CopyToGm(const GlobalTensor<Dtype> &dst, const LocalTensor<float> &src,
                                int mSize, int nSize, int srcStride, int dstStride)
{
    QuantMode_t mode;
    if constexpr (std::is_same<Dtype, float>::value) {
        mode = NoQuant;
    } else if constexpr (std::is_same<Dtype, float16_t>::value) {
        mode = F322F16;
#if !defined(XLITE_DEVICE_310P)
    } else if constexpr (std::is_same<Dtype, bfloat16_t>::value) {
        mode = F322BF16;
#endif
    }
    DataCopyCO12DstParams param(nSize, mSize, dstStride, srcStride, mode, 0, 0, 1);
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(dst, src, param);
}

template <typename OutDtype>
__aicore__ inline void CopyToGm(const GlobalTensor<OutDtype> &dst, const LocalTensor<int32_t> &src,
                                int mSize, int nSize, int srcStride, int dstStride)
{
    QuantMode_t mode = NoQuant;
    if constexpr (std::is_same<OutDtype, float16_t>::value) {
        mode = DEQF16;
        float quant = 1;
        uint64_t deqScalar = static_cast<uint64_t>(*reinterpret_cast<int32_t *>(&quant));
        SetFixpipePreQuantFlag(deqScalar);
#if !defined(XLITE_DEVICE_310P)
        PipeBarrier<PIPE_FIX>();
#endif
    }
    DataCopyCO12DstParams param(nSize, mSize, dstStride, srcStride, mode, 0, 0, 1);
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(dst, src, param);
}

template <typename MatDtype, typename OutDtype>
__aicore__ inline void CopyToGmWithDequant(const GlobalTensor<OutDtype> &dst,
                                           const LocalTensor<MatDtype> &src, int mSize, int nSize,
                                           int srcStride, int dstStride, bool use_dequant,
                                           const LocalTensor<uint64_t> &deqScale)
{
    if (!use_dequant) {
        CopyToGm(dst, src, mSize, nSize, srcStride, dstStride);
        return;
    }

    QuantMode_t mode = NoQuant;
    if constexpr (std::is_same<MatDtype, float>::value) {
        if constexpr (std::is_same<OutDtype, float16_t>::value) {
            mode = F322F16;
#if !defined(XLITE_DEVICE_310P)
        } else if constexpr (std::is_same<OutDtype, bfloat16_t>::value) {
            mode = F322BF16;
#endif
        }
    } else if constexpr (std::is_same<MatDtype, int32_t>::value) {
        if constexpr (std::is_same<OutDtype, half>::value) {
            mode = VDEQF16;
        }
    }
    SetFixPipeConfig(deqScale);
    DataCopyCO12DstParams param(nSize, mSize, dstStride, srcStride, mode, 0, 0, 1);
    SetFixpipeNz2ndFlag(1, 1, 1);
    DataCopy(dst, src, param);
}

inline __aicore__ void DataCacheCleanAndInvalid(__gm__ void *__restrict__ gm)
{
    __asm__ __volatile__("");
    dcci(gm, 0 /*SINGLE_CACHE_LINE*/);
    __asm__ __volatile__("");
}

#if __DAV_C220_VEC__
// make sure that: len != 0
inline __aicore__ void SetMask(int32_t len)
{
    uint64_t temp = len % 64;
    uint64_t mask = ((uint64_t)1 << temp) - 1;

    if (len == 128) {
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    } else if (len >= 64) {
        set_vector_mask(mask, (uint64_t)-1);
    } else {
        set_vector_mask(0x0, mask);
    }
}

// make sure that: len != 0
inline __aicore__ void SetMaskFromHighBit(int32_t high, int32_t len)
{
    uint64_t temp = len % 64;
    uint64_t mask = ~(((uint64_t)1 << (64 - temp)) - 1);

    if (high == 128) {
        // If dtype size is 2 bytes, all 128 bits of mask take effect.
        if (len > 64) {
            set_vector_mask((uint64_t)-1, mask);
        } else {
            set_vector_mask(mask, 0x0);
        }
    } else if (high == 64) {
        // If dtype size is 4 bytes, only lower 64 bits take effect.
        set_vector_mask(0, mask);
    }
}

template <typename Dtype>
__inline__ __aicore__ void ReduceMaxV3(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    constexpr int instPad = VECTOR_MAX_REPEAT * pad;
    set_mask_norm();

    uint32_t repeat = DIV_ROUND_UP(dim, pad);
    if (repeat == 1) {
        SetMask(dim);
        vcmax(dst, src, 1, 1, 1, 8, Order_t::ONLY_VALUE);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        return;
    }

    repeat = dim / pad;
    __ubuf__ Dtype *orgSrc = src;
    __ubuf__ Dtype *tail = src + repeat * pad;
    __ubuf__ Dtype *last;
    int lastValid = 0;
    int remain;
    uint32_t total = repeat;
    while (total > 1) {
        remain = total % 2;
        repeat = total / 2;
        if (lastValid != 0 && remain == 0) {
            repeat = repeat - 1;
        }
        if (repeat > VECTOR_MAX_REPEAT) {
            int instNum = DIV_ROUND_UP(repeat, VECTOR_MAX_REPEAT);
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vmax(dst + i * instPad, src + i * instPad, src + repeat * pad + i * instPad,
                     currRepeat, 1, 1, 1, 8, 8, 8);
            }
        } else {
            vmax(dst, src, src + repeat * pad, repeat, 1, 1, 1, 8, 8, 8);
        }

        if ((lastValid ^ remain) != 0) {
            if (lastValid != 0) {
                vmax(dst + repeat * pad, src + repeat * 2 * pad, last, 1, 1, 1, 1, 8, 8, 8);
                repeat = repeat + 1;
                lastValid = 0;
            } else {
                last = src + ROUND_DOWN(total, 2) * pad;
                lastValid = 1;
            }
        }
        pipe_barrier(PIPE_V);
        total = repeat + remain;
        src = dst;
    }
    remain = dim % pad;
    if (remain != 0) {
        if (src == orgSrc) {
            copy_ubuf_to_ubuf(dst, src, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
            src = dst;
        }
        SetMask(remain);
        vmax(dst, src, tail, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
    vcmax(dst, src, 1, 8, 1, 8, Order_t::ONLY_VALUE);
}

template <typename Dtype>
__inline__ __aicore__ void ReduceSumV3(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    constexpr int instPad = VECTOR_MAX_REPEAT * pad;
    set_mask_norm();

    uint32_t repeat = DIV_ROUND_UP(dim, pad);
    if (repeat == 1) {
        SetMask(dim);
        vcadd(dst, src, 1, 1, 1, 8, 0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        return;
    }

    repeat = dim / pad;
    __ubuf__ Dtype *orgSrc = src;
    __ubuf__ Dtype *tail = src + repeat * pad;
    __ubuf__ Dtype *last;
    int lastValid = 0;
    int remain;
    uint32_t total = repeat;
    while (total > 1) {
        remain = total % 2;
        repeat = total / 2;
        if (lastValid != 0 && remain == 0) {
            repeat = repeat - 1;
        }
        if (repeat > VECTOR_MAX_REPEAT) {
            int instNum = DIV_ROUND_UP(repeat, VECTOR_MAX_REPEAT);
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vadd(dst + i * instPad, src + i * instPad, src + repeat * pad + i * instPad,
                     currRepeat, 1, 1, 1, 8, 8, 8);
            }
        } else {
            vadd(dst, src, src + repeat * pad, repeat, 1, 1, 1, 8, 8, 8);
        }

        if ((lastValid ^ remain) != 0) {
            if (lastValid != 0) {
                vadd(dst + repeat * pad, src + repeat * 2 * pad, last, 1, 1, 1, 1, 8, 8, 8);
                repeat = repeat + 1;
                lastValid = 0;
            } else {
                last = src + ROUND_DOWN(total, 2) * pad;
                lastValid = 1;
            }
        }
        pipe_barrier(PIPE_V);
        total = repeat + remain;
        src = dst;
    }
    remain = dim % pad;
    if (remain != 0) {
        if (src == orgSrc) {
            copy_ubuf_to_ubuf(dst, src, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
            src = dst;
        }
        SetMask(remain);
        vadd(dst, src, tail, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
    vcadd(dst, src, 1, 1, 1, 8, 0);
}

// float16_t: dim <= 32640 (255 * 128), float: dim <= 16320 (255 * 64)
template <typename Dtype>
__inline__ __aicore__ void ReduceMaxV2(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    set_mask_norm();

    uint32_t repeat = DIV_ROUND_UP(dim, pad);
    if (repeat == 1) {
        SetMask(dim);
        vcmax(dst, src, 1, 1, 1, 8, Order_t::ONLY_VALUE);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        return;
    }

    repeat = dim / pad;
    __ubuf__ Dtype *orgSrc = src;
    __ubuf__ Dtype *tail = src + repeat * pad;
    __ubuf__ Dtype *last;
    int lastValid = 0;
    int remain;
    uint32_t total = repeat;
    while (total > 1) {
        remain = total % 2;
        repeat = total / 2;
        if (lastValid != 0 && remain == 0) {
            repeat = repeat - 1;
        }
        vmax(dst, src, src + repeat * pad, repeat, 1, 1, 1, 8, 8, 8);

        if ((lastValid ^ remain) != 0) {
            if (lastValid != 0) {
                vmax(dst + repeat * pad, src + repeat * 2 * pad, last, 1, 1, 1, 1, 8, 8, 8);
                repeat = repeat + 1;
                lastValid = 0;
            } else {
                last = src + ROUND_DOWN(total, 2) * pad;
                lastValid = 1;
            }
        }
        pipe_barrier(PIPE_V);
        total = repeat + remain;
        src = dst;
    }
    remain = dim % pad;
    if (remain != 0) {
        if (src == orgSrc) {
            copy_ubuf_to_ubuf(dst, src, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
            src = dst;
        }
        SetMask(remain);
        vmax(dst, src, tail, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
    vcmax(dst, src, 1, 8, 1, 8, Order_t::ONLY_VALUE);
}

// float16_t: dim <= 32640 (255 * 128), float: dim <= 16320 (255 * 64)
template <typename Dtype>
__inline__ __aicore__ void ReduceSumV2(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    constexpr int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    set_mask_norm();

    uint32_t repeat = DIV_ROUND_UP(dim, pad);
    if (repeat == 1) {
        SetMask(dim);
        vcadd(dst, src, 1, 1, 1, 8, 0);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
        return;
    }

    repeat = dim / pad;
    __ubuf__ Dtype *orgSrc = src;
    __ubuf__ Dtype *tail = src + repeat * pad;
    __ubuf__ Dtype *last;
    int lastValid = 0;
    int remain;
    uint32_t total = repeat;
    while (total > 1) {
        remain = total % 2;
        repeat = total / 2;
        if (lastValid != 0 && remain == 0) {
            repeat = repeat - 1;
        }
        vadd(dst, src, src + repeat * pad, repeat, 1, 1, 1, 8, 8, 8);

        if ((lastValid ^ remain) != 0) {
            if (lastValid != 0) {
                vadd(dst + repeat * pad, src + repeat * 2 * pad, last, 1, 1, 1, 1, 8, 8, 8);
                repeat = repeat + 1;
                lastValid = 0;
            } else {
                last = src + ROUND_DOWN(total, 2) * pad;
                lastValid = 1;
            }
        }
        pipe_barrier(PIPE_V);
        total = repeat + remain;
        src = dst;
    }
    remain = dim % pad;
    if (remain != 0) {
        if (src == orgSrc) {
            copy_ubuf_to_ubuf(dst, src, 0, 1, 8, 1, 1);
            pipe_barrier(PIPE_V);
            src = dst;
        }
        SetMask(remain);
        vadd(dst, src, tail, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask((uint64_t)-1, (uint64_t)-1);
    }
    vcadd(dst, src, 1, 1, 1, 8, 0);
}

template <typename Dtype>
__inline__ __aicore__ void ReduceMax(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    Dtype min;
    uint32_t remain = dim;
    __ubuf__ Dtype *calc = src;

    int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    int instPad = VECTOR_MAX_REPEAT * pad;
    if constexpr (std::is_same<Dtype, half>::value) {
        min = half(-65504);
    } else if constexpr (std::is_same<Dtype, float>::value) {
        min = -3.4028235e+38;
    }

    uint32_t repeat = DIV_ROUND_UP(remain, pad);
    set_mask_norm();
    if (remain == 1) {
        SetMask(remain);
        vcmax(dst, calc, repeat, 1, 1, 8, Order_t::ONLY_VALUE);
        pipe_barrier(PIPE_V);
    }

    while (remain != 1) {
        if (repeat == 1) {
            SetMask(remain);
        } else {
            if (remain % pad != 0) {
                SetMaskFromHighBit(pad, pad - remain % pad);
                vector_dup(calc + ROUND_DOWN(remain, pad), min, 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
            }
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        if (repeat > VECTOR_MAX_REPEAT) {
            int instNum = DIV_ROUND_UP(repeat, VECTOR_MAX_REPEAT);
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vcmax(dst + i * VECTOR_MAX_REPEAT, calc + i * instPad, currRepeat, 1, 1, 8,
                      Order_t::ONLY_VALUE);
            }
        } else {
            vcmax(dst, calc, repeat, 1, 1, 8, Order_t::ONLY_VALUE);
        }
        calc = dst;
        pipe_barrier(PIPE_V);
        remain = repeat;
        repeat = DIV_ROUND_UP(remain, pad);
    }
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

template <typename Dtype>
__inline__ __aicore__ void ReduceSum(__ubuf__ Dtype *dst, __ubuf__ Dtype *src, uint32_t dim)
{
    uint32_t remain = dim;
    __ubuf__ Dtype *calc = src;
    int pad = VECTOR_MAX_BYTESIZE / sizeof(Dtype);
    int instPad = VECTOR_MAX_REPEAT * pad;

    uint32_t repeat = DIV_ROUND_UP(remain, pad);
    set_mask_norm();
    if (remain == 1) {
        SetMask(remain);
        vcadd(dst, calc, repeat, 1, 1, 8, 0);
        pipe_barrier(PIPE_V);
    }

    while (remain != 1) {
        if (repeat == 1) {
            SetMask(remain);
        } else {
            if (remain % pad != 0) {
                SetMaskFromHighBit(pad, pad - remain % pad);
                vector_dup(calc + ROUND_DOWN(remain, pad), Dtype(0), 1, 1, 1, 8, 0);
                pipe_barrier(PIPE_V);
            }
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }

        if (repeat > VECTOR_MAX_REPEAT) {
            int instNum = DIV_ROUND_UP(repeat, VECTOR_MAX_REPEAT);
            for (int i = 0; i < instNum; i++) {
                int currRepeat = VECTOR_MAX_REPEAT;
                if (currRepeat + i * VECTOR_MAX_REPEAT > repeat) {
                    currRepeat = repeat - i * VECTOR_MAX_REPEAT;
                }
                vcadd(dst + i * VECTOR_MAX_REPEAT, calc + i * instPad, currRepeat, 1, 1, 8, 0);
            }
        } else {
            vcadd(dst, calc, repeat, 1, 1, 8, 0);
        }
        calc = dst;
        pipe_barrier(PIPE_V);
        remain = repeat;
        repeat = DIV_ROUND_UP(remain, pad);
    }
    set_vector_mask((uint64_t)-1, (uint64_t)-1);
}

__inline__ __aicore__ void MrgSort(__ubuf__ float *sort0, __ubuf__ float *sort1, int vbitsortRepeat,
                                   int *dstBufIdx)
{
    __ubuf__ float *mrgSortBuf[2] = {sort0, sort1};
    constexpr float min = FLOAT_MIN;
    int pad = VECTOR_MAX_BYTESIZE / sizeof(float);
    uint64_t len = SORT_BLOCK_SIZE;
    int cnt = 4;
    int sortRepeat = vbitsortRepeat;
    int bufIdx = 0;
    uint64_t validBit = 0xF;
    while (sortRepeat > 1) {
        __ubuf__ float *addr = mrgSortBuf[bufIdx];
        if (sortRepeat <= 2) {
            cnt = 2;
            validBit = 0x3;
        } else if (sortRepeat <= 3) {
            cnt = 3;
            validBit = 0x7;
        } else if (sortRepeat % cnt != 0) {
            set_vector_mask(0xAAAAAAAAAAAAAAAA, 0xAAAAAAAAAAAAAAAA);
            vector_dup(addr + sortRepeat * len * 2, 0,
                       DIV_ROUND_UP((cnt - sortRepeat % cnt) * len * 2, pad), 1, 1, 8, 0);
            set_vector_mask(0x5555555555555555, 0x5555555555555555);
            vector_dup(addr + sortRepeat * len * 2, float(min),
                       DIV_ROUND_UP((cnt - sortRepeat % cnt) * len * 2, pad), 1, 1, 8, 0);
            pipe_barrier(PIPE_V);
            set_vector_mask((uint64_t)-1, (uint64_t)-1);
        }
        int stride = len * 2;
        __ubuf__ float *addrs[4] = {addr, addr + stride, addr + 2 * stride, addr + 3 * stride};
        uint64_t lens = len | len << 16 | len << 32 | len << 48;
        sortRepeat = DIV_ROUND_UP(sortRepeat, cnt);
        uint64_t config = sortRepeat | validBit << MGR_SORT_VALID_BITS_OFFSET;
        bufIdx = 1 - bufIdx;
        vmrgsort4(mrgSortBuf[bufIdx], addrs, lens, config);
        pipe_barrier(PIPE_V);
        len *= cnt;
    }
    *dstBufIdx = bufIdx;
}

#define BITS_PER_DWORD 64
#define BIT_DWORD(nr) ((nr) / BITS_PER_DWORD)

inline __aicore__ void bitmapSet(__ubuf__ uint64_t *addr, uint32_t id)
{
    __ubuf__ uint64_t *p = addr + BIT_DWORD(id);
    *p |= (1ULL << (id % BITS_PER_DWORD));
}

inline __aicore__ int bitmapTest(__ubuf__ uint64_t *addr, uint32_t id)
{
    __ubuf__ uint64_t *p = addr + BIT_DWORD(id);
    return ((*p) & (1ULL << (id % BITS_PER_DWORD))) ? 1 : 0;
}

// Copy data from GM to UB with automatic alignment handling.
// Selects the DMA primitive by byte alignment so any size is supported:
//   32B aligned -> copy_gm_to_ubuf (block count);
//   2B  aligned -> copy_gm_to_ubuf_align_b16 (byte size);
//   otherwise  -> copy_gm_to_ubuf_align_b8   (byte size, incl. odd sizes).
// The b16/b8 _align variants take a byte lenBurst and handle the internal
// padding internally. DstT and SrcT can differ (e.g. float* dst with
// uint32_t* src for interleaved sort buffers).
template <typename DstT, typename SrcT = DstT>
__aicore__ inline void CopyGmToUbufAligned(__ubuf__ DstT *dst, __gm__ SrcT *src, uint32_t byteSize)
{
    if (byteSize % BLOCK_SIZE == 0) {
        copy_gm_to_ubuf(dst, src, 0, 1, byteSize / BLOCK_SIZE, 0, 0);
    } else if (byteSize % 2 == 0) {
        copy_gm_to_ubuf_align_b16(dst, src, 0, 1, byteSize, 0, 0, 0, 0);
    } else {
        copy_gm_to_ubuf_align_b8(dst, src, 0, 1, byteSize, 0, 0, 0, 0);
    }
}

// Copy data from UB to GM with automatic alignment handling (see
// CopyGmToUbufAligned above for the primitive selection rationale).
// DstT and SrcT can differ (e.g. uint32_t* dst with float* src for
// interleaved sort buffers).
template <typename DstT, typename SrcT = DstT>
__aicore__ inline void CopyUbufToGmAligned(__gm__ DstT *dst, __ubuf__ SrcT *src, uint32_t byteSize)
{
    if (byteSize % BLOCK_SIZE == 0) {
        copy_ubuf_to_gm(dst, src, 0, 1, byteSize / BLOCK_SIZE, 0, 0);
    } else if (byteSize % 2 == 0) {
        copy_ubuf_to_gm_align_b16(dst, src, 0, 1, byteSize, 0, 0, 0, 0);
    } else {
        copy_ubuf_to_gm_align_b8(dst, src, 0, 1, byteSize, 0, 0, 0, 0);
    }
}

#endif

#endif
