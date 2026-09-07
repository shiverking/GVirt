/*
 * Copyright (C) 2025 - 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "base.h"
#include "ascend.h"
#include "runtime.h"
#include "ccl.h"
#include "op.h"
#include "swizzle.h"
#include "aclrtlaunch_all.h"

#include "kernels/kernel_param.h"
#include "trace/trace.h"
#ifdef XLITE_ARCH_310P
#include "aclnn_310p.h"

// Implemented in the Bisheng host object emitted from the isolated official
// Add baseline translation unit.
extern void xlite_official_add_probe_310p_do(uint32_t blockDim, void *stream, uint8_t *x,
                                             uint8_t *y, uint8_t *z);
#endif

#define KERNEL_PTR_TYPE(name) decltype(aclrtlaunch_##name##_bfloat16_t)

static inline bool IsDummyRuntime(const XRuntime &rt)
{
    return rt.IsDummyRuntime();
}

void XliteOpProbe310P(XRuntime &rt, XTensor &out, uint32_t value)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (out.dtype != INT32 || out.numel < 1) {
        throw std::runtime_error("Ascend310P launch probe requires a non-empty INT32 output");
    }
    // Deliberately use one block so this probe is independent of runtime core
    // discovery and tests only binary registration, launch, and a scalar GM write.
    aclrtlaunch_xlite_probe_310p(1, rt.stream, out.ptr, value);
#else
    (void)out;
    (void)value;
    throw std::runtime_error("Ascend310P launch probe is unavailable in this build");
#endif
}

void XliteOpOfficialAddProbe310P(XRuntime &rt, XTensor &x, XTensor &y, XTensor &z)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    constexpr uint64_t probeElements = 8 * 2048;
    if (x.dtype != FP16 || y.dtype != FP16 || z.dtype != FP16 ||
        x.numel != probeElements || y.numel != probeElements || z.numel != probeElements) {
        throw std::runtime_error(
            "official Ascend310P Add probe requires three FP16 tensors with 8*2048 elements");
    }
    // Fixed at the blockDim and triple-chevron host wrapper used by the
    // passing AddKernelInvocationNeo sample.
    xlite_official_add_probe_310p_do(8, rt.stream, reinterpret_cast<uint8_t *>(x.ptr),
                                     reinterpret_cast<uint8_t *>(y.ptr),
                                     reinterpret_cast<uint8_t *>(z.ptr));
#else
    (void)x;
    (void)y;
    (void)z;
    throw std::runtime_error("official Ascend310P Add probe is unavailable in this build");
#endif
}

// Pick how many AIV blocks to launch for a byte-wise copy kernel (concat/split)
// based on the total bytes. Each block processes ~tilePerCore bytes in series,
// so for small data we launch only a few blocks (avoids the multi-core launch +
// pipe_barrier sync overhead that dominates tiny transfers), while large data
// saturates all aivNum cores. tilePerCore ~= 2 MB keeps the per-block work
// (~tens of us) well above the launch cost (~2 us).
static inline uint32_t CopyKernelBlockNum(const XRuntime &rt, uint64_t totalBytes,
                                          uint64_t tilePerCore = 2 * 1024 * 1024)
{
    if (totalBytes == 0 || rt.aivNum <= 1) {
        return 1;
    }
    uint64_t needed = DIV_ROUND_UP(totalBytes, tilePerCore);
    if (needed >= rt.aivNum) {
        return rt.aivNum;
    }
    return static_cast<uint32_t>(needed);
}

HcclDataType XDtype2HcclDtype(enum XDtype dtype)
{
    switch (dtype) {
        case INT8:
            return HCCL_DATA_TYPE_INT8;
        case INT32:
            return HCCL_DATA_TYPE_INT32;
        case INT64:
            return HCCL_DATA_TYPE_INT64;
        case FP16:
            return HCCL_DATA_TYPE_FP16;
        case BF16:
            return HCCL_DATA_TYPE_BFP16;
        case FP32:
            return HCCL_DATA_TYPE_FP32;
        default:
            throw std::runtime_error(std::string("unknown data type ") + XDtypeStr(dtype));
    }
}

template <typename... Args>
static bool EachXDtype(enum XDtype dtype, Args &&...args)
{
    return (... && (std::forward<Args>(args).dtype == dtype));
}

void XliteOpAllGather(XRuntime &rt, XTensor &in, XTensor &out, enum commType type, bool fetchOffset,
                      DebugSrcLoc loc, uint32_t copySize)
{
    uint32_t rankSize = rt.tpSize();
    if (type == DP) {
        rankSize = rt.dpSize();
    } else if (type == EP) {
        rankSize = rt.moeEpSize();
    }
    if (in.dtype != out.dtype || in.numel * rankSize != out.numel) {
        std::stringstream ss;
        ss << loc.ToStr() << __func__ << ": check tensor failed! in.dtype=" << XDtypeStr(in.dtype)
           << "(" << in.dtype << "), out.dtype=" << XDtypeStr(out.dtype) << "(" << out.dtype
           << "); in.numel=" << in.numel << " rankSize=" << rankSize
           << ", expected out.numel=" << (in.numel * rankSize)
           << ", actual out.numel=" << out.numel;
        throw std::runtime_error(ss.str());
    }
    if ((in.numel * XDtypeBit(in.dtype)) % XDtypeBit(INT8)) {
        throw std::runtime_error(loc.ToStr() + std::string(__func__) +
                                 ": all gather 8bit align check failed!");
    }

    XcclComm *xcclComm = nullptr;
    HcclComm hcclComm = nullptr;
    uint32_t rank = 0;
    uint32_t localRank = 0;
    if (type == TP) {
        xcclComm = rt._tpXcclComm;
        hcclComm = rt._tpComm;
        rank = rt.tpSize();
        localRank = rt.rankId() % rt.tpSize();
    } else if (type == DP) {
        xcclComm = rt._dpXcclComm;
        hcclComm = rt._dpComm;
        rank = rt.dpSize();
        localRank = rt.rankId() / rt.tpSize();
    } else if (type == EP) {
        xcclComm = rt._epXcclComm;
        hcclComm = rt._epComm;
        rank = rt.moeEpSize();
        localRank = rt.rankId() / rt.moeTpSize();
    }

    if (IsDummyRuntime(rt)) {
        if (xcclComm && in.dtype != INT64 && rankSize > 1) {
            bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
            if (needCopy) {
                XTensor &tmpIn =
                    rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
                XTensor &tmpOut = rt.GetTensor(out.shape, out.dtype, DBG_LOC);
                rt.PutTensor(tmpIn);
                rt.PutTensor(tmpOut);
            }
        }
        return;
    }

    if (rankSize <= 1) {
        if (in.ptr != out.ptr) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, out.bytes, in.ptr, in.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
        return;
    }

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpIn = nullptr;
        XTensor *tmpOut = nullptr;

        if (needCopy) {
            tmpIn = &rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            tmpOut = &rt.GetTensor(out.shape, out.dtype, DBG_LOC);
            CHECK_ACL(aclrtMemcpyAsync(tmpIn->ptr, in.bytes, in.ptr, in.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            inPtr = tmpIn->ptr;
            outPtr = tmpOut->ptr;
        }

        // prevents the size of each copy from being too small.
        uint64_t sizePerRank = DIV_ROUND_UP(in.bytes, rank);
        uint64_t maxCoreNum = rank;
        if (sizePerRank >= DOUBLE_AIVNUM_SIZE_BOUND) {
            maxCoreNum = static_cast<uint64_t>(rank) * 2;
        }
        uint32_t coreNum = rt.aivNum;
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        if (coreNum >= rank) {
            coreNum = ROUND_DOWN(coreNum, rank);
            uint32_t corePerRank = coreNum / rank;
            if (corePerRank > 1 && copySize * corePerRank > MAX_TOTAL_COPY_SIZE) {
                copySize = MAX_TOTAL_COPY_SIZE / corePerRank;
            }
        }

        // call correct allreduce kernel
        uintptr_t count;
        KERNEL_PTR_TYPE(allgather) * launchKernel;
        switch (in.dtype) {
            case FP16:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_float16_t;
                break;
            case BF16:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_bfloat16_t;
                break;
            // BIT1 is packed, so count is in.numel / 8.
            case BIT1:
                count = in.numel * XDtypeBit(in.dtype) / XDtypeBit(INT8);
                launchKernel = aclrtlaunch_allgather_int8_t;
                break;
            case INT8:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_int8_t;
                break;
            case INT32:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_int32_t;
                break;
            case FP32:
                count = in.numel;
                launchKernel = aclrtlaunch_allgather_float;
                break;
            default:
                std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
                throw std::runtime_error(err_str + " unsupported dtype for xccl func");
                break;
        }
        launchKernel(coreNum, rt.stream, inPtr, outPtr, count, localRank, rank,
                     xcclComm->generation++, xcclComm->dParam, copySize, fetchOffset || type == DP);

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, out.bytes, outPtr, out.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.PutTensor(*tmpIn);
            rt.PutTensor(*tmpOut);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclAllGather(in.ptr, out.ptr, in.numel * XDtypeBit(in.dtype) / XDtypeBit(INT8),
                             HCCL_DATA_TYPE_INT8, hcclComm, rt.stream));
}

void XliteOpReduceScatter(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                          bool fetchOffset, DebugSrcLoc loc, uint32_t copySize)
{
    uint32_t rankSize = type == TP ? rt.tpSize() : rt.dpSize();
    if (in.dtype != out.dtype || in.numel != out.numel * rankSize) {
        std::stringstream ss;
        ss << loc.ToStr() << __func__ << ": check tensor failed! in.dtype=" << XDtypeStr(in.dtype)
           << "(" << in.dtype << "), out.dtype=" << XDtypeStr(out.dtype) << "(" << out.dtype
           << "); out.numel=" << out.numel << " rankSize=" << rankSize
           << ", expected in.numel=" << (out.numel * rankSize) << ", actual in.numel=" << in.numel;
        throw std::runtime_error(ss.str());
    }

    auto xcclComm = (type == TP) ? rt._tpXcclComm : rt._dpXcclComm;
    auto hcclComm = (type == TP) ? rt._tpComm : rt._dpComm;
    uint32_t rank = (type == TP) ? rt.tpSize() : rt.dpSize();
    uint32_t localRank = (type == TP) ? (rt.rankId() % rt.tpSize()) : (rt.rankId() / rt.tpSize());

    if (IsDummyRuntime(rt)) {
        if (xcclComm && in.dtype != INT64 && rankSize > 1) {
            bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
            if (needCopy) {
                XTensor &tmpIn =
                    rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
                XTensor &tmpOut = rt.GetTensor(out.shape, out.dtype, DBG_LOC);
                rt.PutTensor(tmpIn);
                rt.PutTensor(tmpOut);
            }
        }
        return;
    }

    if (rankSize <= 1) {
        if (in.ptr != out.ptr) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, out.bytes, in.ptr, in.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
        return;
    }

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpIn = nullptr;
        XTensor *tmpOut = nullptr;

        if (needCopy) {
            tmpIn = &rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            tmpOut = &rt.GetTensor(out.shape, out.dtype, DBG_LOC);
            CHECK_ACL(aclrtMemcpyAsync(tmpIn->ptr, in.bytes, in.ptr, in.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            inPtr = tmpIn->ptr;
            outPtr = tmpOut->ptr;
        }

        // prevents the size of each copy from being too small.
        uint64_t sizePerRank = DIV_ROUND_UP(in.bytes, rank);
        uint64_t maxCoreNum = rank;
        if (sizePerRank >= DOUBLE_AIVNUM_SIZE_BOUND) {
            maxCoreNum = static_cast<uint64_t>(rank) * 2;
        }
        uint32_t coreNum = rt.aivNum;
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        if (coreNum >= rank) {
            coreNum = ROUND_DOWN(coreNum, rank);
            uint32_t corePerRank = coreNum / rank;
            if (corePerRank > 1 && copySize * corePerRank > MAX_TOTAL_COPY_SIZE) {
                copySize = MAX_TOTAL_COPY_SIZE / corePerRank;
            }
        }

        KERNEL_PTR_TYPE(reduce_scatter) * launchKernel;
        // call correct allreduce kernel
        switch (in.dtype) {
            case FP16:
                launchKernel = aclrtlaunch_reduce_scatter_float16_t;
                break;
            case BF16:
                launchKernel = aclrtlaunch_reduce_scatter_bfloat16_t;
                break;
            case INT8:
                launchKernel = aclrtlaunch_reduce_scatter_int8_t;
                break;
            case INT32:
                launchKernel = aclrtlaunch_reduce_scatter_int32_t;
                break;
            case FP32:
                launchKernel = aclrtlaunch_reduce_scatter_float;
                break;
            default:
                std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
                throw std::runtime_error(err_str + " unsupported dtype for xccl func");
                break;
        }
        launchKernel(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank, rank,
                     xcclComm->generation++, xcclComm->dParam, copySize, fetchOffset || type == DP);

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, out.bytes, outPtr, out.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.PutTensor(*tmpIn);
            rt.PutTensor(*tmpOut);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclReduceScatter(in.ptr, out.ptr, out.numel, XDtype2HcclDtype(in.dtype),
                                 HCCL_REDUCE_SUM, hcclComm, rt.stream));
}

void XliteOpAllReduceSum(XRuntime &rt, XTensor &in, XTensor &out, enum commType type,
                         bool fetchOffset, DebugSrcLoc loc, uint32_t copySize)
{
    if (in.dtype != out.dtype || in.numel != out.numel) {
        std::stringstream ss;
        ss << loc.ToStr() << __func__ << ": check tensor failed! in.dtype=" << XDtypeStr(in.dtype)
           << "(" << in.dtype << "), out.dtype=" << XDtypeStr(out.dtype) << "(" << out.dtype
           << "); in.numel=" << in.numel << ", out.numel=" << out.numel;
        throw std::runtime_error(ss.str());
    }

    auto xcclComm = (type == TP) ? rt._tpXcclComm : rt._dpXcclComm;
    auto hcclComm = (type == TP) ? rt._tpComm : rt._dpComm;
    uint32_t rank = (type == TP) ? rt.tpSize() : rt.dpSize();
    uint32_t localRank = (type == TP) ? (rt.rankId() % rt.tpSize()) : (rt.rankId() / rt.tpSize());

    if (IsDummyRuntime(rt)) {
        if (xcclComm && in.dtype != INT64 && rank > 1) {
            bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
            if (needCopy) {
                XTensor &tmpBuff =
                    rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
                rt.PutTensor(tmpBuff);
            }
        }
        return;
    }

    if (rank <= 1) {
        if (in.ptr != out.ptr) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, in.bytes, in.ptr, in.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
        }
        return;
    }

    if (xcclComm && in.dtype != INT64) {
        bool needCopy = (!rt.TensorInPool(in) || !rt.TensorInPool(out));
        void *inPtr = in.ptr;
        void *outPtr = out.ptr;
        XTensor *tmpBuff = nullptr;

        if (needCopy) {
            tmpBuff = &rt.GetTensor(in.shape, in.dtype, DBG_LOC);  // tmp to ensure not from pool
            CHECK_ACL(aclrtMemcpyAsync(tmpBuff->ptr, in.bytes, in.ptr, in.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            inPtr = tmpBuff->ptr;
            outPtr = tmpBuff->ptr;
        }

        // prevents the size of each copy from being too small.
        uint64_t sizePerRank = DIV_ROUND_UP(in.bytes, rank);
        uint64_t maxCoreNum = rank;
        if (sizePerRank >= DOUBLE_AIVNUM_SIZE_BOUND) {
            maxCoreNum = static_cast<uint64_t>(rank) * 2;
        }
        uint32_t coreNum = rt.aivNum;
        if (coreNum > maxCoreNum) {
            coreNum = maxCoreNum;
        }

        if (coreNum >= rank) {
            coreNum = ROUND_DOWN(coreNum, rank);
            uint32_t corePerRank = coreNum / rank;
            if (corePerRank > 1 && copySize * corePerRank > MAX_TOTAL_COPY_SIZE) {
                copySize = MAX_TOTAL_COPY_SIZE / corePerRank;
            }
        }

        // call correct allreduce kernel
        KERNEL_PTR_TYPE(allreduce) * launchKernel;
        switch (in.dtype) {
            case FP16:
                launchKernel = aclrtlaunch_allreduce_float16_t;
                break;
            case BF16:
                launchKernel = aclrtlaunch_allreduce_bfloat16_t;
                break;
            case INT8:
                launchKernel = aclrtlaunch_allreduce_int8_t;
                break;
            case INT32:
                launchKernel = aclrtlaunch_allreduce_int32_t;
                break;
            case FP32:
                launchKernel = aclrtlaunch_allreduce_float;
                break;
            default:
                std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
                throw std::runtime_error(err_str + " unsupported dtype for xccl func");
                break;
        }
        launchKernel(coreNum, rt.stream, inPtr, outPtr, in.numel, localRank, rank,
                     xcclComm->generation++, xcclComm->dParam, copySize, fetchOffset || type == DP);

        if (needCopy) {
            CHECK_ACL(aclrtMemcpyAsync(out.ptr, out.bytes, outPtr, out.bytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            rt.PutTensor(*tmpBuff);
        }
        return;
    }

    // fallback to HCCL path
    CHECK_HCCL(HcclAllReduce(in.ptr, out.ptr, in.numel, XDtype2HcclDtype(in.dtype), HCCL_REDUCE_SUM,
                             hcclComm, rt.stream));
}

void XliteOpAlltoAllV(XRuntime &rt, XTensor &in, XTensor &out, XTensor &sendCounts,
                      XTensor &recvCounts, XTensor &sdispls, XTensor &rdispls, enum commType type,
                      DebugSrcLoc loc)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    if (in.dtype != out.dtype) {
        throw std::runtime_error(loc.ToStr() + std::string(__func__) +
                                 ": check tensor failed! input: " + std::to_string(in.dtype) +
                                 " output: " + std::to_string(out.dtype));
    }

    HcclComm hcclComm = rt._tpComm;
    if (type == DP) {
        hcclComm = rt._dpComm;
    } else if (type == EP) {
        hcclComm = rt._epComm;
    }

    CHECK_HCCL(HcclAlltoAllV(in.ptr, sendCounts.ptr, sdispls.ptr, XDtype2HcclDtype(in.dtype),
                             out.ptr, recvCounts.ptr, rdispls.ptr, XDtype2HcclDtype(out.dtype),
                             hcclComm, rt.stream));
}

void XliteOpEmbed(XRuntime &rt, XTensor &in, XTensor &embed, uint32_t start, uint32_t end,
                  XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (in.dtype != INT32 || embed.dtype != FP16 || out.dtype != FP16 || rt.tpSize() != 1) {
        throw std::runtime_error(
            "Ascend310P embedding requires INT32 ids, FP16 tensors and TP=1");
    }
#endif
    KERNEL_PTR_TYPE(embed_kernel) * launchKernel;
    if (EachXDtype(FP16, embed, out)) {
        launchKernel = aclrtlaunch_embed_kernel_float16_t;
    } else if (EachXDtype(BF16, embed, out)) {
        launchKernel = aclrtlaunch_embed_kernel_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(embed) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, embed.ptr, in.ptr, out.ptr, embed.shape[1], in.shape[0],
                 start, end, rt.tpSize());
}

void XliteOpRmsNorm(XRuntime &rt, XTensor &in, const XTensor &norm, XTensor &out, float normEps,
                    uint32_t normDim, bool useNorm, const XTensor &normBias, uint32_t cntPerToken,
                    uint32_t inStartOffset, uint32_t outStartOffset, const XTensor &variance)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (in.dtype != FP16 || norm.dtype != FP16 || out.dtype != FP16 || normBias.ptr != nullptr ||
        variance.ptr != nullptr || !useNorm) {
        throw std::runtime_error(
            "Ascend310P RMSNorm requires FP16 I/O/weight and the standard no-bias path");
    }
#endif
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (in.dtype == FP16 && (out.dtype == FP16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (in.dtype == BF16 && (out.dtype == BF16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(in) + XT_STR(norm) + XT_STR(out) + XT_STR(normBias);
        throw std::runtime_error(err_str + " unsupported!");
    }
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Rms);
    launchKernel(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, normBias.ptr, out.ptr,
                 in.shape[0], normDim, normEps, kind, cntPerToken, in.shape[1], out.shape[1],
                 inStartOffset, outStartOffset, useNorm, variance.ptr, rt.tpSize());
}

void XliteOpLayerNorm(XRuntime &rt, XTensor &in, XTensor &norm, XTensor &normBias, XTensor &out,
                      float normEps, uint32_t normDim, uint32_t cntPerToken, uint32_t inStartOffset,
                      uint32_t outStartOffset)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (EachXDtype(FP16, in, out)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (EachXDtype(BF16, in, out)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(in) + XT_STR(norm) + XT_STR(normBias) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Layer);
    launchKernel(rt.aivNum, rt.stream, in.ptr, nullptr, norm.ptr, normBias.ptr, out.ptr,
                 in.shape[0], normDim, normEps, kind, cntPerToken, in.shape[1], out.shape[1],
                 inStartOffset, outStartOffset, true, nullptr, rt.tpSize());
}

void XliteOpL2Norm(XRuntime &rt, XTensor &in, XTensor &out, float normEps, uint32_t normDim,
                   uint32_t cntPerToken, uint32_t inStartOffset, uint32_t outStartOffset)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (in.dtype == FP16 && (out.dtype == FP16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (in.dtype == BF16 && (out.dtype == BF16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::L2);
    launchKernel(rt.aivNum, rt.stream, in.ptr, nullptr, nullptr, nullptr, out.ptr, in.shape[0],
                 normDim, normEps, kind, cntPerToken, in.shape[1], out.shape[1], inStartOffset,
                 outStartOffset, true, nullptr, rt.tpSize());
}

void XliteOpAdd(XRuntime &rt, XTensor &in1, XTensor &in2, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (!EachXDtype(FP16, in1, in2, out)) {
        throw std::runtime_error("Ascend310P add only supports FP16");
    }
#endif
    KERNEL_PTR_TYPE(add) * launchKernel;
    if (EachXDtype(FP16, in1, in2, out)) {
        launchKernel = aclrtlaunch_add_float16_t;
    } else if (EachXDtype(BF16, in1, in2, out)) {
        launchKernel = aclrtlaunch_add_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in1) + XT_STR(in2) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in1.ptr, in2.ptr, out.ptr, in1.shape[0], in1.shape[1]);
}

void XliteOpAddAndRmsNorm(XRuntime &rt, XTensor &in, XTensor &addInOut, XTensor &norm,
                          float normEps, XTensor &out, const XTensor &normBias)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(norm) * launchKernel;
    if (EachXDtype(FP16, in, addInOut, out)) {
        launchKernel = aclrtlaunch_norm_float16_t;
    } else if (EachXDtype(BF16, in, addInOut, out)) {
        launchKernel = aclrtlaunch_norm_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(in) + XT_STR(addInOut) + XT_STR(norm) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    auto kind = static_cast<std::underlying_type_t<NormKind>>(NormKind::Rms);
    launchKernel(rt.aivNum, rt.stream, in.ptr, addInOut.ptr, norm.ptr, normBias.ptr, out.ptr,
                 in.shape[0], in.shape[1], normEps, kind, 1, in.shape[1], out.shape[1], 0, 0, true,
                 nullptr, rt.tpSize());
}

void XliteOpMatmul(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out, bool weightNZ,
                   const XTensor &bias, const XTensor &deqScale, bool transpose, uint64_t m0,
                   uint64_t n0, uint64_t k0)
{
    if (IsDummyRuntime(rt)) {
#ifdef XLITE_ARCH_310P
        const size_t matmulN = transpose ? weight.shape[1] : weight.shape[0];
        XTensor *chunkOutput = nullptr;
        if (!transpose && matmulN > XLITE_310P_MATMUL_N_CHUNK) {
            chunkOutput = &rt.GetTensor(
                {in.shape[0], std::min(matmulN, XLITE_310P_MATMUL_N_CHUNK)}, FP16, DBG_LOC);
        }
        XTensor &workspace =
            rt.GetTensor({XLITE_310P_ACLNN_WORKSPACE_BYTES}, INT8, DBG_LOC);
        rt.PutTensor(workspace);
        if (chunkOutput != nullptr) {
            rt.PutTensor(*chunkOutput);
        }
#endif
        if (EachXDtype(BF16, in, weight, out) && bias.ptr != nullptr) {
            XTensor &biasFp32 = rt.GetTensor(bias.shape, FP32, DBG_LOC);
            rt.PutTensor(biasFp32);
        } else if (in.dtype == BF16 && weight.dtype == FP32 && out.dtype == FP32 && !transpose) {
            XTensor &tmp = rt.GetTensor(in.shape, FP32, DBG_LOC);
            rt.PutTensor(tmp);
        } else if (in.dtype == BF16 && weight.dtype == FP32 && out.dtype == BF16 && !transpose) {
            XTensor &inFp32 = rt.GetTensor(in.shape, FP32, DBG_LOC);
            XTensor &outFp32 = rt.GetTensor(out.shape, FP32, DBG_LOC);
            rt.PutTensor(inFp32);
            rt.PutTensor(outFp32);
        }
        return;
    }
#ifdef XLITE_ARCH_310P
    XliteAclnn310PMatmul(rt, in, weight, out, weightNZ, bias, deqScale, transpose);
    return;
#else

    uint64_t m = in.shape[0];
    uint64_t n = transpose ? weight.shape[1] : weight.shape[0];
    uint64_t k = transpose ? weight.shape[0] : weight.shape[1];
    bool needExtraSpace = (bias.ptr != nullptr || deqScale.ptr != nullptr);
    uint64_t mLoop;
    uint64_t nLoop;
    uint64_t totalLoops;
    uint64_t swizzle = rt.defaultMatmulSwizzle;

    // Notice: Ensure that no overflow occurs
    // L1(512K): PINGPONG * (sizeof(x) * m0 * 2k0 + sizeof(y) * n0 * k0) + BiasSize(Optional)] +
    // FixPipe(Optional)
    //         = 4 * sizeof(x) * m0 * k0 + 2 * k0 * sizeof(y) * n0 + [4 * n0] + [8 * n0]
    //         = 2 * k0 * (2 * sizeof(x) * m0 + sizeof(y) * n0) + [12 * n0]
    // L0A(64K): PINGPONG * sizeof(x) * m0 * k0 / 4 = sizeof(x) * m0 * k0 / 2
    // L0B(64K): PINGPONG * sizeof(y) * m0 * k0 / 4 = sizeof(y) * m0 * k0 / 2
    // BiasTable(1K): n0 * sizeof(float/int32_t) = 4 * n0
    // FixPipe(2K): n0 * sizeof(uint64_t) = 8 * n0
    if (m0 == MATMUL_M0_N0_K0_DEFAULT_VALUE || n0 == MATMUL_M0_N0_K0_DEFAULT_VALUE ||
        k0 == MATMUL_M0_N0_K0_DEFAULT_VALUE) {
        m0 = ROUND_UP(m, 32);
        if (m0 > 128) {
            m0 = 128;
        }
        // if matmul has bias or dequant scale, L1 buffer will overflow!
        n0 = needExtraSpace ? 128 : 256;
        k0 = 4096 / XDtypeBit(weight.dtype);

        mLoop = DIV_ROUND_UP(m, m0);
        nLoop = DIV_ROUND_UP(n, n0);
        totalLoops = mLoop * nLoop;
        uint64_t lastLoops = totalLoops % rt.aicNum;

        // If the data size is small, we should make a data tiling mode
        // to ensure even loads on each AICore.
        if (totalLoops < static_cast<uint64_t>(3) * rt.aicNum &&
            (lastLoops != 0 && lastLoops < rt.aicNum / 2)) {
            if (n <= static_cast<uint64_t>(32) * rt.aicNum) {
                m0 = m0 > 64 ? 64 : m0;
                n0 = 64;
            } else if (n <= static_cast<uint64_t>(64) * rt.aicNum) {
                n0 = 64;
            } else if (n <= static_cast<uint64_t>(128) * rt.aicNum) {
                n0 = 128;
            } else if (n <= static_cast<uint64_t>(256) * rt.aicNum) {
                n0 = needExtraSpace ? 128 : 256;
            } else {
                m0 = m0 > 64 ? 64 : m0;
                // BiasTable(1K): 4 * n0 <= 1K, so that n0 <= 256
                n0 = needExtraSpace ? 256 : 384;
                k0 /= 2;
            }
        }
    }
    mLoop = DIV_ROUND_UP(m, m0);
    nLoop = DIV_ROUND_UP(n, n0);
    totalLoops = mLoop * nLoop;
    uint32_t aicNum = totalLoops > rt.aicNum ? rt.aicNum : totalLoops;
    if (aicNum == 0) {
        aicNum = 1;
    }

    if (!rt.disableSwizzleTable) {
        XlitePickSwizzle(n, k, &swizzle);
    }

    KERNEL_PTR_TYPE(matmul) * launchKernel;
    XTensor *castedIn = nullptr;
    XTensor *castedBias = nullptr;
    XTensor *castedOut = nullptr;
    if (EachXDtype(FP16, in, weight, out)) {
        launchKernel = aclrtlaunch_matmul_float16_t;
    } else if (EachXDtype(BF16, in, weight, out)) {
        if (bias.ptr != nullptr) {
            castedBias = &rt.GetTensor(bias.shape, FP32, DBG_LOC);
            aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, bias.ptr, castedBias->ptr,
                                              bias.numel);
        }
        launchKernel = aclrtlaunch_matmul_bfloat16_t;
    } else if (EachXDtype(FP32, in, weight, out) && !transpose) {
        launchKernel = aclrtlaunch_matmul_float;
    } else if (in.dtype == BF16 && EachXDtype(FP32, weight, out) && !transpose) {
        castedIn = &rt.GetTensor(in.shape, FP32, DBG_LOC);
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, castedIn->ptr, in.numel);
        launchKernel = aclrtlaunch_matmul_float;
    } else if (in.dtype == BF16 && weight.dtype == FP32 && out.dtype == BF16 && !transpose) {
        castedIn = &rt.GetTensor(in.shape, FP32, DBG_LOC);
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, castedIn->ptr, in.numel);
        castedOut = &rt.GetTensor(out.shape, FP32, DBG_LOC);
        launchKernel = aclrtlaunch_matmul_float;
    } else if (EachXDtype(INT8, in, weight) && out.dtype == FP16) {
        launchKernel = aclrtlaunch_matmul_int8_t;
    } else if (EachXDtype(INT4, in, weight) && out.dtype == FP16) {
        launchKernel = aclrtlaunch_matmul_int4b_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(weight) + XT_STR(out) +
                              " transpose=" + std::to_string(transpose);
        throw std::runtime_error(err_str + " unsupported!");
    }
    auto inPtr = castedIn ? castedIn->ptr : in.ptr;
    auto biasPtr = castedBias ? castedBias->ptr : bias.ptr;
    auto outPtr = castedOut ? castedOut->ptr : out.ptr;

    auto inDtype = castedIn ? castedIn->dtype : in.dtype;
    xlite_trace::RecordMatmul(rt.rankId(), inDtype, transpose, weightNZ, m, n, k);

    launchKernel(aicNum, rt.stream, inPtr, weight.ptr, outPtr, m, n, k, weightNZ, transpose, m0, n0,
                 k0, swizzle, biasPtr, deqScale.ptr);
    if (castedOut) {
        aclrtlaunch_cast_float_bfloat16_t(rt.aivNum, rt.stream, castedOut->ptr, out.ptr, out.numel);
    }
    if (castedIn) {
        rt.PutTensor(*castedIn);
    }
    if (castedBias) {
        rt.PutTensor(*castedBias);
    }
    if (castedOut) {
        rt.PutTensor(*castedOut);
    }
#endif
}

void XliteOpSiluAndMul(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &num)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (!EachXDtype(FP16, in, out) || out.shape.size() != 2 || out.shape[1] != 6144) {
        throw std::runtime_error(
            "Ascend310P SiLU-and-Mul requires FP16 and intermediate_size=6144");
    }
#endif
#ifdef XLITE_ARCH_310P
    decltype(aclrtlaunch_silu_and_mul_float16_t) *launchKernel;
    launchKernel = aclrtlaunch_silu_and_mul_float16_t;
#else
    KERNEL_PTR_TYPE(silu_and_mul) * launchKernel;
    if (EachXDtype(FP16, in, out)) {
        launchKernel = aclrtlaunch_silu_and_mul_float16_t;
    } else if (EachXDtype(BF16, in, out)) {
        launchKernel = aclrtlaunch_silu_and_mul_bfloat16_t;
    } else if (EachXDtype(FP32, in, out)) {
        launchKernel = aclrtlaunch_silu_and_mul_float;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
#endif
    launchKernel(rt.aivNum, rt.stream, in.ptr, out.ptr, num.ptr, in.shape[0], out.shape[1]);
}

void XliteOpCastDown(XRuntime &rt, XTensor &in, XTensor &out, XTensor &outScale)
{
    throw std::runtime_error(std::string(__func__) + ": TODO");
}

void XliteOpCastUp(XRuntime &rt, XTensor &in, XTensor &inScale, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (in.dtype == BF16 && out.dtype == FP32) {
        aclrtlaunch_cast_bfloat16_t_float(rt.aivNum, rt.stream, in.ptr, out.ptr, in.numel);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(inScale) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpPermutation(XRuntime &rt, XTensor &in, XTensor &routing, uint32_t start, uint32_t end,
                        XTensor &out, XTensor &unpIdx, XTensor &counts)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    throw std::runtime_error("Ascend310P LLM FP16 POC does not support permutation/MoE");
#else
    aclrtlaunch_permutation(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr,
                            counts.ptr, in.shape[0], in.shape[1], out.shape[0], counts.shape[0],
                            start, end);
#endif
}

void XliteOpUnpermutation(XRuntime &rt, XTensor &in, XTensor &unpIdx, XTensor &routing,
                          XTensor &weights, uint32_t start, uint32_t end, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(unpermutation) * launchKernel;
    if (EachXDtype(BF16, in, out, weights)) {
        launchKernel = aclrtlaunch_unpermutation_bfloat16_t;
    } else if (in.dtype == BF16 && out.dtype == BF16 && weights.dtype == FP32) {
        launchKernel = aclrtlaunch_unpermutation_float;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(unpIdx) + XT_STR(routing) +
                              XT_STR(weights) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, in.ptr, routing.ptr, out.ptr, unpIdx.ptr, weights.ptr,
                 out.shape[0], in.shape[1], weights.shape[1], start, end);
}

// deqScales: uint64_t, 低 32 位 TF32 格式有效, 1 符号位, 8 指数位, 10 尾数位, 后 13 位不参与计算
void XliteOpGroupMatmul(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                        XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                        long outDim, long inDim, XTensor &output, bool weightNZ, bool transpose)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    KERNEL_PTR_TYPE(group_matmul) * launchKernel;
    if (in.dtype == BF16 && weightDtype == BF16 && output.dtype == BF16) {
        launchKernel = aclrtlaunch_group_matmul_bfloat16_t;
    } else if (in.dtype == FP16 && weightDtype == FP16 && output.dtype == FP16) {
        launchKernel = aclrtlaunch_group_matmul_float16_t;
    } else if (in.dtype == FP32 && weightDtype == FP32 && output.dtype == FP32 && !transpose) {
        launchKernel = aclrtlaunch_group_matmul_float;
    } else if (in.dtype == INT8 && weightDtype == INT8 && output.dtype == FP16) {
        launchKernel = aclrtlaunch_group_matmul_int8_t;
    } else if (in.dtype == INT4 && weightDtype == INT4 && output.dtype == FP16) {
        launchKernel = aclrtlaunch_group_matmul_int4b_t;
    } else {
        std::string err_str = DBG_PREFIX;
        err_str += XT_STR(in) + XT_STR(output) + ", weight dtype:" + XDtypeStr(weightDtype);
        throw std::runtime_error(err_str + " unsupported!");
    }
    xlite_trace::RecordGroupMatmul(rt, counts, start, end, weightDtype, transpose, weightNZ,
                                   static_cast<uint64_t>(outDim), static_cast<uint64_t>(inDim));

    launchKernel(rt.aicNum, rt.stream, in.ptr, weights.ptr, output.ptr, deqScales.ptr, counts.ptr,
                 counts.shape[0], outDim, inDim, -1, -1, -1, start, end, weightNZ, transpose,
                 rt.defaultMatmulSwizzle);
}

void XliteOpRopeCache(XRuntime &rt, XTensor &inout, XTensor &kCache, XTensor &vCache,
                      XTensor &position, XTensor &cossin, XTensor &slotMapping, uint32_t nHeads,
                      uint32_t nKvHeads, uint32_t headDim, uint32_t rotDim, uint32_t blockSize,
                      bool isNeox, uint64_t mropeMaskH, uint64_t mropeMaskW)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (!EachXDtype(FP16, inout, kCache, vCache, cossin) || nHeads != 16 || nKvHeads != 8 ||
        headDim != 128 || blockSize != 128 || kCache.shape.size() != 4 ||
        vCache.shape.size() != 4) {
        throw std::runtime_error(
            "Ascend310P RoPE-and-Cache requires FP16, Q16/KV8, head_dim=128 and 4D block-128 cache");
    }
#endif
    uint32_t localHeads = nHeads / rt.tpSize();
    uint32_t localKvHeads = nKvHeads / rt.tpSize();
    localKvHeads = localKvHeads == 0 ? 1 : localKvHeads;
    uintptr_t qPtr = reinterpret_cast<uintptr_t>(inout.ptr);
    uintptr_t kPtr =
        qPtr + static_cast<uint64_t>(localHeads) * headDim * XDtypeBit(inout.dtype) / 8;
    uintptr_t vPtr =
        kPtr + static_cast<uint64_t>(localKvHeads) * headDim * XDtypeBit(inout.dtype) / 8;
    void *k = reinterpret_cast<void *>(kPtr);
    void *v = reinterpret_cast<void *>(vPtr);
    float scale = 1.0f / sqrtf(static_cast<float>(headDim));

    if (!isNeox) {
        throw std::runtime_error(std::string(__func__) + ": unsupported rope type gptj");
    }

    KERNEL_PTR_TYPE(rope_and_cache) * launchKernel;
    if (EachXDtype(FP16, inout, kCache, vCache, cossin)) {
        launchKernel = aclrtlaunch_rope_and_cache_float16_t;
    } else if (EachXDtype(BF16, inout, kCache, vCache, cossin)) {
        launchKernel = aclrtlaunch_rope_and_cache_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(inout) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(position) + XT_STR(cossin) + XT_STR(slotMapping);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, position.ptr, inout.ptr, k, v, cossin.ptr, kCache.ptr,
                 vCache.ptr, slotMapping.ptr, inout.shape[0], rotDim, inout.shape[1],
                 inout.shape[1], inout.shape[1], localHeads, localKvHeads, headDim, blockSize,
                 scale, mropeMaskH, mropeMaskW);
}

void XliteOpAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache, XTensor &qk,
                      XTensor &output, XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                      XTensor &blockTables, uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim,
                      uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock)
{
    if (IsDummyRuntime(rt)) {
#ifdef XLITE_ARCH_310P
        XTensor &workspace =
            rt.GetTensor({XLITE_310P_ACLNN_WORKSPACE_BYTES}, INT8, DBG_LOC);
        rt.PutTensor(workspace);
#endif
        return;
    }
#ifdef XLITE_ARCH_310P
    (void)qk;
    (void)queryStartLoc;
    (void)lens;
    (void)cachedLens;
    (void)blockTables;
    (void)maxNumBlock;
    XliteAclnn310PAttention(rt, qkv, kCache, vCache, output, lens, cachedLens, blockTables,
                            maxNumBlock, nHeads, nKvHeads, headDim, blockSize, batch, true);
#else
    KERNEL_PTR_TYPE(attention) * launchKernel;
    if (EachXDtype(FP16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_attention_float16_t;
    } else if (EachXDtype(BF16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_attention_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qkv) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(qk) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, vCache.ptr, qk.ptr, output.ptr,
                 queryStartLoc.ptr, lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, nKvHeads,
                 headDim, blockSize, batch, maxNumBlock);
#endif
}

void XliteOpFlashAttention(XRuntime &rt, XTensor &qkv, XTensor &kCache, XTensor &vCache,
                           XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum, XTensor &lastMax,
                           XTensor &lastSum, XTensor &sync, XTensor &output, XTensor &queryStartLoc,
                           XTensor &lens, XTensor &cachedLens, XTensor &blockTables,
                           uint32_t nHeads, uint32_t nKvHeads, uint32_t headDim, uint32_t blockSize,
                           uint32_t batch, uint32_t maxNumBlock, uint32_t tileSizeOfCachedKV)
{
    if (IsDummyRuntime(rt)) {
#ifdef XLITE_ARCH_310P
        XTensor &workspace =
            rt.GetTensor({XLITE_310P_ACLNN_WORKSPACE_BYTES}, INT8, DBG_LOC);
        rt.PutTensor(workspace);
#endif
        return;
    }
#ifdef XLITE_ARCH_310P
    (void)qk;
    (void)sv;
    (void)max;
    (void)sum;
    (void)lastMax;
    (void)lastSum;
    (void)sync;
    (void)queryStartLoc;
    (void)lens;
    (void)cachedLens;
    (void)blockTables;
    (void)maxNumBlock;
    (void)tileSizeOfCachedKV;
    XliteAclnn310PAttention(rt, qkv, kCache, vCache, output, lens, cachedLens, blockTables,
                            maxNumBlock, nHeads, nKvHeads, headDim, blockSize, batch, false);
#else
    KERNEL_PTR_TYPE(flash_attention) * launchKernel;
    if (EachXDtype(FP16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_flash_attention_float16_t;
    } else if (EachXDtype(BF16, qkv, qk, kCache, vCache, output)) {
        launchKernel = aclrtlaunch_flash_attention_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qkv) + XT_STR(kCache) + XT_STR(vCache) +
                              XT_STR(qk) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, qkv.ptr, kCache.ptr, vCache.ptr, qk.ptr, sv.ptr, max.ptr,
                 sum.ptr, lastMax.ptr, lastSum.ptr, sync.ptr, output.ptr, queryStartLoc.ptr,
                 lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, nKvHeads, headDim, blockSize,
                 batch, maxNumBlock, tileSizeOfCachedKV);
#endif
}

void XliteOpMLAV2(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kCache, XTensor &peCache,
                  XTensor &qk, XTensor &oAbsorb, XTensor &queryStartLoc, XTensor &lens,
                  XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                  uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch, uint32_t maxNumBlocks,
                  float scale, uint32_t topK, const XTensor &topkIndices)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (topK != 0 && maxNumBlocks * blockSize > MAX_SOFTMAX_PINGPONG_LEN) {
        throw std::runtime_error(std::string(__func__) +
                                 ": topK > 0 is not supported when maxNumBlocks (" +
                                 std::to_string(maxNumBlocks) + ") * blockSize > " +
                                 std::to_string(MAX_SOFTMAX_PINGPONG_LEN));
    }
    if (topK > MAX_TOPK_NUM) {
        throw std::runtime_error(std::string(__func__) + ": topK should be less than or equal to " +
                                 std::to_string(MAX_TOPK_NUM));
    }
    if (EachXDtype(BF16, qAbsorb, qr, kCache, peCache, oAbsorb)) {
        aclrtlaunch_mla_v2_bfloat16_t(
            rt.aicNum, rt.stream, qAbsorb.ptr, qr.ptr, kCache.ptr, peCache.ptr, topkIndices.ptr,
            qk.ptr, oAbsorb.ptr, queryStartLoc.ptr, lens.ptr, cachedLens.ptr, blockTables.ptr,
            nHeads, ropeHeadDim, kvLoraRank, blockSize, batch, maxNumBlocks, scale, topK);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qAbsorb) + XT_STR(qr) + XT_STR(kCache) +
                              XT_STR(peCache) + XT_STR(oAbsorb);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpFlashMLAV2(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kCache,
                       XTensor &peCache, XTensor &qk, XTensor &sv, XTensor &max, XTensor &sum,
                       XTensor &lastMax, XTensor &lastSum, XTensor &sync, XTensor &oAbsorb,
                       XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                       XTensor &blockTables, uint32_t nHeads, uint32_t ropeHeadDim,
                       uint32_t kvLoraRank, uint32_t blockSize, uint32_t batch,
                       uint32_t maxNumBlocks, float scale, uint32_t tileSizeOfCachedKV,
                       uint32_t topK, const XTensor &topkIndices)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (tileSizeOfCachedKV > MAX_SOFTMAX_PINGPONG_LEN) {
        throw std::runtime_error(std::string(__func__) + ": tile size of kv " +
                                 std::to_string(tileSizeOfCachedKV) + " > " +
                                 std::to_string(MAX_SOFTMAX_PINGPONG_LEN));
    }
    if (topK > MAX_TOPK_NUM) {
        throw std::runtime_error(std::string(__func__) + ": topK should be less than or equal to " +
                                 std::to_string(MAX_TOPK_NUM));
    }
    if (EachXDtype(BF16, qAbsorb, qr, kCache, peCache, oAbsorb)) {
        aclrtlaunch_flash_mla_v2_bfloat16_t(
            rt.aicNum, rt.stream, qAbsorb.ptr, qr.ptr, kCache.ptr, peCache.ptr, topkIndices.ptr,
            qk.ptr, sv.ptr, max.ptr, sum.ptr, lastMax.ptr, lastSum.ptr, sync.ptr, oAbsorb.ptr,
            queryStartLoc.ptr, lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, ropeHeadDim,
            kvLoraRank, blockSize, batch, maxNumBlocks, scale, tileSizeOfCachedKV, topK);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qAbsorb) + XT_STR(qr) + XT_STR(kCache) +
                              XT_STR(peCache) + XT_STR(oAbsorb);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpGatherSparseKVCache(XRuntime &rt, XTensor &kCache, XTensor &peCache,
                                XTensor &blockTables, XTensor &topkIndices, XTensor &queryLens,
                                XTensor &cachedLens, XTensor &kDenseCache, XTensor &peDenseCache,
                                uint32_t batch, uint32_t indexTopK, uint32_t blockSize,
                                uint32_t maxNumBlocks, uint32_t kvLoraRank, uint32_t ropeHeadDim,
                                uint32_t kvHeads)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (kvHeads != 1) {
        throw std::runtime_error(std::string(__func__) +
                                 ": kvHeads should be less than or equal to 1");
    }
    if (EachXDtype(BF16, kCache, peCache, kDenseCache, peDenseCache)) {
        aclrtlaunch_gather_sparse_kv_cache_bfloat16_t(
            rt.aivNum, rt.stream, kCache.ptr, peCache.ptr, blockTables.ptr, topkIndices.ptr,
            queryLens.ptr, cachedLens.ptr, kDenseCache.ptr, peDenseCache.ptr, batch, indexTopK,
            blockSize, maxNumBlocks, kvLoraRank, ropeHeadDim);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(kCache) + XT_STR(peCache) + XT_STR(kDenseCache) +
                              XT_STR(peDenseCache);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpMLAV3(XRuntime &rt, XTensor &qAbsorb, XTensor &qr, XTensor &kDenseCache,
                  XTensor &peDenseCache, XTensor &qk, XTensor &oAbsorb, XTensor &queryStartLoc,
                  XTensor &lens, XTensor &cachedLens, uint32_t nHeads, uint32_t ropeHeadDim,
                  uint32_t kvLoraRank, uint32_t batch, uint32_t indexTopK, float scale)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (EachXDtype(BF16, qAbsorb, qr, kDenseCache, peDenseCache, oAbsorb)) {
        aclrtlaunch_mla_v3_bfloat16_t(rt.aicNum, rt.stream, qAbsorb.ptr, qr.ptr, kDenseCache.ptr,
                                      peDenseCache.ptr, qk.ptr, oAbsorb.ptr, queryStartLoc.ptr,
                                      lens.ptr, cachedLens.ptr, nHeads, ropeHeadDim, kvLoraRank,
                                      batch, indexTopK, scale);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(qAbsorb) + XT_STR(qr) + XT_STR(kDenseCache) +
                              XT_STR(peDenseCache) + XT_STR(oAbsorb);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpRopeComplex(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim, uint32_t outStepDim,
                        uint32_t ropeDim, uint32_t offset, uint32_t outOffset, XTensor &inputWithR,
                        XTensor &freqs, XTensor &position, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

#ifdef XLITE_ARCH_310P
    throw std::runtime_error("Ascend310P LLM FP16 POC does not support complex RoPE/MLA");
#else
    KERNEL_PTR_TYPE(rope_complex_and_cache) * launchKernel;
    if (inputWithR.dtype == FP16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_float16_t;
    } else if (inputWithR.dtype == BF16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(inputWithR);
        throw std::runtime_error(err_str + " TODO");
    }
    launchKernel(rt.aivNum, rt.stream, inputWithR.shape[0], nLocalHeads, stepDim, ropeDim, offset,
                 0, inputWithR.ptr, output.ptr, outStepDim, outOffset, freqs.ptr, position.ptr, 0,
                 nullptr, nullptr);
#endif
}

void XliteOpRopeComplexAndCache(XRuntime &rt, uint32_t nLocalHeads, uint32_t stepDim,
                                uint32_t ropeDim, uint32_t offset, uint32_t vdim,
                                XTensor &inputWithR, XTensor &freqs, XTensor &position,
                                uint32_t blockSize, XTensor &vCache, XTensor &slotMapping)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

#ifdef XLITE_ARCH_310P
    throw std::runtime_error("Ascend310P LLM FP16 POC does not support complex RoPE/MLA cache");
#else
    KERNEL_PTR_TYPE(rope_complex_and_cache) * launchKernel;
    if (inputWithR.dtype == FP16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_float16_t;
    } else if (inputWithR.dtype == BF16) {
        launchKernel = aclrtlaunch_rope_complex_and_cache_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(inputWithR);
        throw std::runtime_error(err_str + " TODO");
    }
    launchKernel(rt.aivNum, rt.stream, inputWithR.shape[0], nLocalHeads, stepDim, ropeDim, offset,
                 vdim, inputWithR.ptr, nullptr, 0, 0, freqs.ptr, position.ptr, blockSize,
                 vCache.ptr, slotMapping.ptr);
#endif
}

void XliteOpMlaPrepare(XRuntime &rt, XTensor &attnQkvc, const XTensor &qNorm,
                       const XTensor &qNormBias, XTensor &attnNormQc, const XTensor &kvNorm,
                       const XTensor &kvNormBias, const XTensor &freqs, const XTensor &position,
                       uint32_t qLoraRank, uint32_t kvLoraRank, uint32_t ropeHeadDim,
                       uint32_t blockSize, XTensor &kCache, XTensor &peCache,
                       const XTensor &slotMapping, float normEps, const XTensor &attnNormKvc)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    KERNEL_PTR_TYPE(mla_prepare) * launchKernel;
    if (attnQkvc.dtype == FP16) {
        launchKernel = aclrtlaunch_mla_prepare_float16_t;
    } else if (attnQkvc.dtype == BF16) {
        launchKernel = aclrtlaunch_mla_prepare_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(attnQkvc);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, attnQkvc.ptr, qNorm.ptr, qNormBias.ptr, attnNormQc.ptr,
                 kvNorm.ptr, kvNormBias.ptr, attnNormKvc.ptr, freqs.ptr, position.ptr, kCache.ptr,
                 peCache.ptr, slotMapping.ptr, attnQkvc.shape[0], qLoraRank, kvLoraRank,
                 ropeHeadDim, blockSize, normEps, rt.tpSize());
}

void XliteOpQkRmsNorm(XRuntime &rt, XTensor &in, const XTensor &qNorm, const XTensor &qNormBias,
                      const XTensor &kNorm, const XTensor &kNormBias, XTensor &out, float normEps,
                      uint32_t qNormDim, uint32_t qCntPerToken, uint32_t kNormDim,
                      uint32_t kCntPerToken, uint32_t kStartOffset, bool useNorm,
                      const XTensor &qVariance, const XTensor &kVariance)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    if (in.dtype != FP16 || out.dtype != FP16 || qNorm.dtype != FP16 || kNorm.dtype != FP16 ||
        qNormBias.ptr != nullptr || kNormBias.ptr != nullptr || qNormDim != 128 ||
        kNormDim != 128 || qCntPerToken != 16 || kCntPerToken != 8 || kStartOffset != 2048 ||
        !useNorm || qVariance.ptr != nullptr || kVariance.ptr != nullptr) {
        throw std::runtime_error(
            "Ascend310P Q/K RMSNorm requires FP16, head_dim=128 and Q16/KV8");
    }
#endif
    KERNEL_PTR_TYPE(qk_rms_norm) * launchKernel;
    if (in.dtype == FP16 && (out.dtype == FP16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_qk_rms_norm_float16_t;
    } else if (in.dtype == BF16 && (out.dtype == BF16 || out.dtype == FP32)) {
        launchKernel = aclrtlaunch_qk_rms_norm_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(out) + XT_STR(qNorm) + XT_STR(kNorm);
        throw std::runtime_error(err_str + " unsupported!");
    }
    void *qOutPtr = useNorm ? out.ptr : qVariance.ptr;
    void *kOutPtr = useNorm ? out.ptr : kVariance.ptr;
    void *qVarArg = useNorm ? qVariance.ptr : nullptr;
    void *kVarArg = useNorm ? kVariance.ptr : nullptr;
    uint32_t outStep = useNorm ? out.shape[1] : 1;
    launchKernel(rt.aivNum, rt.stream, in.ptr, qNorm.ptr, qNormBias.ptr, qOutPtr, kNorm.ptr,
                 kNormBias.ptr, kOutPtr, in.shape[0], qNormDim, qCntPerToken, kNormDim,
                 kCntPerToken, in.shape[1], outStep, normEps, kStartOffset, useNorm, qVarArg,
                 kVarArg, rt.tpSize());
}

void XliteOpIndexerPrepare(XRuntime &rt, XTensor &kw, const XTensor &kNorm,
                           const XTensor &kNormBias, const XTensor &freqs, const XTensor &position,
                           uint32_t indexHeadDim, uint32_t indexNHeads, uint32_t ropeHeadDim,
                           uint32_t blockSize, XTensor &indexKCache, const XTensor &slotMapping,
                           float normEps, const XTensor &q, float scale, uint32_t topK, bool isLong,
                           uint32_t tpSize)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (tpSize != 1) {
        throw std::runtime_error(std::string(__func__) + ": tpSize should be 1, got " +
                                 std::to_string(tpSize));
    }

    KERNEL_PTR_TYPE(indexer_prepare) * launchKernel;
    if (kw.dtype == FP16) {
        launchKernel = aclrtlaunch_indexer_prepare_float16_t;
    } else if (kw.dtype == BF16) {
        launchKernel = aclrtlaunch_indexer_prepare_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(kw);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, kw.ptr, kNorm.ptr, kNormBias.ptr, freqs.ptr, position.ptr,
                 kw.shape[0], indexHeadDim, indexNHeads, ropeHeadDim, blockSize, normEps,
                 indexKCache.ptr, slotMapping.ptr, q.ptr, scale, topK, isLong);
}

void XliteOpAddBias(XRuntime &rt, XTensor &input, XTensor &weight, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(add_bias) * launchKernel;
    if (EachXDtype(FP32, input, weight, output)) {
        launchKernel = aclrtlaunch_add_bias_float;
    } else if (EachXDtype(FP16, input, weight, output)) {
        launchKernel = aclrtlaunch_add_bias_float16_t;
    } else if (EachXDtype(BF16, input, weight, output)) {
        launchKernel = aclrtlaunch_add_bias_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(weight) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, input.ptr, weight.ptr, output.ptr,
                 output.shape[0] * output.shape[1], output.shape[1]);
}

void XliteOpSoftmaxTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outWeights,
                        XTensor &outRouting, uint32_t topK, bool normTopKProb)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(softmax_topk) * launchKernel;
    if (scores.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 &&
        outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_softmax_topk_float;
    } else if (scores.dtype == BF16 && indices.dtype == INT32 && outWeights.dtype == BF16 &&
               outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_softmax_topk_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(scores) + XT_STR(indices) + XT_STR(outWeights) + XT_STR(outRouting);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, scores.ptr, indices.ptr, outWeights.ptr, outRouting.ptr,
                 scores.shape[0], indices.shape[0], topK, normTopKProb);
}

void XliteOpSigmoidTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &bias, float scale,
                        XTensor &outWeights, XTensor &outRouting, uint32_t nGroup,
                        uint32_t nTopkGroup, uint32_t topK, bool normTopKProb)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(sigmoid_topk) * launchKernel;
    if (scores.dtype == FP32 && indices.dtype == INT32 && outWeights.dtype == FP32 &&
        outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_sigmoid_topk_float;
    } else if (scores.dtype == BF16 && indices.dtype == INT32 && outWeights.dtype == BF16 &&
               outRouting.dtype == BIT1) {
        launchKernel = aclrtlaunch_sigmoid_topk_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(scores) + XT_STR(indices) + XT_STR(outWeights) + XT_STR(outRouting);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, scores.ptr, indices.ptr, bias.ptr, scale, outWeights.ptr,
                 outRouting.ptr, scores.shape[0], indices.shape[0], nGroup, nTopkGroup, topK,
                 normTopKProb);
}

void XliteOpTopK(XRuntime &rt, XTensor &scores, XTensor &indices, XTensor &outIndices,
                 XTensor &queryLens, XTensor &cachedLens, uint32_t batch, size_t k)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    uint32_t maxSeqLen = scores.shape[1];
    if (maxSeqLen <= k) {
        return;
    }

    if (k != 2048) {
        throw std::runtime_error(std::string(__func__) + ": only K=2048 is supported, got " +
                                 std::to_string(k));
    }

    KERNEL_PTR_TYPE(topk) * launchKernel;
    if (scores.dtype == BF16 && indices.dtype == INT32) {
        launchKernel = aclrtlaunch_topk_bfloat16_t;
    } else if (scores.dtype == FP32 && indices.dtype == INT32) {
        launchKernel = aclrtlaunch_topk_float;
    } else {
        throw std::runtime_error(std::string(__func__) + ": unsupported!" +
                                 std::to_string(scores.dtype));
    }
    launchKernel(rt.aivNum, rt.stream, scores.ptr, indices.ptr, outIndices.ptr, queryLens.ptr,
                 cachedLens.ptr, maxSeqLen, batch, k);
}

void XliteOpSoftmax(XRuntime &rt, uint32_t calcLen, XTensor &x)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(softmax) * launchKernel;
    if (x.dtype == FP16) {
        launchKernel = aclrtlaunch_softmax_float16_t;
    } else if (x.dtype == BF16) {
        launchKernel = aclrtlaunch_softmax_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(1, rt.stream, x.ptr, x.shape[0], x.shape[1], calcLen);
}

void XliteOpSoftmaxLong(XRuntime &rt, uint32_t calcLen, XTensor &x, XTensor &expBuf)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(softmax_long) * launchKernel;
    if (x.dtype == FP16) {
        launchKernel = aclrtlaunch_softmax_long_float16_t;
    } else if (x.dtype == BF16) {
        launchKernel = aclrtlaunch_softmax_long_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(1, rt.stream, x.ptr, expBuf.ptr, x.shape[0], x.shape[1], calcLen);
}

// out = int8(x / scale + offset), turn scale to 1/scale before calculation
void XliteOpQuant(XRuntime &rt, XTensor &x, XTensor &scale_reciprocal, XTensor &offset,
                  XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (x.ptr == nullptr || scale_reciprocal.ptr == nullptr || offset.ptr == nullptr ||
        out.ptr == nullptr) {
        std::string err_str =
            DBG_PREFIX + XT_STR(x) + XT_STR(scale_reciprocal) + XT_STR(offset) + XT_STR(out);
        throw std::runtime_error(err_str + " null pointer!");
    }
    size_t m = x.shape[0];
    size_t n = x.shape[1];
    if (x.dtype == BF16) {
        aclrtlaunch_quant_bf16_to_i8_static(rt.aivNum, rt.stream, x.ptr, scale_reciprocal.ptr,
                                            offset.ptr, out.ptr, m, n);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpQuantDyn(XRuntime &rt, XTensor &x, XTensor &scale, XTensor &out, const XTensor &num)
{
    if (IsDummyRuntime(rt) || x.numel == 0) {
        return;
    }
    if (x.ptr == nullptr || scale.ptr == nullptr || out.ptr == nullptr) {
        std::string err_str = DBG_PREFIX + XT_STR(x) + XT_STR(scale) + XT_STR(out);
        throw std::runtime_error(err_str + " null pointer!");
    }
    size_t m = x.shape[0];
    size_t n = x.shape[1];
    if (x.dtype == BF16) {
        aclrtlaunch_quant_bf16_to_i8_dynamic(rt.aivNum, rt.stream, x.ptr, scale.ptr, out.ptr,
                                             num.ptr, m, n);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(x);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpMSDMergeDequant(XRuntime &rt, XTensor &yMerged, XTensor &scaleBiasPtrs, XTensor &counts,
                            uint32_t start, uint32_t end, XTensor &perTokenScale, XTensor &out)
{
    if (IsDummyRuntime(rt) || yMerged.numel == 0) {
        return;
    }

    std::string err_str =
        DBG_PREFIX + XT_STR(yMerged) + XT_STR(scaleBiasPtrs) + XT_STR(perTokenScale) + XT_STR(out);

    if (yMerged.dtype == FP16 && perTokenScale.dtype == FP32 && out.dtype == BF16) {
        if (yMerged.shape.size() != 2 || yMerged.shape[0] % 2 == 1) {
            throw std::runtime_error(err_str + " yMerged.shape is invalid!");
        }
        uint32_t m = yMerged.shape[0] / 2;
        uint32_t n = yMerged.shape[1];
        aclrtlaunch_msd_merge_dequant_int8_t(rt.aivNum, rt.stream, yMerged.ptr, scaleBiasPtrs.ptr,
                                             perTokenScale.ptr, out.ptr, nullptr, m, n, counts.ptr,
                                             start, end);
    } else {
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpDeQuant(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &scale,
                    const XTensor &num)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    size_t m = in.shape[0];
    size_t n = in.numel / in.shape[0];
    if (in.dtype == FP16) {
        aclrtlaunch_dequant_float16_t(rt.aivNum, rt.stream, in.ptr, scale.ptr, out.ptr, num.ptr, m,
                                      n);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out,
                          const XTensor &quantBias, const XTensor &weightScale, bool weightNZ,
                          bool transpose, const XTensor &outScale, const XTensor &num)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    if (in.dtype == INT8 && weight.dtype == INT8 && out.dtype == BF16) {
        out.View(FP16);
        XliteOpMatmul(rt, in, weight, out, weightNZ, quantBias, weightScale, transpose);
        XliteOpDeQuant(rt, out, out, outScale, num);
        out.View(BF16);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(in) + XT_STR(weight) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpGroupMatmulDeQuant(XRuntime &rt, XTensor &in, XTensor &weights, XTensor &deqScales,
                               XTensor &counts, uint32_t start, uint32_t end, XDtype weightDtype,
                               long outDim, long inDim, XTensor &output, XTensor &outScale,
                               XTensor &num, bool weightNZ, bool transpose)
{
    if (IsDummyRuntime(rt) || in.numel == 0) {
        return;
    }
    if (in.dtype == INT8 && weightDtype == INT8 && output.dtype == BF16) {
        output.View(FP16);
        XliteOpGroupMatmul(rt, in, weights, deqScales, counts, start, end, weightDtype, outDim,
                           inDim, output, weightNZ, transpose);
        XliteOpDeQuant(rt, output, output, outScale, num);
        output.View(BF16);
    } else {
        std::string err_str = DBG_PREFIX;
        err_str += XT_STR(in) + XT_STR(output) + "weight dtype:" + XDtypeStr(weightDtype);
        throw std::runtime_error(err_str + " unsupported!");
    }
}

void XliteOpConcat(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    // Concat kernel supports up to maxInputs inputs
#ifndef XLITE_ARCH_310P
    constexpr uint32_t maxInputs = 8;
    if (inputs.size() <= maxInputs) {
        void *ptrs[maxInputs] = {nullptr};
        uint64_t sizes[maxInputs] = {0};
        uint64_t totalBytes = 0;
        for (uint32_t i = 0; i < inputs.size(); i++) {
            ptrs[i] = inputs[i].ptr;
            sizes[i] = inputs[i].bytes;
            totalBytes += sizes[i];
        }
        // Scale block count to data size: small transfers use few blocks to avoid
        // multi-core launch+barrier overhead; large transfers saturate all cores.
        uint32_t numBlocks = CopyKernelBlockNum(rt, totalBytes);
        aclrtlaunch_concat(numBlocks, rt.stream, out.ptr, ptrs[0], ptrs[1], ptrs[2], ptrs[3],
                           ptrs[4], ptrs[5], ptrs[6], ptrs[7], sizes[0], sizes[1], sizes[2],
                           sizes[3], sizes[4], sizes[5], sizes[6], sizes[7],
                           static_cast<uint32_t>(inputs.size()), totalBytes);
        return;
    }
#endif

    // Fallback for the rare > maxInputs case
    size_t offset = 0;
    for (const auto &tensor : inputs) {
        size_t bytes = tensor.bytes;
        void *dst = static_cast<uint8_t *>(out.ptr) + offset;
        CHECK_ACL(aclrtMemcpyAsync(dst, bytes, tensor.ptr, bytes, ACL_MEMCPY_DEVICE_TO_DEVICE,
                                   rt.stream));
        offset += bytes;
    }
}
void XliteOpConcatCol(XRuntime &rt, const std::vector<XTensor> &inputs, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    size_t times =
        std::accumulate(inputs[0].shape.begin(), inputs[0].shape.end() - 1, 1, std::multiplies());

    size_t totalLastDim = 0;
    for (const auto &tensor : inputs) {
        totalLastDim += tensor.shape.back();
    }
    size_t elemSize = XDtypeBit(inputs[0].dtype) / 8;
    size_t outRowStride = totalLastDim * elemSize;
    for (size_t h = 0; h < times; ++h) {
        size_t dstOffset = h * outRowStride;
        for (const auto &tensor : inputs) {
            size_t rowBytes = tensor.shape.back() * elemSize;
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(out.ptr) + dstOffset, rowBytes,
                                       static_cast<uint8_t *>(tensor.ptr) + h * rowBytes, rowBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            dstOffset += rowBytes;
        }
    }
}

void XliteOpSplitCol(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    size_t height = std::accumulate(in.shape.begin(), in.shape.end() - 1, 1, std::multiplies());
    size_t elemSize = XDtypeBit(in.dtype) / 8;
    for (size_t h = 0; h < height; ++h) {
        size_t srcOffset = h * in.shape.back() * elemSize;
        for (auto cur : outputs) {
            size_t rowBytes = cur.shape.back() * elemSize;
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(cur.ptr) + h * rowBytes, rowBytes,
                                       static_cast<uint8_t *>(in.ptr) + srcOffset, rowBytes,
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            srcOffset += rowBytes;
        }
    }
}

void XliteOpSplit(XRuntime &rt, XTensor &in, const std::vector<XTensor> &outputs,
                  const std::vector<size_t> &sizes, uint32_t numPackets)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

    // 计算总大小
    size_t totalSize = 0;
    for (size_t size : sizes) {
        totalSize += size;
    }

    // Split kernel supports up to maxOutputs outputs
#ifndef XLITE_ARCH_310P
    constexpr uint32_t maxOutputs = 8;
    if (outputs.size() <= maxOutputs && sizes.size() == outputs.size()) {
        void *ptrs[maxOutputs] = {nullptr};
        uint64_t s[maxOutputs] = {0};
        for (uint32_t i = 0; i < outputs.size(); i++) {
            ptrs[i] = outputs[i].ptr;
            s[i] = sizes[i];
        }
        // Total bytes the kernel copies = numPackets * totalSize; scale block count
        // to data size to avoid multi-core launch+barrier overhead on tiny transfers.
        uint32_t numBlocks = CopyKernelBlockNum(rt, static_cast<uint64_t>(numPackets) * totalSize);
        aclrtlaunch_split(numBlocks, rt.stream, in.ptr, ptrs[0], ptrs[1], ptrs[2], ptrs[3], ptrs[4],
                          ptrs[5], ptrs[6], ptrs[7], s[0], s[1], s[2], s[3], s[4], s[5], s[6], s[7],
                          static_cast<uint32_t>(outputs.size()), numPackets,
                          static_cast<uint64_t>(totalSize));
        return;
    }
#endif

    // Fallback for the rare > maxOutputs case
    for (uint32_t i = 0; i < numPackets; i++) {
        uint8_t *srcBase = static_cast<uint8_t *>(in.ptr) + i * totalSize;
        size_t srcOffset = 0;

        for (size_t j = 0; j < outputs.size(); j++) {
            CHECK_ACL(aclrtMemcpyAsync(static_cast<uint8_t *>(outputs[j].ptr) + i * sizes[j],
                                       sizes[j], srcBase + srcOffset, sizes[j],
                                       ACL_MEMCPY_DEVICE_TO_DEVICE, rt.stream));
            srcOffset += sizes[j];
        }
    }
}

void XliteOpIndexerScores(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight,
                          XTensor &scores, XTensor &queryStartLoc, XTensor &lens,
                          XTensor &cachedLens, XTensor &blockTables, uint32_t nHeads,
                          uint32_t headDim, uint32_t blockSize, uint32_t batch,
                          uint32_t maxNumBlock)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(indexer_scores) * launchKernel;
    if (EachXDtype(FP16, q, kCache, weight, scores)) {
        launchKernel = aclrtlaunch_indexer_scores_float16_t;
    } else if (EachXDtype(BF16, q, kCache, weight, scores)) {
        launchKernel = aclrtlaunch_indexer_scores_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(q) + XT_STR(kCache) + XT_STR(weight) + XT_STR(scores);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, q.ptr, kCache.ptr, weight.ptr, scores.ptr, queryStartLoc.ptr,
                 lens.ptr, cachedLens.ptr, blockTables.ptr, nHeads, headDim, blockSize, batch,
                 maxNumBlock);
}

void XliteOpIndexerTopK(XRuntime &rt, XTensor &q, XTensor &kCache, XTensor &weight, XTensor &scores,
                        XTensor &lastTopk, XTensor &indices, XTensor &topkIndices,
                        XTensor &queryStartLoc, XTensor &lens, XTensor &cachedLens,
                        XTensor &blockTables, XTensor &sync, uint32_t nHeads, uint32_t headDim,
                        uint32_t blockSize, uint32_t batch, uint32_t maxNumBlock, uint32_t topK)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (topK > MAX_TOPK_NUM) {
        throw std::runtime_error(std::string(__func__) + ": topK should be less than or equal to " +
                                 std::to_string(MAX_TOPK_NUM));
    }
    KERNEL_PTR_TYPE(indexer_topk) * launchKernel;
    if (EachXDtype(FP16, q, kCache, weight, scores)) {
        launchKernel = aclrtlaunch_indexer_topk_float16_t;
    } else if (EachXDtype(BF16, q, kCache, weight, scores)) {
        launchKernel = aclrtlaunch_indexer_topk_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(q) + XT_STR(kCache) + XT_STR(weight) +
                              XT_STR(scores) + XT_STR(indices) + XT_STR(topkIndices);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aicNum, rt.stream, q.ptr, kCache.ptr, weight.ptr, queryStartLoc.ptr, lens.ptr,
                 cachedLens.ptr, blockTables.ptr, scores.ptr, lastTopk.ptr, indices.ptr,
                 topkIndices.ptr, sync.ptr, nHeads, headDim, blockSize, batch, maxNumBlock, topK);
}

void XliteOpMuls(XRuntime &rt, XTensor &input, float scale, XTensor &output, uint32_t calcOffset,
                 uint32_t calcNum)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(muls) * launchKernel;
    if (EachXDtype(FP16, input, output)) {
        launchKernel = aclrtlaunch_muls_float16_t;
    } else if (EachXDtype(BF16, input, output)) {
        launchKernel = aclrtlaunch_muls_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    uint32_t shape0 = input.shape[0];
    uint32_t shape1 = input.shape.size() >= 2 ? input.shape[1] : 1;
    if (calcOffset >= shape1) {
        throw std::runtime_error(std::string(__func__) + ": calcOffset " +
                                 std::to_string(calcOffset) + " >= shape1 " +
                                 std::to_string(shape1));
    }
    if (calcNum > shape1 - calcOffset) {
        calcNum = shape1 - calcOffset;
    }
    if (calcNum == 0) {
        throw std::runtime_error(std::string(__func__) + ": calcNum should be > 0");
    } else if (calcNum > MAX_MULS_CALC_NUM) {
        throw std::runtime_error(std::string(__func__) +
                                 ": calcNum should be <= " + std::to_string(MAX_MULS_CALC_NUM));
    }
    launchKernel(rt.aivNum, rt.stream, input.ptr, scale, output.ptr, shape0, shape1, calcOffset,
                 calcNum);
}

void XliteOpExpertsCountsSum(XRuntime &rt, XTensor &expertsCountsInput, XTensor &tokensPerEpgroup,
                             XTensor &expertsCountsOutput, uint32_t nRoutedExperts)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
#ifdef XLITE_ARCH_310P
    throw std::runtime_error("Ascend310P LLM FP16 POC does not support expert/MoE kernels");
#else
    aclrtlaunch_experts_counts_sum(rt.aivNum, rt.stream, expertsCountsInput.ptr,
                                   tokensPerEpgroup.ptr, expertsCountsOutput.ptr, nRoutedExperts,
                                   tokensPerEpgroup.shape[0]);
#endif
}

void XliteOpReorderMoE(XRuntime &rt, XTensor &in, XTensor &out, const XTensor &counts,
                       uint32_t hiddenSize, uint32_t localStart, uint32_t localEnd, bool forward)
{
    if (IsDummyRuntime(rt)) {
        return;
    }

#ifdef XLITE_ARCH_310P
    throw std::runtime_error("Ascend310P LLM FP16 POC does not support reorder-MoE");
#else
    if (in.numel == 0 || localStart >= localEnd) {
        return;
    }

    uint32_t moeEpSize = counts.shape[0];
    uint32_t nRoutedExperts = counts.shape[1];
    uint32_t elemBytes = XDtypeBit(in.dtype) / 8;

    aclrtlaunch_reorder_moe(rt.aivNum, rt.stream, in.ptr, out.ptr, counts.ptr, moeEpSize,
                            nRoutedExperts, hiddenSize, localStart, localEnd, forward ? 1 : 0,
                            elemBytes);
#endif
}
void XliteOpTranspose_1_2(XRuntime &rt, XTensor &input, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(transpose_1_2) * launchKernel;
    if (EachXDtype(FP16, input, output)) {
        launchKernel = aclrtlaunch_transpose_1_2_float16_t;
    } else if (EachXDtype(BF16, input, output)) {
        launchKernel = aclrtlaunch_transpose_1_2_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, input.ptr, output.ptr, input.shape[0], input.shape[1],
                 input.shape[2]);
}

void XliteOpConv1dAndSiLU(XRuntime &rt, XTensor &state, XTensor &input, XTensor &weight,
                          XTensor &output, bool updateState)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (state.shape.size() != 3 || input.shape.size() != 3 || output.shape.size() != 3) {
        throw std::runtime_error("XliteOpConv1dAndSiLU: state/input/output must be 3D [B,C,*]");
    }
    if (state.shape[0] != input.shape[0] || state.shape[1] != input.shape[1] ||
        output.shape[0] != input.shape[0] || output.shape[1] != input.shape[1] ||
        output.shape[2] != input.shape[2]) {
        throw std::runtime_error("XliteOpConv1dAndSiLU: batch/channel/seq shape mismatch");
    }
    uint32_t kernelDim = weight.shape.size() >= 3 ? weight.shape[2] : weight.shape.back();
    if (state.shape[2] != kernelDim) {
        throw std::runtime_error("XliteOpConv1dAndSiLU: state last dim != kernelDim");
    }
    if (weight.shape[0] != input.shape[1]) {
        throw std::runtime_error("XliteOpConv1dAndSiLU: weight channels mismatch");
    }
    uint32_t batch = input.shape[0];
    uint32_t channels = input.shape[1];
    uint32_t seqLen = input.shape[2];

    KERNEL_PTR_TYPE(conv1d_and_silu) * launchKernel;
    if (EachXDtype(FP32, state, input, weight, output)) {
        launchKernel = aclrtlaunch_conv1d_and_silu_float;
    } else if (EachXDtype(FP16, state, input, weight, output)) {
        launchKernel = aclrtlaunch_conv1d_and_silu_float16_t;
    } else if (EachXDtype(BF16, state, input, weight, output)) {
        launchKernel = aclrtlaunch_conv1d_and_silu_bfloat16_t;
    } else {
        std::string err_str =
            DBG_PREFIX + XT_STR(state) + XT_STR(input) + XT_STR(weight) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }

    if (kernelDim > 16 || seqLen > 4096) {
        throw std::runtime_error(
            "XliteOpConv1dAndSiLU: require kernelDim<=16 and seqLen<=4096 for fused kernel");
    }
    launchKernel(rt.aivNum, rt.stream, state.ptr, input.ptr, weight.ptr, output.ptr, batch,
                 channels, seqLen, kernelDim, updateState ? 1u : 0u);
}

void XliteOpBetaDecay(XRuntime &rt, XTensor &b, XTensor &a, XTensor &A_log, XTensor &dt_bias,
                      XTensor &beta, XTensor &g, uint32_t bsz, uint32_t seqlen,
                      uint32_t num_v_heads)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(beta_decay) * launchKernel;
    if (EachXDtype(FP32, b, a)) {
        launchKernel = aclrtlaunch_beta_decay_float;
    } else if (EachXDtype(FP16, b, a)) {
        launchKernel = aclrtlaunch_beta_decay_float16_t;
    } else if (EachXDtype(BF16, b, a)) {
        launchKernel = aclrtlaunch_beta_decay_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(b) + XT_STR(a);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, b.ptr, a.ptr, A_log.ptr, dt_bias.ptr, beta.ptr, g.ptr, bsz,
                 seqlen, num_v_heads);
}

void XliteOpSigmoidGateMul(XRuntime &rt, XTensor &attn, XTensor &gate, XTensor &out)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (attn.shape.size() < 2 || gate.shape.size() < 2 || out.shape.size() < 2) {
        throw std::runtime_error(std::string(__func__) + ": attn/gate/out must be 2D");
    }
    if (attn.shape[0] != gate.shape[0] || attn.shape[1] != gate.shape[1] ||
        attn.shape[0] != out.shape[0] || attn.shape[1] != out.shape[1]) {
        throw std::runtime_error(std::string(__func__) + ": attn/gate/out shape mismatch");
    }
    KERNEL_PTR_TYPE(sigmoid_gate_mul) * launchKernel;
    if (EachXDtype(FP16, attn, gate, out)) {
        launchKernel = aclrtlaunch_sigmoid_gate_mul_float16_t;
    } else if (EachXDtype(BF16, attn, gate, out)) {
        launchKernel = aclrtlaunch_sigmoid_gate_mul_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(attn) + XT_STR(gate) + XT_STR(out);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, attn.ptr, gate.ptr, out.ptr,
                 static_cast<uint32_t>(attn.shape[0]), static_cast<uint32_t>(attn.shape[1]));
}

void XliteOpRecurrentGatedDeltaRule(XRuntime &rt, XTensor &query, XTensor &key, XTensor &value,
                                    XTensor &beta, XTensor &g, XTensor &state, XTensor &out,
                                    uint32_t batch, uint32_t seqlen, uint32_t numHeads,
                                    uint32_t kDim, uint32_t vDim)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (kDim == 0 || vDim == 0 || numHeads == 0 || batch == 0 || seqlen == 0) {
        throw std::runtime_error(std::string(__func__) + ": invalid dims");
    }
    if (kDim > 128 || vDim > 128) {
        throw std::runtime_error(std::string(__func__) +
                                 ": kDim/vDim must be <= 128 for current kernel");
    }
    uint32_t tokens = batch * seqlen;
    if (query.shape[0] != tokens || key.shape[0] != tokens || value.shape[0] != tokens ||
        out.shape[0] != tokens || beta.shape[0] != tokens || g.shape[0] != tokens) {
        throw std::runtime_error(std::string(__func__) + ": token dim mismatch");
    }
    if (query.shape.size() < 2 || key.shape.size() < 2 || value.shape.size() < 2 ||
        out.shape.size() < 2 || query.shape[1] != numHeads * kDim ||
        key.shape[1] != numHeads * kDim || value.shape[1] != numHeads * vDim ||
        out.shape[1] != numHeads * vDim) {
        throw std::runtime_error(std::string(__func__) + ": feature dim mismatch");
    }
    if (beta.shape.size() < 2 || g.shape.size() < 2 || beta.shape[1] != numHeads ||
        g.shape[1] != numHeads) {
        throw std::runtime_error(std::string(__func__) + ": beta/g head dim mismatch");
    }
    if (state.shape.size() != 4 || state.shape[0] < batch || state.shape[1] != numHeads ||
        state.shape[2] != kDim || state.shape[3] != vDim) {
        throw std::runtime_error(std::string(__func__) + ": state shape mismatch");
    }
    KERNEL_PTR_TYPE(recurrent_gated_delta_rule) * launchKernel;
    if (EachXDtype(FP32, query, key, value, beta, g, state, out)) {
        launchKernel = aclrtlaunch_recurrent_gated_delta_rule_float;
    } else if (EachXDtype(FP16, query, key, value, beta, g, state, out)) {
        launchKernel = aclrtlaunch_recurrent_gated_delta_rule_float16_t;
    } else if (EachXDtype(BF16, query, key, value, beta, g, state, out)) {
        launchKernel = aclrtlaunch_recurrent_gated_delta_rule_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(query) + XT_STR(key) + XT_STR(value);
        throw std::runtime_error(err_str + " unsupported!");
    }
    launchKernel(rt.aivNum, rt.stream, query.ptr, key.ptr, value.ptr, beta.ptr, g.ptr, state.ptr,
                 out.ptr, batch, seqlen, numHeads, kDim, vDim,
                 1.0f / sqrtf(static_cast<float>(kDim)));
}

void XliteOpEinsumMhtHdtMhd(XRuntime &rt, XTensor &mht, XTensor &hdt, XTensor &mhd, uint32_t m,
                            uint32_t h, uint32_t t, uint32_t d, bool weightNZ, int T, int D)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(einsum_mht_hdt_mhd) * launchKernel;
    if (EachXDtype(FP16, mht, hdt, mhd)) {
        launchKernel = aclrtlaunch_einsum_mht_hdt_mhd_float16_t;
    } else if (EachXDtype(BF16, mht, hdt, mhd)) {
        launchKernel = aclrtlaunch_einsum_mht_hdt_mhd_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(mht) + XT_STR(hdt) + XT_STR(mhd);
        throw std::runtime_error(err_str + " unsupported!");
    }
    uint64_t swizzle = rt.defaultMatmulSwizzle;
    if (!rt.disableSwizzleTable) {
        XlitePickSwizzle(d, t, &swizzle);
    }
    launchKernel(rt.aicNum, rt.stream, mht.ptr, hdt.ptr, mhd.ptr, m, h, t, d,
                 MATMUL_M0_N0_K0_DEFAULT_VALUE, MATMUL_M0_N0_K0_DEFAULT_VALUE,
                 MATMUL_M0_N0_K0_DEFAULT_VALUE, weightNZ, swizzle, T, D);
}

void XliteOpEinsumMhtHtdMhd(XRuntime &rt, XTensor &mht, XTensor &htd, XTensor &mhd, uint32_t m,
                            uint32_t h, uint32_t t, uint32_t d, bool weightNZ, int T, int D)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    KERNEL_PTR_TYPE(einsum_mht_htd_mhd) * launchKernel;
    if (EachXDtype(FP16, mht, htd, mhd)) {
        launchKernel = aclrtlaunch_einsum_mht_htd_mhd_float16_t;
    } else if (EachXDtype(BF16, mht, htd, mhd)) {
        launchKernel = aclrtlaunch_einsum_mht_htd_mhd_bfloat16_t;
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(mht) + XT_STR(htd) + XT_STR(mhd);
        throw std::runtime_error(err_str + " unsupported!");
    }
    uint64_t swizzle = rt.defaultMatmulSwizzle;
    if (!rt.disableSwizzleTable) {
        XlitePickSwizzle(d, t, &swizzle);
    }
    launchKernel(rt.aicNum, rt.stream, mht.ptr, htd.ptr, mhd.ptr, m, h, t, d,
                 MATMUL_M0_N0_K0_DEFAULT_VALUE, MATMUL_M0_N0_K0_DEFAULT_VALUE,
                 MATMUL_M0_N0_K0_DEFAULT_VALUE, weightNZ, swizzle, T, D);
}

void XliteOpUnpackActivation(XRuntime &rt, XTensor &input, XTensor &output)
{
    if (IsDummyRuntime(rt)) {
        return;
    }
    if (input.dtype == INT8 && input.shape.size() == 2 && input.shape[1] % 2 == 0) {
        aclrtlaunch_unpack_activation_int8_t(rt.aivNum, rt.stream, input.ptr, output.ptr,
                                             input.shape[0], input.shape[1]);
    } else {
        std::string err_str = DBG_PREFIX + XT_STR(input) + XT_STR(output);
        throw std::runtime_error(err_str + " unsupported!");
    }
}
