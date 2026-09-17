/*
 * Ascend310P3 Qwen3-ASR unified-core paged decode attention probe.
 *
 * One M200 AI Core owns a (request, query-head) work item and keeps every
 * intermediate on chip:
 *   paged BSHD GM -> L1 -> QK MMAD -> L0C -> UB -> online softmax
 *                  -> PV MMAD -> L0C -> UB -> FP32 output accumulator.
 *
 * This first candidate is deliberately limited to one partition (KV <= 512)
 * and 16-token tiles. Q is already scaled by 1/sqrt(128).
 *
 * Resources per core:
 *   UB   17,472 bytes
 *   L1   12,288 bytes
 *   L0A   4,096 bytes
 *   L0B   4,096 bytes
 *   L0C   8,192 bytes
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQkvDim = 4096;
constexpr uint32_t kQDim = 2048;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kTileTokens = 16;
constexpr uint32_t kMaxBatch = 20;
constexpr uint32_t kMaxKv = 512;
constexpr uint32_t kCubeElements = 256;

__aicore__ inline void GmRowToL1Nz(const LocalTensor<half> &dst,
                                    const GlobalTensor<half> &src,
                                    uint32_t dstRow)
{
    Nd2NzParams params(1, 1, kHeadDim, 0, kHeadDim, kTileTokens, 1, 0);
    DataCopy(dst[dstRow * 16], src, params);
}

__aicore__ inline void L1ToL0A(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src,
                                uint32_t kBlocks)
{
    LoadData2dParams params(0, 1, 1, 0, kBlocks - 1, 0, inc);
    for (uint32_t kb = 0; kb < kBlocks; ++kb) {
        LoadData(dst[kb * kCubeElements], src[kb * kCubeElements], params);
    }
}

__aicore__ inline void L1ToL0B(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src,
                                uint32_t nBlocks, uint32_t kBlocks)
{
    LoadData2dParams params(0, nBlocks * kBlocks, 1, 0, 0, 0, inc);
    LoadData(dst, src, params);
}

__aicore__ inline void L1ToL0BTranspose(const LocalTensor<half> &dst,
                                         const LocalTensor<half> &src,
                                         uint32_t nBlocks)
{
    // FP16 transpose is the supported regular LoadData form on M200. It does
    // not use the prohibited dedicated transposed-load intrinsic.
    LoadData2dParams params(0, nBlocks, 1, 0, 0, 1, inc);
    LoadData(dst, src, params);
}

class AsrAttentionSinglePartitionProbe {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache,
                                GM_ADDR blockTable, GM_ADDR kvLengths,
                                GM_ADDR output, uint32_t batch,
                                uint32_t tableStride)
    {
        qkv_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qkv));
        kCache_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(kCache));
        vCache_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(vCache));
        blockTable_ = reinterpret_cast<__gm__ int32_t *>(blockTable);
        kvLengths_ = reinterpret_cast<__gm__ int32_t *>(kvLengths);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
        batch_ = batch;
        tableStride_ = tableStride;

        Bind(l1Q_, TPosition::A1, 0);
        Bind(l1B_, TPosition::B1, 4096);
        Bind(l1P_, TPosition::A1, 8192);
        Bind(l0A_, TPosition::A2, 0);
        Bind(l0B_, TPosition::B2, 0);
        Bind(l0C_, TPosition::CO1, 0);

        Bind(zeroH_, TPosition::VECCALC, 0);
        Bind(probabilityH_, TPosition::VECCALC, 4096);
        Bind(outputH_, TPosition::VECCALC, 4608);
        Bind(qkF_, TPosition::VECCALC, 4864);
        Bind(pvF_, TPosition::VECCALC, 5888);
        Bind(expInputF_, TPosition::VECCALC, 14080);
        Bind(expOutputF_, TPosition::VECCALC, 14208);
        Bind(weightF_, TPosition::VECCALC, 14336);
        Bind(pvCompactF_, TPosition::VECCALC, 14400);
        Bind(accAF_, TPosition::VECCALC, 14912);
        Bind(accBF_, TPosition::VECCALC, 15424);
        Bind(correctedF_, TPosition::VECCALC, 15936);
        Bind(scalarF_, TPosition::VECCALC, 16448);
        Bind(finalF_, TPosition::VECCALC, 16960);
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        FetchEvents();
        const uint32_t work = batch_ * kQHeads;
        for (uint32_t item = GetBlockIdx(); item < work; item += GetBlockNum()) {
            const uint32_t request = item / kQHeads;
            const uint32_t queryHead = item % kQHeads;
            const int32_t lengthSigned = kvLengths_[request];
            if (lengthSigned <= 0 || lengthSigned > static_cast<int32_t>(kMaxKv)) {
                continue;
            }
            RunHead(request, queryHead, static_cast<uint32_t>(lengthSigned));
        }
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void FetchEvents()
    {
        vToMte3_ = Fetch(HardEvent::V_MTE3);
        mte3ToMte2_ = Fetch(HardEvent::MTE3_MTE2);
        mte3ToMte1_ = Fetch(HardEvent::MTE3_MTE1);
        mte2ToMte1_ = Fetch(HardEvent::MTE2_MTE1);
        mte1ToM_ = Fetch(HardEvent::MTE1_M);
        mToMte1_ = Fetch(HardEvent::M_MTE1);
        mToV_ = Fetch(HardEvent::M_V);
        vToM_ = Fetch(HardEvent::V_M);
        vToS_ = Fetch(HardEvent::V_S);
        sToV_ = Fetch(HardEvent::S_V);
        mte3ToV_ = Fetch(HardEvent::MTE3_V);
    }

    __aicore__ inline event_t Fetch(HardEvent event)
    {
        return static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    }

    __aicore__ inline void ZeroHalfTile()
    {
        half zero = static_cast<half>(0.0F);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        for (uint32_t row = 0; row < kTileTokens; ++row) {
            // The knowledge-base M200 rule limits each FP16 Duplicate to 128.
            vector_dup(H(zeroH_) + row * kHeadDim, zero, 1, 1, 1, 8, 1);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ZeroL1(LocalTensor<half> &destination,
                                  bool followedByMte2)
    {
        ZeroHalfTile();
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopyParams copy{1, 128, 0, 0};
        DataCopy(destination, zeroH_, copy);
        if (followedByMte2) {
            SetFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
            WaitFlag<HardEvent::MTE3_MTE2>(mte3ToMte2_);
        } else {
            SetFlag<HardEvent::MTE3_MTE1>(mte3ToMte1_);
            WaitFlag<HardEvent::MTE3_MTE1>(mte3ToMte1_);
        }
    }

    __aicore__ inline uint64_t CacheOffset(uint32_t request,
                                            uint32_t logicalToken,
                                            uint32_t kvHead) const
    {
        const uint32_t logicalBlock = logicalToken / kBlockSize;
        const uint32_t tokenInBlock = logicalToken % kBlockSize;
        const uint32_t physicalBlock = static_cast<uint32_t>(
            blockTable_[request * tableStride_ + logicalBlock]);
        return ((static_cast<uint64_t>(physicalBlock) * kBlockSize + tokenInBlock) *
                kKvHeads + kvHead) * kHeadDim;
    }

    __aicore__ inline void StageQuery(uint32_t request, uint32_t queryHead)
    {
        ZeroL1(l1Q_, true);
        GmRowToL1Nz(l1Q_, qkv_[static_cast<uint64_t>(request) * kQkvDim +
                               queryHead * kHeadDim], 0);
        SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(mte2ToMte1_);
    }

    __aicore__ inline void StageCacheRows(const GlobalTensor<half> &cache,
                                           uint32_t request,
                                           uint32_t logicalStart,
                                           uint32_t validTokens,
                                           uint32_t kvHead)
    {
        ZeroL1(l1B_, true);
        for (uint32_t token = 0; token < validTokens; ++token) {
            GmRowToL1Nz(l1B_, cache[CacheOffset(request, logicalStart + token,
                                                kvHead)], token);
        }
        SetFlag<HardEvent::MTE2_MTE1>(mte2ToMte1_);
        WaitFlag<HardEvent::MTE2_MTE1>(mte2ToMte1_);
    }

    __aicore__ inline void RunMmad(uint32_t m, uint32_t n, uint32_t k)
    {
        SetFlag<HardEvent::MTE1_M>(mte1ToM_);
        WaitFlag<HardEvent::MTE1_M>(mte1ToM_);
        MmadParams params;
        params.m = m;
        params.n = n;
        params.k = k;
        params.cmatrixSource = false;
        params.cmatrixInitVal = true;
        Mmad(l0C_, l0A_, l0B_, params);
        SetFlag<HardEvent::M_MTE1>(mToMte1_);
        SetFlag<HardEvent::M_V>(mToV_);
        WaitFlag<HardEvent::M_V>(mToV_);
    }

    __aicore__ inline void DrainL0C(LocalTensor<float> &destination,
                                    uint32_t elements)
    {
        DataCopyParams drain;
        drain.blockCount = 1;
        drain.blockLen = elements * sizeof(float) / 1024;
        drain.srcStride = 0;
        drain.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        enhanced.deqScale = DeqScale::DEQ_NONE;
        DataCopy(destination, l0C_, drain, enhanced);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_M>(vToM_);
        WaitFlag<HardEvent::V_M>(vToM_);
        WaitFlag<HardEvent::M_MTE1>(mToMte1_);
    }

    __aicore__ inline void BuildProbabilities(float &runningMax,
                                               float &runningSum,
                                               uint32_t validTokens,
                                               float &correction)
    {
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        float tileMax = F(qkF_)[0];
        for (uint32_t token = 1; token < validTokens; ++token) {
            const float score = F(qkF_)[token];
            tileMax = score > tileMax ? score : tileMax;
        }
        const float newMax = runningMax > tileMax ? runningMax : tileMax;
        F(expInputF_)[0] = runningMax - newMax;
        for (uint32_t token = 0; token < kTileTokens; ++token) {
            F(expInputF_)[token + 1] = token < validTokens ?
                F(qkF_)[token] - newMax : -65504.0F;
        }
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(0, (static_cast<uint64_t>(1) << 17) - 1);
        vexp(F(expOutputF_), F(expInputF_), 1, 1, 1, 8, 8);
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        correction = F(expOutputF_)[0];
        float tileSum = 0.0F;
        for (uint32_t token = 0; token < kTileTokens; ++token) {
            const float weight = F(expOutputF_)[token + 1];
            F(weightF_)[token] = weight;
            tileSum += weight;
        }
        runningSum = runningSum * correction + tileSum;
        runningMax = newMax;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
    }

    __aicore__ inline void StageProbabilities()
    {
        half zero = static_cast<half>(0.0F);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vector_dup(H(probabilityH_), zero, 1, 1, 1, 8, 1);
        vector_dup(H(probabilityH_) + 128, zero, 1, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, (static_cast<uint64_t>(1) << 16) - 1);
        vconv_f322f16(H(probabilityH_), F(weightF_), 1, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopyParams copy{1, 16, 0, 0};
        DataCopy(l1P_, probabilityH_, copy);
        SetFlag<HardEvent::MTE3_MTE1>(mte3ToMte1_);
        WaitFlag<HardEvent::MTE3_MTE1>(mte3ToMte1_);
        L1ToL0A(l0A_, l1P_, 1);
    }

    __aicore__ inline void CompactPvRow()
    {
        float zero = 0.0F;
        set_vector_mask(0, (static_cast<uint64_t>(1) << 16) - 1);
        for (uint32_t block = 0; block < 8; ++block) {
            vadds(F(pvCompactF_) + block * 16,
                  F(pvF_) + block * kCubeElements, zero,
                  1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void UpdateAccumulator(__ubuf__ float *&accumulator,
                                              __ubuf__ float *&nextAccumulator,
                                              float correction)
    {
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vector_dup(F(scalarF_), correction, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(correctedF_), accumulator, F(scalarF_), 2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(nextAccumulator, F(correctedF_), F(pvCompactF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        __ubuf__ float *old = accumulator;
        accumulator = nextAccumulator;
        nextAccumulator = old;
    }

    __aicore__ inline void RunHead(uint32_t request, uint32_t queryHead,
                                    uint32_t kvLength)
    {
        const uint32_t kvHead = queryHead / 2;
        StageQuery(request, queryHead);
        float zero = 0.0F;
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vector_dup(F(accAF_), zero, 2, 1, 1, 8, 1);
        vector_dup(F(accBF_), zero, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        float runningMax = -65504.0F;
        float runningSum = 0.0F;
        __ubuf__ float *accumulator = F(accAF_);
        __ubuf__ float *nextAccumulator = F(accBF_);

        for (uint32_t logicalStart = 0; logicalStart < kvLength;
             logicalStart += kTileTokens) {
            const uint32_t remaining = kvLength - logicalStart;
            const uint32_t validTokens = remaining < kTileTokens ?
                remaining : kTileTokens;

            StageCacheRows(kCache_, request, logicalStart, validTokens, kvHead);
            L1ToL0A(l0A_, l1Q_, 8);
            L1ToL0B(l0B_, l1B_, 1, 8);
            RunMmad(16, 16, 128);
            DrainL0C(qkF_, 16 * 16);

            float correction = 0.0F;
            BuildProbabilities(runningMax, runningSum, validTokens, correction);
            StageProbabilities();
            StageCacheRows(vCache_, request, logicalStart, validTokens, kvHead);
            L1ToL0BTranspose(l0B_, l1B_, 8);
            RunMmad(16, 128, 16);
            DrainL0C(pvF_, 16 * 128);
            CompactPvRow();
            UpdateAccumulator(accumulator, nextAccumulator, correction);
        }

        float inverseSum = 1.0F / runningSum;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vector_dup(F(scalarF_), inverseSum, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(finalF_), accumulator, F(scalarF_), 2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(H(outputH_), F(finalF_), 2, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        copy_ubuf_to_gm(reinterpret_cast<__gm__ half *>(output_.GetPhyAddr()) +
                        static_cast<uint64_t>(request) * kQDim +
                        queryHead * kHeadDim,
                        H(outputH_), 0, 1, 8, 0, 0);
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
    }

    template <typename T>
    __aicore__ inline void Bind(LocalTensor<T> &tensor, TPosition position,
                                 uint32_t offset)
    {
        tensor.address_.logicPos = static_cast<uint8_t>(position);
        tensor.address_.bufferAddr = offset;
    }
    __aicore__ inline __ubuf__ half *H(LocalTensor<half> &tensor)
    { return reinterpret_cast<__ubuf__ half *>(tensor.GetPhyAddr()); }
    __aicore__ inline __ubuf__ float *F(LocalTensor<float> &tensor)
    { return reinterpret_cast<__ubuf__ float *>(tensor.GetPhyAddr()); }

    TPipe pipe_;
    GlobalTensor<half> qkv_, kCache_, vCache_, output_;
    __gm__ int32_t *blockTable_ = nullptr;
    __gm__ int32_t *kvLengths_ = nullptr;
    LocalTensor<half> l1Q_, l1B_, l1P_, l0A_, l0B_;
    LocalTensor<float> l0C_;
    LocalTensor<half> zeroH_, probabilityH_, outputH_;
    LocalTensor<float> qkF_, pvF_, expInputF_, expOutputF_;
    LocalTensor<float> weightF_, pvCompactF_, accAF_, accBF_;
    LocalTensor<float> correctedF_, scalarF_, finalF_;
    uint32_t batch_ = 0;
    uint32_t tableStride_ = 0;
    event_t vToMte3_, mte3ToMte2_, mte3ToMte1_, mte2ToMte1_;
    event_t mte1ToM_, mToMte1_, mToV_, vToM_, vToS_, sToV_, mte3ToV_;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_attention_single_partition_probe(
    GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable,
    GM_ADDR kvLengths, GM_ADDR output, uint32_t batch, uint32_t tableStride)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_attention_single_partition_probe requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (batch == 0 || batch > kMaxBatch || tableStride == 0 ||
        tableStride > 16) {
        return;
    }
    AsrAttentionSinglePartitionProbe probe;
    probe.Init(qkv, kCache, vCache, blockTable, kvLengths, output,
               batch, tableStride);
    probe.Process();
}
