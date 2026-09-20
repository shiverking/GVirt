/*
 * Ascend310P3 Qwen3-ASR scratch-free paged Decode Attention.
 * One M200 AI Core owns one (request, KV-head), processes both GQA query
 * heads and at most four 512-token partitions, then merges their independent
 * FP32 online-softmax states on chip.
 * K/V are read directly from the 4D BSHD paged cache.  The production entry
 * writes one FP16 output row and has no GM intermediate workspace.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQkvDim = 4096;
constexpr uint32_t kQDim = kQHeads * kHeadDim;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kTileTokens = 16;
constexpr uint32_t kPartitions = 4;
constexpr uint32_t kPartitionTokens = 512;
constexpr uint32_t kStateStride = 144;
constexpr uint32_t kStateBlocks = kStateStride * sizeof(float) / 32;
constexpr uint32_t kMaxBatch = 20;
constexpr uint32_t kMaxKv = 2048;
constexpr uint32_t kCubeElements = 256;

__aicore__ inline void GmRowToL1Nz(const LocalTensor<half> &dst,
                                    const GlobalTensor<half> &src,
                                    uint32_t dstRow)
{
    Nd2NzParams params(1, 1, kHeadDim, 0, kHeadDim, kTileTokens, 1, 0);
    DataCopy(dst[dstRow * 16], src, params);
}

__aicore__ inline void GmCacheRowsToL1Nz(const LocalTensor<half> &dst,
                                          const GlobalTensor<half> &src,
                                          uint32_t rows,
                                          uint32_t dstRow)
{
    // A fixed KV head is strided by all eight heads in BSHD cache.  Nd2Nz
    // accepts that ND row stride, so one MTE2 transaction stages every row
    // up to the next physical 128-token block boundary.  The destination
    // remains a 16-row NZ tile consumed directly by L0B.
    Nd2NzParams params(1, rows, kHeadDim, 0,
                       kKvHeads * kHeadDim, kTileTokens, 1, 0);
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
    LoadData2dParams params(0, nBlocks, 1, 0, 0, 1, inc);
    LoadData(dst, src, params);
}

class AsrPagedDecodeAttentionKernel {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache,
                                GM_ADDR blockTable, GM_ADDR lengths,
                                GM_ADDR states, uint32_t batch,
                                uint32_t tableStride)
    {
        InitCommon(qkv, kCache, vCache, blockTable, lengths,
                   batch, tableStride);
        states_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(states));
        Bind(stateOutF_, TPosition::VECCALC, 16960);
    }

    __aicore__ inline void InitFused(GM_ADDR qkv, GM_ADDR kCache,
                                     GM_ADDR vCache, GM_ADDR blockTable,
                                     GM_ADDR cachedLengths, GM_ADDR output,
                                     uint32_t batch, uint32_t tableStride)
    {
        InitCommon(qkv, kCache, vCache, blockTable, cachedLengths,
                   batch, tableStride);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
        Bind(globalAccF_, TPosition::VECCALC, 16960);
        Bind(scaledPartitionF_, TPosition::VECCALC, 17472);
        Bind(outputH_, TPosition::VECCALC, 17984);
        Bind(weightPairF_, TPosition::VECCALC, 18240);
        Bind(pvPairF_, TPosition::VECCALC, 18304);
        Bind(accPairAF_, TPosition::VECCALC, 18816);
        Bind(accPairBF_, TPosition::VECCALC, 19328);
        Bind(globalPairF_, TPosition::VECCALC, 19840);
    }

    __aicore__ inline void InitFusedPacked(
        GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable,
        GM_ADDR cachedLengths, GM_ADDR queryStartLoc, GM_ADDR queryLens,
        GM_ADDR output, uint32_t batch, uint32_t tableStride)
    {
        InitFused(qkv, kCache, vCache, blockTable, cachedLengths, output,
                  batch, tableStride);
        queryStartLoc_ = reinterpret_cast<__gm__ int32_t *>(queryStartLoc);
        queryLens_ = reinterpret_cast<__gm__ int32_t *>(queryLens);
        packed_ = true;
    }

    __aicore__ inline void InitCommon(GM_ADDR qkv, GM_ADDR kCache,
                                      GM_ADDR vCache, GM_ADDR blockTable,
                                      GM_ADDR lengths, uint32_t batch,
                                      uint32_t tableStride)
    {
        qkv_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(qkv));
        kCache_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(kCache));
        vCache_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(vCache));
        blockTable_ = reinterpret_cast<__gm__ int32_t *>(blockTable);
        lengths_ = reinterpret_cast<__gm__ int32_t *>(lengths);
        queryStartLoc_ = nullptr;
        queryLens_ = nullptr;
        packed_ = false;
        batch_ = batch;
        tableStride_ = tableStride;
        Bind(l1Q_, TPosition::A1, 0);
        Bind(l1B_, TPosition::B1, 4096);
        Bind(l1P_, TPosition::A1, 8192);
        Bind(l1QPair_, TPosition::A1, 12288);
        Bind(l0A_, TPosition::A2, 0);
        Bind(l0B_, TPosition::B2, 0);
        Bind(l0C_, TPosition::CO1, 0);
        Bind(zeroH_, TPosition::VECCALC, 0);
        Bind(probabilityH_, TPosition::VECCALC, 4096);
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
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        FetchEvents();
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        const uint32_t work = batch_ * kQHeads * kPartitions;
        for (uint32_t item = GetBlockIdx(); item < work;
             item += GetBlockNum()) {
            const uint32_t partition = item % kPartitions;
            const uint32_t headItem = item / kPartitions;
            const uint32_t request = headItem / kQHeads;
            const uint32_t queryHead = headItem % kQHeads;
            const int32_t lengthSigned = lengths_[request];
            if (lengthSigned <= 0 || lengthSigned > static_cast<int32_t>(kMaxKv)) {
                continue;
            }
            RunPartition(item, request, queryHead, partition,
                         static_cast<uint32_t>(lengthSigned));
        }
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        pipe_barrier(PIPE_ALL);
    }

    __aicore__ inline void ProcessFused()
    {
        set_atomic_none();
        set_mask_norm();
        FetchEvents();
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        const uint32_t work = batch_ * kKvHeads;
        for (uint32_t item = GetBlockIdx(); item < work;
             item += GetBlockNum()) {
            const uint32_t request = item / kKvHeads;
            const uint32_t kvHead = item % kKvHeads;
            if (packed_ &&
                (queryLens_[request] != 1 || lengths_[request] <= 0)) {
                continue;
            }
            const int32_t lengthSigned = lengths_[request] + 1;
            if (lengthSigned <= 0 ||
                lengthSigned > static_cast<int32_t>(kMaxKv)) {
                continue;
            }
            const uint32_t queryRow = packed_ ?
                static_cast<uint32_t>(queryStartLoc_[request]) : request;
            RunFusedGroup(request, queryRow, kvHead,
                          static_cast<uint32_t>(lengthSigned));
        }
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline event_t Fetch(HardEvent event)
    {
        return static_cast<event_t>(GetTPipePtr()->FetchEventID(event));
    }

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

    __aicore__ inline void ZeroHalfTile()
    {
        half zero = static_cast<half>(0.0F);
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        for (uint32_t row = 0; row < kTileTokens; ++row) {
            vector_dup(H(zeroH_) + row * kHeadDim, zero,
                       1, 1, 1, 8, 1);
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
        return ((static_cast<uint64_t>(physicalBlock) * kBlockSize +
                 tokenInBlock) * kKvHeads + kvHead) * kHeadDim;
    }

    __aicore__ inline void StageQuery(LocalTensor<half> &destination,
                                      uint32_t queryRow,
                                      uint32_t queryHead)
    {
        ZeroL1(destination, true);
        GmRowToL1Nz(destination,
                     qkv_[static_cast<uint64_t>(queryRow) * kQkvDim +
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
        // I20: a short tail must initialize the rows that MMAD still reads.
        // A full tile overwrites all 16x128 elements and therefore needs no
        // UB->L1 zero-fill.  Avoiding that fill removes two barriers and one
        // 4 KiB transfer from the common path.
        if (validTokens < kTileTokens) {
            ZeroL1(l1B_, true);
        }

        uint32_t staged = 0;
        while (staged < validTokens) {
            const uint32_t logicalToken = logicalStart + staged;
            const uint32_t tokenInBlock = logicalToken % kBlockSize;
            const uint32_t beforeBoundary = kBlockSize - tokenInBlock;
            const uint32_t remaining = validTokens - staged;
            const uint32_t rows = remaining < beforeBoundary ?
                remaining : beforeBoundary;
            GmCacheRowsToL1Nz(
                l1B_, cache[CacheOffset(request, logicalToken, kvHead)],
                rows, staged);
            staged += rows;
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
                                               float &correction,
                                               __ubuf__ float *weights)
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
            weights[token] = weight;
            tileSum += weight;
        }
        runningSum = runningSum * correction + tileSum;
        runningMax = newMax;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
    }

    __aicore__ inline void StageProbabilities(__ubuf__ float *weights)
    {
        half zero = static_cast<half>(0.0F);
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(H(probabilityH_), zero, 1, 1, 1, 8, 1);
        vector_dup(H(probabilityH_) + 128, zero, 1, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, (static_cast<uint64_t>(1) << 16) - 1);
        vconv_f322f16(H(probabilityH_), weights, 1, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopyParams copy{1, 16, 0, 0};
        DataCopy(l1P_, probabilityH_, copy);
        SetFlag<HardEvent::MTE3_MTE1>(mte3ToMte1_);
        WaitFlag<HardEvent::MTE3_MTE1>(mte3ToMte1_);
        L1ToL0A(l0A_, l1P_, 1);
    }

    __aicore__ inline void CompactPvRow(__ubuf__ float *destination)
    {
        float zero = 0.0F;
        set_vector_mask(0, (static_cast<uint64_t>(1) << 16) - 1);
        for (uint32_t block = 0; block < 8; ++block) {
            vadds(destination + block * 16,
                  F(pvF_) + block * kCubeElements, zero,
                  1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void UpdateAccumulator(__ubuf__ float *&accumulator,
                                              __ubuf__ float *&nextAccumulator,
                                              float correction,
                                              __ubuf__ float *pvCompact)
    {
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(scalarF_), correction, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(correctedF_), accumulator, F(scalarF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(nextAccumulator, F(correctedF_), pvCompact,
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        __ubuf__ float *old = accumulator;
        accumulator = nextAccumulator;
        nextAccumulator = old;
    }

    __aicore__ inline void WriteState(uint32_t item, float runningMax,
                                      float runningSum,
                                      __ubuf__ float *accumulator)
    {
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        float zero = 0.0F;
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(stateOutF_), zero, 2, 1, 1, 8, 1);
        set_vector_mask(0, (static_cast<uint64_t>(1) << 16) - 1);
        vector_dup(F(stateOutF_) + 128, zero, 1, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        F(stateOutF_)[0] = runningMax;
        F(stateOutF_)[1] = runningSum;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vadds(F(stateOutF_) + 2, accumulator, zero,
              2, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopyParams store{1, kStateBlocks, 0, 0};
        DataCopy(states_[static_cast<uint64_t>(item) * kStateStride],
                 stateOutF_, store);
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void RunPartition(uint32_t item, uint32_t request,
                                        uint32_t queryHead,
                                        uint32_t partition,
                                        uint32_t kvLength)
    {
        float zero = 0.0F;
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(accAF_), zero, 2, 1, 1, 8, 1);
        vector_dup(F(accBF_), zero, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        __ubuf__ float *accumulator = F(accAF_);
        __ubuf__ float *nextAccumulator = F(accBF_);
        const uint32_t partitionStart = partition * kPartitionTokens;
        if (partitionStart >= kvLength) {
            WriteState(item, -65504.0F, 0.0F, accumulator);
            return;
        }

        const uint32_t partitionEnd =
            kvLength < partitionStart + kPartitionTokens ?
            kvLength : partitionStart + kPartitionTokens;
        const uint32_t kvHead = queryHead / 2;
        StageQuery(l1Q_, request, queryHead);
        float runningMax = -65504.0F;
        float runningSum = 0.0F;
        for (uint32_t logicalStart = partitionStart;
             logicalStart < partitionEnd; logicalStart += kTileTokens) {
            const uint32_t remaining = partitionEnd - logicalStart;
            const uint32_t validTokens = remaining < kTileTokens ?
                remaining : kTileTokens;
            StageCacheRows(kCache_, request, logicalStart,
                           validTokens, kvHead);
            L1ToL0A(l0A_, l1Q_, 8);
            L1ToL0B(l0B_, l1B_, 1, 8);
            RunMmad(16, 16, 128);
            DrainL0C(qkF_, 16 * 16);
            float correction = 0.0F;
            BuildProbabilities(runningMax, runningSum,
                               validTokens, correction, F(weightF_));
            StageProbabilities(F(weightF_));
            StageCacheRows(vCache_, request, logicalStart,
                           validTokens, kvHead);
            L1ToL0BTranspose(l0B_, l1B_, 8);
            RunMmad(16, 128, 16);
            DrainL0C(pvF_, 16 * 128);
            CompactPvRow(F(pvCompactF_));
            UpdateAccumulator(accumulator, nextAccumulator, correction,
                              F(pvCompactF_));
        }
        WriteState(item, runningMax, runningSum, accumulator);
    }

    __aicore__ inline void MergePartition(float &globalMax,
                                           float &globalSum,
                                           float partitionMax,
                                           float partitionSum,
                                           __ubuf__ float *partitionOutput,
                                           __ubuf__ float *globalAccumulator)
    {
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        const float newMax = globalMax > partitionMax ?
            globalMax : partitionMax;
        F(expInputF_)[0] = globalMax - newMax;
        F(expInputF_)[1] = partitionMax - newMax;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(0, 3);
        vexp(F(expOutputF_), F(expInputF_), 1, 1, 1, 8, 8);
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        float correction = F(expOutputF_)[0];
        float partitionScale = F(expOutputF_)[1];
        globalSum = globalSum * correction +
                    partitionSum * partitionScale;
        globalMax = newMax;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);

        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(scalarF_), correction, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(correctedF_), globalAccumulator, F(scalarF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vector_dup(F(scalarF_), partitionScale, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(scaledPartitionF_), partitionOutput, F(scalarF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vadd(globalAccumulator, F(correctedF_), F(scaledPartitionF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void StoreFusedOutput(uint32_t queryRow,
                                            uint32_t queryHead,
                                            float globalSum,
                                            __ubuf__ float *globalAccumulator)
    {
        float inverseSum = 1.0F / globalSum;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(scalarF_), inverseSum, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(correctedF_), globalAccumulator, F(scalarF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        vconv_f322f16(H(outputH_), F(correctedF_), 2, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopyParams store{1, kHeadDim * sizeof(half) / 32, 0, 0};
        const uint64_t outputOffset =
            static_cast<uint64_t>(queryRow) * kQDim +
            queryHead * kHeadDim;
        DataCopy(output_[outputOffset], outputH_, store);
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
    }

    __aicore__ inline void RunFusedGroup(uint32_t request,
                                         uint32_t queryRow,
                                         uint32_t kvHead,
                                         uint32_t kvLength)
    {
        float zero = 0.0F;
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(globalAccF_), zero, 2, 1, 1, 8, 1);
        vector_dup(F(globalPairF_), zero, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        float globalMax0 = -65504.0F;
        float globalSum0 = 0.0F;
        float globalMax1 = -65504.0F;
        float globalSum1 = 0.0F;
        const uint32_t queryHead0 = kvHead * 2;
        const uint32_t queryHead1 = queryHead0 + 1;
        StageQuery(l1Q_, queryRow, queryHead0);
        StageQuery(l1QPair_, queryRow, queryHead1);

        for (uint32_t partitionStart = 0; partitionStart < kvLength;
             partitionStart += kPartitionTokens) {
            set_vector_mask(static_cast<uint64_t>(-1),
                            static_cast<uint64_t>(-1));
            vector_dup(F(accAF_), zero, 2, 1, 1, 8, 1);
            vector_dup(F(accBF_), zero, 2, 1, 1, 8, 1);
            vector_dup(F(accPairAF_), zero, 2, 1, 1, 8, 1);
            vector_dup(F(accPairBF_), zero, 2, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            __ubuf__ float *accumulator0 = F(accAF_);
            __ubuf__ float *nextAccumulator0 = F(accBF_);
            __ubuf__ float *accumulator1 = F(accPairAF_);
            __ubuf__ float *nextAccumulator1 = F(accPairBF_);
            float partitionMax0 = -65504.0F;
            float partitionSum0 = 0.0F;
            float partitionMax1 = -65504.0F;
            float partitionSum1 = 0.0F;
            const uint32_t partitionEnd =
                kvLength < partitionStart + kPartitionTokens ?
                kvLength : partitionStart + kPartitionTokens;

            for (uint32_t logicalStart = partitionStart;
                 logicalStart < partitionEnd;
                 logicalStart += kTileTokens) {
                const uint32_t remaining = partitionEnd - logicalStart;
                const uint32_t validTokens = remaining < kTileTokens ?
                    remaining : kTileTokens;
                // Stage the GQA group's K tile once, then issue the two QK
                // MMADs against the resident L0B operand.
                StageCacheRows(kCache_, request, logicalStart,
                               validTokens, kvHead);
                L1ToL0A(l0A_, l1Q_, 8);
                L1ToL0B(l0B_, l1B_, 1, 8);
                RunMmad(16, 16, 128);
                DrainL0C(qkF_, 16 * 16);
                float correction0 = 0.0F;
                BuildProbabilities(partitionMax0, partitionSum0,
                                   validTokens, correction0, F(weightF_));

                L1ToL0A(l0A_, l1QPair_, 8);
                RunMmad(16, 16, 128);
                DrainL0C(qkF_, 16 * 16);
                float correction1 = 0.0F;
                BuildProbabilities(partitionMax1, partitionSum1,
                                   validTokens, correction1,
                                   F(weightPairF_));

                // Stage V and load L0B once.  Each query head supplies its
                // own probability A operand and online-softmax state.
                StageCacheRows(vCache_, request, logicalStart,
                               validTokens, kvHead);
                L1ToL0BTranspose(l0B_, l1B_, 8);
                StageProbabilities(F(weightF_));
                RunMmad(16, 128, 16);
                DrainL0C(pvF_, 16 * 128);
                CompactPvRow(F(pvCompactF_));
                UpdateAccumulator(accumulator0, nextAccumulator0,
                                  correction0, F(pvCompactF_));

                StageProbabilities(F(weightPairF_));
                RunMmad(16, 128, 16);
                DrainL0C(pvF_, 16 * 128);
                CompactPvRow(F(pvPairF_));
                UpdateAccumulator(accumulator1, nextAccumulator1,
                                  correction1, F(pvPairF_));
            }
            MergePartition(globalMax0, globalSum0, partitionMax0,
                           partitionSum0, accumulator0, F(globalAccF_));
            MergePartition(globalMax1, globalSum1, partitionMax1,
                           partitionSum1, accumulator1, F(globalPairF_));
        }
        StoreFusedOutput(queryRow, queryHead0, globalSum0, F(globalAccF_));
        StoreFusedOutput(queryRow, queryHead1, globalSum1, F(globalPairF_));
    }

    template <typename T>
    __aicore__ inline void Bind(LocalTensor<T> &tensor,
                                 TPosition position, uint32_t offset)
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
    GlobalTensor<float> states_;
    __gm__ int32_t *blockTable_ = nullptr;
    __gm__ int32_t *lengths_ = nullptr;
    __gm__ int32_t *queryStartLoc_ = nullptr;
    __gm__ int32_t *queryLens_ = nullptr;
    bool packed_ = false;
    LocalTensor<half> l1Q_, l1B_, l1P_, l1QPair_, l0A_, l0B_;
    LocalTensor<float> l0C_;
    LocalTensor<half> zeroH_, probabilityH_;
    LocalTensor<float> qkF_, pvF_, expInputF_, expOutputF_;
    LocalTensor<float> weightF_, pvCompactF_, accAF_, accBF_;
    LocalTensor<float> correctedF_, scalarF_, stateOutF_;
    LocalTensor<float> globalAccF_, scaledPartitionF_;
    LocalTensor<float> weightPairF_, pvPairF_, accPairAF_, accPairBF_;
    LocalTensor<float> globalPairF_;
    LocalTensor<half> outputH_;
    uint32_t batch_ = 0;
    uint32_t tableStride_ = 0;
    event_t vToMte3_, mte3ToMte2_, mte3ToMte1_, mte2ToMte1_;
    event_t mte1ToM_, mToMte1_, mToV_, vToM_, vToS_, sToV_, mte3ToV_;
};
}  // namespace

#ifdef XLITE_ASR_ATTENTION_PARTITION_PROBE
extern "C" __global__ __aicore__ void asr_attention_partition_state_probe(
    GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable,
    GM_ADDR lengths, GM_ADDR states, uint32_t batch, uint32_t tableStride)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_attention_partition_state_probe requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (batch == 0 || batch > kMaxBatch || tableStride == 0 ||
        tableStride > 16) {
        return;
    }
    AsrPagedDecodeAttentionKernel probe;
    probe.Init(qkv, kCache, vCache, blockTable, lengths, states,
               batch, tableStride);
    probe.Process();
}
#endif

extern "C" __global__ __aicore__ void asr_paged_decode_attention_fp16(
    GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable,
    GM_ADDR cachedLengths, GM_ADDR output, uint32_t batch, uint32_t tableStride)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_paged_decode_attention_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (batch == 0 || batch > kMaxBatch || tableStride == 0 ||
        tableStride > 16) {
        return;
    }
    AsrPagedDecodeAttentionKernel kernel;
    kernel.InitFused(qkv, kCache, vCache, blockTable, cachedLengths, output,
                     batch, tableStride);
    kernel.ProcessFused();
}

extern "C" __global__ __aicore__ void asr_paged_mixed_decode_attention_fp16(
    GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable,
    GM_ADDR cachedLengths, GM_ADDR queryStartLoc, GM_ADDR queryLens,
    GM_ADDR output, uint32_t batch, uint32_t tableStride)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_paged_mixed_decode_attention_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (batch == 0 || batch > kMaxBatch || tableStride == 0 ||
        tableStride > 16) {
        return;
    }
    AsrPagedDecodeAttentionKernel kernel;
    kernel.InitFusedPacked(qkv, kCache, vCache, blockTable, cachedLengths,
                           queryStartLoc, queryLens, output, batch,
                           tableStride);
    kernel.ProcessFused();
}
