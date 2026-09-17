/* Ascend310P3 Qwen3-ASR paged BSHD -> attention MMAD tile probe.
 *
 * K output:  [16 tokens, 128 dims] (MMAD B native [N,K]).
 * Vt output: [128 dims, 16 tokens] (MMAD B native [N,K]).
 * Invalid tail rows/columns are explicitly zeroed. No UB is assumed clean.
 * UB=5120 bytes, L1/L0=0. Dynamic events only.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kTileTokens = 16;
constexpr uint32_t kDimBlocks = 8;

class PagedKvStageProbe {
public:
    __aicore__ inline void Init(GM_ADDR kCache, GM_ADDR vCache,
                                GM_ADDR blockTable, GM_ADDR kTile,
                                GM_ADDR vTransposeTile, uint32_t logicalStart,
                                uint32_t validTokens, uint32_t kvHead)
    {
        kCache_ = reinterpret_cast<__gm__ half *>(kCache);
        vCache_ = reinterpret_cast<__gm__ half *>(vCache);
        blockTable_ = reinterpret_cast<__gm__ int32_t *>(blockTable);
        kTile_ = reinterpret_cast<__gm__ half *>(kTile);
        vTransposeTile_ = reinterpret_cast<__gm__ half *>(vTransposeTile);
        logicalStart_ = logicalStart;
        validTokens_ = validTokens;
        kvHead_ = kvHead;
        Bind(kTileUb_, 0);
        Bind(vCompactUb_, 4096);
        Bind(vTransposeUb_, 4608);
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        const event_t loadReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t loadFree = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const event_t storeReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t storeFree = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        half zero = static_cast<half>(0.0F);
        SetFlag<HardEvent::MTE3_V>(storeFree);

        // The complete K tile is initialized before valid paged rows overwrite
        // it. FP16 Duplicate is deliberately limited to one 128-element row.
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        for (uint32_t token = 0; token < kTileTokens; ++token) {
            vector_dup(H(kTileUb_) + token * kHeadDim, zero, 1, 1, 1, 8, 1);
        }
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE2>(loadFree);
        WaitFlag<HardEvent::V_MTE2>(loadFree);
        for (uint32_t token = 0; token < validTokens_; ++token) {
            const uint64_t offset = CacheOffset(logicalStart_ + token, 0);
            copy_gm_to_ubuf(H(kTileUb_) + token * kHeadDim,
                            kCache_ + offset, 0, 1, 8, 0, 0);
        }
        SetFlag<HardEvent::MTE2_V>(loadReady);
        WaitFlag<HardEvent::MTE2_V>(loadReady);
        WaitFlag<HardEvent::MTE3_V>(storeFree);
        SetFlag<HardEvent::V_MTE3>(storeReady);
        WaitFlag<HardEvent::V_MTE3>(storeReady);
        copy_ubuf_to_gm(kTile_, H(kTileUb_), 0, 1, 128, 0, 0);
        SetFlag<HardEvent::MTE3_V>(storeFree);

        // Stage each [token,16-dim] slice contiguously, then use the basic
        // 16x16 vector transpose. This handles a physical-block crossing on a
        // per-token basis and never reads block-table padding.
        for (uint32_t dimBlock = 0; dimBlock < kDimBlocks; ++dimBlock) {
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            vector_dup(H(vCompactUb_), zero, 1, 1, 1, 8, 1);
            vector_dup(H(vCompactUb_) + 128, zero, 1, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE2>(loadFree);
            WaitFlag<HardEvent::V_MTE2>(loadFree);
            for (uint32_t token = 0; token < validTokens_; ++token) {
                const uint64_t offset = CacheOffset(logicalStart_ + token,
                                                    dimBlock * 16);
                copy_gm_to_ubuf(H(vCompactUb_) + token * 16,
                                vCache_ + offset, 0, 1, 1, 0, 0);
            }
            SetFlag<HardEvent::MTE2_V>(loadReady);
            WaitFlag<HardEvent::MTE2_V>(loadReady);
            vtranspose(reinterpret_cast<__ubuf__ uint16_t *>(H(vTransposeUb_)),
                       reinterpret_cast<__ubuf__ uint16_t *>(H(vCompactUb_)));
            pipe_barrier(PIPE_V);
            WaitFlag<HardEvent::MTE3_V>(storeFree);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            copy_ubuf_to_gm(vTransposeTile_ + dimBlock * 256,
                            H(vTransposeUb_), 0, 1, 16, 0, 0);
            SetFlag<HardEvent::MTE3_V>(storeFree);
        }
        WaitFlag<HardEvent::MTE3_V>(storeFree);
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline uint64_t CacheOffset(uint32_t logicalToken,
                                           uint32_t dimension) const
    {
        const uint32_t logicalBlock = logicalToken / kBlockSize;
        const uint32_t tokenInBlock = logicalToken % kBlockSize;
        const uint32_t physicalBlock = static_cast<uint32_t>(blockTable_[logicalBlock]);
        return ((static_cast<uint64_t>(physicalBlock) * kBlockSize + tokenInBlock) *
                kKvHeads + kvHead_) * kHeadDim + dimension;
    }
    template <typename T>
    __aicore__ inline void Bind(LocalTensor<T> &tensor, uint32_t offset)
    { tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC); tensor.address_.bufferAddr = offset; }
    __aicore__ inline __ubuf__ half *H(LocalTensor<half> &tensor)
    { return reinterpret_cast<__ubuf__ half *>(tensor.GetPhyAddr()); }

    TPipe pipe_;
    __gm__ half *kCache_ = nullptr;
    __gm__ half *vCache_ = nullptr;
    __gm__ int32_t *blockTable_ = nullptr;
    __gm__ half *kTile_ = nullptr;
    __gm__ half *vTransposeTile_ = nullptr;
    LocalTensor<half> kTileUb_, vCompactUb_, vTransposeUb_;
    uint32_t logicalStart_ = 0, validTokens_ = 0, kvHead_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_paged_kv_stage_probe(
    GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable, GM_ADDR kTile,
    GM_ADDR vTransposeTile, uint32_t logicalStart, uint32_t validTokens,
    uint32_t kvHead)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_paged_kv_stage_probe requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (validTokens == 0 || validTokens > 16 || kvHead >= 8 || logicalStart >= 2048 ||
        logicalStart + validTokens > 2048) return;
    PagedKvStageProbe probe;
    probe.Init(kCache, vCache, blockTable, kTile, vTransposeTile,
               logicalStart, validTokens, kvHead);
    probe.Process();
}
