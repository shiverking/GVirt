/*
 * Qwen3-ASR-only paged decode attention correctness kernel for Ascend310P3.
 *
 * This is the production-path micro-probe for paged addressing and FP32 online
 * softmax.  It deliberately uses vector dot/PV updates; the subsequent stage
 * replaces those two blocks with audited low-level LoadData/Mmad blocks before
 * Runtime integration.  Q is already scaled by 1/sqrt(128).
 *
 * Contract:
 *   qkv:         [batch, 4096] FP16 (Q occupies the first 2048 values)
 *   k/v cache:   [num_blocks, 128, 8, 128] FP16 BSHD
 *   block table: [batch, table_stride] INT32
 *   kv lengths:  [batch] INT32, inclusive of the current token
 *   output:      [batch, 2048] FP16
 *
 * UB layout (all offsets are bytes):
 *   four FP16 heads                         1024
 *   Q/K/V/product/reduce/acc0/acc1/weighted 4096
 *   exp input/output (8 FP32 lanes each)      64
 *   total                                    5184 bytes
 * L1/L0A/L0B/L0C: 0 bytes.
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
constexpr uint32_t kMaxBatch = 20;
constexpr uint32_t kMaxKv = 2048;

__aicore__ inline __ubuf__ float *ReduceSum128(__ubuf__ float *source,
                                                __ubuf__ float *scratch)
{
    uint32_t remain = kHeadDim;
    while (remain > 1) {
        const uint32_t half = remain / 2;
        const uint64_t mask = half >= 64 ? static_cast<uint64_t>(-1)
                                         : (static_cast<uint64_t>(1) << half) - 1;
        set_vector_mask(0, mask);
        vadd(scratch, source, source + half, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        __ubuf__ float *old = source;
        source = scratch;
        scratch = old;
        remain = half;
    }
    return source;
}

class AsrPagedDecodeAttentionProbeKernel {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache,
                                GM_ADDR blockTable, GM_ADDR kvLengths,
                                GM_ADDR output, uint32_t batch,
                                uint32_t tableStride)
    {
        qkv_ = reinterpret_cast<__gm__ half *>(qkv);
        kCache_ = reinterpret_cast<__gm__ half *>(kCache);
        vCache_ = reinterpret_cast<__gm__ half *>(vCache);
        blockTable_ = reinterpret_cast<__gm__ int32_t *>(blockTable);
        kvLengths_ = reinterpret_cast<__gm__ int32_t *>(kvLengths);
        output_ = reinterpret_cast<__gm__ half *>(output);
        batch_ = batch;
        tableStride_ = tableStride;

        Bind(qH_, 0); Bind(kH_, 256); Bind(vH_, 512); Bind(outH_, 768);
        Bind(qF_, 1024); Bind(kF_, 1536); Bind(vF_, 2048);
        Bind(productF_, 2560); Bind(reduceF_, 3072);
        Bind(accAF_, 3584); Bind(accBF_, 4096); Bind(weightedF_, 4608);
        Bind(expInputF_, 5120); Bind(expOutputF_, 5152);
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        const event_t loadReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t loadFree = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const event_t vectorToScalar = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        const event_t scalarToVector = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        const event_t storeReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t storeFree = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        float one = 1.0F;
        SetFlag<HardEvent::V_MTE2>(loadFree);
        SetFlag<HardEvent::MTE3_V>(storeFree);

        const uint32_t work = batch_ * kQHeads;
        for (uint32_t item = GetBlockIdx(); item < work; item += GetBlockNum()) {
            const uint32_t request = item / kQHeads;
            const uint32_t queryHead = item % kQHeads;
            const uint32_t kvHead = queryHead / 2;
            const int32_t kvLengthSigned = kvLengths_[request];
            if (kvLengthSigned <= 0 || kvLengthSigned > static_cast<int32_t>(kMaxKv)) {
                continue;
            }
            const uint32_t kvLength = static_cast<uint32_t>(kvLengthSigned);

            WaitFlag<HardEvent::V_MTE2>(loadFree);
            copy_gm_to_ubuf(H(qH_), qkv_ + static_cast<uint64_t>(request) * kQkvDim +
                            queryHead * kHeadDim, 0, 1, 8, 0, 0);
            SetFlag<HardEvent::MTE2_V>(loadReady);
            WaitFlag<HardEvent::MTE2_V>(loadReady);
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            vconv_f162f32(F(qF_), H(qH_), 2, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE2>(loadFree);

            float runningMax = 0.0F;
            float runningSum = 0.0F;
            __ubuf__ float *accumulator = F(accAF_);
            __ubuf__ float *nextAccumulator = F(accBF_);

            for (uint32_t token = 0; token < kvLength; ++token) {
                const uint32_t logicalBlock = token / kBlockSize;
                const int32_t physicalBlockSigned =
                    blockTable_[request * tableStride_ + logicalBlock];
                if (physicalBlockSigned < 0) {
                    continue;
                }
                const uint32_t physicalBlock = static_cast<uint32_t>(physicalBlockSigned);
                const uint32_t tokenInBlock = token % kBlockSize;
                const uint64_t cacheOffset =
                    ((static_cast<uint64_t>(physicalBlock) * kBlockSize + tokenInBlock) *
                     kKvHeads + kvHead) * kHeadDim;
                WaitFlag<HardEvent::V_MTE2>(loadFree);
                copy_gm_to_ubuf(H(kH_), kCache_ + cacheOffset, 0, 1, 8, 0, 0);
                copy_gm_to_ubuf(H(vH_), vCache_ + cacheOffset, 0, 1, 8, 0, 0);
                SetFlag<HardEvent::MTE2_V>(loadReady);
                WaitFlag<HardEvent::MTE2_V>(loadReady);
                set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
                vconv_f162f32(F(kF_), H(kH_), 2, 1, 1, 8, 4);
                vconv_f162f32(F(vF_), H(vH_), 2, 1, 1, 8, 4);
                pipe_barrier(PIPE_V);
                SetFlag<HardEvent::V_MTE2>(loadFree);
                vmul(F(productF_), F(qF_), F(kF_), 2, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                __ubuf__ float *scoreAddress = ReduceSum128(F(productF_), F(reduceF_));
                SetFlag<HardEvent::V_S>(vectorToScalar);
                WaitFlag<HardEvent::V_S>(vectorToScalar);
                const float score = *scoreAddress;

                if (token == 0) {
                    runningMax = score;
                    runningSum = one;
                    SetFlag<HardEvent::S_V>(scalarToVector);
                    WaitFlag<HardEvent::S_V>(scalarToVector);
                    set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
                    vmuls(accumulator, F(vF_), one, 2, 1, 1, 8, 8);
                    pipe_barrier(PIPE_V);
                    continue;
                }

                const float newMax = score > runningMax ? score : runningMax;
                F(expInputF_)[0] = runningMax - newMax;
                F(expInputF_)[1] = score - newMax;
                SetFlag<HardEvent::S_V>(scalarToVector);
                WaitFlag<HardEvent::S_V>(scalarToVector);
                set_vector_mask(0, 3);
                vexp(F(expOutputF_), F(expInputF_), 1, 1, 1, 8, 8);
                SetFlag<HardEvent::V_S>(vectorToScalar);
                WaitFlag<HardEvent::V_S>(vectorToScalar);
                const float correction = F(expOutputF_)[0];
                const float weight = F(expOutputF_)[1];
                runningSum = runningSum * correction + weight;
                runningMax = newMax;
                SetFlag<HardEvent::S_V>(scalarToVector);
                WaitFlag<HardEvent::S_V>(scalarToVector);
                set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
                // CANN 9.1 beta1 does not accept a runtime Scalar value as the
                // third vmuls operand on M200.  Materialize both scalars in UB
                // after the explicit S_V handoff, then use ordinary vmul.
                vector_dup(F(kF_), correction, 2, 1, 1, 8, 1);
                vector_dup(F(productF_), weight, 2, 1, 1, 8, 1);
                pipe_barrier(PIPE_V);
                vmul(nextAccumulator, accumulator, F(kF_), 2, 1, 1, 1, 8, 8, 8);
                vmul(F(weightedF_), F(vF_), F(productF_), 2, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vadd(nextAccumulator, nextAccumulator, F(weightedF_),
                     2, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                __ubuf__ float *old = accumulator;
                accumulator = nextAccumulator;
                nextAccumulator = old;
            }

            const float inverseSum = one / runningSum;
            SetFlag<HardEvent::S_V>(scalarToVector);
            WaitFlag<HardEvent::S_V>(scalarToVector);
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            vector_dup(F(kF_), inverseSum, 2, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            vmul(F(weightedF_), accumulator, F(kF_), 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            WaitFlag<HardEvent::MTE3_V>(storeFree);
            vconv_f322f16(H(outH_), F(weightedF_), 2, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE3>(storeReady);
            WaitFlag<HardEvent::V_MTE3>(storeReady);
            copy_ubuf_to_gm(output_ + static_cast<uint64_t>(request) * kQDim +
                            queryHead * kHeadDim, H(outH_), 0, 1, 8, 0, 0);
            SetFlag<HardEvent::MTE3_V>(storeFree);
        }
        WaitFlag<HardEvent::V_MTE2>(loadFree);
        WaitFlag<HardEvent::MTE3_V>(storeFree);
        pipe_barrier(PIPE_ALL);
    }

private:
    template <typename T>
    __aicore__ inline void Bind(LocalTensor<T> &tensor, uint32_t offset)
    {
        tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        tensor.address_.bufferAddr = offset;
    }
    __aicore__ inline __ubuf__ half *H(LocalTensor<half> &tensor)
    { return reinterpret_cast<__ubuf__ half *>(tensor.GetPhyAddr()); }
    __aicore__ inline __ubuf__ float *F(LocalTensor<float> &tensor)
    { return reinterpret_cast<__ubuf__ float *>(tensor.GetPhyAddr()); }

    TPipe pipe_;
    __gm__ half *qkv_ = nullptr;
    __gm__ half *kCache_ = nullptr;
    __gm__ half *vCache_ = nullptr;
    __gm__ int32_t *blockTable_ = nullptr;
    __gm__ int32_t *kvLengths_ = nullptr;
    __gm__ half *output_ = nullptr;
    LocalTensor<half> qH_, kH_, vH_, outH_;
    LocalTensor<float> qF_, kF_, vF_, productF_, reduceF_;
    LocalTensor<float> accAF_, accBF_, weightedF_, expInputF_, expOutputF_;
    uint32_t batch_ = 0;
    uint32_t tableStride_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_paged_decode_attention_fp16(
    GM_ADDR qkv, GM_ADDR kCache, GM_ADDR vCache, GM_ADDR blockTable,
    GM_ADDR kvLengths, GM_ADDR output, uint32_t batch, uint32_t tableStride)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_paged_decode_attention_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (batch == 0 || batch > kMaxBatch || tableStride == 0 || tableStride > 16) {
        return;
    }
    AsrPagedDecodeAttentionProbeKernel kernel;
    kernel.Init(qkv, kCache, vCache, blockTable, kvLengths, output,
                batch, tableStride);
    kernel.Process();
}
