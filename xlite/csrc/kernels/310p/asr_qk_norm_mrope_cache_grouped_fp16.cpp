/*
 * Qwen3-ASR-only grouped Q/K RMSNorm, three-axis interleaved MRoPE,
 * Q scaling and BSHD K/V cache write for Ascend310P3.
 *
 * One AIV owns two Q heads and the matching K/V head for every token.  Q/K
 * weights are converted once per launch and the three position rows are
 * composed once per token, then reused by all three heads.
 *
 * UB=4864 bytes, no L1/L0. Seven event channels use FetchEventID.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQDim = kQHeads * kHeadDim;
constexpr uint32_t kKDim = kKvHeads * kHeadDim;
constexpr uint32_t kQkvDim = kQDim + 2 * kKDim;
constexpr uint64_t kMropeMaskH = 0x0492492492492492ULL;
constexpr uint64_t kMropeMaskW = 0x0924924924924924ULL;

__aicore__ inline void Reduce128(__ubuf__ float *source,
                                  __ubuf__ float *destination)
{
    uint32_t remain = kHeadDim;
    while (remain > 1) {
        const uint32_t half = remain / 2;
        const uint64_t mask = half >= 64 ? static_cast<uint64_t>(-1)
                                         : (static_cast<uint64_t>(1) << half) - 1;
        set_vector_mask(0, mask);
        vadd(destination, source, source + half, 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        __ubuf__ float *old = source;
        source = destination;
        destination = old;
        remain = half;
    }
}

class GroupedQkNormMropeCacheKernel {
public:
    __aicore__ inline void Init(GM_ADDR qkv, GM_ADDR qWeight, GM_ADDR kWeight,
                                GM_ADDR positions, GM_ADDR cossin, GM_ADDR slots,
                                GM_ADDR kCache, GM_ADDR vCache, uint32_t tokens,
                                float eps, float qScale)
    {
        qkv_ = reinterpret_cast<__gm__ half *>(qkv);
        qWeight_ = reinterpret_cast<__gm__ half *>(qWeight);
        kWeight_ = reinterpret_cast<__gm__ half *>(kWeight);
        positions_ = reinterpret_cast<__gm__ int64_t *>(positions);
        cossin_ = reinterpret_cast<__gm__ half *>(cossin);
        slots_ = reinterpret_cast<__gm__ int32_t *>(slots);
        kCache_ = reinterpret_cast<__gm__ half *>(kCache);
        vCache_ = reinterpret_cast<__gm__ half *>(vCache);
        tokens_ = tokens;
        eps_ = eps;
        qScale_ = qScale;

        Bind(inputH_, 0);
        Bind(normH_, 256);
        Bind(outputH_, 512);
        Bind(cosH_, 768);
        Bind(sinH_, 1024);
        Bind(freqTH_, 1280);
        Bind(freqHH_, 1536);
        Bind(freqWH_, 1792);
        Bind(valueH_, 2048);
        Bind(inputF_, 2304);
        Bind(squareAF_, 2816);
        Bind(squareBF_, 3328);
        Bind(qWeightF_, 3840);
        Bind(kWeightF_, 4352);

        load_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        store_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        loadFree_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        storeFree_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        valueReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE3));
        vToS_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        sToV_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        float invHead = 0.0078125F;
        half zero = static_cast<half>(0.0F);
        half qScaleHalf = static_cast<half>(qScale_);

        SetFlag<HardEvent::V_MTE2>(loadFree_);
        SetFlag<HardEvent::MTE3_V>(storeFree_);

        // Persistent FP32 weights: two GM reads and conversions per AIV launch,
        // independent of token count and the three heads owned by this core.
        WaitFlag<HardEvent::V_MTE2>(loadFree_);
        copy_gm_to_ubuf(H(freqTH_), qWeight_, 0, 1, 8, 0, 0);
        copy_gm_to_ubuf(H(freqHH_), kWeight_, 0, 1, 8, 0, 0);
        SetFlag<HardEvent::MTE2_V>(load_);
        WaitFlag<HardEvent::MTE2_V>(load_);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vconv_f162f32(F(qWeightF_), H(freqTH_), 2, 1, 1, 8, 4);
        vconv_f162f32(F(kWeightF_), H(freqHH_), 2, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE2>(loadFree_);

        const uint32_t head = GetBlockIdx();
        if (head >= kKvHeads) {
            WaitFlag<HardEvent::V_MTE2>(loadFree_);
            WaitFlag<HardEvent::MTE3_V>(storeFree_);
            return;
        }
        for (uint32_t token = 0; token < tokens_; ++token) {
            const uint64_t posT = static_cast<uint64_t>(positions_[token]);
            const uint64_t posH = static_cast<uint64_t>(positions_[tokens_ + token]);
            const uint64_t posW = static_cast<uint64_t>(positions_[2 * tokens_ + token]);

            WaitFlag<HardEvent::V_MTE2>(loadFree_);
            copy_gm_to_ubuf(H(freqTH_), cossin_ + posT * kHeadDim,
                            0, 1, 8, 0, 0);
            copy_gm_to_ubuf(H(freqHH_), cossin_ + posH * kHeadDim,
                            0, 1, 8, 0, 0);
            copy_gm_to_ubuf(H(freqWH_), cossin_ + posW * kHeadDim,
                            0, 1, 8, 0, 0);
            SetFlag<HardEvent::MTE2_V>(load_);
            WaitFlag<HardEvent::MTE2_V>(load_);
            ComposeMrope(zero);
            SetFlag<HardEvent::V_MTE2>(loadFree_);

            ProcessHead(token, 2 * head, true, qWeightF_, invHead,
                        zero, qScaleHalf);
            ProcessHead(token, 2 * head + 1, true, qWeightF_, invHead,
                        zero, qScaleHalf);
            ProcessHead(token, head, false, kWeightF_, invHead,
                        zero, qScaleHalf);
            StoreValue(token, head);
        }
        WaitFlag<HardEvent::V_MTE2>(loadFree_);
        WaitFlag<HardEvent::MTE3_V>(storeFree_);
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void ComposeMrope(half zero)
    {
        set_vector_mask(0, static_cast<uint64_t>(-1));
        vadds(H(cosH_), H(freqTH_), zero, 1, 1, 1, 8, 8);
        vadds(H(sinH_), H(freqTH_) + 64, zero, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, kMropeMaskH);
        vadds(H(cosH_), H(freqHH_), zero, 1, 1, 1, 8, 8);
        vadds(H(sinH_), H(freqHH_) + 64, zero, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(0, kMropeMaskW);
        vadds(H(cosH_), H(freqWH_), zero, 1, 1, 1, 8, 8);
        vadds(H(sinH_), H(freqWH_) + 64, zero, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
    }

    __aicore__ inline void ProcessHead(uint32_t token, uint32_t head, bool isQ,
                                       LocalTensor<float> &weight, float invHead,
                                       half zero, half qScaleHalf)
    {
        WaitFlag<HardEvent::MTE3_V>(storeFree_);
        const uint32_t qkvOffset = token * kQkvDim +
            (isQ ? head * kHeadDim : kQDim + head * kHeadDim);
        WaitFlag<HardEvent::V_MTE2>(loadFree_);
        copy_gm_to_ubuf(H(inputH_), qkv_ + qkvOffset, 0, 1, 8, 0, 0);
        SetFlag<HardEvent::MTE2_V>(load_);
        WaitFlag<HardEvent::MTE2_V>(load_);

        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vconv_f162f32(F(inputF_), H(inputH_), 2, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        vmul(F(squareAF_), F(inputF_), F(inputF_), 2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmuls(F(squareBF_), F(squareAF_), invHead, 2, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        Reduce128(F(squareBF_), F(squareAF_));
        set_vector_mask(0, 1);
        vadds(F(squareBF_), F(squareAF_), eps_, 1, 1, 1, 8, 8);
        pipe_barrier(PIPE_V);
        vsqrt(F(squareAF_), F(squareBF_), 1, 1, 1, 8, 8);
        SetFlag<HardEvent::V_S>(vToS_);
        WaitFlag<HardEvent::V_S>(vToS_);
        float invRms = 1.0F / *F(squareAF_);
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        vector_dup(F(squareBF_), invRms, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(squareAF_), F(inputF_), F(squareBF_), 2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vmul(F(inputF_), F(squareAF_), F(weight), 2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        vconv_f322f16(H(normH_), F(inputF_), 2, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);

        set_vector_mask(0, static_cast<uint64_t>(-1));
        vmul(H(inputH_), H(normH_), H(cosH_), 1, 1, 1, 1, 8, 8, 0);
        vmul(H(freqWH_), H(normH_) + 64, H(cosH_), 1, 1, 1, 1, 8, 8, 0);
        vmul(H(freqTH_), H(normH_), H(sinH_), 1, 1, 1, 1, 8, 8, 0);
        vmul(H(freqHH_), H(normH_) + 64, H(sinH_), 1, 1, 1, 1, 8, 8, 0);
        pipe_barrier(PIPE_V);
        vsub(H(outputH_), H(inputH_), H(freqHH_), 1, 1, 1, 1, 8, 8, 8);
        vadd(H(outputH_) + 64, H(freqWH_), H(freqTH_), 1, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        if (isQ) {
            vmuls(H(normH_), H(outputH_), qScaleHalf, 1, 1, 1, 8, 8);
        } else {
            vadds(H(normH_), H(outputH_), zero, 1, 1, 1, 8, 8);
        }
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE2>(loadFree_);

        SetFlag<HardEvent::V_MTE3>(store_);
        WaitFlag<HardEvent::V_MTE3>(store_);
        copy_ubuf_to_gm(qkv_ + qkvOffset, H(normH_), 0, 1, 8, 0, 0);
        if (!isQ) {
            const uint32_t slot = static_cast<uint32_t>(slots_[token]);
            const uint64_t cacheOffset =
                (static_cast<uint64_t>(slot) * kKvHeads + head) * kHeadDim;
            copy_ubuf_to_gm(kCache_ + cacheOffset, H(normH_), 0, 1, 8, 0, 0);
        }
        SetFlag<HardEvent::MTE3_V>(storeFree_);
    }

    __aicore__ inline void StoreValue(uint32_t token, uint32_t head)
    {
        // loadFree_ was returned after the K vector stage. valueH_ bypasses V
        // and is ordered directly from MTE2 to MTE3.
        WaitFlag<HardEvent::V_MTE2>(loadFree_);
        copy_gm_to_ubuf(H(valueH_),
                        qkv_ + token * kQkvDim + kQDim + kKDim +
                            head * kHeadDim,
                        0, 1, 8, 0, 0);
        SetFlag<HardEvent::MTE2_MTE3>(valueReady_);
        WaitFlag<HardEvent::MTE2_MTE3>(valueReady_);
        WaitFlag<HardEvent::MTE3_V>(storeFree_);
        const uint32_t slot = static_cast<uint32_t>(slots_[token]);
        const uint64_t cacheOffset =
            (static_cast<uint64_t>(slot) * kKvHeads + head) * kHeadDim;
        copy_ubuf_to_gm(vCache_ + cacheOffset, H(valueH_), 0, 1, 8, 0, 0);
        SetFlag<HardEvent::MTE3_V>(storeFree_);
        SetFlag<HardEvent::V_MTE2>(loadFree_);
    }

    template <typename T>
    __aicore__ inline void Bind(LocalTensor<T> &tensor, uint32_t offset)
    {
        tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        tensor.address_.bufferAddr = offset;
    }
    __aicore__ inline __ubuf__ half *H(LocalTensor<half> &tensor)
    {
        return reinterpret_cast<__ubuf__ half *>(tensor.GetPhyAddr());
    }
    __aicore__ inline __ubuf__ float *F(LocalTensor<float> &tensor)
    {
        return reinterpret_cast<__ubuf__ float *>(tensor.GetPhyAddr());
    }

    TPipe pipe_;
    __gm__ half *qkv_ = nullptr;
    __gm__ half *qWeight_ = nullptr;
    __gm__ half *kWeight_ = nullptr;
    __gm__ int64_t *positions_ = nullptr;
    __gm__ half *cossin_ = nullptr;
    __gm__ int32_t *slots_ = nullptr;
    __gm__ half *kCache_ = nullptr;
    __gm__ half *vCache_ = nullptr;
    LocalTensor<half> inputH_, normH_, outputH_, cosH_, sinH_;
    LocalTensor<half> freqTH_, freqHH_, freqWH_, valueH_;
    LocalTensor<float> inputF_, squareAF_, squareBF_, qWeightF_, kWeightF_;
    event_t load_, store_, loadFree_, storeFree_, valueReady_, vToS_, sToV_;
    uint32_t tokens_ = 0;
    float eps_ = 0.0F;
    float qScale_ = 0.0F;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_qk_norm_mrope_cache_grouped_fp16(
    GM_ADDR qkv, GM_ADDR qWeight, GM_ADDR kWeight, GM_ADDR positions,
    GM_ADDR cossin, GM_ADDR slots, GM_ADDR kCache, GM_ADDR vCache,
    uint32_t tokens, float eps, float qScale)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_qk_norm_mrope_cache_grouped_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tokens == 0 || tokens > 20 || GetBlockNum() != kKvHeads) {
        return;
    }
    GroupedQkNormMropeCacheKernel kernel;
    kernel.Init(qkv, qWeight, kWeight, positions, cossin, slots, kCache,
                vCache, tokens, eps, qScale);
    kernel.Process();
}
