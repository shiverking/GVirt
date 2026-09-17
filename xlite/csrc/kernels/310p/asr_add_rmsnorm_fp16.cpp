/*
 * Qwen3-ASR-only FP16 residual-add + RMSNorm for Ascend310P3.
 * Fixed contract: [tokens, 2048], tokens in [1, 20].
 *
 * UB layout (bytes): input 4096, residual 4096, weight 4096,
 * output 4096, then six FP32 rows (input, residual, sum, reduce-A,
 * reduce-B, weight) at 8192 bytes each. Total: 65536 bytes.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kFp32Lanes = 64;

__aicore__ inline void SetFp32Mask(uint32_t len)
{
    const uint64_t tail = len % kFp32Lanes;
    const uint64_t mask = tail == 0 ? static_cast<uint64_t>(-1)
                                    : (static_cast<uint64_t>(1) << tail) - 1;
    if (len >= kFp32Lanes) {
        set_vector_mask(mask, static_cast<uint64_t>(-1));
    } else {
        set_vector_mask(0, mask);
    }
}

__aicore__ inline void ReducePowerOfTwoFp32(
    __ubuf__ float *source, __ubuf__ float *destination)
{
    uint32_t remain = kHidden;
    while (remain > 1) {
        const uint32_t half = remain / 2;
        uint32_t repeat = 1;
        if (half >= kFp32Lanes) {
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            repeat = half / kFp32Lanes;
        } else {
            SetFp32Mask(half);
        }
        vadd(destination, source, source + half, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        __ubuf__ float *previousSource = source;
        source = destination;
        destination = previousSource;
        remain = half;
    }
}

class AsrAddRmsNormKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR residual, GM_ADDR weight,
                                GM_ADDR output, uint32_t tokens, float eps)
    {
        input_ = reinterpret_cast<__gm__ half *>(input);
        residual_ = reinterpret_cast<__gm__ half *>(residual);
        weight_ = reinterpret_cast<__gm__ half *>(weight);
        output_ = reinterpret_cast<__gm__ half *>(output);
        tokens_ = tokens;
        eps_ = eps;

        inputUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        inputUb_.address_.bufferAddr = 0;
        residualUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        residualUb_.address_.bufferAddr = kHidden * sizeof(half);
        weightUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        weightUb_.address_.bufferAddr = 2 * kHidden * sizeof(half);
        outputUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        outputUb_.address_.bufferAddr = 3 * kHidden * sizeof(half);
        inputFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        inputFp32_.address_.bufferAddr = 4 * kHidden * sizeof(half);
        residualFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        residualFp32_.address_.bufferAddr =
            4 * kHidden * sizeof(half) + kHidden * sizeof(float);
        sumFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        sumFp32_.address_.bufferAddr =
            4 * kHidden * sizeof(half) + 2 * kHidden * sizeof(float);
        reduceAFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        reduceAFp32_.address_.bufferAddr =
            4 * kHidden * sizeof(half) + 3 * kHidden * sizeof(float);
        reduceBFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        reduceBFp32_.address_.bufferAddr =
            4 * kHidden * sizeof(half) + 4 * kHidden * sizeof(float);
        weightFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        weightFp32_.address_.bufferAddr =
            4 * kHidden * sizeof(half) + 5 * kHidden * sizeof(float);
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        const event_t loadEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        const event_t storeEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t inputFreeEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE2));
        const event_t outputFreeEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
        const event_t vectorToScalarEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_S));
        const event_t scalarToVectorEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::S_V));
        constexpr uint32_t halfBlocks = kHidden * sizeof(half) / 32;
        constexpr uint32_t fp16ToFp32Repeats = kHidden / kFp32Lanes;
        // Keep floating scalar operands stack-local on M200.
        float invHidden = 0.00048828125F;

        copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half *>(weightUb_.GetPhyAddr()),
                        weight_, 0, 1, halfBlocks, 0, 0);
        SetFlag<HardEvent::MTE2_V>(loadEvent);
        WaitFlag<HardEvent::MTE2_V>(loadEvent);
        vconv_f162f32(reinterpret_cast<__ubuf__ float *>(weightFp32_.GetPhyAddr()),
                      reinterpret_cast<__ubuf__ half *>(weightUb_.GetPhyAddr()),
                      fp16ToFp32Repeats, 1, 1, 8, 4);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE2>(inputFreeEvent);
        SetFlag<HardEvent::MTE3_V>(outputFreeEvent);

        for (uint32_t row = GetBlockIdx(); row < tokens_; row += GetBlockNum()) {
            auto *inputFp32 = reinterpret_cast<__ubuf__ float *>(inputFp32_.GetPhyAddr());
            auto *residualFp32 =
                reinterpret_cast<__ubuf__ float *>(residualFp32_.GetPhyAddr());
            auto *sum = reinterpret_cast<__ubuf__ float *>(sumFp32_.GetPhyAddr());
            auto *reduceA = reinterpret_cast<__ubuf__ float *>(reduceAFp32_.GetPhyAddr());
            auto *reduceB = reinterpret_cast<__ubuf__ float *>(reduceBFp32_.GetPhyAddr());

            WaitFlag<HardEvent::V_MTE2>(inputFreeEvent);
            copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half *>(inputUb_.GetPhyAddr()),
                            input_ + row * kHidden, 0, 1, halfBlocks, 0, 0);
            SetFlag<HardEvent::MTE2_V>(loadEvent);
            WaitFlag<HardEvent::MTE2_V>(loadEvent);
            vconv_f162f32(inputFp32,
                          reinterpret_cast<__ubuf__ half *>(inputUb_.GetPhyAddr()),
                          fp16ToFp32Repeats, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE2>(inputFreeEvent);

            WaitFlag<HardEvent::V_MTE2>(inputFreeEvent);
            copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half *>(residualUb_.GetPhyAddr()),
                            residual_ + row * kHidden, 0, 1, halfBlocks, 0, 0);
            SetFlag<HardEvent::MTE2_V>(loadEvent);
            WaitFlag<HardEvent::MTE2_V>(loadEvent);
            vconv_f162f32(residualFp32,
                          reinterpret_cast<__ubuf__ half *>(residualUb_.GetPhyAddr()),
                          fp16ToFp32Repeats, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE2>(inputFreeEvent);

            vadd(sum, inputFp32, residualFp32, kHidden / kFp32Lanes,
                 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            // Preserve the legacy contract: residual is FP16-rounded in GM,
            // while normalization consumes the unrounded FP32 sum.
            WaitFlag<HardEvent::MTE3_V>(outputFreeEvent);
            vconv_f322f16(reinterpret_cast<__ubuf__ half *>(outputUb_.GetPhyAddr()),
                          sum, fp16ToFp32Repeats, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE3>(storeEvent);
            WaitFlag<HardEvent::V_MTE3>(storeEvent);
            copy_ubuf_to_gm(residual_ + row * kHidden,
                            reinterpret_cast<__ubuf__ half *>(outputUb_.GetPhyAddr()),
                            0, 1, halfBlocks, 0, 0);
            SetFlag<HardEvent::MTE3_V>(outputFreeEvent);

            vmul(reduceA, sum, sum, kHidden / kFp32Lanes,
                 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(reduceB, reduceA, invHidden, kHidden / kFp32Lanes,
                  1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            ReducePowerOfTwoFp32(reduceB, reduceA);

            SetFp32Mask(1);
            // 2048 has eleven reduction stages, so the result is in reduceA.
            // Keep the following Vector operations out-of-place regardless:
            // reduced -> reduceB -> reduceA.
            vadds(reduceB, reduceA, eps_, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vsqrt(reduceA, reduceB, 1, 1, 1, 8, 8);
            SetFlag<HardEvent::V_S>(vectorToScalarEvent);
            WaitFlag<HardEvent::V_S>(vectorToScalarEvent);
            float inverseRms = *reduceA;
            inverseRms = 1.0F / inverseRms;
            SetFlag<HardEvent::S_V>(scalarToVectorEvent);
            WaitFlag<HardEvent::S_V>(scalarToVectorEvent);
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            vector_dup(reduceB, inverseRms, kHidden / kFp32Lanes, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            vmul(reduceA, sum, reduceB, kHidden / kFp32Lanes,
                 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmul(inputFp32, reduceA,
                 reinterpret_cast<__ubuf__ float *>(weightFp32_.GetPhyAddr()),
                 kHidden / kFp32Lanes, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            WaitFlag<HardEvent::MTE3_V>(outputFreeEvent);
            vconv_f322f16(reinterpret_cast<__ubuf__ half *>(outputUb_.GetPhyAddr()),
                          inputFp32, fp16ToFp32Repeats, 1, 1, 4, 8);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE3>(storeEvent);
            WaitFlag<HardEvent::V_MTE3>(storeEvent);
            copy_ubuf_to_gm(output_ + row * kHidden,
                            reinterpret_cast<__ubuf__ half *>(outputUb_.GetPhyAddr()),
                            0, 1, halfBlocks, 0, 0);
            SetFlag<HardEvent::MTE3_V>(outputFreeEvent);
        }
        WaitFlag<HardEvent::V_MTE2>(inputFreeEvent);
        WaitFlag<HardEvent::MTE3_V>(outputFreeEvent);
        pipe_barrier(PIPE_ALL);
    }

private:
    TPipe pipe_;
    __gm__ half *input_ = nullptr;
    __gm__ half *residual_ = nullptr;
    __gm__ half *weight_ = nullptr;
    __gm__ half *output_ = nullptr;
    LocalTensor<half> inputUb_;
    LocalTensor<half> residualUb_;
    LocalTensor<half> weightUb_;
    LocalTensor<half> outputUb_;
    LocalTensor<float> inputFp32_;
    LocalTensor<float> residualFp32_;
    LocalTensor<float> sumFp32_;
    LocalTensor<float> reduceAFp32_;
    LocalTensor<float> reduceBFp32_;
    LocalTensor<float> weightFp32_;
    uint32_t tokens_ = 0;
    float eps_ = 0.0F;
};

}  // namespace

extern "C" __global__ __aicore__ void asr_add_rmsnorm_fp16(
    GM_ADDR input, GM_ADDR residual, GM_ADDR weight, GM_ADDR output,
    uint32_t tokens, float eps)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_add_rmsnorm_fp16 requires the real Ascend310P3 M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tokens == 0 || tokens > 20) {
        return;
    }
    AsrAddRmsNormKernel kernel;
    kernel.Init(input, residual, weight, output, tokens, eps);
    kernel.Process();
}
