/*
 * Qwen3-ASR-only FP16 RMSNorm for Ascend310P3.
 * Fixed contract: [tokens, 2048], tokens in [1, 20].
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

__aicore__ inline void ReducePowerOfTwoFp32(__ubuf__ float *value)
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
        vadd(value, value, value + half, repeat, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        remain = half;
    }
}

class AsrRmsNormKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR output,
                                uint32_t tokens, float eps)
    {
        input_ = reinterpret_cast<__gm__ half *>(input);
        weight_ = reinterpret_cast<__gm__ half *>(weight);
        output_ = reinterpret_cast<__gm__ half *>(output);
        tokens_ = tokens;
        eps_ = eps;

        inputUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        inputUb_.address_.bufferAddr = 0;
        weightUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        weightUb_.address_.bufferAddr = kHidden * sizeof(half);
        outputUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        outputUb_.address_.bufferAddr = 2 * kHidden * sizeof(half);
        inputFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        inputFp32_.address_.bufferAddr = 3 * kHidden * sizeof(half);
        squareFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        squareFp32_.address_.bufferAddr =
            3 * kHidden * sizeof(half) + kHidden * sizeof(float);
        weightFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        weightFp32_.address_.bufferAddr =
            3 * kHidden * sizeof(half) + 2 * kHidden * sizeof(float);
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
        constexpr uint32_t halfBlocks = kHidden * sizeof(half) / 32;
        constexpr uint32_t fp16ToFp32Repeats = kHidden / kFp32Lanes;
        // M200 places file-scope floating constants in GM. Vector scalar
        // operands must instead be stack-local scalar values.
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
            WaitFlag<HardEvent::V_MTE2>(inputFreeEvent);
            copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half *>(inputUb_.GetPhyAddr()),
                            input_ + row * kHidden,
                            0, 1, halfBlocks, 0, 0);
            SetFlag<HardEvent::MTE2_V>(loadEvent);
            WaitFlag<HardEvent::MTE2_V>(loadEvent);

            auto *inputFp32 = reinterpret_cast<__ubuf__ float *>(inputFp32_.GetPhyAddr());
            auto *square = reinterpret_cast<__ubuf__ float *>(squareFp32_.GetPhyAddr());
            vconv_f162f32(inputFp32,
                          reinterpret_cast<__ubuf__ half *>(inputUb_.GetPhyAddr()),
                          fp16ToFp32Repeats, 1, 1, 8, 4);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE2>(inputFreeEvent);
            vmul(square, inputFp32, inputFp32, kHidden / kFp32Lanes,
                 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmuls(square, square, invHidden,
                  kHidden / kFp32Lanes, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            ReducePowerOfTwoFp32(square);

            SetFp32Mask(1);
            vadds(square, square, eps_, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vsqrt(square, square, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            // On M200 scalar reads are synchronous with V. A scalar FP32
            // reciprocal is more accurate than the approximate vrec path and
            // does not require the redundant C220-style V_S/S_V event pair.
            float inverseRms = *square;
            inverseRms = 1.0F / inverseRms;
            set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
            vector_dup(square, inverseRms, kHidden / kFp32Lanes, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            vmul(inputFp32, inputFp32, square, kHidden / kFp32Lanes,
                 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmul(inputFp32, inputFp32,
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
    __gm__ half *weight_ = nullptr;
    __gm__ half *output_ = nullptr;
    LocalTensor<half> inputUb_;
    LocalTensor<half> weightUb_;
    LocalTensor<half> outputUb_;
    LocalTensor<float> inputFp32_;
    LocalTensor<float> squareFp32_;
    LocalTensor<float> weightFp32_;
    uint32_t tokens_ = 0;
    float eps_ = 0.0F;
};

}  // namespace

extern "C" __global__ __aicore__ void asr_rmsnorm_fp16(
    GM_ADDR input, GM_ADDR weight, GM_ADDR output, uint32_t tokens, float eps)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_rmsnorm_fp16 requires the real Ascend310P3 M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tokens == 0 || tokens > 20) {
        return;
    }
    AsrRmsNormKernel kernel;
    kernel.Init(input, weight, output, tokens, eps);
    kernel.Process();
}
