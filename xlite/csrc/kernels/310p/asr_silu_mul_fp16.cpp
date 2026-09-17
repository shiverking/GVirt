/*
 * Qwen3-ASR-only FP16 SiLU(gate) * up for Ascend310P3.
 * Fixed contract: input [tokens, 12288], output [tokens, 6144].
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kIntermediate = 6144;
constexpr uint32_t kInputWidth = 2 * kIntermediate;
constexpr uint32_t kTile = 2048;
constexpr uint32_t kHalfLanes = 128;

class AsrSiluMulKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t tokens)
    {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input), tokens * kInputWidth);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output), tokens * kIntermediate);
        tokens_ = tokens;
        gateUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        gateUb_.address_.bufferAddr = 0;
        upUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        upUb_.address_.bufferAddr = kTile * sizeof(half);
        tmpUb_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        tmpUb_.address_.bufferAddr = 2 * kTile * sizeof(half);
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
        const event_t bufferFreeEvent = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
        constexpr uint32_t blocks = kTile * sizeof(half) / 32;
        constexpr uint32_t repeats = kTile / kHalfLanes;

        SetFlag<HardEvent::MTE3_MTE2>(bufferFreeEvent);
        for (uint32_t row = GetBlockIdx(); row < tokens_; row += GetBlockNum()) {
            for (uint32_t offset = 0; offset < kIntermediate; offset += kTile) {
                const uint32_t inputBase = row * kInputWidth + offset;
                WaitFlag<HardEvent::MTE3_MTE2>(bufferFreeEvent);
                copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half *>(gateUb_.GetPhyAddr()),
                                reinterpret_cast<__gm__ half *>(input_.GetPhyAddr()) +
                                    inputBase,
                                0, 1, blocks, 0, 0);
                copy_gm_to_ubuf(reinterpret_cast<__ubuf__ half *>(upUb_.GetPhyAddr()),
                                reinterpret_cast<__gm__ half *>(input_.GetPhyAddr()) +
                                    inputBase + kIntermediate,
                                0, 1, blocks, 0, 0);
                SetFlag<HardEvent::MTE2_V>(loadEvent);
                WaitFlag<HardEvent::MTE2_V>(loadEvent);

                auto *gate = reinterpret_cast<__ubuf__ half *>(gateUb_.GetPhyAddr());
                auto *up = reinterpret_cast<__ubuf__ half *>(upUb_.GetPhyAddr());
                auto *tmp = reinterpret_cast<__ubuf__ half *>(tmpUb_.GetPhyAddr());
                vmuls(tmp, gate, static_cast<half>(-1.0F), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vexp(tmp, tmp, repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vadds(tmp, tmp, static_cast<half>(1.0F), repeats, 1, 1, 8, 8);
                pipe_barrier(PIPE_V);
                vdiv(tmp, gate, tmp, repeats, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);
                vmul(gate, up, tmp, repeats, 1, 1, 1, 8, 8, 8);
                pipe_barrier(PIPE_V);

                SetFlag<HardEvent::V_MTE3>(storeEvent);
                WaitFlag<HardEvent::V_MTE3>(storeEvent);
                copy_ubuf_to_gm(
                    reinterpret_cast<__gm__ half *>(output_.GetPhyAddr()) +
                    row * kIntermediate + offset,
                    gate, 0, 1, blocks, 0, 0);
                SetFlag<HardEvent::MTE3_MTE2>(bufferFreeEvent);
            }
        }
        WaitFlag<HardEvent::MTE3_MTE2>(bufferFreeEvent);
        pipe_barrier(PIPE_ALL);
    }

private:
    TPipe pipe_;
    GlobalTensor<half> input_;
    GlobalTensor<half> output_;
    LocalTensor<half> gateUb_;
    LocalTensor<half> upUb_;
    LocalTensor<half> tmpUb_;
    uint32_t tokens_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void asr_silu_mul_fp16(
    GM_ADDR input, GM_ADDR output, uint32_t tokens)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_silu_mul_fp16 requires the real Ascend310P3 M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tokens == 0 || tokens > 20) {
        return;
    }
    AsrSiluMulKernel kernel;
    kernel.Init(input, output, tokens);
    kernel.Process();
}
