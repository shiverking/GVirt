/*
 * Qwen3-ASR-only whole-row FP16 SiLU(gate) * up for Ascend310P3.
 * Fixed contract: input [tokens,12288], output [tokens,6144], tokens 1..20.
 *
 * One 6144-element row is processed per pipeline iteration, removing the
 * baseline's three 2048-element load/event/store rounds.
 * UB=36864 bytes, no L1/L0. Three events use FetchEventID.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kIntermediate = 6144;
constexpr uint32_t kInputWidth = 2 * kIntermediate;
constexpr uint32_t kHalfLanes = 128;

class AsrSiluMulRowKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR output, uint32_t tokens)
    {
        input_ = reinterpret_cast<__gm__ half *>(input);
        output_ = reinterpret_cast<__gm__ half *>(output);
        tokens_ = tokens;
        Bind(gate_, 0);
        Bind(up_, kIntermediate * sizeof(half));
        Bind(tmp_, 2 * kIntermediate * sizeof(half));
        load_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_V));
        store_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        reusable_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_MTE2));
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        set_vector_mask(static_cast<uint64_t>(-1), static_cast<uint64_t>(-1));
        constexpr uint32_t blocks = kIntermediate * sizeof(half) / 32;
        constexpr uint32_t repeats = kIntermediate / kHalfLanes;
        half negativeOne = static_cast<half>(-1.0F);
        half one = static_cast<half>(1.0F);

        SetFlag<HardEvent::MTE3_MTE2>(reusable_);
        for (uint32_t row = GetBlockIdx(); row < tokens_; row += GetBlockNum()) {
            const uint32_t inputBase = row * kInputWidth;
            WaitFlag<HardEvent::MTE3_MTE2>(reusable_);
            copy_gm_to_ubuf(H(gate_), input_ + inputBase,
                            0, 1, blocks, 0, 0);
            copy_gm_to_ubuf(H(up_), input_ + inputBase + kIntermediate,
                            0, 1, blocks, 0, 0);
            SetFlag<HardEvent::MTE2_V>(load_);
            WaitFlag<HardEvent::MTE2_V>(load_);

            vmuls(H(tmp_), H(gate_), negativeOne, repeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vexp(H(tmp_), H(tmp_), repeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vadds(H(tmp_), H(tmp_), one, repeats, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            vdiv(H(tmp_), H(gate_), H(tmp_), repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vmul(H(gate_), H(up_), H(tmp_), repeats, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);

            SetFlag<HardEvent::V_MTE3>(store_);
            WaitFlag<HardEvent::V_MTE3>(store_);
            copy_ubuf_to_gm(output_ + row * kIntermediate, H(gate_),
                            0, 1, blocks, 0, 0);
            SetFlag<HardEvent::MTE3_MTE2>(reusable_);
        }
        WaitFlag<HardEvent::MTE3_MTE2>(reusable_);
        pipe_barrier(PIPE_ALL);
    }

private:
    __aicore__ inline void Bind(LocalTensor<half> &tensor, uint32_t offset)
    {
        tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        tensor.address_.bufferAddr = offset;
    }
    __aicore__ inline __ubuf__ half *H(LocalTensor<half> &tensor)
    {
        return reinterpret_cast<__ubuf__ half *>(tensor.GetPhyAddr());
    }

    TPipe pipe_;
    __gm__ half *input_ = nullptr;
    __gm__ half *output_ = nullptr;
    LocalTensor<half> gate_, up_, tmp_;
    event_t load_, store_, reusable_;
    uint32_t tokens_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_silu_mul_row_fp16(
    GM_ADDR input, GM_ADDR output, uint32_t tokens)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_silu_mul_row_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (tokens == 0 || tokens > 20) {
        return;
    }
    AsrSiluMulRowKernel kernel;
    kernel.Init(input, output, tokens);
    kernel.Process();
}
