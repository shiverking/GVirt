/*
 * Ascend310P3 Qwen3-ASR four-partition attention-state merge probe.
 *
 * State layout is [work_items, 4, 144] FP32.  Each aligned record stores
 * {partition_max, partition_sum, unnormalised_output[128]}; the remaining
 * 14 floats are padding and must never participate in arithmetic.  A zero sum
 * marks an empty partition.  All merge arithmetic remains FP32.
 *
 * Resources per core: UB 3,968 bytes; no L1/L0; six dynamic event pairs.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kPartitions = 4;
constexpr uint32_t kStateStride = 144;
constexpr uint32_t kStateBlocks = kStateStride * sizeof(float) / 32;
constexpr uint32_t kQDim = kQHeads * kHeadDim;
constexpr uint32_t kMaxBatch = 20;

class AsrAttentionPartitionMergeProbe {
public:
    __aicore__ inline void Init(GM_ADDR states, GM_ADDR output,
                                uint32_t batch)
    {
        states_.SetGlobalBuffer(reinterpret_cast<__gm__ float *>(states));
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output));
        batch_ = batch;
        Bind(stateF_, 0);
        Bind(accumulatorF_, 576);
        Bind(correctedF_, 1088);
        Bind(scaledPartialF_, 1600);
        Bind(correctionF_, 2112);
        Bind(partitionScaleF_, 2624);
        Bind(expInputF_, 3136);
        Bind(expOutputF_, 3168);
        Bind(finalF_, 3200);
        Bind(outputH_, 3712);
    }

    __aicore__ inline void Process()
    {
        set_atomic_none();
        set_mask_norm();
        FetchEvents();
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
        const uint32_t workItems = batch_ * kQHeads;
        for (uint32_t item = GetBlockIdx(); item < workItems;
             item += GetBlockNum()) {
            Merge(item);
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
        mte2ToV_ = Fetch(HardEvent::MTE2_V);
        vToMte2_ = Fetch(HardEvent::V_MTE2);
        vToS_ = Fetch(HardEvent::V_S);
        sToV_ = Fetch(HardEvent::S_V);
        vToMte3_ = Fetch(HardEvent::V_MTE3);
        mte3ToV_ = Fetch(HardEvent::MTE3_V);
    }

    __aicore__ inline void Merge(uint32_t item)
    {
        float zero = 0.0F;
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(accumulatorF_), zero, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE2>(vToMte2_);

        float globalMax = -65504.0F;
        float globalSum = 0.0F;
        for (uint32_t partition = 0; partition < kPartitions; ++partition) {
            WaitFlag<HardEvent::V_MTE2>(vToMte2_);
            const uint64_t stateOffset =
                (static_cast<uint64_t>(item) * kPartitions + partition) *
                kStateStride;
            DataCopyParams load{1, kStateBlocks, 0, 0};
            DataCopy(stateF_, states_[stateOffset], load);
            SetFlag<HardEvent::MTE2_V>(mte2ToV_);
            WaitFlag<HardEvent::MTE2_V>(mte2ToV_);
            set_vector_mask(0, 3);
            vadds(F(expInputF_), F(stateF_), zero, 1, 1, 1, 8, 8);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_S>(vToS_);
            WaitFlag<HardEvent::V_S>(vToS_);
            const float partitionMax = F(expInputF_)[0];
            const float partitionSum = F(expInputF_)[1];
            if (partitionSum <= 0.0F) {
                SetFlag<HardEvent::V_MTE2>(vToMte2_);
                continue;
            }

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
            vector_dup(F(correctionF_), correction, 2, 1, 1, 8, 1);
            vector_dup(F(partitionScaleF_), partitionScale,
                       2, 1, 1, 8, 1);
            pipe_barrier(PIPE_V);
            vmul(F(correctedF_), F(accumulatorF_), F(correctionF_),
                 2, 1, 1, 1, 8, 8, 8);
            vmul(F(scaledPartialF_), F(stateF_) + 2,
                 F(partitionScaleF_), 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            vadd(F(accumulatorF_), F(correctedF_), F(scaledPartialF_),
                 2, 1, 1, 1, 8, 8, 8);
            pipe_barrier(PIPE_V);
            SetFlag<HardEvent::V_MTE2>(vToMte2_);
        }
        WaitFlag<HardEvent::V_MTE2>(vToMte2_);

        float inverseSum = 1.0F / globalSum;
        SetFlag<HardEvent::S_V>(sToV_);
        WaitFlag<HardEvent::S_V>(sToV_);
        set_vector_mask(static_cast<uint64_t>(-1),
                        static_cast<uint64_t>(-1));
        vector_dup(F(correctionF_), inverseSum, 2, 1, 1, 8, 1);
        pipe_barrier(PIPE_V);
        vmul(F(finalF_), F(accumulatorF_), F(correctionF_),
             2, 1, 1, 1, 8, 8, 8);
        pipe_barrier(PIPE_V);
        WaitFlag<HardEvent::MTE3_V>(mte3ToV_);
        vconv_f322f16(H(outputH_), F(finalF_), 2, 1, 1, 4, 8);
        pipe_barrier(PIPE_V);
        SetFlag<HardEvent::V_MTE3>(vToMte3_);
        WaitFlag<HardEvent::V_MTE3>(vToMte3_);
        DataCopyParams store{1, kHeadDim * sizeof(half) / 32, 0, 0};
        const uint64_t outputOffset =
            static_cast<uint64_t>(item / kQHeads) * kQDim +
            (item % kQHeads) * kHeadDim;
        DataCopy(output_[outputOffset], outputH_, store);
        SetFlag<HardEvent::MTE3_V>(mte3ToV_);
    }

    template <typename T>
    __aicore__ inline void Bind(LocalTensor<T> &tensor, uint32_t offset)
    {
        tensor.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        tensor.address_.bufferAddr = offset;
    }
    __aicore__ inline __ubuf__ float *F(LocalTensor<float> &tensor)
    { return reinterpret_cast<__ubuf__ float *>(tensor.GetPhyAddr()); }
    __aicore__ inline __ubuf__ half *H(LocalTensor<half> &tensor)
    { return reinterpret_cast<__ubuf__ half *>(tensor.GetPhyAddr()); }

    TPipe pipe_;
    GlobalTensor<float> states_;
    GlobalTensor<half> output_;
    LocalTensor<float> stateF_, accumulatorF_, correctedF_;
    LocalTensor<float> scaledPartialF_, correctionF_, partitionScaleF_;
    LocalTensor<float> expInputF_, expOutputF_, finalF_;
    LocalTensor<half> outputH_;
    uint32_t batch_ = 0;
    event_t mte2ToV_, vToMte2_, vToS_, sToV_, vToMte3_, mte3ToV_;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_attention_partition_merge_probe(
    GM_ADDR states, GM_ADDR output, uint32_t batch)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_attention_partition_merge_probe requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIV_ONLY);
    if (batch == 0 || batch > kMaxBatch) {
        return;
    }
    AsrAttentionPartitionMergeProbe probe;
    probe.Init(states, output, batch);
    probe.Process();
}
