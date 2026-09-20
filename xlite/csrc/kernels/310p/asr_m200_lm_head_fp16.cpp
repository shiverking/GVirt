/*
 * Qwen3-ASR-1.7B-only FP16 LM Head for Ascend310P3.
 *
 * Fixed contract: [M,2048] x transpose([151936,2048]) -> [M,151936],
 * M in [1,20].  Eight AICs traverse the vocabulary tiles in one launch and
 * write the final contiguous logits directly.  This first implementation is
 * the independently testable correctness baseline; activation reuse and
 * pipe overlap remain behind the device-timing promotion gate.
 *
 * Resources per AIC: UB=24576, L1=40960, L0A=8192, L0B=32768,
 * L0C=16384 bytes.  All eight cross-pipe event identifiers are dynamic.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kVocabulary = 151936;
constexpr uint32_t kMaxBatch = 20;
constexpr uint32_t kTileM = 32;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 128;
constexpr uint32_t kBlock = 16;
constexpr uint32_t kCubeElements = 256;

__aicore__ inline void GmToL1Nz(const LocalTensor<half> &dst,
                                 const GlobalTensor<half> &src,
                                 uint32_t rows, uint32_t cols,
                                 uint32_t srcStride, uint32_t dstRows)
{
    Nd2NzParams params(1, rows, cols, 0, srcStride, dstRows, 1, 0);
    DataCopy(dst, src, params);
}

__aicore__ inline void L1ToL0A(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src)
{
    constexpr uint32_t mBlocks = kTileM / kBlock;
    constexpr uint32_t kBlocks = kTileK / kBlock;
    LoadData2dParams params(0, mBlocks, 1, 0, kBlocks - 1, 0, inc);
    for (uint32_t kb = 0; kb < kBlocks; ++kb) {
        LoadData(dst[kb * kCubeElements],
                 src[kb * mBlocks * kCubeElements], params);
    }
}

__aicore__ inline void L1ToL0B(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src)
{
    constexpr uint32_t nBlocks = kTileN / kBlock;
    constexpr uint32_t kBlocks = kTileK / kBlock;
    LoadData2dParams params(0, kBlocks * nBlocks, 1, 0, 0, 0, inc);
    LoadData(dst, src, params);
}

class AsrLmHeadKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight, GM_ADDR logits,
                                uint32_t batch)
    {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input),
                               batch * kHidden);
        weight_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight),
                                kVocabulary * kHidden);
        logits_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(logits),
                                batch * kVocabulary);
        batch_ = batch;

        l1A_.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        l1A_.address_.bufferAddr = 0;
        l1B_.address_.logicPos = static_cast<uint8_t>(TPosition::B1);
        l1B_.address_.bufferAddr = kTileM * kTileK * sizeof(half);
        l0A_.address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0A_.address_.bufferAddr = 0;
        l0B_.address_.logicPos = static_cast<uint8_t>(TPosition::B2);
        l0B_.address_.bufferAddr = 0;
        l0C_.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0C_.address_.bufferAddr = 0;
        outFp32_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        outFp32_.address_.bufferAddr = 0;
        outFp16_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        outFp16_.address_.bufferAddr = kTileM * kTileN * sizeof(float);

        l1Free_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
        l1Ready_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        cubeReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        cubeDone_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        vectorReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::M_V));
        l0Reusable_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_M));
        storeReady_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        storeDone_ = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        constexpr uint32_t nTiles = kVocabulary / kTileN;
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        for (uint32_t nTile = core; nTile < nTiles; nTile += cores) {
            ProcessTile(nTile);
        }
    }

private:
    __aicore__ inline void ProcessTile(uint32_t nTile)
    {
        constexpr uint32_t kTiles = kHidden / kTileK;
        const uint32_t nOffset = nTile * kTileN;

        SetFlag<HardEvent::MTE1_MTE2>(l1Free_);
        for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
            const uint32_t kOffset = kTile * kTileK;
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);
            GmToL1Nz(l1A_, input_[kOffset], batch_, kTileK,
                     kHidden, kTileM);
            GmToL1Nz(l1B_, weight_[nOffset * kHidden + kOffset],
                     kTileN, kTileK, kHidden, kTileN);
            SetFlag<HardEvent::MTE2_MTE1>(l1Ready_);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Ready_);

            L1ToL0A(l0A_, l1A_);
            L1ToL0B(l0B_, l1B_);
            SetFlag<HardEvent::MTE1_MTE2>(l1Free_);
            SetFlag<HardEvent::MTE1_M>(cubeReady_);
            WaitFlag<HardEvent::MTE1_M>(cubeReady_);

            MmadParams params;
            params.m = kTileM;
            params.n = kTileN;
            params.k = kTileK;
            params.cmatrixSource = false;
            params.cmatrixInitVal = (kTile == 0);
            Mmad(l0C_, l0A_, l0B_, params);
            SetFlag<HardEvent::M_MTE1>(cubeDone_);
            WaitFlag<HardEvent::M_MTE1>(cubeDone_);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);

        DataCopyParams drain;
        drain.blockCount = 1;
        drain.blockLen = kTileM * kTileN * sizeof(float) / 1024;
        drain.srcStride = 0;
        drain.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        enhanced.deqScale = DeqScale::DEQ_NONE;
        SetFlag<HardEvent::M_V>(vectorReady_);
        WaitFlag<HardEvent::M_V>(vectorReady_);
        DataCopy(outFp32_, l0C_, drain, enhanced);
        PipeBarrier<PIPE_V>();
        Cast(outFp16_, outFp32_, RoundMode::CAST_NONE, kTileM * kTileN);
        SetFlag<HardEvent::V_MTE3>(storeReady_);
        WaitFlag<HardEvent::V_MTE3>(storeReady_);

        DataCopyParams write;
        write.blockCount = batch_;
        write.blockLen = kBlock * sizeof(half) / 32;
        write.srcStride = 0;
        write.dstStride = kVocabulary * sizeof(half) / 32 - write.blockLen;
        constexpr uint32_t nBlocks = kTileN / kBlock;
        for (uint32_t nb = 0; nb < nBlocks; ++nb) {
            DataCopy(logits_[nOffset + nb * kBlock],
                     outFp16_[nb * kTileM * kBlock], write);
        }
        SetFlag<HardEvent::MTE3_V>(storeDone_);
        WaitFlag<HardEvent::MTE3_V>(storeDone_);
        SetFlag<HardEvent::V_M>(l0Reusable_);
        WaitFlag<HardEvent::V_M>(l0Reusable_);
    }

    GlobalTensor<half> input_, weight_, logits_;
    LocalTensor<half> l1A_, l1B_, l0A_, l0B_, outFp16_;
    LocalTensor<float> l0C_, outFp32_;
    event_t l1Free_ = 0;
    event_t l1Ready_ = 0;
    event_t cubeReady_ = 0;
    event_t cubeDone_ = 0;
    event_t vectorReady_ = 0;
    event_t l0Reusable_ = 0;
    event_t storeReady_ = 0;
    event_t storeDone_ = 0;
    uint32_t batch_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void asr_m200_lm_head_fp16(
    GM_ADDR input, GM_ADDR weight, GM_ADDR logits, uint32_t batch)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_m200_lm_head_fp16 requires the real Ascend310P3 M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (batch == 0 || batch > kMaxBatch) {
        return;
    }
    AsrLmHeadKernel kernel;
    kernel.Init(input, weight, logits, batch);
    kernel.Process();
}
