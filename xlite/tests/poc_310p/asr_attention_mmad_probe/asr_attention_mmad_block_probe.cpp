/*
 * Ascend310P3 Qwen3-ASR attention MMAD block probe.
 *
 * QK contract: [16,128] x transpose([16,128]) -> FP32 [16,16].
 * PV contract: [16,16]  x transpose([128,16]) -> FP32 [16,128].
 *
 * The host supplies B in the model-native [N,K] form, exactly as the verified
 * projection kernel does.  This probe validates the low-level cube block only;
 * paged BSHD staging remains a separate block and is not hidden in this file.
 *
 * Resources: L1=36 KiB, L0A=4 KiB, L0B=32 KiB, L0C=8 KiB, UB=12 KiB.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBlock = 16;
constexpr uint32_t kTileM = 16;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 128;
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
                                const LocalTensor<half> &src,
                                uint32_t mBlocks, uint32_t kBlocks)
{
    LoadData2dParams params(0, mBlocks, 1, 0, kBlocks - 1, 0, inc);
    for (uint32_t kb = 0; kb < kBlocks; ++kb) {
        LoadData(dst[kb * kCubeElements], src[kb * mBlocks * kCubeElements], params);
    }
}

__aicore__ inline void L1ToL0B(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src,
                                uint32_t nBlocks, uint32_t kBlocks)
{
    LoadData2dParams params(0, kBlocks * nBlocks, 1, 0, 0, 0, inc);
    LoadData(dst, src, params);
}

class AttentionMmadBlockProbe {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                uint32_t m, uint32_t n, uint32_t k)
    {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), m * k);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), n * k);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), m * n);
        m_ = m; n_ = n; k_ = k;

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
    }

    __aicore__ inline void Process()
    {
        const uint32_t mPadded = kTileM;
        const uint32_t nPadded = n_;
        const uint32_t kPadded = k_;
        const uint32_t mBlocks = mPadded / kBlock;
        const uint32_t nBlocks = nPadded / kBlock;
        const uint32_t kBlocks = kPadded / kBlock;

        const event_t l1Free = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
        const event_t l1Ready = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        const event_t cubeReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        const event_t cubeDone = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        const event_t vectorReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::M_V));
        const event_t storeReady = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        const event_t storeDone = static_cast<event_t>(
            GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));

        SetFlag<HardEvent::MTE1_MTE2>(l1Free);
        WaitFlag<HardEvent::MTE1_MTE2>(l1Free);
        GmToL1Nz(l1A_, aGm_, m_, k_, k_, mPadded);
        GmToL1Nz(l1B_, bGm_, n_, k_, k_, nPadded);
        SetFlag<HardEvent::MTE2_MTE1>(l1Ready);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Ready);
        L1ToL0A(l0A_, l1A_, mBlocks, kBlocks);
        L1ToL0B(l0B_, l1B_, nBlocks, kBlocks);
        SetFlag<HardEvent::MTE1_MTE2>(l1Free);
        SetFlag<HardEvent::MTE1_M>(cubeReady);
        WaitFlag<HardEvent::MTE1_M>(cubeReady);

        MmadParams params;
        params.m = mPadded;
        params.n = nPadded;
        params.k = kPadded;
        params.cmatrixSource = false;
        params.cmatrixInitVal = true;
        Mmad(l0C_, l0A_, l0B_, params);
        SetFlag<HardEvent::M_MTE1>(cubeDone);
        WaitFlag<HardEvent::M_MTE1>(cubeDone);
        WaitFlag<HardEvent::MTE1_MTE2>(l1Free);

        DataCopyParams drain;
        drain.blockCount = 1;
        drain.blockLen = mPadded * nPadded * sizeof(float) / 1024;
        drain.srcStride = 0;
        drain.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        enhanced.deqScale = DeqScale::DEQ_NONE;
        SetFlag<HardEvent::M_V>(vectorReady);
        WaitFlag<HardEvent::M_V>(vectorReady);
        DataCopy(outFp32_, l0C_, drain, enhanced);
        PipeBarrier<PIPE_V>();
        Cast(outFp16_, outFp32_, RoundMode::CAST_NONE, mPadded * nPadded);

        SetFlag<HardEvent::V_MTE3>(storeReady);
        WaitFlag<HardEvent::V_MTE3>(storeReady);
        DataCopyParams write;
        write.blockCount = m_;
        write.blockLen = kBlock * sizeof(half) / 32;
        write.srcStride = 0;
        write.dstStride = n_ * sizeof(half) / 32 - write.blockLen;
        for (uint32_t nb = 0; nb < nBlocks; ++nb) {
            DataCopy(cGm_[nb * kBlock], outFp16_[nb * mPadded * kBlock], write);
        }
        SetFlag<HardEvent::MTE3_V>(storeDone);
        WaitFlag<HardEvent::MTE3_V>(storeDone);
        pipe_barrier(PIPE_ALL);
    }

private:
    TPipe pipe_;
    GlobalTensor<half> aGm_, bGm_, cGm_;
    LocalTensor<half> l1A_, l1B_, l0A_, l0B_, outFp16_;
    LocalTensor<float> l0C_, outFp32_;
    uint32_t m_ = 0, n_ = 0, k_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_attention_mmad_block_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_attention_mmad_block_probe requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (m != 16 || (n != 16 && n != 128) || (k != 16 && k != 128)) return;
    AttentionMmadBlockProbe probe;
    probe.Init(a, b, c, m, n, k);
    probe.Process();
}
