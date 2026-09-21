/*
 * Qwen3-ASR FP16 ND x FRACTAL_NZ -> ND MatMul for Ascend310P3.
 *
 * The format-29 weight physical order is
 *   [N/16][K/16][16][16]
 * and is copied to L1 without an ND2NZ conversion.  Activations remain ND
 * and are staged once per launch.  Supported M is deliberately limited to
 * 1..20 until the independent prefill tiling is validated.
 *
 * Max resources per AIC (Down): UB=24576, L1=425984, L0A=8192,
 * L0B=32768, L0C=16384 bytes.  All cross-pipe events come from TPipe.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBlock = 16;
constexpr uint32_t kTileM = 32;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 128;
constexpr uint32_t kFractalElements = 256;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline void NdActivationToL1(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint32_t rows, uint32_t sourceStride, uint32_t paddedRows)
{
    Nd2NzParams params(1, rows, kTileK, 0, sourceStride,
                       paddedRows, 1, 0);
    DataCopy(dst, src, params);
}

__aicore__ inline void NzWeightTileToL1(
    const LocalTensor<half> &dst, const GlobalTensor<half> &src,
    uint32_t nBlockStart, uint32_t kBlockStart,
    uint32_t nBlocks, uint32_t totalKBlocks)
{
    constexpr uint32_t tileKBlocks = kTileK / kBlock;
    const uint32_t sourceOffset =
        (nBlockStart * totalKBlocks + kBlockStart) * kFractalElements;
    DataCopyParams copy;
    copy.blockCount = nBlocks;
    copy.blockLen = tileKBlocks * kFractalElements * sizeof(half) / 32;
    copy.srcStride =
        (totalKBlocks - tileKBlocks) * kFractalElements * sizeof(half) / 32;
    copy.dstStride = 0;
    DataCopy(dst, src[sourceOffset], copy);
}

__aicore__ inline void L1ToL0A(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src,
                                uint32_t mBlocks)
{
    constexpr uint32_t kBlocks = kTileK / kBlock;
    LoadData2dParams params(0, mBlocks, 1, 0, kBlocks - 1, 0, inc);
    for (uint32_t kb = 0; kb < kBlocks; ++kb) {
        LoadData(dst[kb * kFractalElements],
                 src[kb * mBlocks * kFractalElements], params);
    }
}

__aicore__ inline void L1ToL0B(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src)
{
    constexpr uint32_t blocks =
        (kTileN / kBlock) * (kTileK / kBlock);
    LoadData2dParams params(0, blocks, 1, 0, 0, 0, inc);
    LoadData(dst, src, params);
}

class AsrNzMatmulKernel {
public:
    __aicore__ inline void Init(GM_ADDR input, GM_ADDR weight,
                                GM_ADDR output, uint32_t m,
                                uint32_t n, uint32_t k)
    {
        input_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(input), m * k);
        weight_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(weight), n * k);
        output_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(output), m * n);
        m_ = m;
        n_ = n;
        k_ = k;
        mPadded_ = AlignUp(m, kBlock);

        l1A_.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        l1A_.address_.bufferAddr = 0;
        l1B_.address_.logicPos = static_cast<uint8_t>(TPosition::B1);
        l1B_.address_.bufferAddr = kTileM * k * sizeof(half);
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
        const uint32_t kTiles = k_ / kTileK;
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            NdActivationToL1(l1A_[kt * kTileM * kTileK],
                             input_[kt * kTileK], m_, k_, mPadded_);
        }
        SetFlag<HardEvent::MTE2_MTE1>(l1Ready_);
        WaitFlag<HardEvent::MTE2_MTE1>(l1Ready_);

        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        const uint32_t nTiles = n_ / kTileN;
        for (uint32_t nt = core; nt < nTiles; nt += cores) {
            ProcessNTile(nt, kTiles);
        }
    }

private:
    __aicore__ inline void ProcessNTile(uint32_t nTile, uint32_t kTiles)
    {
        constexpr uint32_t nBlocks = kTileN / kBlock;
        constexpr uint32_t kBlocks = kTileK / kBlock;
        const uint32_t nOffset = nTile * kTileN;
        const uint32_t mBlocks = mPadded_ / kBlock;
        const uint32_t totalKBlocks = k_ / kBlock;

        SetFlag<HardEvent::MTE1_MTE2>(l1Free_);
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);
            NzWeightTileToL1(l1B_, weight_, nOffset / kBlock,
                             kt * kBlocks, nBlocks, totalKBlocks);
            SetFlag<HardEvent::MTE2_MTE1>(l1Ready_);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Ready_);
            L1ToL0A(l0A_, l1A_[kt * kTileM * kTileK], mBlocks);
            L1ToL0B(l0B_, l1B_);
            SetFlag<HardEvent::MTE1_MTE2>(l1Free_);
            SetFlag<HardEvent::MTE1_M>(cubeReady_);
            WaitFlag<HardEvent::MTE1_M>(cubeReady_);

            MmadParams params;
            params.m = mPadded_;
            params.n = kTileN;
            params.k = kTileK;
            params.cmatrixSource = false;
            params.cmatrixInitVal = (kt == 0);
            Mmad(l0C_, l0A_, l0B_, params);
            SetFlag<HardEvent::M_MTE1>(cubeDone_);
            WaitFlag<HardEvent::M_MTE1>(cubeDone_);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);

        DataCopyParams drain;
        drain.blockCount = 1;
        drain.blockLen = mPadded_ * kTileN * sizeof(float) / 1024;
        drain.srcStride = 0;
        drain.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        enhanced.deqScale = DeqScale::DEQ_NONE;
        SetFlag<HardEvent::M_V>(vectorReady_);
        WaitFlag<HardEvent::M_V>(vectorReady_);
        DataCopy(outFp32_, l0C_, drain, enhanced);
        PipeBarrier<PIPE_V>();
        Cast(outFp16_, outFp32_, RoundMode::CAST_NONE, mPadded_ * kTileN);
        SetFlag<HardEvent::V_MTE3>(storeReady_);
        WaitFlag<HardEvent::V_MTE3>(storeReady_);

        DataCopyParams write;
        write.blockCount = m_;
        write.blockLen = kBlock * sizeof(half) / 32;
        write.srcStride = 0;
        write.dstStride = n_ * sizeof(half) / 32 - write.blockLen;
        for (uint32_t nb = 0; nb < nBlocks; ++nb) {
            DataCopy(output_[nOffset + nb * kBlock],
                     outFp16_[nb * mPadded_ * kBlock], write);
        }
        SetFlag<HardEvent::MTE3_V>(storeDone_);
        WaitFlag<HardEvent::MTE3_V>(storeDone_);
        SetFlag<HardEvent::V_M>(l0Reusable_);
        WaitFlag<HardEvent::V_M>(l0Reusable_);
    }

    TPipe pipe_;
    GlobalTensor<half> input_, weight_, output_;
    LocalTensor<half> l1A_, l1B_, l0A_, l0B_, outFp16_;
    LocalTensor<float> l0C_, outFp32_;
    event_t l1Free_, l1Ready_, cubeReady_, cubeDone_;
    event_t vectorReady_, l0Reusable_, storeReady_, storeDone_;
    uint32_t m_ = 0, n_ = 0, k_ = 0, mPadded_ = 0;
};

__aicore__ inline bool IsSupported(uint32_t m, uint32_t n, uint32_t k)
{
    const bool projection =
        (k == 2048 && (n == 2048 || n == 4096 || n == 12288)) ||
        (k == 6144 && n == 2048);
    const bool lmHead = k == 2048 && n == 151936;
    return m >= 1 && m <= 20 && (projection || lmHead);
}
}  // namespace

extern "C" __global__ __aicore__ void asr_m200_matmul_nz_fp16(
    GM_ADDR input, GM_ADDR weight, GM_ADDR output,
    uint32_t m, uint32_t n, uint32_t k)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_m200_matmul_nz_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (!IsSupported(m, n, k)) {
        return;
    }
    AsrNzMatmulKernel kernel;
    kernel.Init(input, weight, output, m, n, k);
    kernel.Process();
}
