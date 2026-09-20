/*
 * Qwen3-ASR-only cached-input FP16 projections for Ascend310P3.
 *
 * Fixed supported shapes are QKV/O/Gate-Up [M,2048] and Down [M,6144],
 * M=1..20. Each AIC stages every A K tile once in L1 and reuses it for all
 * assigned N tiles. B remains streamed one 128x128 tile at a time.
 *
 * Maximum resources (Down): UB=24576, L1=425984, L0A=8192,
 * L0B=32768, L0C=16384 bytes. Eight event channels use FetchEventID.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {
constexpr uint32_t kBlock = 16;
constexpr uint32_t kTileM = 32;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 128;
constexpr uint32_t kCubeElements = 256;

__aicore__ inline uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

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
                                uint32_t mBlocks)
{
    constexpr uint32_t kBlocks = kTileK / kBlock;
    LoadData2dParams params(0, mBlocks, 1, 0, kBlocks - 1, 0, inc);
    for (uint32_t kb = 0; kb < kBlocks; ++kb) {
        LoadData(dst[kb * kCubeElements],
                 src[kb * mBlocks * kCubeElements], params);
    }
}

__aicore__ inline void L1ToL0B(const LocalTensor<half> &dst,
                                const LocalTensor<half> &src,
                                uint32_t nBlocks)
{
    constexpr uint32_t kBlocks = kTileK / kBlock;
    LoadData2dParams params(0, kBlocks * nBlocks, 1, 0, 0, 0, inc);
    LoadData(dst, src, params);
}

class CachedProjectionKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                uint32_t m, uint32_t n, uint32_t k)
    {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), m * k);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), n * k);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), m * n);
        m_ = m; n_ = n; k_ = k;
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

        l1Free_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_MTE2));
        l1Ready_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE2_MTE1));
        cubeReady_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE1_M));
        cubeDone_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_MTE1));
        vectorReady_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::M_V));
        l0Reusable_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_M));
        storeReady_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::V_MTE3));
        storeDone_ = static_cast<event_t>(GetTPipePtr()->FetchEventID(HardEvent::MTE3_V));
    }

    __aicore__ inline void Process()
    {
        const uint32_t kTiles = k_ / kTileK;
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            GmToL1Nz(l1A_[kt * kTileM * kTileK], aGm_[kt * kTileK],
                     m_, kTileK, k_, mPadded_);
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
        const uint32_t nOffset = nTile * kTileN;
        const uint32_t mBlocks = mPadded_ / kBlock;
        constexpr uint32_t nBlocks = kTileN / kBlock;
        SetFlag<HardEvent::MTE1_MTE2>(l1Free_);
        for (uint32_t kt = 0; kt < kTiles; ++kt) {
            const uint32_t kOffset = kt * kTileK;
            WaitFlag<HardEvent::MTE1_MTE2>(l1Free_);
            GmToL1Nz(l1B_, bGm_[nOffset * k_ + kOffset],
                     kTileN, kTileK, k_, kTileN);
            SetFlag<HardEvent::MTE2_MTE1>(l1Ready_);
            WaitFlag<HardEvent::MTE2_MTE1>(l1Ready_);
            L1ToL0A(l0A_, l1A_[kt * kTileM * kTileK], mBlocks);
            L1ToL0B(l0B_, l1B_, nBlocks);
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
            DataCopy(cGm_[nOffset + nb * kBlock],
                     outFp16_[nb * mPadded_ * kBlock], write);
        }
        SetFlag<HardEvent::MTE3_V>(storeDone_);
        WaitFlag<HardEvent::MTE3_V>(storeDone_);
        SetFlag<HardEvent::V_M>(l0Reusable_);
        WaitFlag<HardEvent::V_M>(l0Reusable_);
    }

    TPipe pipe_;
    GlobalTensor<half> aGm_, bGm_, cGm_;
    LocalTensor<half> l1A_, l1B_, l0A_, l0B_, outFp16_;
    LocalTensor<float> l0C_, outFp32_;
    event_t l1Free_, l1Ready_, cubeReady_, cubeDone_;
    event_t vectorReady_, l0Reusable_, storeReady_, storeDone_;
    uint32_t m_ = 0, n_ = 0, k_ = 0, mPadded_ = 0;
};
}  // namespace

extern "C" __global__ __aicore__ void asr_m200_projection_cached_fp16(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_m200_projection_cached_fp16 requires Ascend310P3 M200"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    const bool supported =
        (k == 2048 && (n == 2048 || n == 4096 || n == 12288)) ||
        (k == 6144 && n == 2048);
    if (m == 0 || m > 20 || !supported) return;
    CachedProjectionKernel kernel;
    kernel.Init(a, b, c, m, n, k);
    kernel.Process();
}
