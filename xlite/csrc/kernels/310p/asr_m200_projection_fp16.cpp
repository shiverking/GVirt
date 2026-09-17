/*
 * Qwen3-ASR-only low-level FP16 projection probe for Ascend310P3.
 *
 * This source intentionally uses explicit GM/L1/L0 transfers and Mmad.  It
 * must remain independent of the high-level Matmul library so its resource and event
 * schedule can be audited before promotion into the runtime backend.
 */
#include "kernel_operator.h"

using namespace AscendC;

namespace {

constexpr uint32_t kMBlock = 16;
constexpr uint32_t kNBlock = 16;
constexpr uint32_t kKBlock = 16;
constexpr uint32_t kTileM = 32;
constexpr uint32_t kTileN = 128;
constexpr uint32_t kTileK = 128;
constexpr uint32_t kCubeElements = 256;

__aicore__ inline uint32_t AsrAlignUp(uint32_t value, uint32_t alignment)
{
    return ((value + alignment - 1) / alignment) * alignment;
}

__aicore__ inline void AsrGmToL1Nz(const LocalTensor<half> &dst,
                                    const GlobalTensor<half> &src,
                                    uint32_t rows, uint32_t cols,
                                    uint32_t srcStride, uint32_t dstRows)
{
    Nd2NzParams params(1, rows, cols, 0, srcStride, dstRows, 1, 0);
    DataCopy(dst, src, params);
}

__aicore__ inline void AsrL1ToL0A(const LocalTensor<half> &dst,
                                   const LocalTensor<half> &src,
                                   uint32_t mBlocks, uint32_t kBlocks)
{
    LoadData2dParams params(0, mBlocks, 1, 0, kBlocks - 1, 0, inc);
    for (uint32_t kb = 0; kb < kBlocks; ++kb) {
        LoadData(dst[kb * kCubeElements], src[kb * mBlocks * kCubeElements], params);
    }
}

__aicore__ inline void AsrL1ToL0B(const LocalTensor<half> &dst,
                                   const LocalTensor<half> &src,
                                   uint32_t nBlocks, uint32_t kBlocks)
{
    // GM weights use the model's native [N,K] layout.  ND2NZ above has
    // already arranged the tile for the cube's ZN B operand, so this is a
    // contiguous L1-to-L0B load.  Applying the transposed-B loader here
    // interleaves N rows into K and collapses every output column to their
    // average.
    LoadData2dParams params(0, kBlocks * nBlocks, 1, 0, 0, 0, inc);
    LoadData(dst, src, params);
}

class AsrProjectionKernel {
public:
    __aicore__ inline void Init(GM_ADDR a, GM_ADDR b, GM_ADDR c,
                                uint32_t m, uint32_t n, uint32_t k)
    {
        aGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), m * k);
        bGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), n * k);
        cGm_.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), m * n);
        m_ = m;
        n_ = n;
        k_ = k;

        uint64_t l1Offset = 0;
        l1A_.address_.logicPos = static_cast<uint8_t>(TPosition::A1);
        l1A_.address_.bufferAddr = l1Offset;
        l1Offset += kTileM * kTileK * sizeof(half);
        l1B_.address_.logicPos = static_cast<uint8_t>(TPosition::B1);
        l1B_.address_.bufferAddr = l1Offset;

        l0A_.address_.logicPos = static_cast<uint8_t>(TPosition::A2);
        l0A_.address_.bufferAddr = 0;
        l0B_.address_.logicPos = static_cast<uint8_t>(TPosition::B2);
        l0B_.address_.bufferAddr = 0;
        l0C_.address_.logicPos = static_cast<uint8_t>(TPosition::CO1);
        l0C_.address_.bufferAddr = 0;
        outFp32Ub_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        outFp32Ub_.address_.bufferAddr = 0;
        outFp16Ub_.address_.logicPos = static_cast<uint8_t>(TPosition::VECCALC);
        outFp16Ub_.address_.bufferAddr = kTileM * kTileN * sizeof(float);
    }

    __aicore__ inline void Process()
    {
        const uint32_t core = GetBlockIdx();
        const uint32_t cores = GetBlockNum();
        const uint32_t nTiles = (n_ + kTileN - 1) / kTileN;
        for (uint32_t nTile = core; nTile < nTiles; nTile += cores) {
            ProcessNTile(nTile);
        }
    }

private:
    __aicore__ inline void ProcessNTile(uint32_t nTile)
    {
        const uint32_t nOffset = nTile * kTileN;
        const uint32_t nActual = n_ - nOffset < kTileN ? n_ - nOffset : kTileN;
        const uint32_t nPadded = AsrAlignUp(nActual, kNBlock);
        const uint32_t mPadded = AsrAlignUp(m_, kMBlock);
        const uint32_t mBlocks = mPadded / kMBlock;
        const uint32_t nBlocks = nPadded / kNBlock;
        const uint32_t kTiles = (k_ + kTileK - 1) / kTileK;

        // L1 is single-buffered.  Prime its ownership and return it from
        // MTE1 to MTE2 after every L1-to-L0 transfer before the next K
        // slice is allowed to overwrite it.
        SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
        for (uint32_t kTile = 0; kTile < kTiles; ++kTile) {
            const uint32_t kOffset = kTile * kTileK;
            const uint32_t kActual = k_ - kOffset < kTileK ? k_ - kOffset : kTileK;
            const uint32_t kPadded = AsrAlignUp(kActual, kKBlock);
            const uint32_t kBlocks = kPadded / kKBlock;

            WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            AsrGmToL1Nz(l1A_, aGm_[kOffset], m_, kActual, k_, mPadded);
            AsrGmToL1Nz(l1B_, bGm_[nOffset * k_ + kOffset], nActual,
                        kActual, k_, nPadded);
            SetFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::MTE2_MTE1>(EVENT_ID0);

            AsrL1ToL0A(l0A_, l1A_, mBlocks, kBlocks);
            AsrL1ToL0B(l0B_, l1B_, nBlocks, kBlocks);
            SetFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);
            SetFlag<HardEvent::MTE1_M>(EVENT_ID0);
            WaitFlag<HardEvent::MTE1_M>(EVENT_ID0);

            MmadParams params;
            params.m = mPadded;
            params.n = nPadded;
            params.k = kPadded;
            params.cmatrixSource = false;
            params.cmatrixInitVal = (kTile == 0);
            Mmad(l0C_, l0A_, l0B_, params);
            SetFlag<HardEvent::M_MTE1>(EVENT_ID0);
            WaitFlag<HardEvent::M_MTE1>(EVENT_ID0);
        }
        WaitFlag<HardEvent::MTE1_MTE2>(EVENT_ID0);

        // Ascend310P3 has no usable FixPipe path from L0C directly to GM.
        // Design reference: l0c-ub-nd-readout.md, pinned by
        // doc/310p_asr_ascendc/shared_assets.json.
        // On Ascend310P, enhanced DataCopy is the only supported L0C-to-UB
        // readback path.  One MATRIX-mode burst covers the complete padded
        // FP32 tile; per-fractal/basic copies silently return corrupt data.
        DataCopyParams drain;
        drain.blockCount = 1;
        drain.blockLen = mPadded * nPadded * sizeof(float) / 1024;
        drain.srcStride = 0;
        drain.dstStride = 0;
        DataCopyEnhancedParams enhanced;
        enhanced.blockMode = BlockMode::BLOCK_MODE_MATRIX;
        enhanced.deqScale = DeqScale::DEQ_NONE;
        SetFlag<HardEvent::M_V>(EVENT_ID0);
        WaitFlag<HardEvent::M_V>(EVENT_ID0);
        DataCopy(outFp32Ub_, l0C_, drain, enhanced);
        PipeBarrier<PIPE_V>();
        Cast(outFp16Ub_, outFp32Ub_, RoundMode::CAST_NONE,
             mPadded * nPadded);
        SetFlag<HardEvent::V_MTE3>(EVENT_ID0);
        WaitFlag<HardEvent::V_MTE3>(EVENT_ID0);

        DataCopyParams write;
        write.blockCount = m_;
        write.blockLen = nActual * sizeof(half) / 32;
        write.srcStride = (nPadded - nActual) * sizeof(half) / 32;
        write.dstStride = (n_ - nActual) * sizeof(half) / 32;
        DataCopy(cGm_[nOffset], outFp16Ub_, write);
        SetFlag<HardEvent::MTE3_V>(EVENT_ID0);
        WaitFlag<HardEvent::MTE3_V>(EVENT_ID0);
        SetFlag<HardEvent::V_M>(EVENT_ID0);
        WaitFlag<HardEvent::V_M>(EVENT_ID0);
    }

    GlobalTensor<half> aGm_;
    GlobalTensor<half> bGm_;
    GlobalTensor<half> cGm_;
    LocalTensor<half> l1A_;
    LocalTensor<half> l1B_;
    LocalTensor<half> l0A_;
    LocalTensor<half> l0B_;
    LocalTensor<float> outFp32Ub_;
    LocalTensor<half> outFp16Ub_;
    LocalTensor<float> l0C_;
    uint32_t m_ = 0;
    uint32_t n_ = 0;
    uint32_t k_ = 0;
};

}  // namespace

extern "C" __global__ __aicore__ void asr_m200_projection_fp16(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, uint32_t m, uint32_t n, uint32_t k)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "asr_m200_projection_fp16 requires the real Ascend310P3 M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    if (m == 0 || m > 20 || n == 0 || k == 0 || (n % 16) != 0 ||
        (k % 128) != 0) {
        return;
    }
    AsrProjectionKernel kernel;
    kernel.Init(a, b, c, m, n, k);
    kernel.Process();
}
