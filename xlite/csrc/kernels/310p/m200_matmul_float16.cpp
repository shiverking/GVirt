/*
 * Qwen3-ASR-only FP16 ND MatMul for Ascend310P3 (M200).
 */
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using namespace matmul;

namespace {

__aicore__ inline uint32_t XliteM200CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

__aicore__ inline void XliteM200CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize,
                                           GM_ADDR tilingGm)
{
    uint32_t *dst = reinterpret_cast<uint32_t *>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGm + sizeof(TCubeTiling));
}

}  // namespace

extern "C" __global__ __aicore__ void xlite_m200_matmul_float16(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "xlite_m200_matmul_float16 requires the real M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    XliteM200CopyTiling(&tiling, localMemSize, tilingGm);

    const uint32_t mBlocks = XliteM200CeilDiv(tiling.M, tiling.singleCoreM);
    const uint32_t nBlocks = XliteM200CeilDiv(tiling.N, tiling.singleCoreN);
    const uint32_t blockIdx = GetBlockIdx();
    const uint32_t totalBlocks = mBlocks * nBlocks;
    if (blockIdx >= tiling.usedCoreNum || blockIdx >= totalBlocks) {
        return;
    }

    AscendC::GlobalTensor<half> aGlobal;
    AscendC::GlobalTensor<half> bGlobal;
    AscendC::GlobalTensor<half> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ half *>(c), tiling.M * tiling.N);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, half>> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);
    AscendC::TBuf<> localWorkspace;
    pipe.InitBuffer(localWorkspace, localMemSize);
    auto workspaceTensor = localWorkspace.Get<uint8_t>(localMemSize);
    mm.SetLocalWorkspace(workspaceTensor);
    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    // A large prefill exposes more (M,N) tiles than physical Cube cores. Each
    // core owns a deterministic grid-stride sequence so a single host launch
    // covers the complete output without adding per-tile dispatch overhead.
    for (uint32_t task = blockIdx; task < totalBlocks; task += tiling.usedCoreNum) {
        const uint32_t mIndex = task % mBlocks;
        const uint32_t nIndex = task / mBlocks;
        const uint32_t offsetA = mIndex * tiling.Ka * tiling.singleCoreM;
        const uint32_t offsetB = nIndex * tiling.singleCoreN * tiling.Kb;
        const uint32_t offsetC = mIndex * tiling.N * tiling.singleCoreM +
                                 nIndex * tiling.singleCoreN;
        const uint32_t remainingM = tiling.M - mIndex * tiling.singleCoreM;
        const uint32_t remainingN = tiling.N - nIndex * tiling.singleCoreN;
        const uint32_t tailM =
            remainingM < tiling.singleCoreM ? remainingM : tiling.singleCoreM;
        const uint32_t tailN =
            remainingN < tiling.singleCoreN ? remainingN : tiling.singleCoreN;
        mm.SetTensorA(aGlobal[offsetA], false);
        // Xlite stores linear weights as physical [N,K].
        mm.SetTensorB(bGlobal[offsetB], true);
        mm.SetTail(tailM, tailN);
        mm.IterateAll(cGlobal[offsetC]);
    }
    mm.End();
}
