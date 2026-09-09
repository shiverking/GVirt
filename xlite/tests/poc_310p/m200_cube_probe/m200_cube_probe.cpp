/*
 * Ascend310P3 Cube capability probe for the Qwen3-ASR-only Xlite path.
 * Based on Ascend's MatmulInvocationNeo programming model, with the output
 * kept in FP16 to match the decoder tensors.
 */
#include "kernel_operator.h"
#define ASCENDC_CUBE_ONLY
#include "lib/matmul_intf.h"

using namespace matmul;

namespace {

__aicore__ inline uint32_t CeilDiv(uint32_t a, uint32_t b)
{
    return (a + b - 1) / b;
}

__aicore__ inline void CopyTiling(TCubeTiling *tiling, uint64_t &localMemSize,
                                  GM_ADDR tilingGm)
{
    uint32_t *dst = reinterpret_cast<uint32_t *>(tiling);
    auto src = reinterpret_cast<__gm__ uint32_t *>(tilingGm);
    for (uint32_t i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); ++i) {
        dst[i] = src[i];
    }
    localMemSize = *reinterpret_cast<__gm__ uint64_t *>(tilingGm + sizeof(TCubeTiling));
}

__aicore__ inline void CalcOffsets(uint32_t blockIdx, const TCubeTiling &tiling,
                                   uint32_t &offsetA, uint32_t &offsetB,
                                   uint32_t &offsetC, uint32_t &tailM,
                                   uint32_t &tailN)
{
    const uint32_t mBlocks = CeilDiv(tiling.M, tiling.singleCoreM);
    const uint32_t mIndex = blockIdx % mBlocks;
    const uint32_t nIndex = blockIdx / mBlocks;
    offsetA = mIndex * tiling.Ka * tiling.singleCoreM;
    // B is physically [N,K] and logically transposed by Matmul.
    offsetB = nIndex * tiling.singleCoreN * tiling.Kb;
    offsetC = mIndex * tiling.N * tiling.singleCoreM + nIndex * tiling.singleCoreN;
    tailM = tiling.M - mIndex * tiling.singleCoreM;
    tailM = tailM < tiling.singleCoreM ? tailM : tiling.singleCoreM;
    tailN = tiling.N - nIndex * tiling.singleCoreN;
    tailN = tailN < tiling.singleCoreN ? tailN : tiling.singleCoreN;
}

}  // namespace

extern "C" __global__ __aicore__ void xlite_m200_cube_probe(
    GM_ADDR a, GM_ADDR b, GM_ADDR c, GM_ADDR workspace, GM_ADDR tilingGm)
{
#if defined(__NPU_ARCH__) && (__NPU_ARCH__ != 2002)
#error "xlite_m200_cube_probe must be compiled for the real M200 architecture"
#endif
    KERNEL_TASK_TYPE_DEFAULT(KERNEL_TYPE_AIC_ONLY);
    using AType = half;
    using BType = half;
    using CType = half;

    AscendC::TPipe pipe;
    TCubeTiling tiling;
    uint64_t localMemSize = 0;
    CopyTiling(&tiling, localMemSize, tilingGm);
    const uint32_t totalBlocks = CeilDiv(tiling.M, tiling.singleCoreM) *
                                 CeilDiv(tiling.N, tiling.singleCoreN);
    if (GetBlockIdx() >= tiling.usedCoreNum || GetBlockIdx() >= totalBlocks) {
        return;
    }

    AscendC::GlobalTensor<AType> aGlobal;
    AscendC::GlobalTensor<BType> bGlobal;
    AscendC::GlobalTensor<CType> cGlobal;
    aGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ AType *>(a), tiling.M * tiling.Ka);
    bGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ BType *>(b), tiling.Ka * tiling.N);
    cGlobal.SetGlobalBuffer(reinterpret_cast<__gm__ CType *>(c), tiling.M * tiling.N);

    uint32_t offsetA = 0;
    uint32_t offsetB = 0;
    uint32_t offsetC = 0;
    uint32_t tailM = 0;
    uint32_t tailN = 0;
    CalcOffsets(GetBlockIdx(), tiling, offsetA, offsetB, offsetC, tailM, tailN);

    Matmul<MatmulType<AscendC::TPosition::GM, CubeFormat::ND, AType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, BType>,
           MatmulType<AscendC::TPosition::GM, CubeFormat::ND, CType>> mm;
    REGIST_MATMUL_OBJ(&pipe, GetSysWorkSpacePtr(), mm, &tiling);

    AscendC::TBuf<> localWorkspace;
    pipe.InitBuffer(localWorkspace, localMemSize);
    auto workspaceTensor = localWorkspace.Get<uint8_t>(localMemSize);
    mm.SetLocalWorkspace(workspaceTensor);

    mm.SetOrgShape(tiling.M, tiling.N, tiling.Ka, tiling.Kb);
    mm.SetTensorA(aGlobal[offsetA], false);
    mm.SetTensorB(bGlobal[offsetB], true);
    mm.SetTail(tailM, tailN);
    mm.IterateAll(cGlobal[offsetC]);
    mm.End();
}
