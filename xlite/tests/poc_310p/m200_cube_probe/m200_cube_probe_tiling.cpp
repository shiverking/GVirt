#include "m200_cube_probe_tiling.h"

#include <algorithm>
#include <cstring>
#include <stdexcept>
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

namespace {

uint32_t CeilDiv(uint32_t value, uint32_t divisor)
{
    return (value + divisor - 1) / divisor;
}

uint32_t AlignUp(uint32_t value, uint32_t alignment)
{
    return CeilDiv(value, alignment) * alignment;
}

bool IsAsrProjection(uint32_t n, uint32_t k)
{
    return (n == 256 && k == 2048) || (n == 4480 && k == 2048) ||
           (n == 4096 && k == 2048) ||
           (n == 2048 && k == 2048) || (n == 12288 && k == 2048) ||
           (n == 2048 && k == 6144) || (n == 151936 && k == 2048);
}

}  // namespace

M200CubeProbeTiling GenerateM200CubeProbeTiling(uint32_t m, uint32_t n, uint32_t k)
{
    if ((m != 1 && m != 8 && m != 20) || !IsAsrProjection(n, k)) {
        throw std::invalid_argument(
            "M200 Cube probe only supports Qwen3-ASR projections with M=1/8/20");
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
    const uint32_t availableCores = platform->GetCoreNumAic();
    if (availableCores == 0) {
        throw std::runtime_error("Ascend310P reported zero Cube cores");
    }
    const uint32_t requestedCores = std::min(availableCores, CeilDiv(n, 256U));
    const uint32_t singleCoreN = AlignUp(CeilDiv(n, requestedCores), 256U);
    MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    // Xlite linear weights are stored as [N,K].  The logical multiplication is
    // [M,K] x [N,K].T, matching the existing ACLNN path without a weight copy.
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, true);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    tilingApi.SetOrgShape(m, n, k);
    tilingApi.SetShape(m, n, k);
    tilingApi.SetSingleShape(32, singleCoreN, -1);
    tilingApi.SetDim(requestedCores);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tiling;
    if (tilingApi.GetTiling(tiling) == -1) {
        throw std::runtime_error("Ascend310P Matmul tiling failed");
    }

    M200CubeProbeTiling result;
    result.usedCores = static_cast<uint32_t>(tiling.get_usedCoreNum());
    if (result.usedCores == 0 || result.usedCores > availableCores) {
        throw std::runtime_error("M200 Cube probe generated an invalid Cube core count");
    }
    uint64_t ubBytes = 0;
    platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
    result.localWorkspaceBytes = ubBytes;
    result.systemWorkspaceBytes = static_cast<uint64_t>(platform->GetLibApiWorkSpaceSize());
    const uint32_t tilingBytes = tiling.GetDataSize();
    if (tilingBytes > sizeof(optiling::TCubeTiling)) {
        throw std::runtime_error("TCubeTiling serialized size exceeds its ABI storage");
    }
    result.bytes.resize(sizeof(optiling::TCubeTiling) + sizeof(uint64_t), 0);
    tiling.SaveToBuffer(result.bytes.data(), tilingBytes);
    std::memcpy(result.bytes.data() + sizeof(optiling::TCubeTiling), &ubBytes, sizeof(ubBytes));
    return result;
}
