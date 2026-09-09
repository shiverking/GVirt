#include "m200_cube_probe_tiling.h"

#include <cstring>
#include <stdexcept>
#include "kernel_tiling/kernel_tiling.h"
#include "tiling/platform/platform_ascendc.h"
#include "tiling/tiling_api.h"

using namespace matmul_tiling;

M200CubeProbeTiling GenerateM200CubeProbeTiling(uint32_t m, uint32_t n, uint32_t k)
{
    if ((m != 1 && m != 20) || n != 256 || k != 2048) {
        throw std::invalid_argument("M200 Cube probe only supports M=1/20, N=256, K=2048");
    }

    auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
    MultiCoreMatmulTiling tilingApi(*platform);
    tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetBType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
    tilingApi.SetCType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16);
    tilingApi.SetOrgShape(m, n, k);
    tilingApi.SetShape(m, n, k);
    tilingApi.SetSingleShape(32, 256, -1);
    tilingApi.SetDim(1);
    tilingApi.SetBias(false);
    tilingApi.SetBufferSpace(-1, -1, -1);

    optiling::TCubeTiling tiling;
    if (tilingApi.GetTiling(tiling) == -1) {
        throw std::runtime_error("Ascend310P Matmul tiling failed");
    }

    M200CubeProbeTiling result;
    result.usedCores = static_cast<uint32_t>(tiling.get_usedCoreNum());
    if (result.usedCores != 1) {
        throw std::runtime_error("M200 Cube probe expected exactly one used core");
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
