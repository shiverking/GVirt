/* Isolated CANN host-tiling bridge. This target intentionally uses CANN's
 * libstdc++ ABI and exposes only a C interface to the Torch-ABI Xlite library. */
#include "m200_tiling_310p_c.h"

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <exception>
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

void SetError(char *error, size_t capacity, const char *message)
{
    if (error != nullptr && capacity != 0) {
        std::snprintf(error, capacity, "%s", message);
    }
}

}  // namespace

extern "C" int XliteM200BuildTiling310P(uint32_t m, uint32_t n, uint32_t k,
                                         void *buffer, size_t capacity, size_t *written,
                                         uint32_t *usedCores,
                                         uint64_t *systemWorkspaceBytes,
                                         char *error, size_t errorCapacity)
{
    try {
        if (buffer == nullptr || written == nullptr || usedCores == nullptr ||
            systemWorkspaceBytes == nullptr) {
            throw std::invalid_argument("null M200 tiling output argument");
        }
        auto platform = platform_ascendc::PlatformAscendCManager::GetInstance(SOC_VERSION);
        const uint32_t availableCores = platform->GetCoreNumAic();
        if (availableCores == 0) {
            throw std::runtime_error("Ascend310P reported zero Cube cores");
        }
        const uint32_t targetCores = std::min(availableCores, CeilDiv(n, 256U));
        const uint32_t singleCoreN = AlignUp(CeilDiv(n, targetCores), 256U);
        const uint32_t requestedCores = CeilDiv(n, singleCoreN);

        MultiCoreMatmulTiling tilingApi(*platform);
        tilingApi.SetAType(TPosition::GM, CubeFormat::ND, DataType::DT_FLOAT16, false);
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
            throw std::runtime_error("Ascend310P M200 MatMul tiling failed");
        }
        *usedCores = static_cast<uint32_t>(tiling.get_usedCoreNum());
        if (*usedCores == 0 || *usedCores > availableCores) {
            throw std::runtime_error("Ascend310P M200 MatMul generated an invalid core count");
        }
        uint64_t ubBytes = 0;
        platform->GetCoreMemSize(platform_ascendc::CoreMemType::UB, ubBytes);
        *systemWorkspaceBytes = static_cast<uint64_t>(platform->GetLibApiWorkSpaceSize());
        const uint32_t tilingBytes = tiling.GetDataSize();
        const size_t required = sizeof(optiling::TCubeTiling) + sizeof(uint64_t);
        if (tilingBytes > sizeof(optiling::TCubeTiling) || capacity < required) {
            throw std::runtime_error("Ascend310P TCubeTiling buffer is too small");
        }
        std::memset(buffer, 0, required);
        tiling.SaveToBuffer(buffer, tilingBytes);
        std::memcpy(static_cast<uint8_t *>(buffer) + sizeof(optiling::TCubeTiling),
                    &ubBytes, sizeof(ubBytes));
        *written = required;
        return 0;
    } catch (const std::exception &exception) {
        SetError(error, errorCapacity, exception.what());
        return -1;
    } catch (...) {
        SetError(error, errorCapacity, "unknown M200 tiling failure");
        return -1;
    }
}
