#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_xlite_m200_cube_probe.h"
#include "m200_cube_probe_tiling.h"

namespace {

constexpr uint32_t kN = 256;
constexpr uint32_t kK = 2048;
constexpr uint16_t kHalfOne = 0x3c00;
constexpr uint16_t kHalfPointFive = 0x3800;
constexpr uint16_t kHalfPointTwoFive = 0x3400;
constexpr uint16_t kHalfMinusPointFive = 0xb800;
constexpr uint16_t kHalfNan = 0x7e00;

void Check(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) + " failed, aclError=" +
                                 std::to_string(status));
    }
}

struct DeviceBuffer {
    void *ptr = nullptr;
    explicit DeviceBuffer(size_t bytes)
    {
        Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc");
    }
    ~DeviceBuffer()
    {
        if (ptr != nullptr) {
            (void)aclrtFree(ptr);
        }
    }
    DeviceBuffer(const DeviceBuffer &) = delete;
    DeviceBuffer &operator=(const DeviceBuffer &) = delete;
};

uint16_t Expected(uint32_t row, uint32_t col)
{
    static constexpr uint16_t full[] = {0x6000, 0x6400, 0x6800, 0xe400};
    static constexpr uint16_t half[] = {0x5c00, 0x6000, 0x6400, 0xe000};
    return (row & 1U) == 0 ? full[col & 3U] : half[col & 3U];
}

void Run(uint32_t m)
{
    const auto tiling = GenerateM200CubeProbeTiling(m, kN, kK);
    std::vector<uint16_t> a(static_cast<size_t>(m) * kK);
    std::vector<uint16_t> b(static_cast<size_t>(kK) * kN);
    std::vector<uint16_t> c(static_cast<size_t>(m) * kN, kHalfNan);
    for (uint32_t row = 0; row < m; ++row) {
        const uint16_t value = (row & 1U) == 0 ? kHalfOne : kHalfPointFive;
        for (uint32_t col = 0; col < kK; ++col) {
            a[static_cast<size_t>(row) * kK + col] = value;
        }
    }
    static constexpr uint16_t bPattern[] = {
        kHalfPointTwoFive, kHalfPointFive, kHalfOne, kHalfMinusPointFive};
    for (uint32_t row = 0; row < kK; ++row) {
        for (uint32_t col = 0; col < kN; ++col) {
            b[static_cast<size_t>(row) * kN + col] = bPattern[col & 3U];
        }
    }

    DeviceBuffer aDevice(a.size() * sizeof(uint16_t));
    DeviceBuffer bDevice(b.size() * sizeof(uint16_t));
    DeviceBuffer cDevice(c.size() * sizeof(uint16_t));
    DeviceBuffer workspace(tiling.systemWorkspaceBytes);
    DeviceBuffer tilingDevice(tiling.bytes.size());
    Check(aclrtMemcpy(aDevice.ptr, a.size() * sizeof(uint16_t), a.data(),
                     a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
    Check(aclrtMemcpy(bDevice.ptr, b.size() * sizeof(uint16_t), b.data(),
                     b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
    Check(aclrtMemcpy(cDevice.ptr, c.size() * sizeof(uint16_t), c.data(),
                     c.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy sentinel C");
    Check(aclrtMemcpy(tilingDevice.ptr, tiling.bytes.size(), tiling.bytes.data(),
                     tiling.bytes.size(), ACL_MEMCPY_HOST_TO_DEVICE), "copy tiling");

    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    ACLRT_LAUNCH_KERNEL(xlite_m200_cube_probe)
    (tiling.usedCores, stream, aDevice.ptr, bDevice.ptr, cDevice.ptr, workspace.ptr,
     tilingDevice.ptr);
    Check(aclrtSynchronizeStream(stream), "aclrtSynchronizeStream");
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtMemcpy(c.data(), c.size() * sizeof(uint16_t), cDevice.ptr,
                     c.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy C");

    size_t mismatches = 0;
    size_t sentinels = 0;
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < kN; ++col) {
            const auto actual = c[static_cast<size_t>(row) * kN + col];
            sentinels += actual == kHalfNan;
            mismatches += actual != Expected(row, col);
        }
    }
    if (sentinels != 0 || mismatches != 0) {
        throw std::runtime_error("incorrect output: sentinel=" + std::to_string(sentinels) +
                                 ", mismatches=" + std::to_string(mismatches));
    }
    std::cout << "M200 Cube probe PASS: M=" << m << ", N=" << kN << ", K=" << kK
              << ", cores=" << tiling.usedCores
              << ", local_workspace=" << tiling.localWorkspaceBytes
              << ", system_workspace=" << tiling.systemWorkspaceBytes << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 2) {
            throw std::invalid_argument("usage: xlite_m200_cube_probe_runner M(1|20)");
        }
        const uint32_t m = static_cast<uint32_t>(std::stoul(argv[1]));
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Run(m);
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "M200 Cube probe FAILED: " << error.what() << std::endl;
        return 1;
    }
}

