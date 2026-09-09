#include <chrono>
#include <cstdint>
#include <cstring>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_xlite_m200_cube_probe.h"
#include "m200_cube_probe_tiling.h"

namespace {

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

uint16_t HalfBits(float value)
{
    const __fp16 halfValue = static_cast<__fp16>(value);
    uint16_t bits = 0;
    static_assert(sizeof(bits) == sizeof(halfValue));
    std::memcpy(&bits, &halfValue, sizeof(bits));
    return bits;
}

float PatternValue(uint32_t index)
{
    static constexpr float values[] = {0.25F, 0.5F, 1.0F, -0.5F};
    return values[index & 3U];
}

void Launch(const M200CubeProbeTiling &tiling, aclrtStream stream, const DeviceBuffer &a,
            const DeviceBuffer &b, DeviceBuffer &c, const DeviceBuffer &workspace,
            const DeviceBuffer &tilingDevice)
{
    ACLRT_LAUNCH_KERNEL(xlite_m200_cube_probe)
    (tiling.usedCores, stream, a.ptr, b.ptr, c.ptr, workspace.ptr, tilingDevice.ptr);
}

void Run(const std::string &name, uint32_t m, uint32_t n, uint32_t k, uint32_t warmup,
         uint32_t iterations)
{
    if (iterations == 0) {
        throw std::invalid_argument("iterations must be positive");
    }
    const auto tiling = GenerateM200CubeProbeTiling(m, n, k);
    std::vector<uint16_t> a(static_cast<size_t>(m) * k);
    // Match the Xlite linear weight layout [N,K].
    std::vector<uint16_t> b(static_cast<size_t>(n) * k);
    std::vector<uint16_t> c(static_cast<size_t>(m) * n, kHalfNan);
    for (uint32_t row = 0; row < m; ++row) {
        const uint16_t value = (row & 1U) == 0 ? kHalfOne : kHalfPointFive;
        for (uint32_t col = 0; col < k; ++col) {
            a[static_cast<size_t>(row) * k + col] = value;
        }
    }
    static constexpr uint16_t bPattern[] = {
        kHalfPointTwoFive, kHalfPointFive, kHalfOne, kHalfMinusPointFive};
    for (uint32_t output = 0; output < n; ++output) {
        const uint16_t value = bPattern[output & 3U];
        for (uint32_t inner = 0; inner < k; ++inner) {
            b[static_cast<size_t>(output) * k + inner] = value;
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
    for (uint32_t i = 0; i < warmup; ++i) {
        Launch(tiling, stream, aDevice, bDevice, cDevice, workspace, tilingDevice);
    }
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        Launch(tiling, stream, aDevice, bDevice, cDevice, workspace, tilingDevice);
    }
    Check(aclrtSynchronizeStream(stream), "benchmark synchronize");
    const auto stopped = std::chrono::steady_clock::now();
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtMemcpy(c.data(), c.size() * sizeof(uint16_t), cDevice.ptr,
                     c.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy C");

    size_t mismatches = 0;
    size_t sentinels = 0;
    for (uint32_t row = 0; row < m; ++row) {
        const float aValue = (row & 1U) == 0 ? 1.0F : 0.5F;
        for (uint32_t col = 0; col < n; ++col) {
            const auto actual = c[static_cast<size_t>(row) * n + col];
            const auto expected = HalfBits(aValue * PatternValue(col) * static_cast<float>(k));
            sentinels += actual == kHalfNan;
            mismatches += actual != expected;
        }
    }
    if (sentinels != 0 || mismatches != 0) {
        throw std::runtime_error("incorrect output: sentinel=" + std::to_string(sentinels) +
                                 ", mismatches=" + std::to_string(mismatches));
    }

    const double elapsedMs =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    const double averageMs = elapsedMs / static_cast<double>(iterations);
    const double tflops =
        (2.0 * static_cast<double>(m) * n * k) / (averageMs * 1.0e9);
    std::cout << std::fixed << std::setprecision(6)
              << "M200 Cube probe PASS: name=" << name << ", M=" << m << ", N=" << n
              << ", K=" << k << ", cores=" << tiling.usedCores << ", warmup=" << warmup
              << ", iterations=" << iterations << ", average_ms=" << averageMs
              << ", tflops=" << tflops
              << ", local_workspace=" << tiling.localWorkspaceBytes
              << ", system_workspace=" << tiling.systemWorkspaceBytes << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 7) {
            throw std::invalid_argument(
                "usage: xlite_m200_cube_probe_runner NAME M N K WARMUP ITERATIONS");
        }
        const std::string name = argv[1];
        const uint32_t m = static_cast<uint32_t>(std::stoul(argv[2]));
        const uint32_t n = static_cast<uint32_t>(std::stoul(argv[3]));
        const uint32_t k = static_cast<uint32_t>(std::stoul(argv[4]));
        const uint32_t warmup = static_cast<uint32_t>(std::stoul(argv[5]));
        const uint32_t iterations = static_cast<uint32_t>(std::stoul(argv[6]));
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Run(name, m, n, k, warmup, iterations);
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "M200 Cube probe FAILED: " << error.what() << std::endl;
        return 1;
    }
}
