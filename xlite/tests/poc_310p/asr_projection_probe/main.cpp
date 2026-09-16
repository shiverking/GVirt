#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_asr_m200_projection_fp16.h"

namespace {

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
    const __fp16 converted = static_cast<__fp16>(value);
    uint16_t bits = 0;
    std::memcpy(&bits, &converted, sizeof(bits));
    return bits;
}

float WeightValue(uint32_t row)
{
    static constexpr float values[] = {0.25F, 0.5F, 1.0F, -0.5F};
    return values[row & 3U];
}

void Run(uint32_t m, uint32_t n, uint32_t k, uint32_t warmup, uint32_t iterations)
{
    if (m == 0 || m > 20 || n == 0 || k == 0 || iterations == 0) {
        throw std::invalid_argument("invalid probe shape or iteration count");
    }
    std::vector<uint16_t> a(static_cast<size_t>(m) * k);
    std::vector<uint16_t> b(static_cast<size_t>(n) * k);
    std::vector<uint16_t> c(static_cast<size_t>(m) * n, kHalfNan);
    for (uint32_t row = 0; row < m; ++row) {
        const uint16_t value = HalfBits((row & 1U) == 0 ? 1.0F : 0.5F);
        std::fill(a.begin() + static_cast<size_t>(row) * k,
                  a.begin() + static_cast<size_t>(row + 1) * k, value);
    }
    for (uint32_t row = 0; row < n; ++row) {
        const uint16_t value = HalfBits(WeightValue(row));
        std::fill(b.begin() + static_cast<size_t>(row) * k,
                  b.begin() + static_cast<size_t>(row + 1) * k, value);
    }

    DeviceBuffer aDevice(a.size() * sizeof(uint16_t));
    DeviceBuffer bDevice(b.size() * sizeof(uint16_t));
    DeviceBuffer cDevice(c.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(aDevice.ptr, a.size() * sizeof(uint16_t), a.data(),
                     a.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
    Check(aclrtMemcpy(bDevice.ptr, b.size() * sizeof(uint16_t), b.data(),
                     b.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
    Check(aclrtMemcpy(cDevice.ptr, c.size() * sizeof(uint16_t), c.data(),
                     c.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy C");

    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    const uint32_t blockDim = std::min(8U, (n + 127U) / 128U);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_m200_projection_fp16)
        (blockDim, stream, aDevice.ptr, bDevice.ptr, cDevice.ptr, m, n, k);
    };
    for (uint32_t i = 0; i < warmup; ++i) {
        launch();
    }
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) {
        launch();
    }
    Check(aclrtSynchronizeStream(stream), "benchmark synchronize");
    const auto stopped = std::chrono::steady_clock::now();
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtMemcpy(c.data(), c.size() * sizeof(uint16_t), cDevice.ptr,
                     c.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy output");

    size_t mismatches = 0;
    size_t sentinels = 0;
    for (uint32_t row = 0; row < m; ++row) {
        const float activation = (row & 1U) == 0 ? 1.0F : 0.5F;
        for (uint32_t col = 0; col < n; ++col) {
            const uint16_t actual = c[static_cast<size_t>(row) * n + col];
            const uint16_t expected = HalfBits(activation * WeightValue(col) * k);
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
    const double tflops = 2.0 * m * n * k / (averageMs * 1.0e9);
    std::cout << std::fixed << std::setprecision(6)
              << "ASR low-level projection PASS: M=" << m << ", N=" << n
              << ", K=" << k << ", cores=" << blockDim
              << ", average_ms=" << averageMs << ", tflops=" << tflops << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 6) {
            throw std::invalid_argument("usage: runner M N K WARMUP ITERATIONS");
        }
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Run(static_cast<uint32_t>(std::stoul(argv[1])),
            static_cast<uint32_t>(std::stoul(argv[2])),
            static_cast<uint32_t>(std::stoul(argv[3])),
            static_cast<uint32_t>(std::stoul(argv[4])),
            static_cast<uint32_t>(std::stoul(argv[5])));
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR low-level projection FAILED: " << error.what() << std::endl;
        return 1;
    }
}
