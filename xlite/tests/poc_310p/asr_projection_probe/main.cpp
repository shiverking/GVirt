#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_asr_m200_projection_fp16.h"
#include "aclrtlaunch_asr_m200_projection_cached_fp16.h"

namespace {

constexpr uint16_t kHalfNan = 0x7e00;
constexpr size_t kGuardElements = 256;

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

float InputValue(uint32_t row, uint32_t k)
{
    return static_cast<float>(static_cast<int>((k * 13 + row * 7) % 17) - 8) /
           8.0F;
}

float WeightValue(uint32_t row)
{
    return static_cast<float>(static_cast<int>((row * 37) % 251) - 125) /
           256.0F;
}

float WeightKValue(uint32_t k)
{
    static constexpr float values[] = {1.0F, -1.0F, 0.5F, -0.5F};
    return values[(k / 128 + k / 16 + k) % 4];
}

float HalfValue(uint16_t bits)
{
    __fp16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

void Run(uint32_t m, uint32_t n, uint32_t k, uint32_t warmup,
         uint32_t iterations, bool cached)
{
    if (m == 0 || m > 20 || n == 0 || k == 0 || iterations == 0) {
        throw std::invalid_argument("invalid probe shape or iteration count");
    }
    std::vector<uint16_t> a(static_cast<size_t>(m) * k);
    std::vector<uint16_t> b(static_cast<size_t>(n) * k);
    std::vector<float> rowDots(m, 0.0F);
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t inner = 0; inner < k; ++inner) {
            const float value = InputValue(row, inner);
            a[static_cast<size_t>(row) * k + inner] = HalfBits(value);
            const float weightK = HalfValue(HalfBits(WeightKValue(inner)));
            rowDots[row] += HalfValue(HalfBits(value)) * weightK;
        }
    }
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t inner = 0; inner < k; ++inner) {
            b[static_cast<size_t>(row) * k + inner] =
                HalfBits(WeightValue(row) * WeightKValue(inner));
        }
    }

    const size_t outputElements = static_cast<size_t>(m) * n;
    std::vector<uint16_t> c(outputElements + 2 * kGuardElements, kHalfNan);

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
    auto *output = static_cast<uint8_t *>(cDevice.ptr) +
                   kGuardElements * sizeof(uint16_t);
    const auto launch = [&]() {
        if (cached) {
            ACLRT_LAUNCH_KERNEL(asr_m200_projection_cached_fp16)
            (blockDim, stream, aDevice.ptr, bDevice.ptr, output, m, n, k);
        } else {
            ACLRT_LAUNCH_KERNEL(asr_m200_projection_fp16)
            (blockDim, stream, aDevice.ptr, bDevice.ptr, output, m, n, k);
        }
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

    size_t guardErrors = 0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        guardErrors += c[i] != kHalfNan;
        guardErrors += c[kGuardElements + outputElements + i] != kHalfNan;
    }
    size_t mismatches = 0;
    size_t sentinels = 0;
    size_t finite = 0;
    size_t nonfinite = 0;
    size_t reported = 0;
    double dot = 0.0, actualSquare = 0.0, expectedSquare = 0.0;
    float maxAbs = 0.0F;
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            const uint16_t actual = c[kGuardElements +
                                      static_cast<size_t>(row) * n + col];
            const uint16_t expected = HalfBits(rowDots[row] * WeightValue(col));
            sentinels += actual == kHalfNan;
            const float actualValue = HalfValue(actual);
            const float expectedValue = HalfValue(expected);
            finite += std::isfinite(actualValue);
            nonfinite += !std::isfinite(actualValue);
            maxAbs = std::max(maxAbs, std::abs(actualValue - expectedValue));
            dot += static_cast<double>(actualValue) * expectedValue;
            actualSquare += static_cast<double>(actualValue) * actualValue;
            expectedSquare += static_cast<double>(expectedValue) * expectedValue;
            if (!std::isfinite(actualValue) ||
                std::abs(actualValue - expectedValue) >
                    0.01F + 0.01F * std::abs(expectedValue)) {
                ++mismatches;
                if (reported < 12) {
                    std::cerr << "mismatch[" << row << "," << col << "] actual="
                              << actualValue << " bits=0x" << std::hex << actual
                              << std::dec << " expected=" << HalfValue(expected)
                              << " bits=0x" << std::hex << expected << std::dec << '\n';
                    ++reported;
                }
            }
        }
    }
    const double cosine = dot / std::sqrt(actualSquare * expectedSquare);
    if (guardErrors != 0 || sentinels != 0 || mismatches != 0 ||
        nonfinite != 0 || !std::isfinite(cosine) || cosine < 0.999) {
        throw std::runtime_error("projection contract failed: sentinel=" +
                                 std::to_string(sentinels) + ", mismatches=" +
                                 std::to_string(mismatches) + ", finite=" +
                                 std::to_string(finite) + ", cosine=" +
                                 std::to_string(cosine) + ", max_abs=" +
                                 std::to_string(maxAbs) + ", guard_errors=" +
                                 std::to_string(guardErrors));
    }

    const double elapsedMs =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    const double averageMs = elapsedMs / static_cast<double>(iterations);
    const double tflops = 2.0 * m * n * k / (averageMs * 1.0e9);
    std::cout << std::fixed << std::setprecision(6)
              << "ASR low-level projection PASS: M=" << m << ", N=" << n
              << ", K=" << k << ", cores=" << blockDim
              << ", variant=" << (cached ? "cached" : "baseline")
              << ", average_ms=" << averageMs << ", tflops=" << tflops
              << ", cosine=" << cosine << ", max_abs=" << maxAbs
              << ", guard_errors=" << guardErrors << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 6 && argc != 7) {
            throw std::invalid_argument(
                "usage: runner M N K WARMUP ITERATIONS [baseline|cached]");
        }
        const std::string variant = argc == 7 ? argv[6] : "baseline";
        if (variant != "baseline" && variant != "cached") {
            throw std::invalid_argument("unknown projection variant: " + variant);
        }
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Run(static_cast<uint32_t>(std::stoul(argv[1])),
            static_cast<uint32_t>(std::stoul(argv[2])),
            static_cast<uint32_t>(std::stoul(argv[3])),
            static_cast<uint32_t>(std::stoul(argv[4])),
            static_cast<uint32_t>(std::stoul(argv[5])), variant == "cached");
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR low-level projection FAILED: " << error.what() << std::endl;
        return 1;
    }
}
