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
#include "aclrtlaunch_asr_rmsnorm_fp16.h"
#include "aclrtlaunch_asr_silu_mul_fp16.h"

namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kIntermediate = 6144;
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

float HalfValue(uint16_t bits)
{
    __fp16 value;
    std::memcpy(&value, &bits, sizeof(value));
    return static_cast<float>(value);
}

struct Metrics {
    double cosine = 0.0;
    double maxAbs = 0.0;
    size_t sentinels = 0;
    size_t nonFinite = 0;
};

Metrics Compare(const std::vector<uint16_t> &actual,
                const std::vector<uint16_t> &expected)
{
    if (actual.size() != expected.size()) {
        throw std::invalid_argument("comparison sizes differ");
    }
    double dot = 0.0;
    double actualNorm = 0.0;
    double expectedNorm = 0.0;
    Metrics result;
    for (size_t i = 0; i < actual.size(); ++i) {
        result.sentinels += actual[i] == kHalfNan;
        const double a = HalfValue(actual[i]);
        const double e = HalfValue(expected[i]);
        result.nonFinite += !std::isfinite(a);
        result.maxAbs = std::max(result.maxAbs, std::abs(a - e));
        dot += a * e;
        actualNorm += a * a;
        expectedNorm += e * e;
    }
    result.cosine = dot / std::sqrt(actualNorm * expectedNorm);
    return result;
}

template <typename Launch>
double Benchmark(aclrtStream stream, uint32_t warmup, uint32_t iterations,
                 Launch launch)
{
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
    return std::chrono::duration<double, std::milli>(stopped - started).count() /
           static_cast<double>(iterations);
}

void RunRmsNorm(uint32_t tokens, uint32_t warmup, uint32_t iterations)
{
    const size_t elements = static_cast<size_t>(tokens) * kHidden;
    std::vector<uint16_t> input(elements);
    std::vector<uint16_t> weight(kHidden);
    std::vector<uint16_t> output(elements, kHalfNan);
    std::vector<uint16_t> expected(elements);
    for (uint32_t col = 0; col < kHidden; ++col) {
        weight[col] = HalfBits(0.75F + static_cast<float>(col % 17) / 32.0F);
    }
    constexpr float eps = 1.0e-6F;
    for (uint32_t row = 0; row < tokens; ++row) {
        double sumSquares = 0.0;
        for (uint32_t col = 0; col < kHidden; ++col) {
            const float value = static_cast<float>(static_cast<int32_t>((row * 13 + col) % 31) -
                                                   15) /
                                16.0F;
            input[static_cast<size_t>(row) * kHidden + col] = HalfBits(value);
            const double rounded = HalfValue(input[static_cast<size_t>(row) * kHidden + col]);
            sumSquares += rounded * rounded;
        }
        const float inverse = 1.0F /
                              std::sqrt(static_cast<float>(sumSquares / kHidden) + eps);
        for (uint32_t col = 0; col < kHidden; ++col) {
            const size_t index = static_cast<size_t>(row) * kHidden + col;
            expected[index] = HalfBits(HalfValue(input[index]) * inverse *
                                       HalfValue(weight[col]));
        }
    }

    DeviceBuffer inputDevice(input.size() * sizeof(uint16_t));
    DeviceBuffer weightDevice(weight.size() * sizeof(uint16_t));
    DeviceBuffer outputDevice(output.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(inputDevice.ptr, input.size() * sizeof(uint16_t), input.data(),
                     input.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    Check(aclrtMemcpy(weightDevice.ptr, weight.size() * sizeof(uint16_t), weight.data(),
                     weight.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy weight");
    Check(aclrtMemcpy(outputDevice.ptr, output.size() * sizeof(uint16_t), output.data(),
                     output.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy output");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    const uint32_t cores = std::min(8U, tokens);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_rmsnorm_fp16)
        (cores, stream, inputDevice.ptr, weightDevice.ptr, outputDevice.ptr, tokens, eps);
    };
    const double averageMs = Benchmark(stream, warmup, iterations, launch);
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputDevice.ptr,
                     output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    const Metrics metrics = Compare(output, expected);
    if (metrics.sentinels != 0 || metrics.nonFinite != 0 ||
        metrics.cosine < 0.999 || metrics.maxAbs > 0.03) {
        throw std::runtime_error("RMSNorm mismatch: cosine=" +
                                 std::to_string(metrics.cosine) + ", max_abs=" +
                                 std::to_string(metrics.maxAbs) + ", sentinel=" +
                                 std::to_string(metrics.sentinels));
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR RMSNorm PASS: tokens=" << tokens << ", cores=" << cores
              << ", average_ms=" << averageMs << ", cosine=" << metrics.cosine
              << ", max_abs=" << metrics.maxAbs << std::endl;
}

void RunSiluMul(uint32_t tokens, uint32_t warmup, uint32_t iterations)
{
    const size_t inputElements = static_cast<size_t>(tokens) * 2 * kIntermediate;
    const size_t outputElements = static_cast<size_t>(tokens) * kIntermediate;
    std::vector<uint16_t> input(inputElements);
    std::vector<uint16_t> output(outputElements, kHalfNan);
    std::vector<uint16_t> expected(outputElements);
    for (uint32_t row = 0; row < tokens; ++row) {
        for (uint32_t col = 0; col < kIntermediate; ++col) {
            const float gate = static_cast<float>(static_cast<int32_t>((row * 7 + col) % 41) -
                                                  20) /
                               10.0F;
            const float up = static_cast<float>(static_cast<int32_t>((row * 11 + col) % 29) -
                                                14) /
                             12.0F;
            const size_t base = static_cast<size_t>(row) * 2 * kIntermediate;
            input[base + col] = HalfBits(gate);
            input[base + kIntermediate + col] = HalfBits(up);
            const float roundedGate = HalfValue(input[base + col]);
            const float roundedUp = HalfValue(input[base + kIntermediate + col]);
            expected[static_cast<size_t>(row) * kIntermediate + col] =
                HalfBits((roundedGate / (1.0F + std::exp(-roundedGate))) * roundedUp);
        }
    }
    DeviceBuffer inputDevice(input.size() * sizeof(uint16_t));
    DeviceBuffer outputDevice(output.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(inputDevice.ptr, input.size() * sizeof(uint16_t), input.data(),
                     input.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    Check(aclrtMemcpy(outputDevice.ptr, output.size() * sizeof(uint16_t), output.data(),
                     output.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy output");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    const uint32_t cores = std::min(8U, tokens);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_silu_mul_fp16)
        (cores, stream, inputDevice.ptr, outputDevice.ptr, tokens);
    };
    const double averageMs = Benchmark(stream, warmup, iterations, launch);
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtMemcpy(output.data(), output.size() * sizeof(uint16_t), outputDevice.ptr,
                     output.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "read output");
    const Metrics metrics = Compare(output, expected);
    if (metrics.sentinels != 0 || metrics.nonFinite != 0 ||
        metrics.cosine < 0.999 || metrics.maxAbs > 0.04) {
        throw std::runtime_error("SiLU-Mul mismatch: cosine=" +
                                 std::to_string(metrics.cosine) + ", max_abs=" +
                                 std::to_string(metrics.maxAbs) + ", sentinel=" +
                                 std::to_string(metrics.sentinels));
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR SiLU-Mul PASS: tokens=" << tokens << ", cores=" << cores
              << ", average_ms=" << averageMs << ", cosine=" << metrics.cosine
              << ", max_abs=" << metrics.maxAbs << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 5) {
            throw std::invalid_argument("usage: runner rmsnorm|silu TOKENS WARMUP ITERATIONS");
        }
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        const std::string op = argv[1];
        const uint32_t tokens = static_cast<uint32_t>(std::stoul(argv[2]));
        const uint32_t warmup = static_cast<uint32_t>(std::stoul(argv[3]));
        const uint32_t iterations = static_cast<uint32_t>(std::stoul(argv[4]));
        if (tokens == 0 || tokens > 20 || iterations == 0) {
            throw std::invalid_argument("tokens must be 1-20 and iterations positive");
        }
        if (op == "rmsnorm") {
            RunRmsNorm(tokens, warmup, iterations);
        } else if (op == "silu") {
            RunSiluMul(tokens, warmup, iterations);
        } else {
            throw std::invalid_argument("unknown op: " + op);
        }
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR vector probe FAILED: " << error.what() << std::endl;
        return 1;
    }
}
