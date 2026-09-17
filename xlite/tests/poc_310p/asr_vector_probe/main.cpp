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
#include "aclrtlaunch_asr_add_rmsnorm_fp16.h"
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
    size_t maxAbsIndex = 0;
    size_t sentinels = 0;
    size_t nonFinite = 0;
    size_t zeros = 0;
    size_t mismatches = 0;
    double actualMin = INFINITY;
    double actualMax = -INFINITY;
    double expectedMin = INFINITY;
    double expectedMax = -INFINITY;
};

Metrics Compare(const std::vector<uint16_t> &actual,
                const std::vector<uint16_t> &expected,
                double tolerance)
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
        result.zeros += a == 0.0;
        const double absoluteError = std::abs(a - e);
        result.mismatches += absoluteError > tolerance;
        if (absoluteError > result.maxAbs) {
            result.maxAbs = absoluteError;
            result.maxAbsIndex = i;
        }
        if (std::isfinite(a)) {
            result.actualMin = std::min(result.actualMin, a);
            result.actualMax = std::max(result.actualMax, a);
        }
        result.expectedMin = std::min(result.expectedMin, e);
        result.expectedMax = std::max(result.expectedMax, e);
        dot += a * e;
        actualNorm += a * a;
        expectedNorm += e * e;
    }
    result.cosine = dot / std::sqrt(actualNorm * expectedNorm);
    return result;
}

void PrintDiagnostics(const char *name, const std::vector<uint16_t> &actual,
                      const std::vector<uint16_t> &expected, uint32_t rowWidth,
                      double tolerance, const Metrics &metrics)
{
    std::cerr << std::fixed << std::setprecision(6)
              << name << " diagnostics: elements=" << actual.size()
              << ", mismatches(abs>" << tolerance << ")=" << metrics.mismatches
              << ", max_abs=" << metrics.maxAbs
              << " at [" << metrics.maxAbsIndex / rowWidth << ","
              << metrics.maxAbsIndex % rowWidth << "]"
              << ", cosine=" << metrics.cosine
              << ", actual_range=[" << metrics.actualMin << "," << metrics.actualMax << "]"
              << ", expected_range=[" << metrics.expectedMin << "," << metrics.expectedMax << "]"
              << ", zeros=" << metrics.zeros
              << ", non_finite=" << metrics.nonFinite
              << ", sentinels=" << metrics.sentinels << std::endl;
    size_t printed = 0;
    for (size_t i = 0; i < actual.size() && printed < 16; ++i) {
        const double a = HalfValue(actual[i]);
        const double e = HalfValue(expected[i]);
        if (!std::isfinite(a) || std::abs(a - e) > tolerance) {
            std::cerr << "  mismatch[" << i / rowWidth << "," << i % rowWidth
                      << "] actual=" << a << " bits=0x" << std::hex << actual[i]
                      << " expected=" << std::dec << e << " bits=0x" << std::hex
                      << expected[i] << std::dec << " abs=" << std::abs(a - e)
                      << std::endl;
            ++printed;
        }
    }
    const uint32_t rows = static_cast<uint32_t>(actual.size() / rowWidth);
    for (uint32_t row = 0; row < rows; ++row) {
        double dot = 0.0;
        double actualNorm = 0.0;
        double expectedNorm = 0.0;
        double rowMaxAbs = 0.0;
        size_t rowMismatches = 0;
        for (uint32_t col = 0; col < rowWidth; ++col) {
            const size_t index = static_cast<size_t>(row) * rowWidth + col;
            const double a = HalfValue(actual[index]);
            const double e = HalfValue(expected[index]);
            dot += a * e;
            actualNorm += a * a;
            expectedNorm += e * e;
            rowMaxAbs = std::max(rowMaxAbs, std::abs(a - e));
            rowMismatches += std::abs(a - e) > tolerance;
        }
        const double rowCosine = dot / std::sqrt(actualNorm * expectedNorm);
        std::cerr << "  row=" << row << " cosine=" << rowCosine
                  << " max_abs=" << rowMaxAbs
                  << " mismatches=" << rowMismatches << std::endl;
    }
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
    const Metrics metrics = Compare(output, expected, 0.03);
    if (metrics.sentinels != 0 || metrics.nonFinite != 0 ||
        metrics.cosine < 0.999 || metrics.maxAbs > 0.03) {
        PrintDiagnostics("RMSNorm", output, expected, kHidden, 0.03, metrics);
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

void RunAddRmsNorm(uint32_t tokens, uint32_t warmup, uint32_t iterations)
{
    constexpr size_t guardElements = 16;
    constexpr float eps = 1.0e-6F;
    const size_t elements = static_cast<size_t>(tokens) * kHidden;
    std::vector<uint16_t> input(elements);
    std::vector<uint16_t> weight(kHidden);
    std::vector<uint16_t> initialResidual(elements);
    std::vector<uint16_t> expectedResidual(elements);
    std::vector<uint16_t> expectedOutput(elements);
    std::vector<uint16_t> residualStorage(elements + 2 * guardElements, kHalfNan);
    std::vector<uint16_t> outputStorage(elements + 2 * guardElements, kHalfNan);

    for (uint32_t col = 0; col < kHidden; ++col) {
        weight[col] = HalfBits(0.75F + static_cast<float>(col % 17) / 32.0F);
    }
    for (uint32_t row = 0; row < tokens; ++row) {
        double sumSquares = 0.0;
        for (uint32_t col = 0; col < kHidden; ++col) {
            const size_t index = static_cast<size_t>(row) * kHidden + col;
            input[index] = HalfBits(
                static_cast<float>(static_cast<int32_t>((row * 13 + col) % 31) - 15) /
                16.0F);
            initialResidual[index] = HalfBits(
                static_cast<float>(static_cast<int32_t>((row * 7 + col) % 23) - 11) /
                20.0F);
            const float sum = HalfValue(input[index]) + HalfValue(initialResidual[index]);
            expectedResidual[index] = HalfBits(sum);
            sumSquares += static_cast<double>(sum) * static_cast<double>(sum);
        }
        const float inverse = 1.0F /
                              std::sqrt(static_cast<float>(sumSquares / kHidden) + eps);
        for (uint32_t col = 0; col < kHidden; ++col) {
            const size_t index = static_cast<size_t>(row) * kHidden + col;
            const float sum = HalfValue(input[index]) + HalfValue(initialResidual[index]);
            expectedOutput[index] =
                HalfBits(sum * inverse * HalfValue(weight[col]));
        }
    }
    std::copy(initialResidual.begin(), initialResidual.end(),
              residualStorage.begin() + guardElements);

    DeviceBuffer inputDevice(input.size() * sizeof(uint16_t));
    DeviceBuffer weightDevice(weight.size() * sizeof(uint16_t));
    DeviceBuffer residualDevice(residualStorage.size() * sizeof(uint16_t));
    DeviceBuffer outputDevice(outputStorage.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(inputDevice.ptr, input.size() * sizeof(uint16_t), input.data(),
                     input.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    Check(aclrtMemcpy(weightDevice.ptr, weight.size() * sizeof(uint16_t), weight.data(),
                     weight.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy weight");
    Check(aclrtMemcpy(residualDevice.ptr, residualStorage.size() * sizeof(uint16_t),
                     residualStorage.data(), residualStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "copy residual with guards");
    Check(aclrtMemcpy(outputDevice.ptr, outputStorage.size() * sizeof(uint16_t),
                     outputStorage.data(), outputStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "copy output guards");
    auto *residualData = static_cast<uint8_t *>(residualDevice.ptr) +
                         guardElements * sizeof(uint16_t);
    auto *outputData = static_cast<uint8_t *>(outputDevice.ptr) +
                       guardElements * sizeof(uint16_t);
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    const uint32_t cores = std::min(8U, tokens);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_add_rmsnorm_fp16)
        (cores, stream, inputDevice.ptr, residualData, weightDevice.ptr,
         outputData, tokens, eps);
    };

    launch();
    Check(aclrtSynchronizeStream(stream), "correctness synchronize");
    Check(aclrtMemcpy(residualStorage.data(), residualStorage.size() * sizeof(uint16_t),
                     residualDevice.ptr, residualStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST), "read residual with guards");
    Check(aclrtMemcpy(outputStorage.data(), outputStorage.size() * sizeof(uint16_t),
                     outputDevice.ptr, outputStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST), "read output with guards");

    const bool guardsIntact = std::all_of(
        residualStorage.begin(), residualStorage.begin() + guardElements,
        [](uint16_t value) { return value == kHalfNan; }) &&
        std::all_of(residualStorage.end() - guardElements, residualStorage.end(),
                    [](uint16_t value) { return value == kHalfNan; }) &&
        std::all_of(outputStorage.begin(), outputStorage.begin() + guardElements,
                    [](uint16_t value) { return value == kHalfNan; }) &&
        std::all_of(outputStorage.end() - guardElements, outputStorage.end(),
                    [](uint16_t value) { return value == kHalfNan; });
    std::vector<uint16_t> actualResidual(residualStorage.begin() + guardElements,
                                         residualStorage.end() - guardElements);
    std::vector<uint16_t> actualOutput(outputStorage.begin() + guardElements,
                                       outputStorage.end() - guardElements);
    const Metrics residualMetrics = Compare(actualResidual, expectedResidual, 0.002);
    const Metrics outputMetrics = Compare(actualOutput, expectedOutput, 0.04);
    if (!guardsIntact || residualMetrics.sentinels != 0 ||
        residualMetrics.nonFinite != 0 || residualMetrics.maxAbs > 0.002 ||
        outputMetrics.sentinels != 0 || outputMetrics.nonFinite != 0 ||
        outputMetrics.cosine < 0.999 || outputMetrics.maxAbs > 0.04) {
        PrintDiagnostics("AddRMSNorm residual", actualResidual, expectedResidual,
                         kHidden, 0.002, residualMetrics);
        PrintDiagnostics("AddRMSNorm output", actualOutput, expectedOutput,
                         kHidden, 0.04, outputMetrics);
        throw std::runtime_error(
            "AddRMSNorm mismatch: guards=" + std::to_string(guardsIntact) +
            ", residual_max_abs=" + std::to_string(residualMetrics.maxAbs) +
            ", output_cosine=" + std::to_string(outputMetrics.cosine) +
            ", output_max_abs=" + std::to_string(outputMetrics.maxAbs));
    }

    std::fill(residualStorage.begin(), residualStorage.end(), kHalfNan);
    std::copy(initialResidual.begin(), initialResidual.end(),
              residualStorage.begin() + guardElements);
    Check(aclrtMemcpy(residualDevice.ptr, residualStorage.size() * sizeof(uint16_t),
                     residualStorage.data(), residualStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "reset residual for benchmark");
    const double averageMs = Benchmark(stream, warmup, iterations, launch);
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    std::cout << std::fixed << std::setprecision(6)
              << "ASR AddRMSNorm PASS: tokens=" << tokens << ", cores=" << cores
              << ", average_ms=" << averageMs
              << ", residual_max_abs=" << residualMetrics.maxAbs
              << ", output_cosine=" << outputMetrics.cosine
              << ", output_max_abs=" << outputMetrics.maxAbs
              << ", guards_intact=" << guardsIntact << std::endl;
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
    const Metrics metrics = Compare(output, expected, 0.04);
    if (metrics.sentinels != 0 || metrics.nonFinite != 0 ||
        metrics.cosine < 0.999 || metrics.maxAbs > 0.04) {
        PrintDiagnostics("SiLU-Mul", output, expected, kIntermediate, 0.04, metrics);
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
            throw std::invalid_argument(
                "usage: runner rmsnorm|add-rmsnorm|silu TOKENS WARMUP ITERATIONS");
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
        } else if (op == "add-rmsnorm") {
            RunAddRmsNorm(tokens, warmup, iterations);
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
