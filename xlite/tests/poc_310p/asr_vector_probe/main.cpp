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
#include "aclrtlaunch_asr_qk_norm_mrope_cache_fp16.h"
#include "aclrtlaunch_asr_qk_norm_mrope_cache_grouped_fp16.h"
#include "aclrtlaunch_asr_rmsnorm_fp16.h"
#include "aclrtlaunch_asr_silu_mul_fp16.h"
#include "aclrtlaunch_asr_silu_mul_row_fp16.h"

namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kIntermediate = 6144;
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQkvDim = (kQHeads + 2 * kKvHeads) * kHeadDim;
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

void RunSiluMul(uint32_t tokens, uint32_t warmup, uint32_t iterations,
                bool wholeRow)
{
    const size_t inputElements = static_cast<size_t>(tokens) * 2 * kIntermediate;
    const size_t outputElements = static_cast<size_t>(tokens) * kIntermediate;
    std::vector<uint16_t> input(inputElements);
    constexpr size_t guardElements = 256;
    std::vector<uint16_t> outputStorage(outputElements + 2 * guardElements,
                                        kHalfNan);
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
    DeviceBuffer outputDevice(outputStorage.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(inputDevice.ptr, input.size() * sizeof(uint16_t), input.data(),
                     input.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    Check(aclrtMemcpy(outputDevice.ptr, outputStorage.size() * sizeof(uint16_t),
                     outputStorage.data(), outputStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "copy guarded output");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    const uint32_t cores = std::min(8U, tokens);
    auto *output = static_cast<uint8_t *>(outputDevice.ptr) +
                   guardElements * sizeof(uint16_t);
    const auto launch = [&]() {
        if (wholeRow) {
            ACLRT_LAUNCH_KERNEL(asr_silu_mul_row_fp16)
            (cores, stream, inputDevice.ptr, output, tokens);
        } else {
            ACLRT_LAUNCH_KERNEL(asr_silu_mul_fp16)
            (cores, stream, inputDevice.ptr, output, tokens);
        }
    };
    const double averageMs = Benchmark(stream, warmup, iterations, launch);
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");
    Check(aclrtMemcpy(outputStorage.data(), outputStorage.size() * sizeof(uint16_t),
                     outputDevice.ptr, outputStorage.size() * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST), "read guarded output");
    const bool guardsIntact = std::all_of(
        outputStorage.begin(), outputStorage.begin() + guardElements,
        [](uint16_t value) { return value == kHalfNan; }) &&
        std::all_of(outputStorage.end() - guardElements, outputStorage.end(),
                    [](uint16_t value) { return value == kHalfNan; });
    std::vector<uint16_t> actual(outputStorage.begin() + guardElements,
                                 outputStorage.end() - guardElements);
    const Metrics metrics = Compare(actual, expected, 0.04);
    if (!guardsIntact || metrics.sentinels != 0 || metrics.nonFinite != 0 ||
        metrics.cosine < 0.999 || metrics.maxAbs > 0.04) {
        PrintDiagnostics("SiLU-Mul", actual, expected, kIntermediate, 0.04, metrics);
        throw std::runtime_error("SiLU-Mul mismatch: cosine=" +
                                 std::to_string(metrics.cosine) + ", max_abs=" +
                                 std::to_string(metrics.maxAbs) + ", sentinel=" +
                                 std::to_string(metrics.sentinels) + ", guards=" +
                                 std::to_string(guardsIntact));
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR SiLU-Mul PASS: tokens=" << tokens << ", cores=" << cores
              << ", variant=" << (wholeRow ? "row" : "baseline")
              << ", iterations=" << iterations
              << ", average_ms=" << averageMs << ", cosine=" << metrics.cosine
              << ", max_abs=" << metrics.maxAbs
              << ", guards_intact=" << guardsIntact << std::endl;
}

float RoundHalf(float value) { return HalfValue(HalfBits(value)); }

void RunQkMropeCache(uint32_t tokens, uint32_t warmup, uint32_t iterations,
                     bool grouped)
{
    constexpr float eps = 1.0e-6F;
    constexpr float qScale = 0.08838834764831845F;
    constexpr uint32_t maxPosition = 64;
    constexpr uint32_t cacheSlots = 512;
    std::vector<uint16_t> qkv(static_cast<size_t>(tokens) * kQkvDim);
    std::vector<uint16_t> qWeight(kHeadDim), kWeight(kHeadDim);
    std::vector<int64_t> positions(static_cast<size_t>(3) * tokens);
    std::vector<int32_t> slots(tokens);
    std::vector<uint16_t> cossin(static_cast<size_t>(maxPosition) * kHeadDim);
    const size_t cacheElements = static_cast<size_t>(cacheSlots) * kKvHeads * kHeadDim;
    std::vector<uint16_t> kCache(cacheElements, kHalfNan), vCache(cacheElements, kHalfNan);
    for (uint32_t d = 0; d < kHeadDim; ++d) {
        qWeight[d] = HalfBits(0.8F + static_cast<float>(d % 11) / 64.0F);
        kWeight[d] = HalfBits(0.9F + static_cast<float>(d % 7) / 64.0F);
    }
    for (uint32_t p = 0; p < maxPosition; ++p) {
        for (uint32_t d = 0; d < 64; ++d) {
            const float angle = static_cast<float>(p) /
                std::pow(10000.0F, static_cast<float>(2 * d) / kHeadDim);
            cossin[static_cast<size_t>(p) * kHeadDim + d] = HalfBits(std::cos(angle));
            cossin[static_cast<size_t>(p) * kHeadDim + 64 + d] = HalfBits(std::sin(angle));
        }
    }
    for (uint32_t t = 0; t < tokens; ++t) {
        positions[t] = 3 + t;
        positions[tokens + t] = 11 + (t * 3) % 19;
        positions[2 * tokens + t] = 23 + (t * 5) % 29;
        slots[t] = static_cast<int32_t>((t * 131 + 17) % cacheSlots);
        for (uint32_t d = 0; d < kQkvDim; ++d) {
            qkv[static_cast<size_t>(t) * kQkvDim + d] = HalfBits(
                static_cast<float>(static_cast<int32_t>((t * 17 + d * 7) % 47) - 23) / 16.0F);
        }
    }
    const std::vector<uint16_t> original = qkv;
    std::vector<uint16_t> expected = qkv, expectedK = kCache, expectedV = vCache;
    auto axisForPair = [](uint32_t pair) {
        if (pair < 60 && pair % 3 == 1) return 1U;
        if (pair < 60 && pair % 3 == 2) return 2U;
        return 0U;
    };
    for (uint32_t t = 0; t < tokens; ++t) {
        for (uint32_t packed = 0; packed < kQHeads + kKvHeads; ++packed) {
            const bool isQ = packed < kQHeads;
            const uint32_t head = isQ ? packed : packed - kQHeads;
            const size_t base = static_cast<size_t>(t) * kQkvDim +
                (isQ ? head * kHeadDim : kQHeads * kHeadDim + head * kHeadDim);
            const auto &weight = isQ ? qWeight : kWeight;
            double sum = 0.0;
            for (uint32_t d = 0; d < kHeadDim; ++d) {
                const float x = HalfValue(original[base + d]); sum += x * x;
            }
            const float inv = 1.0F / std::sqrt(static_cast<float>(sum / kHeadDim) + eps);
            std::vector<float> norm(kHeadDim);
            for (uint32_t d = 0; d < kHeadDim; ++d)
                norm[d] = RoundHalf(HalfValue(original[base + d]) * inv * HalfValue(weight[d]));
            for (uint32_t d = 0; d < 64; ++d) {
                const uint32_t axis = axisForPair(d);
                const uint32_t pos = static_cast<uint32_t>(positions[axis * tokens + t]);
                const float c = HalfValue(cossin[static_cast<size_t>(pos) * kHeadDim + d]);
                const float s = HalfValue(cossin[static_cast<size_t>(pos) * kHeadDim + 64 + d]);
                const float first = RoundHalf(RoundHalf(norm[d] * c) - RoundHalf(norm[d + 64] * s));
                const float second = RoundHalf(RoundHalf(norm[d + 64] * c) + RoundHalf(norm[d] * s));
                expected[base + d] = HalfBits(isQ ? RoundHalf(first * RoundHalf(qScale)) : first);
                expected[base + 64 + d] = HalfBits(isQ ? RoundHalf(second * RoundHalf(qScale)) : second);
            }
            if (!isQ) {
                const size_t cacheBase = (static_cast<size_t>(slots[t]) * kKvHeads + head) * kHeadDim;
                std::copy(expected.begin() + base, expected.begin() + base + kHeadDim,
                          expectedK.begin() + cacheBase);
                const size_t valueBase = static_cast<size_t>(t) * kQkvDim +
                    (kQHeads + kKvHeads) * kHeadDim + head * kHeadDim;
                std::copy(original.begin() + valueBase, original.begin() + valueBase + kHeadDim,
                          expectedV.begin() + cacheBase);
            }
        }
    }
    DeviceBuffer qkvD(qkv.size()*2), qwD(qWeight.size()*2), kwD(kWeight.size()*2),
        posD(positions.size()*8), csD(cossin.size()*2), slotD(slots.size()*4),
        kcD(kCache.size()*2), vcD(vCache.size()*2);
    auto h2d = [&](void *d, const void *h, size_t n, const char *name) {
        Check(aclrtMemcpy(d, n, h, n, ACL_MEMCPY_HOST_TO_DEVICE), name);
    };
    h2d(qkvD.ptr,qkv.data(),qkv.size()*2,"copy qkv"); h2d(qwD.ptr,qWeight.data(),qWeight.size()*2,"copy qw");
    h2d(kwD.ptr,kWeight.data(),kWeight.size()*2,"copy kw"); h2d(posD.ptr,positions.data(),positions.size()*8,"copy positions");
    h2d(csD.ptr,cossin.data(),cossin.size()*2,"copy cossin"); h2d(slotD.ptr,slots.data(),slots.size()*4,"copy slots");
    h2d(kcD.ptr,kCache.data(),kCache.size()*2,"copy kcache"); h2d(vcD.ptr,vCache.data(),vCache.size()*2,"copy vcache");
    aclrtStream stream = nullptr; Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    auto launch = [&]() {
        if (grouped) {
            ACLRT_LAUNCH_KERNEL(asr_qk_norm_mrope_cache_grouped_fp16)
            (8, stream, qkvD.ptr, qwD.ptr, kwD.ptr, posD.ptr, csD.ptr,
             slotD.ptr, kcD.ptr, vcD.ptr, tokens, eps, qScale);
        } else {
            ACLRT_LAUNCH_KERNEL(asr_qk_norm_mrope_cache_fp16)
            (8, stream, qkvD.ptr, qwD.ptr, kwD.ptr, posD.ptr, csD.ptr,
             slotD.ptr, kcD.ptr, vcD.ptr, tokens, eps, qScale);
        }
    };
    launch(); Check(aclrtSynchronizeStream(stream), "correctness sync");
    auto d2h = [&](void *h, void *d, size_t n, const char *name) {
        Check(aclrtMemcpy(h, n, d, n, ACL_MEMCPY_DEVICE_TO_HOST), name);
    };
    d2h(qkv.data(),qkvD.ptr,qkv.size()*2,"read qkv"); d2h(kCache.data(),kcD.ptr,kCache.size()*2,"read kcache");
    d2h(vCache.data(),vcD.ptr,vCache.size()*2,"read vcache");
    const Metrics qm=Compare(qkv,expected,0.05), km=Compare(kCache,expectedK,0.05), vm=Compare(vCache,expectedV,0.002);
    const size_t untouched = cacheElements - static_cast<size_t>(tokens)*kKvHeads*kHeadDim;
    if(qm.sentinels||qm.nonFinite||qm.cosine<0.999||qm.maxAbs>0.05||
       km.sentinels!=untouched||km.maxAbs>0.05||vm.sentinels!=untouched||vm.maxAbs>0.002) {
        PrintDiagnostics("QKV",qkv,expected,kQkvDim,0.05,qm);
        throw std::runtime_error("QK-MRoPE-Cache mismatch q_cos="+std::to_string(qm.cosine)+
            ", q_max="+std::to_string(qm.maxAbs)+", k_max="+std::to_string(km.maxAbs)+
            ", v_max="+std::to_string(vm.maxAbs));
    }
    h2d(qkvD.ptr,original.data(),original.size()*2,"reset qkv");
    const double ms=Benchmark(stream,warmup,iterations,launch);
    Check(aclrtDestroyStream(stream),"destroy stream");
    std::cout<<std::fixed<<std::setprecision(6)<<"ASR QK-Norm-MRoPE-Cache PASS: tokens="<<tokens
             <<", variant="<<(grouped ? "grouped" : "baseline")
             <<", average_ms="<<ms<<", qkv_cosine="<<qm.cosine<<", qkv_max_abs="<<qm.maxAbs
             <<", cache_max_abs="<<std::max(km.maxAbs,vm.maxAbs)<<std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 5 && argc != 6) {
            throw std::invalid_argument(
                "usage: runner rmsnorm|add-rmsnorm|qk-mrope-cache|silu "
                "TOKENS WARMUP ITERATIONS [baseline|grouped|row]");
        }
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        const std::string op = argv[1];
        const uint32_t tokens = static_cast<uint32_t>(std::stoul(argv[2]));
        const uint32_t warmup = static_cast<uint32_t>(std::stoul(argv[3]));
        const uint32_t iterations = static_cast<uint32_t>(std::stoul(argv[4]));
        const std::string variant = argc == 6 ? argv[5] : "baseline";
        if (variant != "baseline" && variant != "grouped" && variant != "row") {
            throw std::invalid_argument("unknown vector variant: " + variant);
        }
        if (tokens == 0 || tokens > 20 || iterations == 0) {
            throw std::invalid_argument("tokens must be 1-20 and iterations positive");
        }
        if (op == "rmsnorm") {
            RunRmsNorm(tokens, warmup, iterations);
        } else if (op == "add-rmsnorm") {
            RunAddRmsNorm(tokens, warmup, iterations);
        } else if (op == "qk-mrope-cache") {
            RunQkMropeCache(tokens, warmup, iterations, variant == "grouped");
        } else if (op == "silu") {
            RunSiluMul(tokens, warmup, iterations, variant == "row");
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
