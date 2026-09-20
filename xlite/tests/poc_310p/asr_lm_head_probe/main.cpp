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
#include "aclrtlaunch_asr_m200_lm_head_fp16.h"
#include "aclrtlaunch_asr_m200_lm_head_cached_fp16.h"

namespace {

constexpr uint32_t kHidden = 2048;
constexpr uint32_t kVocabulary = 151936;
constexpr uint32_t kWeightChunkRows = 12288;
constexpr size_t kGuardElements = 256;
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

float InputValue(uint32_t row, uint32_t k)
{
    return static_cast<float>(static_cast<int>((k * 13 + row * 7) % 17) - 8) / 8.0F;
}

float WeightValue(uint32_t row)
{
    return static_cast<float>(static_cast<int>((row * 37) % 251) - 125) / 256.0F;
}

float WeightKValue(uint32_t k)
{
    static constexpr float values[] = {1.0F, -1.0F, 0.5F, -0.5F};
    return values[(k / 128 + k / 16 + k) % 4];
}

void UploadWeight(void *device)
{
    const size_t maxElements = static_cast<size_t>(kWeightChunkRows) * kHidden;
    std::vector<uint16_t> chunk(maxElements);
    for (uint32_t first = 0; first < kVocabulary; first += kWeightChunkRows) {
        const uint32_t rows = std::min(kWeightChunkRows, kVocabulary - first);
        for (uint32_t row = 0; row < rows; ++row) {
            for (uint32_t k = 0; k < kHidden; ++k) {
                chunk[static_cast<size_t>(row) * kHidden + k] =
                    HalfBits(WeightValue(first + row) * WeightKValue(k));
            }
        }
        const size_t bytes = static_cast<size_t>(rows) * kHidden * sizeof(uint16_t);
        auto *destination = static_cast<uint8_t *>(device) +
                            static_cast<size_t>(first) * kHidden * sizeof(uint16_t);
        Check(aclrtMemcpy(destination, bytes, chunk.data(), bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE), "copy weight chunk");
    }
}

void Run(uint32_t batch, uint32_t warmup, uint32_t iterations, bool cached)
{
    if (batch == 0 || batch > 20 || iterations == 0) {
        throw std::invalid_argument("batch must be 1..20 and iterations nonzero");
    }
    std::vector<uint16_t> input(static_cast<size_t>(batch) * kHidden);
    std::vector<float> rowDots(batch, 0.0F);
    for (uint32_t row = 0; row < batch; ++row) {
        for (uint32_t k = 0; k < kHidden; ++k) {
            input[static_cast<size_t>(row) * kHidden + k] = HalfBits(InputValue(row, k));
            rowDots[row] += InputValue(row, k) * WeightKValue(k);
        }
    }

    const size_t outputElements = static_cast<size_t>(batch) * kVocabulary;
    std::vector<uint16_t> guardedOutput(outputElements + 2 * kGuardElements,
                                        kHalfNan);
    DeviceBuffer inputDevice(input.size() * sizeof(uint16_t));
    DeviceBuffer weightDevice(static_cast<size_t>(kVocabulary) * kHidden *
                              sizeof(uint16_t));
    DeviceBuffer outputDevice(guardedOutput.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(inputDevice.ptr, input.size() * sizeof(uint16_t),
                     input.data(), input.size() * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "copy input");
    UploadWeight(weightDevice.ptr);
    Check(aclrtMemcpy(outputDevice.ptr, guardedOutput.size() * sizeof(uint16_t),
                     guardedOutput.data(), guardedOutput.size() * sizeof(uint16_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "initialize guarded output");

    auto *logits = static_cast<uint8_t *>(outputDevice.ptr) +
                   kGuardElements * sizeof(uint16_t);
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    constexpr uint32_t blockDim = 8;
    const auto launch = [&]() {
        if (cached) {
            ACLRT_LAUNCH_KERNEL(asr_m200_lm_head_cached_fp16)
            (blockDim, stream, inputDevice.ptr, weightDevice.ptr, logits, batch);
        } else {
            ACLRT_LAUNCH_KERNEL(asr_m200_lm_head_fp16)
            (blockDim, stream, inputDevice.ptr, weightDevice.ptr, logits, batch);
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
    Check(aclrtMemcpy(guardedOutput.data(),
                     guardedOutput.size() * sizeof(uint16_t), outputDevice.ptr,
                     guardedOutput.size() * sizeof(uint16_t),
                     ACL_MEMCPY_DEVICE_TO_HOST), "copy guarded output");

    size_t guardErrors = 0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        guardErrors += guardedOutput[i] != kHalfNan;
        guardErrors += guardedOutput[kGuardElements + outputElements + i] != kHalfNan;
    }
    size_t mismatches = 0;
    size_t sentinels = 0;
    size_t nonfinite = 0;
    size_t reported = 0;
    double dot = 0.0, actualSquare = 0.0, expectedSquare = 0.0;
    float maxAbs = 0.0F;
    for (uint32_t row = 0; row < batch; ++row) {
        for (uint32_t col = 0; col < kVocabulary; ++col) {
            const uint16_t actual = guardedOutput[
                kGuardElements + static_cast<size_t>(row) * kVocabulary + col];
            const uint16_t expected = HalfBits(rowDots[row] * WeightValue(col));
            sentinels += actual == kHalfNan;
            nonfinite += !std::isfinite(HalfValue(actual));
            const float av = HalfValue(actual), ev = HalfValue(expected);
            maxAbs = std::max(maxAbs, std::abs(av - ev));
            dot += static_cast<double>(av) * ev;
            actualSquare += static_cast<double>(av) * av;
            expectedSquare += static_cast<double>(ev) * ev;
            if (!std::isfinite(av) || std::abs(av - ev) > 0.01F + 0.01F * std::abs(ev)) {
                ++mismatches;
                if (reported < 12) {
                    std::cerr << "mismatch[" << row << ',' << col << "] actual="
                              << HalfValue(actual) << " bits=0x" << std::hex
                              << actual << std::dec << " expected="
                              << HalfValue(expected) << " bits=0x" << std::hex
                              << expected << std::dec << '\n';
                    ++reported;
                }
            }
        }
    }
    const double cosine = dot / std::sqrt(actualSquare * expectedSquare);
    if (guardErrors != 0 || mismatches != 0 || sentinels != 0 || nonfinite != 0 ||
        !std::isfinite(cosine) || cosine < 0.999) {
        throw std::runtime_error(
            "full-logit contract failed: mismatches=" + std::to_string(mismatches) +
            ", sentinels=" + std::to_string(sentinels) +
            ", nonfinite=" + std::to_string(nonfinite) +
            ", cosine=" + std::to_string(cosine) +
            ", max_abs=" + std::to_string(maxAbs) +
            ", guard_errors=" + std::to_string(guardErrors));
    }

    const double elapsedMs =
        std::chrono::duration<double, std::milli>(stopped - started).count();
    const double averageMs = elapsedMs / static_cast<double>(iterations);
    const double tflops = 2.0 * batch * kVocabulary * kHidden /
                          (averageMs * 1.0e9);
    std::cout << std::fixed << std::setprecision(6)
              << "ASR LM Head PASS: M=" << batch
              << ", N=" << kVocabulary << ", K=" << kHidden
              << ", cores=" << blockDim << ", launches_per_iteration=1"
              << ", variant=" << (cached ? "cached" : "baseline")
              << ", average_ms=" << averageMs << ", tflops=" << tflops
              << ", cosine=" << cosine << ", max_abs=" << maxAbs
              << ", guard_errors=" << guardErrors << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 4 && argc != 5) {
            throw std::invalid_argument("usage: runner BATCH WARMUP ITERATIONS [baseline|cached]");
        }
        const std::string variant = argc == 5 ? argv[4] : "baseline";
        if (variant != "baseline" && variant != "cached") {
            throw std::invalid_argument("unknown LM Head variant: " + variant);
        }
        // Device 0 is the process-local logical device selected by the
        // deployment.  This probe intentionally does not rewrite visibility
        // or logical-to-physical mapping.
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice(0)");
        Run(static_cast<uint32_t>(std::stoul(argv[1])),
            static_cast<uint32_t>(std::stoul(argv[2])),
            static_cast<uint32_t>(std::stoul(argv[3])), variant == "cached");
        Check(aclrtResetDevice(0), "aclrtResetDevice(0)");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR LM Head FAILED: " << error.what() << std::endl;
        return 1;
    }
}
