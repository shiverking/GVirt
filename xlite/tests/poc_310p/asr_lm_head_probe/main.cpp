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

float InputValue(uint32_t row)
{
    static constexpr float values[] = {1.0F, 0.5F, -0.25F, 0.125F};
    return values[row & 3U];
}

float WeightValue(uint32_t row)
{
    static constexpr float values[] = {0.25F, 0.5F, 1.0F, -0.5F};
    return values[row & 3U];
}

void UploadWeight(void *device)
{
    const size_t maxElements = static_cast<size_t>(kWeightChunkRows) * kHidden;
    std::vector<uint16_t> chunk(maxElements);
    for (uint32_t first = 0; first < kVocabulary; first += kWeightChunkRows) {
        const uint32_t rows = std::min(kWeightChunkRows, kVocabulary - first);
        for (uint32_t row = 0; row < rows; ++row) {
            const uint16_t value = HalfBits(WeightValue(first + row));
            std::fill_n(chunk.begin() + static_cast<size_t>(row) * kHidden,
                        kHidden, value);
        }
        const size_t bytes = static_cast<size_t>(rows) * kHidden * sizeof(uint16_t);
        auto *destination = static_cast<uint8_t *>(device) +
                            static_cast<size_t>(first) * kHidden * sizeof(uint16_t);
        Check(aclrtMemcpy(destination, bytes, chunk.data(), bytes,
                         ACL_MEMCPY_HOST_TO_DEVICE), "copy weight chunk");
    }
}

void Run(uint32_t batch, uint32_t warmup, uint32_t iterations)
{
    if (batch == 0 || batch > 20 || iterations == 0) {
        throw std::invalid_argument("batch must be 1..20 and iterations nonzero");
    }
    std::vector<uint16_t> input(static_cast<size_t>(batch) * kHidden);
    for (uint32_t row = 0; row < batch; ++row) {
        std::fill_n(input.begin() + static_cast<size_t>(row) * kHidden,
                    kHidden, HalfBits(InputValue(row)));
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
        ACLRT_LAUNCH_KERNEL(asr_m200_lm_head_fp16)
        (blockDim, stream, inputDevice.ptr, weightDevice.ptr, logits, batch);
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
    for (uint32_t row = 0; row < batch; ++row) {
        const float inputValue = InputValue(row);
        for (uint32_t col = 0; col < kVocabulary; ++col) {
            const uint16_t actual = guardedOutput[
                kGuardElements + static_cast<size_t>(row) * kVocabulary + col];
            const uint16_t expected = HalfBits(inputValue * WeightValue(col) * kHidden);
            sentinels += actual == kHalfNan;
            nonfinite += !std::isfinite(HalfValue(actual));
            if (actual != expected) {
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
    if (guardErrors != 0 || mismatches != 0 || sentinels != 0 || nonfinite != 0) {
        throw std::runtime_error(
            "full-logit contract failed: mismatches=" + std::to_string(mismatches) +
            ", sentinels=" + std::to_string(sentinels) +
            ", nonfinite=" + std::to_string(nonfinite) +
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
              << ", average_ms=" << averageMs << ", tflops=" << tflops
              << ", guard_errors=" << guardErrors << std::endl;
}

}  // namespace

int main(int argc, char **argv)
{
    try {
        if (argc != 4) {
            throw std::invalid_argument("usage: runner BATCH WARMUP ITERATIONS");
        }
        // Device 0 is the process-local logical device selected by the
        // deployment.  This probe intentionally does not rewrite visibility
        // or logical-to-physical mapping.
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice(0)");
        Run(static_cast<uint32_t>(std::stoul(argv[1])),
            static_cast<uint32_t>(std::stoul(argv[2])),
            static_cast<uint32_t>(std::stoul(argv[3])));
        Check(aclrtResetDevice(0), "aclrtResetDevice(0)");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR LM Head FAILED: " << error.what() << std::endl;
        return 1;
    }
}
