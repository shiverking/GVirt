#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "acl/acl.h"
#include "aclrtlaunch_asr_attention_partition_merge_probe.h"

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kPartitions = 4;
constexpr uint32_t kPartitionTokens = 512;
constexpr uint32_t kStateStride = 144;
constexpr uint32_t kQDim = kQHeads * kHeadDim;
constexpr size_t kGuardElements = 16;
constexpr uint16_t kGuard = 0x7e00;

void Check(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS) {
        throw std::runtime_error(std::string(operation) +
                                 " failed, aclError=" +
                                 std::to_string(status));
    }
}

struct DeviceBuffer {
    void *ptr = nullptr;
    explicit DeviceBuffer(size_t bytes)
    { Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc"); }
    ~DeviceBuffer() { if (ptr != nullptr) (void)aclrtFree(ptr); }
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

std::vector<int32_t> ParseLengths(const std::string &text)
{
    std::vector<int32_t> values;
    std::stringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) values.push_back(std::stoi(token));
    if (values.empty() || values.size() > 20) {
        throw std::invalid_argument("batch must be in 1..20");
    }
    for (int32_t value : values) {
        if (value < 513 || value > 2048) {
            throw std::invalid_argument("KV length must be in 513..2048");
        }
    }
    return values;
}

double Score(uint32_t request, uint32_t head, uint32_t token)
{
    const int32_t raw = static_cast<int32_t>(
        (request * 29 + head * 17 + token * 7 + token / 31) % 41) - 20;
    return static_cast<double>(raw) / 8.0;
}

double Value(uint32_t request, uint32_t head, uint32_t token,
             uint32_t dim)
{
    const uint32_t mix = request * 19 + head * 13 + token * 5 +
                         dim * 11 + token / 17;
    const int32_t raw = static_cast<int32_t>(mix % 37) - 18;
    return static_cast<double>(raw) / 32.0;
}

struct Metrics {
    double cosine = 0.0;
    double maxAbs = 0.0;
    size_t maxIndex = 0;
    size_t mismatches = 0;
    size_t nonFinite = 0;
};

Metrics Compare(const std::vector<uint16_t> &actual,
                const std::vector<uint16_t> &expected)
{
    Metrics result;
    double dot = 0.0;
    double actualNorm = 0.0;
    double expectedNorm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = HalfValue(actual[i]);
        const double e = HalfValue(expected[i]);
        const double error = std::abs(a - e);
        if (error > result.maxAbs) {
            result.maxAbs = error;
            result.maxIndex = i;
        }
        result.mismatches += error > 0.01;
        result.nonFinite += !std::isfinite(a);
        dot += a * e;
        actualNorm += a * a;
        expectedNorm += e * e;
    }
    result.cosine = dot / std::sqrt(actualNorm * expectedNorm);
    return result;
}

void BuildStates(const std::vector<int32_t> &lengths,
                 std::vector<float> &states,
                 std::vector<uint16_t> &expected)
{
    const float poison = std::numeric_limits<float>::quiet_NaN();
    std::fill(states.begin(), states.end(), poison);
    const uint32_t batch = static_cast<uint32_t>(lengths.size());
    for (uint32_t request = 0; request < batch; ++request) {
        for (uint32_t head = 0; head < kQHeads; ++head) {
            const uint32_t item = request * kQHeads + head;
            for (uint32_t partition = 0; partition < kPartitions; ++partition) {
                float *state = states.data() +
                    (static_cast<size_t>(item) * kPartitions + partition) *
                    kStateStride;
                const uint32_t begin = partition * kPartitionTokens;
                const uint32_t end = std::min<uint32_t>(
                    static_cast<uint32_t>(lengths[request]),
                    begin + kPartitionTokens);
                if (begin >= end) {
                    state[1] = 0.0F;
                    continue;
                }
                double maximum = -std::numeric_limits<double>::infinity();
                for (uint32_t token = begin; token < end; ++token) {
                    maximum = std::max(maximum, Score(request, head, token));
                }
                std::vector<double> accumulator(kHeadDim, 0.0);
                double sum = 0.0;
                for (uint32_t token = begin; token < end; ++token) {
                    const double weight = std::exp(
                        Score(request, head, token) - maximum);
                    sum += weight;
                    for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                        accumulator[dim] += weight *
                            Value(request, head, token, dim);
                    }
                }
                state[0] = static_cast<float>(maximum);
                state[1] = static_cast<float>(sum);
                for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    state[2 + dim] = static_cast<float>(accumulator[dim]);
                }
            }

            double globalMax = -std::numeric_limits<double>::infinity();
            for (uint32_t partition = 0; partition < kPartitions; ++partition) {
                const float *state = states.data() +
                    (static_cast<size_t>(item) * kPartitions + partition) *
                    kStateStride;
                if (state[1] > 0.0F) globalMax = std::max(globalMax,
                                                          static_cast<double>(state[0]));
            }
            double denominator = 0.0;
            std::vector<double> merged(kHeadDim, 0.0);
            for (uint32_t partition = 0; partition < kPartitions; ++partition) {
                const float *state = states.data() +
                    (static_cast<size_t>(item) * kPartitions + partition) *
                    kStateStride;
                if (state[1] <= 0.0F) continue;
                const double scale = std::exp(static_cast<double>(state[0]) -
                                              globalMax);
                denominator += static_cast<double>(state[1]) * scale;
                for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    merged[dim] += static_cast<double>(state[2 + dim]) * scale;
                }
            }
            for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                expected[static_cast<size_t>(request) * kQDim +
                         head * kHeadDim + dim] =
                    HalfBits(static_cast<float>(merged[dim] / denominator));
            }
        }
    }
}

void Run(const std::vector<int32_t> &lengths, uint32_t warmup,
         uint32_t iterations)
{
    if (iterations == 0) throw std::invalid_argument("iterations must be positive");
    const uint32_t batch = static_cast<uint32_t>(lengths.size());
    const size_t workItems = static_cast<size_t>(batch) * kQHeads;
    std::vector<float> states(workItems * kPartitions * kStateStride);
    std::vector<uint16_t> expected(static_cast<size_t>(batch) * kQDim);
    BuildStates(lengths, states, expected);
    const std::vector<float> statesBefore = states;
    std::vector<uint16_t> guarded(expected.size() + 2 * kGuardElements, kGuard);

    DeviceBuffer statesDevice(states.size() * sizeof(float));
    DeviceBuffer outputDevice(guarded.size() * sizeof(uint16_t));
    Check(aclrtMemcpy(statesDevice.ptr, states.size() * sizeof(float),
                      states.data(), states.size() * sizeof(float),
                      ACL_MEMCPY_HOST_TO_DEVICE), "copy states");
    Check(aclrtMemcpy(outputDevice.ptr, guarded.size() * sizeof(uint16_t),
                      guarded.data(), guarded.size() * sizeof(uint16_t),
                      ACL_MEMCPY_HOST_TO_DEVICE), "copy guarded output");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create stream");
    void *output = static_cast<uint8_t *>(outputDevice.ptr) +
                   kGuardElements * sizeof(uint16_t);
    const uint32_t blockDim = std::min<uint32_t>(8, batch * kQHeads);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_attention_partition_merge_probe)
        (blockDim, stream, statesDevice.ptr, output, batch);
    };
    for (uint32_t i = 0; i < warmup; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "warmup sync");
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "timed sync");
    const auto end = std::chrono::steady_clock::now();
    Check(aclrtMemcpy(guarded.data(), guarded.size() * sizeof(uint16_t),
                      outputDevice.ptr, guarded.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "copy output back");
    Check(aclrtMemcpy(states.data(), states.size() * sizeof(float),
                      statesDevice.ptr, states.size() * sizeof(float),
                      ACL_MEMCPY_DEVICE_TO_HOST), "copy states back");
    Check(aclrtDestroyStream(stream), "destroy stream");

    size_t guardErrors = 0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        guardErrors += guarded[i] != kGuard;
        guardErrors += guarded[guarded.size() - 1 - i] != kGuard;
    }
    const std::vector<uint16_t> actual(guarded.begin() + kGuardElements,
                                       guarded.end() - kGuardElements);
    const Metrics metrics = Compare(actual, expected);
    const bool statesChanged = std::memcmp(states.data(), statesBefore.data(),
                                           states.size() * sizeof(float)) != 0;
    const double averageMs =
        std::chrono::duration<double, std::milli>(end - begin).count() /
        iterations;
    if (metrics.cosine < 0.999 || metrics.maxAbs > 0.01 ||
        metrics.nonFinite != 0 || guardErrors != 0 || statesChanged) {
        const size_t request = metrics.maxIndex / kQDim;
        const size_t within = metrics.maxIndex % kQDim;
        std::cerr << std::fixed << std::setprecision(8)
                  << "Partition-merge diagnostics: cosine=" << metrics.cosine
                  << ", max_abs=" << metrics.maxAbs
                  << " at request=" << request
                  << ", head=" << within / kHeadDim
                  << ", dim=" << within % kHeadDim
                  << ", mismatches=" << metrics.mismatches
                  << ", non_finite=" << metrics.nonFinite
                  << ", guard_errors=" << guardErrors
                  << ", states_changed=" << statesChanged << std::endl;
        throw std::runtime_error("partition merge contract failed");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR attention partition merge PASS: batch=" << batch
              << ", average_ms=" << averageMs
              << ", cosine=" << metrics.cosine
              << ", max_abs=" << metrics.maxAbs
              << ", guards=" << guardErrors
              << ", states_changed=" << statesChanged << std::endl;
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: " << argv[0]
                  << " KV_LENGTHS WARMUP ITERATIONS" << std::endl;
        return 2;
    }
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "set device");
        Run(ParseLengths(argv[1]), static_cast<uint32_t>(std::stoul(argv[2])),
            static_cast<uint32_t>(std::stoul(argv[3])));
        Check(aclrtResetDevice(0), "reset device");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR attention partition merge FAILED: "
                  << error.what() << std::endl;
        return 1;
    }
}
