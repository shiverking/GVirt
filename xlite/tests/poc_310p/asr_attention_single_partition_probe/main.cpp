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
#include "aclrtlaunch_asr_attention_single_partition_probe.h"

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQkvDim = 4096;
constexpr uint32_t kQDim = 2048;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kTableStride = 4;
constexpr size_t kGuardElements = 16;
constexpr uint16_t kGuard = 0x7e00;

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
        if (value < 1 || value > 512) {
            throw std::invalid_argument("KV length must be in 1..512");
        }
    }
    return values;
}

size_t CacheIndex(uint32_t block, uint32_t token, uint32_t head, uint32_t dim)
{
    return (((static_cast<size_t>(block) * kBlockSize + token) * kKvHeads + head) *
            kHeadDim + dim);
}

float QueryValue(uint32_t request, uint32_t head, uint32_t dim)
{
    const int32_t raw = static_cast<int32_t>((request * 17 + head * 11 + dim * 3) % 29) - 14;
    return static_cast<float>(raw) / 256.0F;
}

float CacheValue(uint32_t block, uint32_t token, uint32_t head,
                 uint32_t dim, bool value)
{
    const uint32_t mix = block * 19 + token * 7 + head * 13 +
                         dim * (value ? 5 : 3);
    const int32_t raw = static_cast<int32_t>(mix % (value ? 37 : 31)) -
                        (value ? 18 : 15);
    return static_cast<float>(raw) / (value ? 64.0F : 128.0F);
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

void CopyToDevice(DeviceBuffer &device, const void *host, size_t bytes,
                  const char *operation)
{
    Check(aclrtMemcpy(device.ptr, bytes, host, bytes,
                      ACL_MEMCPY_HOST_TO_DEVICE), operation);
}

void Run(const std::vector<int32_t> &lengths, uint32_t warmup,
         uint32_t iterations)
{
    if (iterations == 0) throw std::invalid_argument("iterations must be positive");
    const uint32_t batch = static_cast<uint32_t>(lengths.size());
    const uint32_t numBlocks = batch * kTableStride + 7;
    const size_t cacheElements = static_cast<size_t>(numBlocks) * kBlockSize *
                                 kKvHeads * kHeadDim;
    std::vector<uint16_t> qkv(static_cast<size_t>(batch) * kQkvDim,
                              HalfBits(-0.75F));
    std::vector<uint16_t> kCache(cacheElements);
    std::vector<uint16_t> vCache(cacheElements);
    std::vector<int32_t> blockTable(static_cast<size_t>(batch) * kTableStride, -1);
    std::vector<uint16_t> expected(static_cast<size_t>(batch) * kQDim);
    std::vector<uint16_t> guarded(expected.size() + 2 * kGuardElements, kGuard);

    for (uint32_t request = 0; request < batch; ++request) {
        for (uint32_t head = 0; head < kQHeads; ++head) {
            for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                qkv[static_cast<size_t>(request) * kQkvDim +
                    head * kHeadDim + dim] = HalfBits(QueryValue(request, head, dim));
            }
        }
        for (uint32_t logical = 0; logical < kTableStride; ++logical) {
            blockTable[static_cast<size_t>(request) * kTableStride + logical] =
                static_cast<int32_t>((request * 11 + logical * 17 + 3) % numBlocks);
        }
    }
    for (uint32_t block = 0; block < numBlocks; ++block) {
        for (uint32_t token = 0; token < kBlockSize; ++token) {
            for (uint32_t head = 0; head < kKvHeads; ++head) {
                for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    const size_t index = CacheIndex(block, token, head, dim);
                    kCache[index] = HalfBits(CacheValue(block, token, head, dim, false));
                    vCache[index] = HalfBits(CacheValue(block, token, head, dim, true));
                }
            }
        }
    }
    const std::vector<uint16_t> kBefore = kCache;
    const std::vector<uint16_t> vBefore = vCache;

    for (uint32_t request = 0; request < batch; ++request) {
        for (uint32_t queryHead = 0; queryHead < kQHeads; ++queryHead) {
            const uint32_t kvHead = queryHead / 2;
            std::vector<double> scores(lengths[request]);
            double maximum = -std::numeric_limits<double>::infinity();
            for (int32_t token = 0; token < lengths[request]; ++token) {
                const uint32_t physicalBlock = static_cast<uint32_t>(blockTable[
                    static_cast<size_t>(request) * kTableStride + token / kBlockSize]);
                double score = 0.0;
                for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    score += static_cast<double>(HalfValue(qkv[
                        static_cast<size_t>(request) * kQkvDim +
                        queryHead * kHeadDim + dim])) *
                        HalfValue(kCache[CacheIndex(physicalBlock,
                            token % kBlockSize, kvHead, dim)]);
                }
                scores[token] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            for (double &score : scores) {
                score = std::exp(score - maximum);
                denominator += score;
            }
            for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                double value = 0.0;
                for (int32_t token = 0; token < lengths[request]; ++token) {
                    const uint32_t physicalBlock = static_cast<uint32_t>(blockTable[
                        static_cast<size_t>(request) * kTableStride + token / kBlockSize]);
                    value += scores[token] * HalfValue(vCache[CacheIndex(
                        physicalBlock, token % kBlockSize, kvHead, dim)]);
                }
                expected[static_cast<size_t>(request) * kQDim +
                         queryHead * kHeadDim + dim] =
                    HalfBits(static_cast<float>(value / denominator));
            }
        }
    }

    DeviceBuffer qkvDevice(qkv.size() * sizeof(uint16_t));
    DeviceBuffer kDevice(kCache.size() * sizeof(uint16_t));
    DeviceBuffer vDevice(vCache.size() * sizeof(uint16_t));
    DeviceBuffer tableDevice(blockTable.size() * sizeof(int32_t));
    DeviceBuffer lengthsDevice(lengths.size() * sizeof(int32_t));
    DeviceBuffer outputDevice(guarded.size() * sizeof(uint16_t));
    CopyToDevice(qkvDevice, qkv.data(), qkv.size() * sizeof(uint16_t), "copy qkv");
    CopyToDevice(kDevice, kCache.data(), kCache.size() * sizeof(uint16_t), "copy K");
    CopyToDevice(vDevice, vCache.data(), vCache.size() * sizeof(uint16_t), "copy V");
    CopyToDevice(tableDevice, blockTable.data(), blockTable.size() * sizeof(int32_t),
                 "copy block table");
    CopyToDevice(lengthsDevice, lengths.data(), lengths.size() * sizeof(int32_t),
                 "copy lengths");
    CopyToDevice(outputDevice, guarded.data(), guarded.size() * sizeof(uint16_t),
                 "copy guarded output");

    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create stream");
    void *output = static_cast<uint8_t *>(outputDevice.ptr) +
                   kGuardElements * sizeof(uint16_t);
    const uint32_t blockDim = std::min(8U, batch * kQHeads);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_attention_single_partition_probe)
        (blockDim, stream, qkvDevice.ptr, kDevice.ptr, vDevice.ptr,
         tableDevice.ptr, lengthsDevice.ptr, output, batch, kTableStride);
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
    Check(aclrtMemcpy(kCache.data(), kCache.size() * sizeof(uint16_t),
                      kDevice.ptr, kCache.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "copy K back");
    Check(aclrtMemcpy(vCache.data(), vCache.size() * sizeof(uint16_t),
                      vDevice.ptr, vCache.size() * sizeof(uint16_t),
                      ACL_MEMCPY_DEVICE_TO_HOST), "copy V back");
    Check(aclrtDestroyStream(stream), "destroy stream");

    size_t guardErrors = 0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        guardErrors += guarded[i] != kGuard;
        guardErrors += guarded[guarded.size() - 1 - i] != kGuard;
    }
    const std::vector<uint16_t> actual(guarded.begin() + kGuardElements,
                                       guarded.end() - kGuardElements);
    const Metrics metrics = Compare(actual, expected);
    const size_t cacheChanged =
        static_cast<size_t>(!std::equal(kCache.begin(), kCache.end(), kBefore.begin())) +
        static_cast<size_t>(!std::equal(vCache.begin(), vCache.end(), vBefore.begin()));
    const double averageMs = std::chrono::duration<double, std::milli>(end - begin).count() /
                             iterations;
    if (metrics.cosine < 0.999 || metrics.maxAbs > 0.01 ||
        metrics.nonFinite != 0 || guardErrors != 0 || cacheChanged != 0) {
        const size_t request = metrics.maxIndex / kQDim;
        const size_t within = metrics.maxIndex % kQDim;
        std::cerr << std::fixed << std::setprecision(8)
                  << "Single-partition diagnostics: cosine=" << metrics.cosine
                  << ", max_abs=" << metrics.maxAbs
                  << " at request=" << request
                  << ", head=" << within / kHeadDim
                  << ", dim=" << within % kHeadDim
                  << ", mismatches=" << metrics.mismatches
                  << ", non_finite=" << metrics.nonFinite
                  << ", guard_errors=" << guardErrors
                  << ", cache_changed=" << cacheChanged << std::endl;
        size_t printed = 0;
        for (size_t i = 0; i < actual.size() && printed < 16; ++i) {
            const double error = std::abs(HalfValue(actual[i]) - HalfValue(expected[i]));
            if (error > 0.01) {
                std::cerr << "  mismatch request=" << i / kQDim
                          << " head=" << (i % kQDim) / kHeadDim
                          << " dim=" << i % kHeadDim
                          << " actual=" << HalfValue(actual[i])
                          << " expected=" << HalfValue(expected[i])
                          << " abs=" << error << std::endl;
                ++printed;
            }
        }
        throw std::runtime_error("single-partition attention contract failed");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR attention single-partition PASS: batch=" << batch
              << ", average_ms=" << averageMs
              << ", cosine=" << metrics.cosine
              << ", max_abs=" << metrics.maxAbs
              << ", guards=" << guardErrors
              << ", cache_changed=" << cacheChanged << std::endl;
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
        std::cerr << "ASR attention single-partition FAILED: "
                  << error.what() << std::endl;
        return 1;
    }
}
