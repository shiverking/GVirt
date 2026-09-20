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
#include "aclrtlaunch_asr_paged_decode_attention_fp16.h"

namespace {
constexpr uint32_t kHeadDim = 128;
constexpr uint32_t kQHeads = 16;
constexpr uint32_t kKvHeads = 8;
constexpr uint32_t kQkvDim = 4096;
constexpr uint32_t kQDim = 2048;
constexpr uint32_t kBlockSize = 128;
constexpr uint32_t kTableStride = 16;
constexpr uint16_t kHalfNan = 0x7e00;
constexpr size_t kGuardElements = 16;

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

std::vector<int32_t> ParseLengths(const std::string &value)
{
    std::vector<int32_t> result;
    std::stringstream stream(value);
    std::string token;
    while (std::getline(stream, token, ',')) result.push_back(std::stoi(token));
    if (result.empty()) throw std::invalid_argument("empty KV length list");
    for (int32_t length : result) {
        if (length < 1 || length > 2048) throw std::invalid_argument("KV length outside 1..2048");
    }
    return result;
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

float CacheValue(uint32_t block, uint32_t token, uint32_t head, uint32_t dim,
                 bool value)
{
    const uint32_t mix = block * 19 + token * 7 + head * 13 + dim * (value ? 5 : 3);
    const int32_t raw = static_cast<int32_t>(mix % (value ? 37 : 31)) - (value ? 18 : 15);
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
    double dot = 0.0, actualNorm = 0.0, expectedNorm = 0.0;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = HalfValue(actual[i]);
        const double e = HalfValue(expected[i]);
        const double error = std::abs(a - e);
        if (error > result.maxAbs) { result.maxAbs = error; result.maxIndex = i; }
        result.mismatches += error > 0.01;
        result.nonFinite += !std::isfinite(a);
        dot += a * e; actualNorm += a * a; expectedNorm += e * e;
    }
    result.cosine = dot / std::sqrt(actualNorm * expectedNorm);
    return result;
}

void Run(const std::vector<int32_t> &lengths, uint32_t warmup, uint32_t iterations)
{
    const uint32_t batch = static_cast<uint32_t>(lengths.size());
    if (batch > 20 || iterations == 0) throw std::invalid_argument("invalid batch/iterations");
    const uint32_t numBlocks = batch * kTableStride + 7;
    const size_t cacheElements = static_cast<size_t>(numBlocks) * kBlockSize * kKvHeads * kHeadDim;
    std::vector<uint16_t> qkv(static_cast<size_t>(batch) * kQkvDim, HalfBits(-0.75F));
    std::vector<uint16_t> kCache(cacheElements), vCache(cacheElements);
    std::vector<int32_t> blockTable(static_cast<size_t>(batch) * kTableStride, -1);
    std::vector<uint16_t> expected(static_cast<size_t>(batch) * kQDim);
    // Keep the payload 32-byte aligned while protecting both adjacent regions.
    std::vector<uint16_t> guarded(expected.size() + 2 * kGuardElements, kHalfNan);

    for (uint32_t request = 0; request < batch; ++request) {
        for (uint32_t head = 0; head < kQHeads; ++head) {
            for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                qkv[static_cast<size_t>(request) * kQkvDim + head * kHeadDim + dim] =
                    HalfBits(QueryValue(request, head, dim));
            }
        }
        for (uint32_t logical = 0; logical < kTableStride; ++logical) {
            // A coprime affine permutation creates non-contiguous physical blocks.
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
                const uint32_t block = static_cast<uint32_t>(blockTable[
                    static_cast<size_t>(request) * kTableStride + token / kBlockSize]);
                double score = 0.0;
                for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    const float q = HalfValue(qkv[static_cast<size_t>(request) * kQkvDim +
                                                  queryHead * kHeadDim + dim]);
                    const float k = HalfValue(kCache[CacheIndex(block, token % kBlockSize,
                                                                kvHead, dim)]);
                    score += static_cast<double>(q) * k;
                }
                scores[token] = score;
                maximum = std::max(maximum, score);
            }
            double denominator = 0.0;
            for (double &score : scores) { score = std::exp(score - maximum); denominator += score; }
            for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                double value = 0.0;
                for (int32_t token = 0; token < lengths[request]; ++token) {
                    const uint32_t block = static_cast<uint32_t>(blockTable[
                        static_cast<size_t>(request) * kTableStride + token / kBlockSize]);
                    value += scores[token] * HalfValue(vCache[CacheIndex(
                        block, token % kBlockSize, kvHead, dim)]);
                }
                expected[static_cast<size_t>(request) * kQDim + queryHead * kHeadDim + dim] =
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
    Check(aclrtMemcpy(qkvDevice.ptr, qkv.size() * sizeof(uint16_t), qkv.data(),
                     qkv.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy qkv");
    Check(aclrtMemcpy(kDevice.ptr, kCache.size() * sizeof(uint16_t), kCache.data(),
                     kCache.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy K cache");
    Check(aclrtMemcpy(vDevice.ptr, vCache.size() * sizeof(uint16_t), vCache.data(),
                     vCache.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy V cache");
    Check(aclrtMemcpy(tableDevice.ptr, blockTable.size() * sizeof(int32_t), blockTable.data(),
                     blockTable.size() * sizeof(int32_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy table");
    std::vector<int32_t> cachedLengths = lengths;
    for (int32_t &length : cachedLengths) --length;
    Check(aclrtMemcpy(lengthsDevice.ptr, cachedLengths.size() * sizeof(int32_t),
                     cachedLengths.data(), cachedLengths.size() * sizeof(int32_t),
                     ACL_MEMCPY_HOST_TO_DEVICE), "copy cached lengths");
    Check(aclrtMemcpy(outputDevice.ptr, guarded.size() * sizeof(uint16_t), guarded.data(),
                     guarded.size() * sizeof(uint16_t), ACL_MEMCPY_HOST_TO_DEVICE), "copy output");

    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "aclrtCreateStream");
    void *output = static_cast<void *>(static_cast<uint8_t *>(outputDevice.ptr) +
                                      kGuardElements * sizeof(uint16_t));
    const uint32_t blockDim = std::min(7U, batch * kKvHeads);
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_paged_decode_attention_fp16)
        (blockDim, stream, qkvDevice.ptr, kDevice.ptr, vDevice.ptr, tableDevice.ptr,
         lengthsDevice.ptr, output, batch, kTableStride);
    };
    for (uint32_t i = 0; i < warmup; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    const auto started = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "timed synchronize");
    const auto stopped = std::chrono::steady_clock::now();
    Check(aclrtMemcpy(guarded.data(), guarded.size() * sizeof(uint16_t), outputDevice.ptr,
                     guarded.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy output back");
    Check(aclrtMemcpy(kCache.data(), kCache.size() * sizeof(uint16_t), kDevice.ptr,
                     kCache.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy K back");
    Check(aclrtMemcpy(vCache.data(), vCache.size() * sizeof(uint16_t), vDevice.ptr,
                     vCache.size() * sizeof(uint16_t), ACL_MEMCPY_DEVICE_TO_HOST), "copy V back");
    Check(aclrtDestroyStream(stream), "aclrtDestroyStream");

    size_t guardErrors = 0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        guardErrors += guarded[i] != kHalfNan;
        guardErrors += guarded[guarded.size() - 1 - i] != kHalfNan;
    }
    std::vector<uint16_t> actual(guarded.begin() + kGuardElements,
                                 guarded.end() - kGuardElements);
    const Metrics metrics = Compare(actual, expected);
    const size_t cacheMismatches =
        static_cast<size_t>(!std::equal(kCache.begin(), kCache.end(), kBefore.begin())) +
        static_cast<size_t>(!std::equal(vCache.begin(), vCache.end(), vBefore.begin()));
    const double averageMs = std::chrono::duration<double, std::milli>(stopped - started).count() /
                             iterations;
    if (metrics.nonFinite != 0 || metrics.cosine < 0.999 || metrics.maxAbs > 0.01 ||
        guardErrors != 0 || cacheMismatches != 0) {
        const size_t request = metrics.maxIndex / kQDim;
        const size_t within = metrics.maxIndex % kQDim;
        std::cerr << std::fixed << std::setprecision(8)
                  << "ASR paged attention diagnostics: cosine=" << metrics.cosine
                  << ", max_abs=" << metrics.maxAbs << " at request=" << request
                  << ", head=" << within / kHeadDim << ", dim=" << within % kHeadDim
                  << ", mismatches=" << metrics.mismatches
                  << ", non_finite=" << metrics.nonFinite
                  << ", guard_errors=" << guardErrors
                  << ", cache_mismatch_arrays=" << cacheMismatches << std::endl;
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
        throw std::runtime_error("paged decode attention contract failed");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR paged attention probe PASS: batch=" << batch
              << ", kv_lengths=";
    for (size_t i = 0; i < lengths.size(); ++i) std::cout << (i == 0 ? "" : ",") << lengths[i];
    std::cout << ", cores=" << blockDim << ", average_ms=" << averageMs
              << ", cosine=" << metrics.cosine << ", max_abs=" << metrics.maxAbs
              << ", guard_errors=" << guardErrors
              << ", cache_mismatch_arrays=" << cacheMismatches << std::endl;
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cerr << "usage: " << argv[0] << " KV_LENGTHS WARMUP ITERATIONS" << std::endl;
        return 2;
    }
    try {
        Check(aclInit(nullptr), "aclInit");
        Check(aclrtSetDevice(0), "aclrtSetDevice");
        Run(ParseLengths(argv[1]), static_cast<uint32_t>(std::stoul(argv[2])),
            static_cast<uint32_t>(std::stoul(argv[3])));
        Check(aclrtResetDevice(0), "aclrtResetDevice");
        Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR paged attention probe FAILED: " << error.what() << std::endl;
        return 1;
    }
}
