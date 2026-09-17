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
#include "aclrtlaunch_asr_paged_kv_stage_probe.h"

namespace {
constexpr uint32_t kHeadDim = 128, kHeads = 8, kBlockSize = 128;
constexpr uint32_t kTableStride = 16, kTileTokens = 16;
constexpr size_t kTileElements = kTileTokens * kHeadDim, kGuardElements = 16;
constexpr uint16_t kGuard = 0x7e00;

void Check(aclError status, const char *op)
{ if (status != ACL_SUCCESS) throw std::runtime_error(std::string(op) + " failed, aclError=" + std::to_string(status)); }
struct Buffer {
    void *ptr = nullptr;
    explicit Buffer(size_t bytes) { Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc"); }
    ~Buffer() { if (ptr) (void)aclrtFree(ptr); }
    Buffer(const Buffer &) = delete;
};
uint16_t HalfBits(float x) { const __fp16 y = static_cast<__fp16>(x); uint16_t b; std::memcpy(&b, &y, 2); return b; }
float HalfValue(uint16_t b) { __fp16 x; std::memcpy(&x, &b, 2); return static_cast<float>(x); }
size_t CacheIndex(uint32_t block, uint32_t token, uint32_t head, uint32_t dim)
{ return (((static_cast<size_t>(block) * kBlockSize + token) * kHeads + head) * kHeadDim + dim); }
float Value(uint32_t block, uint32_t token, uint32_t head, uint32_t dim, bool value)
{
    const uint32_t mixed = block * 23 + token * 11 + head * 7 + dim * (value ? 5 : 3);
    return static_cast<float>(static_cast<int32_t>(mixed % 41) - 20) / (value ? 64.0F : 96.0F);
}

struct Metrics { double maxAbs = 0.0; size_t maxIndex = 0, mismatches = 0, nonFinite = 0; };
Metrics Compare(const std::vector<uint16_t> &actual, const std::vector<uint16_t> &expected)
{
    Metrics result;
    for (size_t i = 0; i < actual.size(); ++i) {
        const double a = HalfValue(actual[i]), e = HalfValue(expected[i]), error = std::abs(a - e);
        if (error > result.maxAbs) { result.maxAbs = error; result.maxIndex = i; }
        result.mismatches += error != 0.0; result.nonFinite += !std::isfinite(a);
    }
    return result;
}
size_t GuardErrors(const std::vector<uint16_t> &buffer)
{
    size_t errors = 0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        errors += buffer[i] != kGuard; errors += buffer[buffer.size() - 1 - i] != kGuard;
    }
    return errors;
}

void Run(uint32_t start, uint32_t valid, uint32_t head, uint32_t warmup, uint32_t iterations)
{
    if (valid == 0 || valid > 16 || start + valid > 2048 || head >= 8 || iterations == 0)
        throw std::invalid_argument("invalid staging case");
    constexpr uint32_t numBlocks = 23;
    const size_t cacheElements = static_cast<size_t>(numBlocks) * kBlockSize * kHeads * kHeadDim;
    std::vector<uint16_t> kCache(cacheElements), vCache(cacheElements);
    std::vector<int32_t> table(kTableStride);
    for (uint32_t logical = 0; logical < kTableStride; ++logical)
        table[logical] = static_cast<int32_t>((logical * 17 + 3) % numBlocks);
    for (uint32_t block = 0; block < numBlocks; ++block)
        for (uint32_t token = 0; token < kBlockSize; ++token)
            for (uint32_t h = 0; h < kHeads; ++h)
                for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
                    const size_t index = CacheIndex(block, token, h, dim);
                    kCache[index] = HalfBits(Value(block, token, h, dim, false));
                    vCache[index] = HalfBits(Value(block, token, h, dim, true));
                }
    const auto kBefore = kCache, vBefore = vCache;
    std::vector<uint16_t> expectedK(kTileElements, HalfBits(0.0F));
    std::vector<uint16_t> expectedVt(kTileElements, HalfBits(0.0F));
    for (uint32_t token = 0; token < valid; ++token) {
        const uint32_t logicalToken = start + token;
        const uint32_t block = static_cast<uint32_t>(table[logicalToken / kBlockSize]);
        const uint32_t inBlock = logicalToken % kBlockSize;
        for (uint32_t dim = 0; dim < kHeadDim; ++dim) {
            expectedK[token * kHeadDim + dim] = kCache[CacheIndex(block, inBlock, head, dim)];
            expectedVt[dim * kTileTokens + token] = vCache[CacheIndex(block, inBlock, head, dim)];
        }
    }
    std::vector<uint16_t> kGuarded(kTileElements + 2 * kGuardElements, kGuard);
    std::vector<uint16_t> vGuarded(kTileElements + 2 * kGuardElements, kGuard);
    Buffer kd(kCache.size() * 2), vd(vCache.size() * 2), td(table.size() * 4);
    Buffer ko(kGuarded.size() * 2), vo(vGuarded.size() * 2);
    Check(aclrtMemcpy(kd.ptr, kCache.size() * 2, kCache.data(), kCache.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy K");
    Check(aclrtMemcpy(vd.ptr, vCache.size() * 2, vCache.data(), vCache.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy V");
    Check(aclrtMemcpy(td.ptr, table.size() * 4, table.data(), table.size() * 4, ACL_MEMCPY_HOST_TO_DEVICE), "copy table");
    Check(aclrtMemcpy(ko.ptr, kGuarded.size() * 2, kGuarded.data(), kGuarded.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy K output");
    Check(aclrtMemcpy(vo.ptr, vGuarded.size() * 2, vGuarded.data(), vGuarded.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy V output");
    void *kp = static_cast<uint8_t *>(ko.ptr) + kGuardElements * 2;
    void *vp = static_cast<uint8_t *>(vo.ptr) + kGuardElements * 2;
    aclrtStream stream = nullptr; Check(aclrtCreateStream(&stream), "create stream");
    const auto launch = [&]() { ACLRT_LAUNCH_KERNEL(asr_paged_kv_stage_probe)
        (1, stream, kd.ptr, vd.ptr, td.ptr, kp, vp, start, valid, head); };
    for (uint32_t i = 0; i < warmup; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "warmup sync");
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "timed sync");
    const auto end = std::chrono::steady_clock::now();
    Check(aclrtMemcpy(kGuarded.data(), kGuarded.size() * 2, ko.ptr, kGuarded.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST), "copy K output back");
    Check(aclrtMemcpy(vGuarded.data(), vGuarded.size() * 2, vo.ptr, vGuarded.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST), "copy V output back");
    Check(aclrtMemcpy(kCache.data(), kCache.size() * 2, kd.ptr, kCache.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST), "copy K back");
    Check(aclrtMemcpy(vCache.data(), vCache.size() * 2, vd.ptr, vCache.size() * 2, ACL_MEMCPY_DEVICE_TO_HOST), "copy V back");
    Check(aclrtDestroyStream(stream), "destroy stream");
    std::vector<uint16_t> actualK(kGuarded.begin() + kGuardElements, kGuarded.end() - kGuardElements);
    std::vector<uint16_t> actualV(vGuarded.begin() + kGuardElements, vGuarded.end() - kGuardElements);
    const Metrics km = Compare(actualK, expectedK), vm = Compare(actualV, expectedVt);
    const size_t guards = GuardErrors(kGuarded) + GuardErrors(vGuarded);
    const size_t cacheArraysChanged = !std::equal(kCache.begin(), kCache.end(), kBefore.begin()) +
                                      !std::equal(vCache.begin(), vCache.end(), vBefore.begin());
    const double ms = std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
    if (km.mismatches || vm.mismatches || km.nonFinite || vm.nonFinite || guards || cacheArraysChanged) {
        std::cerr << "Paged KV stage diagnostics: start=" << start << ", valid=" << valid
                  << ", head=" << head << ", K_mismatches=" << km.mismatches
                  << ", K_max_abs=" << km.maxAbs << " at token=" << km.maxIndex / kHeadDim
                  << ", dim=" << km.maxIndex % kHeadDim << ", Vt_mismatches=" << vm.mismatches
                  << ", Vt_max_abs=" << vm.maxAbs << " at dim=" << vm.maxIndex / kTileTokens
                  << ", token=" << vm.maxIndex % kTileTokens << ", guard_errors=" << guards
                  << ", cache_arrays_changed=" << cacheArraysChanged << std::endl;
        throw std::runtime_error("paged KV staging contract failed");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR paged KV stage PASS: start=" << start << ", valid=" << valid
              << ", head=" << head << ", average_ms=" << ms
              << ", guard_errors=" << guards << ", cache_arrays_changed=" << cacheArraysChanged << std::endl;
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc != 6) { std::cerr << "usage: " << argv[0] << " START VALID HEAD WARMUP ITERATIONS\n"; return 2; }
    try {
        Check(aclInit(nullptr), "aclInit"); Check(aclrtSetDevice(0), "set device");
        Run(std::stoul(argv[1]), std::stoul(argv[2]), std::stoul(argv[3]), std::stoul(argv[4]), std::stoul(argv[5]));
        Check(aclrtResetDevice(0), "reset device"); Check(aclFinalize(), "aclFinalize"); return 0;
    } catch (const std::exception &error) { std::cerr << "ASR paged KV stage FAILED: " << error.what() << std::endl; return 1; }
}
