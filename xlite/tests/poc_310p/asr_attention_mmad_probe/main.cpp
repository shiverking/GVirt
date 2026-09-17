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
#include "aclrtlaunch_asr_attention_mmad_block_probe.h"

namespace {
constexpr uint16_t kGuard = 0x7e00;
constexpr size_t kGuardElements = 16;

void Check(aclError status, const char *operation)
{
    if (status != ACL_SUCCESS) throw std::runtime_error(
        std::string(operation) + " failed, aclError=" + std::to_string(status));
}
struct Buffer {
    void *ptr = nullptr;
    explicit Buffer(size_t bytes) { Check(aclrtMalloc(&ptr, bytes, ACL_MEM_MALLOC_HUGE_FIRST), "aclrtMalloc"); }
    ~Buffer() { if (ptr != nullptr) (void)aclrtFree(ptr); }
    Buffer(const Buffer &) = delete;
};
uint16_t HalfBits(float x) { const __fp16 y = static_cast<__fp16>(x); uint16_t b; std::memcpy(&b, &y, 2); return b; }
float HalfValue(uint16_t b) { __fp16 x; std::memcpy(&x, &b, 2); return static_cast<float>(x); }

void Run(const std::string &operation, uint32_t warmup, uint32_t iterations)
{
    const bool qk = operation == "qk";
    if (!qk && operation != "pv") throw std::invalid_argument("operation must be qk or pv");
    const uint32_t m = 16, n = qk ? 16 : 128, k = qk ? 128 : 16;
    std::vector<uint16_t> a(static_cast<size_t>(m) * k);
    std::vector<uint16_t> b(static_cast<size_t>(n) * k);
    std::vector<uint16_t> expected(static_cast<size_t>(m) * n);
    std::vector<uint16_t> guarded(expected.size() + 2 * kGuardElements, kGuard);

    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < k; ++col) {
            const int32_t raw = static_cast<int32_t>((row * 7 + col * 3) % 23) - 11;
            a[static_cast<size_t>(row) * k + col] = HalfBits(static_cast<float>(raw) / 64.0F);
        }
    }
    // B is [N,K]. QK rows represent 16 cache tokens; PV rows represent
    // 128 value dimensions gathered from the same 16-token BSHD tile.
    for (uint32_t row = 0; row < n; ++row) {
        for (uint32_t col = 0; col < k; ++col) {
            const int32_t raw = static_cast<int32_t>((row * 11 + col * 5 + 3) % 29) - 14;
            b[static_cast<size_t>(row) * k + col] = HalfBits(static_cast<float>(raw) / 96.0F);
        }
    }
    for (uint32_t row = 0; row < m; ++row) {
        for (uint32_t col = 0; col < n; ++col) {
            float sum = 0.0F;
            for (uint32_t inner = 0; inner < k; ++inner) {
                sum += HalfValue(a[static_cast<size_t>(row) * k + inner]) *
                       HalfValue(b[static_cast<size_t>(col) * k + inner]);
            }
            expected[static_cast<size_t>(row) * n + col] = HalfBits(sum);
        }
    }

    Buffer aDevice(a.size() * 2), bDevice(b.size() * 2), cDevice(guarded.size() * 2);
    Check(aclrtMemcpy(aDevice.ptr, a.size() * 2, a.data(), a.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy A");
    Check(aclrtMemcpy(bDevice.ptr, b.size() * 2, b.data(), b.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy B");
    Check(aclrtMemcpy(cDevice.ptr, guarded.size() * 2, guarded.data(), guarded.size() * 2, ACL_MEMCPY_HOST_TO_DEVICE), "copy C");
    aclrtStream stream = nullptr;
    Check(aclrtCreateStream(&stream), "create stream");
    void *payload = static_cast<uint8_t *>(cDevice.ptr) + kGuardElements * 2;
    const auto launch = [&]() {
        ACLRT_LAUNCH_KERNEL(asr_attention_mmad_block_probe)
        (1, stream, aDevice.ptr, bDevice.ptr, payload, m, n, k);
    };
    for (uint32_t i = 0; i < warmup; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "warmup synchronize");
    const auto begin = std::chrono::steady_clock::now();
    for (uint32_t i = 0; i < iterations; ++i) launch();
    Check(aclrtSynchronizeStream(stream), "timed synchronize");
    const auto end = std::chrono::steady_clock::now();
    Check(aclrtMemcpy(guarded.data(), guarded.size() * 2, cDevice.ptr, guarded.size() * 2,
                     ACL_MEMCPY_DEVICE_TO_HOST), "copy C back");
    Check(aclrtDestroyStream(stream), "destroy stream");

    size_t guardErrors = 0, mismatches = 0, nonFinite = 0, maxIndex = 0;
    double dot = 0.0, an = 0.0, en = 0.0, maxAbs = 0.0;
    for (size_t i = 0; i < kGuardElements; ++i) {
        guardErrors += guarded[i] != kGuard;
        guardErrors += guarded[guarded.size() - 1 - i] != kGuard;
    }
    for (size_t i = 0; i < expected.size(); ++i) {
        const double actual = HalfValue(guarded[kGuardElements + i]);
        const double reference = HalfValue(expected[i]);
        const double error = std::abs(actual - reference);
        if (error > maxAbs) { maxAbs = error; maxIndex = i; }
        mismatches += error > 0.02;
        nonFinite += !std::isfinite(actual);
        dot += actual * reference; an += actual * actual; en += reference * reference;
    }
    const double cosine = dot / std::sqrt(an * en);
    const double averageMs = std::chrono::duration<double, std::milli>(end - begin).count() / iterations;
    if (guardErrors || nonFinite || cosine < 0.999 || maxAbs > 0.02) {
        std::cerr << std::fixed << std::setprecision(8)
                  << "Attention MMAD diagnostics: op=" << operation
                  << ", cosine=" << cosine << ", max_abs=" << maxAbs
                  << " at row=" << maxIndex / n << ", col=" << maxIndex % n
                  << ", mismatches=" << mismatches << ", non_finite=" << nonFinite
                  << ", guard_errors=" << guardErrors << std::endl;
        throw std::runtime_error("attention MMAD block contract failed");
    }
    std::cout << std::fixed << std::setprecision(6)
              << "ASR attention MMAD block PASS: op=" << operation
              << ", M=" << m << ", N=" << n << ", K=" << k
              << ", average_ms=" << averageMs << ", cosine=" << cosine
              << ", max_abs=" << maxAbs << ", guard_errors=" << guardErrors << std::endl;
}
}  // namespace

int main(int argc, char **argv)
{
    if (argc != 4) { std::cerr << "usage: " << argv[0] << " qk|pv WARMUP ITERATIONS\n"; return 2; }
    try {
        Check(aclInit(nullptr), "aclInit"); Check(aclrtSetDevice(0), "set device");
        Run(argv[1], static_cast<uint32_t>(std::stoul(argv[2])), static_cast<uint32_t>(std::stoul(argv[3])));
        Check(aclrtResetDevice(0), "reset device"); Check(aclFinalize(), "aclFinalize");
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ASR attention MMAD block FAILED: " << error.what() << std::endl;
        return 1;
    }
}
