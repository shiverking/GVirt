#pragma once

#include <cstdint>
#include <vector>

struct M200CubeProbeTiling {
    std::vector<uint8_t> bytes;
    uint32_t usedCores = 0;
    uint64_t localWorkspaceBytes = 0;
    uint64_t systemWorkspaceBytes = 0;
};

M200CubeProbeTiling GenerateM200CubeProbeTiling(uint32_t m, uint32_t n, uint32_t k);

