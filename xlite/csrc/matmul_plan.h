#pragma once
#include <cstdint>
#include <vector>

struct XMatmulChunk {
    int64_t offset;
    std::vector<int64_t> weightDims, weightStorage, weightStrides;
    std::vector<int64_t> outputDims, outputStrides, outputStorage;
};
struct XMatmulPlan {
    bool enabled = true, direct = false;
    int64_t chunkSize = 12288;
    std::vector<int64_t> inputDims, inputStrides;
    std::vector<XMatmulChunk> chunks;
};
struct XMatmulStats {
    uint64_t calls = 0, chunks = 0, copyBytes = 0, workspacePeak = 0, forcedSyncs = 0;
    uint64_t optimizedCalls = 0;
    double hostPrepareMs = 0, deviceMs = 0;
};
