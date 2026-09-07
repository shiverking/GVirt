/*
 * Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include <map>
#include "base.h"

using Key = std::tuple<uint64_t, uint64_t>;

// Values were picked by running matmul_swizzle_perf.py script on Ascend 910B3 NPU.
static const std::map<Key, uint64_t> BestSwizzles = {
    {{6144, 2048}, 0xe01},
    {{2048, 6144}, 0xe01},
};

void XlitePickSwizzle(uint64_t n, uint64_t k, uint64_t *swizzle)
{
#ifdef XLITE_DISABLE_SWIZZLE_TABLE
    (void)n;
    (void)k;
    // The table below was tuned on 910B3.  Keep the caller-provided default on
    // 310P until a device-specific sweep has been completed.
    (void)swizzle;
    return;
#else
    const auto best = BestSwizzles.find({n, k});
    if (best != BestSwizzles.end()) {
        *swizzle = best->second;
    }
#endif
}
