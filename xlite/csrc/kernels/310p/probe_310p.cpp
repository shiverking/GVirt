/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#include "entry_guard.h"

extern "C" __global__ __aicore__ void xlite_probe_310p(GM_ADDR output, uint32_t value)
{
    if (block_idx == 0) {
        *((__gm__ uint32_t *)output) = value;
    }
}
