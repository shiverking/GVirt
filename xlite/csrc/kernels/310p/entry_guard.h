/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include "kernel_operator.h"

// This header is included only by the dedicated Ascend310P kernel entry
// translation units.  Select the project implementation branch from the real
// compiler architecture, and fail the build instead of emitting a no-op kernel
// when the device compiler target is missing or mismatched.
#if !defined(__NPU_ARCH__)
#error "Ascend310P kernel entry requires the device compiler to define __NPU_ARCH__"
#elif __NPU_ARCH__ != 2002
#error "Ascend310P kernel entry requires __NPU_ARCH__=2002"
#endif

#ifndef XLITE_ARCH_310P
#define XLITE_ARCH_310P 1
#endif
