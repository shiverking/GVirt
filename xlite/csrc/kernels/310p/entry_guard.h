/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#pragma once

#include "kernel_operator.h"

// This header is included only by the dedicated Ascend310P kernel entry
// translation units.  Select the project implementation branch from the real
// compiler architecture.  CANN compiles the same translation unit once more
// for its host stub, where __NPU_ARCH__ is intentionally absent; validate the
// value whenever the device compiler provides it.
#if defined(__NPU_ARCH__) && __NPU_ARCH__ != 2002
#error "Ascend310P kernel entry requires __NPU_ARCH__=2002"
#endif

#ifndef XLITE_ARCH_310P
#define XLITE_ARCH_310P 1
#endif
