/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_M200_MATMUL_310P_H_
#define _XLITE_M200_MATMUL_310P_H_

#include <cstdint>

#include "base.h"
#include "runtime.h"

bool XliteM200Matmul310PSupported(const XTensor &in, const XTensor &weight,
                                  const XTensor &out, bool weightNZ,
                                  const XTensor &bias, const XTensor &deqScale,
                                  bool transpose, bool enablePrefill);
void XliteM200Matmul310P(XRuntime &rt, XTensor &in, XTensor &weight, XTensor &out);
void XliteM200Matmul310PDestroy(XRuntime &rt);

#endif
