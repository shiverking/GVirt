/*
 * Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
 */
#ifndef _XLITE_ASCENDC_ASR_MATMUL_310P_H_
#define _XLITE_ASCENDC_ASR_MATMUL_310P_H_

#include "base.h"
#include "runtime.h"

bool XliteAscendCAsrProjection310PSupported(const XTensor &in,
                                             const XTensor &weight,
                                             const XTensor &out, bool weightNZ,
                                             const XTensor &bias,
                                             const XTensor &deqScale,
                                             bool transpose);
bool XliteAscendCAsrKnownMatmul310P(const XTensor &in, const XTensor &weight,
                                    const XTensor &out, bool weightNZ,
                                    const XTensor &bias,
                                    const XTensor &deqScale, bool transpose);
void XliteAscendCAsrProjection310P(XRuntime &rt, XTensor &in,
                                   XTensor &weight, XTensor &out, bool cached);
bool XliteAscendCAsrLmHead310PSupported(const XTensor &in,
                                        const XTensor &weight,
                                        const XTensor &out, bool weightNZ,
                                        const XTensor &bias,
                                        const XTensor &deqScale,
                                        bool transpose);
void XliteAscendCAsrLmHead310P(XRuntime &rt, XTensor &in,
                               XTensor &weight, XTensor &out);

#endif
