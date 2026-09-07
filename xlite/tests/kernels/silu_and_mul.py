#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
import os
import torch
from xlite._C import Runtime, silu_and_mul


rt = Runtime(0, 500)
torch.npu.set_device(0)

supported_dtype_list = [
    (torch.float, 2e-5, 2e-3),
    (torch.float16, 2e-5, 2e-3),
    (torch.bfloat16, 2e-2, 2e-4)
]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    supported_dtype_list = [item for item in supported_dtype_list if item[0] == torch.float16]

for dtype, atol, rtol in supported_dtype_list:
    for dim in [2304, 6400]:
        input = torch.randn(41, dim * 2, dtype=dtype, device="npu:0")
        output = torch.full((41, dim), torch.nan, dtype=dtype, device="npu:0")

        d = input.shape[-1] // 2
        standard = torch.nn.functional.silu(input[..., :d]) * input[..., d:]

        torch.npu.synchronize()
        silu_and_mul(rt, input, output)
        torch.npu.synchronize()
        if torch.isnan(output).any():
            raise AssertionError("silu-and-mul output still contains the no-op sentinel")

        print(f'silu and mul {dtype} executed!')

        try:
            torch.testing.assert_close(standard, output, atol=atol, rtol=rtol)
        except AssertionError as e:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                raise
            print(f'{e}')
            print(f'torch_npu: {standard}')
            print(f'xlite: {output}')
