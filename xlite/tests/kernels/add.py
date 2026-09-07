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
from xlite._C import Runtime, add


rt = Runtime(0, 500)
torch.npu.set_device(0)

supported_dtype_list = [torch.float16, torch.bfloat16]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    supported_dtype_list = [torch.float16]

for dtype in supported_dtype_list:
    x = torch.randn(8, 2048, dtype=dtype, device="npu:0")
    y = torch.randn(8, 2048, dtype=dtype, device="npu:0")
    z = torch.full((8, 2048), torch.nan, dtype=dtype, device="npu:0")

    standard = x + y

    torch.npu.synchronize()
    add(rt, x, y, z)
    torch.npu.synchronize()
    if torch.isnan(z).any():
        raise AssertionError("add output still contains the no-op sentinel")
    print(f'add {dtype} executed!')

    try:
        torch.testing.assert_close(standard, z, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
            raise
        print(f'{e}')
        print(f'x: {x}')
        print(f'y: {y}')
        print(f'torch_npu: {standard}')
        print(f'xlite: {z}')
