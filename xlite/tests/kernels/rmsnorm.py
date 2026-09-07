#!/usr/bin/python3
# coding=utf-8
#
# Copyright (C) 2025. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# ===============================================================================
from __future__ import absolute_import
import logging
import os
import os
import torch
import torch_npu
import torch.distributed as dist
from xlite._C import Runtime, rmsnorm, rmsnorm_with_bias

logging.getLogger().setLevel(logging.INFO)

npu_id = 0
rt = Runtime(npu_id, 500)
torch.npu.set_device(npu_id)

def stand_rmsnorm(x: torch.Tensor, weight: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    input_dtype = x.dtype
    x = x.to(torch.float32)
    variance = x.pow(2)
    variance = variance.mean(-1, keepdim=True)
    x = x * torch.rsqrt(variance + eps)
    return (weight.float() * x).to(input_dtype)

BATCH_SIZE = 64
DIM = 8192
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    dtype_list = [torch.float16]

for test_dtype in dtype_list:
    for has_bias in [True, False]:
        with torch.device("npu"):
            x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype)
            x_standard = x.clone()
            weight = torch.randn(DIM, dtype=test_dtype)
            bias = torch.randn(DIM, dtype=test_dtype)
            y = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)
            z = torch.empty(BATCH_SIZE, DIM, dtype=test_dtype)

        # standard
        standard = stand_rmsnorm(x_standard, weight, NORMEPS)
        if has_bias:
            standard = standard.add_(bias)

        # xlite
        torch.npu.synchronize()
        if has_bias:
            rmsnorm_with_bias(rt, x, weight, bias, y, NORMEPS)
        else:
            rmsnorm(rt, x, weight, y, NORMEPS)
        torch.npu.synchronize()
        logging.info(f'rmsnorm({test_dtype}) {"with" if has_bias else "without"} bias executed!')
        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                raise
            logging.error(f'{e}')
            print(f"standard: {standard}")
            print(f"bias: {bias}")
            logging.error(f'torch_npu: {standard}')
            logging.error(f'xlite: {y}')

BATCH_SIZE = 64
DIM = 128
CNT = 8
SIZE1 = DIM * (CNT + 2)
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    dtype_list = [torch.float16]

for test_dtype in dtype_list:
    for has_bias in [True, False]:
        with torch.device("npu"):
            x = torch.randn(BATCH_SIZE, SIZE1, dtype=test_dtype)
            x_standard = x.clone()
            weight = torch.randn(DIM, dtype=test_dtype)
            bias1 = torch.randn(DIM, dtype=test_dtype)
            bias2 = torch.randn(DIM, dtype=test_dtype)
            y = x.clone()

        x_split1, x_split2, x_split3 = x_standard.split([CNT * DIM, DIM, DIM], dim = 1)
        x_split1 = x_split1.view(BATCH_SIZE, CNT, DIM)
        x_split2 = x_split2.view(BATCH_SIZE, 1, DIM)
        # standard
        x_split1 = stand_rmsnorm(x_split1, weight, NORMEPS)
        x_split2 = stand_rmsnorm(x_split2, weight, NORMEPS)
        if has_bias:
            x_split1 = x_split1.add_(bias1)
            x_split2 = x_split2.add_(bias2)
        x_split1 = x_split1.view(BATCH_SIZE, CNT * DIM)
        x_split2 = x_split2.view(BATCH_SIZE, DIM)
        standard = torch.cat([x_split1, x_split2, x_split3], dim=-1)

        # xlite
        torch.npu.synchronize()
        if has_bias:
            rmsnorm_with_bias(rt, x, weight, bias1, y, NORMEPS, DIM, CNT)
            rmsnorm_with_bias(rt, x, weight, bias2, y, NORMEPS, DIM, 1, DIM * CNT, DIM * CNT)
        else:
            rmsnorm(rt, x, weight, y, NORMEPS, DIM, CNT)
            rmsnorm(rt, x, weight, y, NORMEPS, DIM, 1, DIM * CNT, DIM * CNT)
        torch.npu.synchronize()

        logging.info(f'rmsnorm with stride ({test_dtype}) {"with" if has_bias else "without"} bias executed!')

        try:
            torch.testing.assert_close(standard, y, atol=1e-5, rtol=1e-3)
        except AssertionError as e:
            if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
                raise
            logging.error(f'{e}')
            logging.error(f'torch_npu: {standard}')
            logging.error(f'xlite: {y}')
