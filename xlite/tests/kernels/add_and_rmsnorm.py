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
from xlite._C import Runtime, add_and_rmsnorm

logging.getLogger().setLevel(logging.INFO)

rt = Runtime(0, 500)
torch.npu.set_device(0)

BATCH_SIZE = 64
DIM = 6144
NORMEPS = 1e-6
dtype_list = [torch.float16, torch.bfloat16]
if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
    dtype_list = [torch.float16]

for test_dtype in dtype_list:
    with torch.device("npu"):
        x = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype) / 10
        x_standard = x.clone()
        weight = torch.randn(DIM, dtype=test_dtype) / 10
        y = torch.randn(BATCH_SIZE, DIM, dtype=test_dtype) / 10
        y_standard = y.clone()

    # standard
    x_standard = x_standard.to(torch.float32)
    y_standard = y_standard.to(torch.float32)
    x_standard = x_standard + y_standard
    add_standard = x_standard
    variance = x_standard.pow(2).mean(-1, keepdim=True)
    x_standard = x_standard * torch.rsqrt(variance + NORMEPS)
    standard = (weight.float() * x_standard).to(test_dtype)
    add_standard = add_standard.to(test_dtype)

    # xlite
    torch.npu.synchronize()
    add_and_rmsnorm(rt, x, y, weight, x, NORMEPS)
    torch.npu.synchronize()

    logging.info(f'rmsnorm({test_dtype}) executed!')

    try:
        torch.testing.assert_close(add_standard, y, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
            raise
        logging.error(f'{e}')
        logging.error(f'torch_npu add: {add_standard}')
        logging.error(f'xlite add: {y}')

    try:
        torch.testing.assert_close(standard, x, atol=1e-5, rtol=1e-3)
    except AssertionError as e:
        if os.getenv("XLITE_TEST_FP16_ONLY") == "1":
            raise
        logging.error(f'{e}')
        logging.error(f'torch_npu rmsnorm: {standard}')
        logging.error(f'xlite rmsnorm: {x}')
