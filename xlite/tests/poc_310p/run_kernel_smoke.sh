#!/usr/bin/env bash
set -euo pipefail

export XLITE_TEST_FP16_ONLY=1

python tests/kernels/add.py
python tests/kernels/rmsnorm.py
python tests/kernels/add_and_rmsnorm.py
python tests/kernels/silu_and_mul.py
python tests/kernels/matmul.py
python tests/kernels/rope_and_cache.py
python tests/kernels/attention.py
