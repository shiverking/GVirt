#!/usr/bin/env bash
set -euo pipefail

export XLITE_TEST_FP16_ONLY=1

python tests/poc_310p/test_vector_kernels.py
python tests/kernels/matmul.py
python tests/kernels/rope_and_cache.py
python tests/kernels/attention.py
