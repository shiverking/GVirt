#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == "--rerun-failed" ]]; then
  exec python3 tests/poc_310p/tune_p3_matmul.py --rerun-failed
fi
if (( $# )); then
  echo "Usage: bash tests/poc_310p/validate_p3_310p.sh [--rerun-failed]" >&2
  exit 2
fi
mkdir -p poc_310p_results
export SOC_VERSION=Ascend310P3
export XLITE_KERNEL_SET=llm_fp16
echo "[P3 1/2] Rebuild. Existing MatMul synchronization flags are preserved."
python3 -m pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall -v -e . \
  2>&1 | tee poc_310p_results/p3_build.log
echo "[P3 2/2] Screen candidates and validate changed shapes; no Attention/Vector/Audio tests."
python3 tests/poc_310p/tune_p3_matmul.py 2>&1 | tee poc_310p_results/p3_matmul.log
