#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
if [[ "${1:-}" == "--rerun-failed" ]]; then
  exec python3 tests/kernels/attention.py --decode-attention-backend paged_310p --rerun-failed
fi
if (( $# )); then
  echo "Usage: bash tests/poc_310p/validate_p2_310p.sh [--rerun-failed]" >&2
  exit 2
fi
mkdir -p poc_310p_results
echo "[P2 1/3] Build Ascend310P3 llm_fp16 (MatMul sync policy is not changed)"
export SOC_VERSION=Ascend310P3
export XLITE_KERNEL_SET=llm_fp16
python3 -m pip install --no-build-isolation --no-cache-dir --no-deps --force-reinstall -v -e . \
  2>&1 | tee poc_310p_results/p2_build.log
echo "[P2 2/3] Check installed capability"
python3 - <<'PY'
from xlite._C import get_build_info
info = dict(get_build_info())
print(info)
assert info.get("soc") == "Ascend310P3" and info.get("paged_decode_310p") is True, info
PY
echo "[P2 3/3] Aggregate NEW attention paths only (no Vector/MatMul/Audio rerun)"
python3 tests/kernels/attention.py --decode-attention-backend paged_310p \
  2>&1 | tee poc_310p_results/p2_attention.log
echo "Next: the documented 129-token model check, then the c1/c20 ASR comparison."
