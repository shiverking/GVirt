#!/usr/bin/env bash
set -euo pipefail

report_dir=${1:-native_eager_contract_report}

python3 tests/poc_310p/audit_native_eager_contract.py \
  --report-dir "${report_dir}" \
  --minimum-ms 200

python3 tests/poc_310p/test_matmul.py \
  --backend ascendc_asr_perf \
  --case qkv-m1 --case qkv-m8 --case qkv-m20 \
  --case o-m1 --case o-m8 --case o-m20 \
  --case gate-up-m1 --case gate-up-m8 --case gate-up-m20 \
  --case down-m1 --case down-m8 --case down-m20 \
  --case lm-head-m1 --case lm-head-m8 --case lm-head-m20 \
  --report-dir "${report_dir}/xlite-nd-vs-native-nz"
