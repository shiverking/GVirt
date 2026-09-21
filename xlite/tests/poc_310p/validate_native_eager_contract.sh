#!/usr/bin/env bash
set -euo pipefail

report_dir=${1:-native_eager_contract_report}

python3 tests/poc_310p/audit_native_eager_contract.py \
  --report-dir "${report_dir}" \
  --minimum-ms 200

python3 tests/poc_310p/test_matmul.py \
  --backend ascendc_asr_nz \
  --case qkv-m1 --case qkv-m2 --case qkv-m4 --case qkv-m6 \
  --case qkv-m8 --case qkv-m12 --case qkv-m16 --case qkv-m20 \
  --case o-m1 --case o-m2 --case o-m4 --case o-m6 \
  --case o-m8 --case o-m12 --case o-m16 --case o-m20 \
  --case gate-up-m1 --case gate-up-m2 --case gate-up-m4 --case gate-up-m6 \
  --case gate-up-m8 --case gate-up-m12 --case gate-up-m16 --case gate-up-m20 \
  --case down-m1 --case down-m2 --case down-m4 --case down-m6 \
  --case down-m8 --case down-m12 --case down-m16 --case down-m20 \
  --case lm-head-m1 --case lm-head-m2 --case lm-head-m4 --case lm-head-m6 \
  --case lm-head-m8 --case lm-head-m12 --case lm-head-m16 --case lm-head-m20 \
  --report-dir "${report_dir}/xlite-nz-vs-native-nz"

python3 tests/poc_310p/summarize_ascendc_asr_nz_matmul.py \
  "${report_dir}/xlite-nz-vs-native-nz" \
  --report "${report_dir}/xlite-nz-vs-native-nz-summary.json"
