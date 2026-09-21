#!/usr/bin/env bash
set -euo pipefail

report_dir=${1:-ascendc_asr_nz_matmul_report}

python3 tests/poc_310p/check_ascendc_asr_gates.py --require-resources

args=()
for projection in qkv o gate-up down lm-head; do
  for batch in 1 2 4 6 8 12 16 20; do
    args+=(--case "${projection}-m${batch}")
  done
done

python3 tests/poc_310p/test_matmul.py \
  --backend ascendc_asr_nz \
  --report-dir "${report_dir}" \
  "${args[@]}"

python3 - "${report_dir}" <<'PY'
import json
import sys
from pathlib import Path

root = Path(sys.argv[1])
errors = []
for log in sorted(root.glob("*.log")):
    records = [json.loads(line) for line in log.read_text().splitlines()
               if line.startswith("{")]
    if len(records) != 1:
        errors.append(f"{log}: expected one JSON record")
        continue
    item = records[0]
    if item["formats"]["weight_nz"] != 29 or not item["guards_intact"]:
        errors.append(f"{log}: NZ/guard contract failed")
    if item["xlite_candidate"]["synchronized_window_ms"] < 200.0:
        errors.append(f"{log}: candidate timing window below 200 ms")
    native = item["native_nz"]["device_ms"]
    candidate = item["xlite_candidate"]["device_and_submit_ms"]
    print(f"NZ_AB {log.stem} native_ms={native:.6f} "
          f"candidate_ms={candidate:.6f} speedup={native/candidate:.3f}x")
if errors:
    raise SystemExit("\n".join(errors))
print("AscendC ASR NZ MatMul correctness/timing contract PASS")
PY
