#!/usr/bin/env bash
set -uo pipefail

checkpoint=${1:?usage: validate_ascendc_asr_perf_runtime.sh CHECKPOINT REPORT_DIR}
report_dir=${2:?usage: validate_ascendc_asr_perf_runtime.sh CHECKPOINT REPORT_DIR}
mkdir -p "${report_dir}"
failures=0

run_stage() {
    local name=$1
    shift
    local log="${report_dir}/${name}.log"
    echo "[ RUN      ] ${name}"
    if "$@" >"${log}" 2>&1; then
        echo "[       OK ] ${name}"
        tail -20 "${log}"
    else
        echo "[  FAILED  ] ${name} (recorded; continuing)"
        failures=$((failures + 1))
    fi
}

run_stage source-gates \
    python3 tests/poc_310p/check_ascendc_asr_gates.py --require-resources

matmul_args=(
    --backend ascendc_asr_perf
    --report-dir "${report_dir}/matmul"
)
for projection in qkv o gate-up down lm-head; do
    for batch in 1 8 20; do
        matmul_args+=(--case "${projection}-m${batch}")
    done
done
run_stage random-weight-matmul python3 tests/poc_310p/test_matmul.py "${matmul_args[@]}"

common_args=(
    --checkpoint "${checkpoint}"
    --input-mode synthetic
    --prompt-tokens 129
    --decode-tokens 16
    --max-seq-len 512
    --stability-iters 1
    --decode-attention-backend legacy
)
run_stage whole-model-baseline \
    python3 tests/poc_310p/run_qwen3_asr_llm.py "${common_args[@]}" \
    --matmul-backend ascendc_asr \
    --report "${report_dir}/whole-model-baseline.json"
run_stage whole-model-perf \
    python3 tests/poc_310p/run_qwen3_asr_llm.py "${common_args[@]}" \
    --matmul-backend ascendc_asr_perf \
    --report "${report_dir}/whole-model-perf.json"

if [[ -f "${report_dir}/whole-model-baseline.json" &&
      -f "${report_dir}/whole-model-perf.json" ]]; then
    run_stage compare-whole-model \
        python3 tests/poc_310p/compare_ascendc_asr_perf_runtime.py \
        "${report_dir}/whole-model-baseline.json" \
        "${report_dir}/whole-model-perf.json"
else
    echo "[  FAILED  ] compare-whole-model (missing report; recorded)"
    failures=$((failures + 1))
fi

echo
echo "AscendC ASR performance-runtime validation: ${failures} failed"
if [[ ${failures} -ne 0 ]]; then
    echo "AGGREGATED FAILURES"
    for log in "${report_dir}"/*.log; do
        echo "--- ${log}"
        tail -80 "${log}"
    done
    exit 1
fi
