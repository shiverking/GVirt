#!/usr/bin/env bash
set -uo pipefail

checkpoint=${1:?usage: validate_ascendc_asr_nz_iteration.sh CHECKPOINT REPORT_DIR [smoke|full] [build|skip-build]}
report_dir=${2:?usage: validate_ascendc_asr_nz_iteration.sh CHECKPOINT REPORT_DIR [smoke|full] [build|skip-build]}
mode=${3:-smoke}
build_mode=${4:-build}
if [[ "${mode}" != "smoke" && "${mode}" != "full" ]]; then
    echo "mode must be smoke or full" >&2
    exit 2
fi
if [[ "${build_mode}" != "build" && "${build_mode}" != "skip-build" ]]; then
    echo "build mode must be build or skip-build" >&2
    exit 2
fi
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

if [[ "${build_mode}" == "build" ]]; then
    run_stage rebuild-extension \
        python3 -m pip install -v -e . --no-build-isolation
fi
run_stage source-gates \
    python3 tests/poc_310p/check_ascendc_asr_gates.py --require-resources
run_stage nz-physical-layout \
    python3 tests/poc_310p/probe_nz_physical_layout.py

if [[ "${mode}" == "full" ]]; then
    run_stage native-kv-physical-layout \
        python3 tests/poc_310p/probe_native_kv_physical_layout.py \
        --report "${report_dir}/native-kv-physical-layout.json"
    run_stage nz-matmul-full \
        bash tests/poc_310p/validate_ascendc_asr_nz_matmul.sh \
        "${report_dir}/matmul"
else
    run_stage nz-matmul-smoke \
        python3 tests/poc_310p/test_matmul.py \
        --backend ascendc_asr_nz \
        --report-dir "${report_dir}/matmul" \
        --case qkv-m1 --case qkv-m20 --case qkv-m129 \
        --case down-m20 --case down-m129 \
        --case lm-head-m1 --case lm-head-m20
fi

common_args=(
    --checkpoint "${checkpoint}"
    --input-mode synthetic
    --prompt-tokens 129
    --decode-tokens 16
    --max-seq-len 512
    --stability-iters 1
    --decode-attention-backend legacy
)
run_stage whole-model-aclnn-same-nz \
    env XLITE_WEIGHT_NZ=1 python3 tests/poc_310p/run_qwen3_asr_llm.py \
    "${common_args[@]}" --matmul-backend aclnn \
    --report "${report_dir}/whole-model-aclnn-same-nz.json"
run_stage whole-model-ascendc-nz \
    env XLITE_WEIGHT_NZ=1 python3 tests/poc_310p/run_qwen3_asr_llm.py \
    "${common_args[@]}" --matmul-backend ascendc_asr_nz \
    --report "${report_dir}/whole-model-ascendc-nz.json"

if [[ -f "${report_dir}/whole-model-aclnn-same-nz.json" &&
      -f "${report_dir}/whole-model-ascendc-nz.json" ]]; then
    run_stage compare-whole-model \
        python3 tests/poc_310p/compare_ascendc_asr_nz_runtime.py \
        "${report_dir}/whole-model-aclnn-same-nz.json" \
        "${report_dir}/whole-model-ascendc-nz.json" \
        --report "${report_dir}/comparison.json"
else
    echo "[  FAILED  ] compare-whole-model (missing report; recorded)"
    failures=$((failures + 1))
fi

echo
echo "AscendC ASR NZ iteration (${mode}): ${failures} failed"
if [[ ${failures} -ne 0 ]]; then
    echo "AGGREGATED FAILURES"
    for log in "${report_dir}"/*.log; do
        echo "--- ${log}"
        tail -100 "${log}"
    done
    exit 1
fi
