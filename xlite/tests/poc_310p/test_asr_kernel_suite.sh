#!/usr/bin/env bash
set -uo pipefail

# One entry point for every Qwen3-ASR 310P kernel gate.  Each stage writes a
# complete log and independent stages continue after a failure so one device
# run returns all actionable diagnostics.
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
xlite_dir=$(cd -- "${script_dir}/../.." && pwd)
cann_path=${1:-/usr/local/Ascend/cann-9.1.0-beta.1}
report_dir=${2:-asr_kernel_suite_report}
mkdir -p "${report_dir}"
report_dir=$(cd -- "${report_dir}" && pwd)

passed=()
failed=()

run_stage() {
    local name=$1
    shift
    local log="${report_dir}/${name}.log"
    echo "[ SUITE RUN ] ${name}"
    (cd "${xlite_dir}" && "$@") >"${log}" 2>&1
    local status=$?
    if [[ ${status} -eq 0 ]]; then
        passed+=("${name}")
        echo "[ SUITE  OK ] ${name}"
        tail -n 12 "${log}"
    else
        failed+=("${name}:${status}:${log}")
        echo "[ SUITE FAIL ] ${name} (recorded; continuing)"
    fi
}

run_stage static-gates python3 tests/poc_310p/check_ascendc_asr_gates.py
run_stage projection bash tests/poc_310p/test_asr_projection_probe.sh "${cann_path}" all
run_stage ascendc-lm-head bash \
    tests/poc_310p/test_asr_lm_head_probe.sh "${cann_path}" all
run_stage vector-fused bash tests/poc_310p/test_asr_vector_probe.sh "${cann_path}" all
run_stage paged-attention-probe bash \
    tests/poc_310p/test_asr_paged_attention_probe.sh "${cann_path}" all
run_stage attention-mmad-blocks bash \
    tests/poc_310p/test_asr_attention_mmad_probe.sh "${cann_path}" all
run_stage paged-kv-stage bash \
    tests/poc_310p/test_asr_paged_kv_stage_probe.sh "${cann_path}" all
run_stage attention-block-chain bash \
    tests/poc_310p/test_asr_attention_chain_probe.sh "${cann_path}" all
run_stage attention-single-partition bash \
    tests/poc_310p/test_asr_attention_single_partition_probe.sh "${cann_path}" all
run_stage attention-partition-merge bash \
    tests/poc_310p/test_asr_attention_partition_merge_probe.sh "${cann_path}" all
run_stage attention-partition-chain bash \
    tests/poc_310p/test_asr_attention_partition_chain_probe.sh "${cann_path}" all
run_stage attention-fused-partition bash \
    tests/poc_310p/test_asr_attention_fused_partition_probe.sh "${cann_path}" all
run_stage compatibility-runtime-vector python3 tests/poc_310p/test_vector_kernels.py
run_stage compatibility-qk-mrope-cache env XLITE_TEST_FP16_ONLY=1 python3 tests/kernels/rope_and_cache.py
run_stage compatibility-decode-attention env XLITE_TEST_FP16_ONLY=1 \
    python3 tests/kernels/attention.py --batched-decode-only
run_stage compatibility-lm-head python3 tests/poc_310p/test_matmul.py \
    --case lm-head-m1 --case lm-head-m8 --case lm-head-m20 \
    --report-dir "${report_dir}/lm_head"

echo
echo "ASR kernel suite summary: ${#passed[@]} stages passed, ${#failed[@]} stages failed"
if [[ ${#failed[@]} -ne 0 ]]; then
    echo
    echo "================================================================================"
    echo "AGGREGATED STAGE FAILURES"
    for failure in "${failed[@]}"; do
        IFS=: read -r name status log <<<"${failure}"
        echo
        echo "[${name}] exit=${status} log=${log}"
        echo "--------------------------------------------------------------------------------"
        cat "${log}"
    done
    exit 1
fi

printf '%s\n' "${passed[@]}" >"${report_dir}/passed_stages.txt"
