#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
xlite_dir=$(cd -- "${script_dir}/../.." && pwd)
cann_path=${1:-/usr/local/Ascend/cann-9.1.0-beta.1}
report_dir=${2:-ascendc_asr_phase0_report}
shared_root=${3:-}

mkdir -p "${report_dir}"
failures=()

run_case() {
    local name=$1
    shift
    local log="${report_dir}/${name}.log"
    echo "[ RUN      ] ${name}"
    "$@" >"${log}" 2>&1
    local status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] ${name}"
    else
        failures+=("${name}:${status}:${log}")
        echo "[  FAILED  ] ${name} (exit=${status}, recorded; continuing)"
    fi
}

gate_args=()
if [[ -n "${shared_root}" ]]; then
    gate_args+=(--shared-root "${shared_root}")
fi

cd "${xlite_dir}" || exit $?
run_case source-gates python3 tests/poc_310p/check_ascendc_asr_gates.py "${gate_args[@]}"
run_case environment python3 tests/poc_310p/probe_environment.py \
    --cann-path "${cann_path}" --report "${report_dir}/environment.json"

# These existing probes establish the CANN 9.1 hardware baseline. They are not
# production implementations of the new backend: the projection probe still
# uses the historical high-level Matmul API, and is retained only for an A/B
# reference until the low-level Mmad projection lands.
run_case cube-baseline env \
    ASCEND_CANN_PACKAGE_PATH="${cann_path}" \
    XLITE_M200_PROBE_BUILD_DIR="/tmp/xlite_m200_probe_phase0" \
    XLITE_M200_PROBE_WARMUP=3 XLITE_M200_PROBE_ITERATIONS=20 \
    bash tests/poc_310p/test_m200_cube_probe.sh qkv down
run_case vector-baseline python3 tests/poc_310p/test_vector_kernels.py

echo
echo "AscendC ASR phase-0 summary: $((4 - ${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
    echo
    echo "================================================================================"
    echo "AGGREGATED FAILURES"
    for failure in "${failures[@]}"; do
        IFS=: read -r name status log <<<"${failure}"
        echo
        echo "[${name}] exit=${status} log=${log}"
        echo "--------------------------------------------------------------------------------"
        cat "${log}"
    done
    exit 1
fi

echo "Reports: ${report_dir}"
