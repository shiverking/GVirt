#!/usr/bin/env bash
set -uo pipefail

export XLITE_TEST_FP16_ONLY=1

log_dir=$(mktemp -d /tmp/xlite-310p-smoke.XXXXXX)
failures=()

run_test() {
    local name=$1
    shift
    local log_file="${log_dir}/${name}.log"

    echo "[ RUN      ] ${name}"
    "$@" >"${log_file}" 2>&1
    local status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] ${name}"
    else
        failures+=("${name}:${status}:${log_file}")
        echo "[  FAILED  ] ${name} (exit=${status}, recorded; continuing)"
    fi
}

run_test vector-kernels python3 tests/poc_310p/test_vector_kernels.py
run_test matmul python3 tests/poc_310p/test_matmul.py --report-dir "${log_dir}/matmul-shapes"
run_test rope-and-cache python3 tests/kernels/rope_and_cache.py
run_test attention python3 tests/kernels/attention.py

echo
echo "Kernel smoke summary: $((4 - ${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
    echo
    echo "================================================================================"
    echo "AGGREGATED FAILURES"
    for failure in "${failures[@]}"; do
        IFS=: read -r name status log_file <<<"${failure}"
        echo
        echo "[${name}] exit=${status} log=${log_file}"
        echo "--------------------------------------------------------------------------------"
        cat "${log_file}"
    done
    echo
    echo "All logs: ${log_dir}"
    exit 1
fi

echo "All logs: ${log_dir}"
