#!/usr/bin/env bash
set -uo pipefail

export XLITE_TEST_FP16_ONLY=1

parallel=0
skip_vector=0
while [[ $# -gt 0 ]]; do
    case "$1" in
        --parallel)
            parallel=1
            ;;
        --skip-vector)
            skip_vector=1
            ;;
        -h|--help)
            echo "usage: $0 [--parallel] [--skip-vector]"
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            echo "usage: $0 [--parallel] [--skip-vector]" >&2
            exit 2
            ;;
    esac
    shift
done

log_dir=$(mktemp -d /tmp/xlite-310p-smoke.XXXXXX)
failure_names=()
failure_statuses=()
failure_logs=()
passed=0
total=0
pids=()
parallel_names=()
parallel_logs=()

run_test() {
    local name=$1
    shift
    local log_file="${log_dir}/${name}.log"

    total=$((total + 1))
    echo "[ RUN      ] ${name}"
    "$@" >"${log_file}" 2>&1
    local status=$?
    if [[ ${status} -eq 0 ]]; then
        passed=$((passed + 1))
        echo "[       OK ] ${name}"
    else
        failure_names+=("${name}")
        failure_statuses+=("${status}")
        failure_logs+=("${log_file}")
        echo "[  FAILED  ] ${name} (exit=${status}, recorded; continuing)"
    fi
}

start_test() {
    local name=$1
    shift
    local log_file="${log_dir}/${name}.log"

    total=$((total + 1))
    echo "[ START    ] ${name}"
    "$@" >"${log_file}" 2>&1 &
    pids+=("$!")
    parallel_names+=("${name}")
    parallel_logs+=("${log_file}")
}

finish_parallel_tests() {
    local index status name log_file
    for index in "${!pids[@]}"; do
        name=${parallel_names[$index]}
        log_file=${parallel_logs[$index]}
        if wait "${pids[$index]}"; then
            passed=$((passed + 1))
            echo "[       OK ] ${name}"
        else
            status=$?
            failure_names+=("${name}")
            failure_statuses+=("${status}")
            failure_logs+=("${log_file}")
            echo "[  FAILED  ] ${name} (exit=${status}, recorded; continuing)"
        fi
    done
}

if [[ ${parallel} -eq 1 ]]; then
    if [[ ${skip_vector} -eq 0 ]]; then
        start_test vector-kernels python3 tests/poc_310p/test_vector_kernels.py
    fi
    start_test matmul python3 tests/poc_310p/test_matmul.py \
        --report-dir "${log_dir}/matmul-shapes"
    start_test rope-and-cache python3 tests/kernels/rope_and_cache.py
    start_test attention python3 tests/kernels/attention.py
    finish_parallel_tests
else
    if [[ ${skip_vector} -eq 0 ]]; then
        run_test vector-kernels python3 tests/poc_310p/test_vector_kernels.py
    fi
    run_test matmul python3 tests/poc_310p/test_matmul.py \
        --report-dir "${log_dir}/matmul-shapes"
    run_test rope-and-cache python3 tests/kernels/rope_and_cache.py
    run_test attention python3 tests/kernels/attention.py
fi

echo
echo "Kernel smoke summary: ${passed} passed, ${#failure_names[@]} failed"
if [[ ${#failure_names[@]} -ne 0 ]]; then
    echo
    echo "================================================================================"
    echo "AGGREGATED FAILURES"
    for index in "${!failure_names[@]}"; do
        name=${failure_names[$index]}
        status=${failure_statuses[$index]}
        log_file=${failure_logs[$index]}
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
