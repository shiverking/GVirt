#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_vector_probe"
build_dir=${XLITE_ASR_VECTOR_BUILD_DIR:-/tmp/xlite_asr_vector_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_VECTOR_WARMUP:-3}
iterations=${XLITE_ASR_VECTOR_ITERATIONS:-20}

echo "[ ASR VECTOR PROBE ] source=${source_dir}"
echo "[ ASR VECTOR PROBE ] build=${build_dir}"
echo "[ ASR VECTOR PROBE ] cann=${cann_path}"
cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
failures=()
executed=0
cases=(
    "add-rmsnorm-m1:add-rmsnorm:1"
    "add-rmsnorm-m2:add-rmsnorm:2"
    "add-rmsnorm-m4:add-rmsnorm:4"
    "add-rmsnorm-m6:add-rmsnorm:6"
    "add-rmsnorm-m8:add-rmsnorm:8"
    "add-rmsnorm-m12:add-rmsnorm:12"
    "add-rmsnorm-m16:add-rmsnorm:16"
    "add-rmsnorm-m20:add-rmsnorm:20"
    "rmsnorm-m1:rmsnorm:1"
    "rmsnorm-m2:rmsnorm:2"
    "rmsnorm-m4:rmsnorm:4"
    "rmsnorm-m6:rmsnorm:6"
    "rmsnorm-m8:rmsnorm:8"
    "rmsnorm-m12:rmsnorm:12"
    "rmsnorm-m16:rmsnorm:16"
    "rmsnorm-m20:rmsnorm:20"
    "silu-m1:silu:1"
    "silu-m2:silu:2"
    "silu-m4:silu:4"
    "silu-m6:silu:6"
    "silu-m8:silu:8"
    "silu-m12:silu:12"
    "silu-m16:silu:16"
    "silu-m20:silu:20"
)
for spec in "${cases[@]}"; do
    IFS=: read -r name op tokens <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${op}" &&
          "${case_filter}" != "${name}" ]]; then
        continue
    fi
    executed=$((executed + 1))
    log="${build_dir}/${name}.log"
    echo "[ RUN      ] ${name}"
    "${build_dir}/xlite_asr_vector_probe_runner" \
        "${op}" "${tokens}" "${warmup}" "${iterations}" >"${log}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] ${name}"
        cat "${log}"
    else
        failures+=("${name}:${status}:${log}")
        echo "[  FAILED  ] ${name} (recorded; continuing)"
    fi
done

echo
if [[ ${executed} -eq 0 ]]; then
    echo "Unknown ASR vector case: ${case_filter}" >&2
    exit 2
fi
echo "ASR vector probe summary: $((executed - ${#failures[@]})) passed, ${#failures[@]} failed"
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
