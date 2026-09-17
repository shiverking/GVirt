#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_projection_probe"
build_dir=${XLITE_ASR_PROJECTION_BUILD_DIR:-/tmp/xlite_asr_projection_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_PROJECTION_WARMUP:-3}
iterations=${XLITE_ASR_PROJECTION_ITERATIONS:-20}

echo "[ ASR LOW-LEVEL PROJECTION ] source=${source_dir}"
echo "[ ASR LOW-LEVEL PROJECTION ] build=${build_dir}"
echo "[ ASR LOW-LEVEL PROJECTION ] cann=${cann_path}"

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
failures=()
executed=0
cases=(
    "qkv-m1:1:4096:2048"
    "qkv-m8:8:4096:2048"
    "qkv-m20:20:4096:2048"
    "o-m20:20:2048:2048"
    "gate-up-m20:20:12288:2048"
    "down-m20:20:2048:6144"
)

for spec in "${cases[@]}"; do
    IFS=: read -r name m n k <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${name}" ]]; then
        continue
    fi
    executed=$((executed + 1))
    log="${build_dir}/${name}.log"
    echo "[ RUN      ] ${name} M=${m} N=${n} K=${k}"
    "${build_dir}/xlite_asr_projection_probe_runner" \
        "${m}" "${n}" "${k}" "${warmup}" "${iterations}" >"${log}" 2>&1
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
    echo "Unknown ASR projection case: ${case_filter}" >&2
    exit 2
fi
echo "ASR projection probe summary: $((executed - ${#failures[@]})) passed, ${#failures[@]} failed"
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
