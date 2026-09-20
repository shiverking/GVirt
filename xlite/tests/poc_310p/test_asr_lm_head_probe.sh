#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_lm_head_probe"
build_dir=${XLITE_ASR_LM_HEAD_BUILD_DIR:-/tmp/xlite_asr_lm_head_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_LM_HEAD_WARMUP:-1}
iterations=${XLITE_ASR_LM_HEAD_ITERATIONS:-3}

echo "[ ASR LM HEAD ] source=${source_dir}"
echo "[ ASR LM HEAD ] build=${build_dir}"
echo "[ ASR LM HEAD ] cann=${cann_path}"
echo "[ ASR LM HEAD ] device mapping is externally owned and unchanged"

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
failures=()
executed=0
cases=("m1:1" "m8:8" "m20:20")
for spec in "${cases[@]}"; do
    IFS=: read -r name batch <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${name}" ]]; then
        continue
    fi
    executed=$((executed + 1))
    log="${build_dir}/${name}.log"
    echo "[ RUN      ] lm-head-${name} M=${batch} N=151936 K=2048"
    "${build_dir}/xlite_asr_lm_head_probe_runner" \
        "${batch}" "${warmup}" "${iterations}" >"${log}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] lm-head-${name}"
        cat "${log}"
    else
        failures+=("${name}:${status}:${log}")
        echo "[  FAILED  ] lm-head-${name} (recorded; continuing)"
    fi
done

echo
if [[ ${executed} -eq 0 ]]; then
    echo "Unknown ASR LM Head case: ${case_filter}" >&2
    exit 2
fi
echo "ASR LM Head summary: $((executed - ${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
    echo
    echo "================================================================================"
    echo "AGGREGATED FAILURES"
    for failure in "${failures[@]}"; do
        IFS=: read -r name status log <<<"${failure}"
        echo
        echo "[lm-head-${name}] exit=${status} log=${log}"
        echo "--------------------------------------------------------------------------------"
        cat "${log}"
    done
    exit 1
fi
