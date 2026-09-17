#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_attention_mmad_probe"
build_dir=${XLITE_ASR_ATTN_MMAD_BUILD_DIR:-/tmp/xlite_asr_attention_mmad_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_ATTN_MMAD_WARMUP:-3}
iterations=${XLITE_ASR_ATTN_MMAD_ITERATIONS:-20}

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?
export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"

failures=(); executed=0
for operation in qk pv; do
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${operation}" ]]; then continue; fi
    executed=$((executed + 1)); log="${build_dir}/${operation}.log"
    echo "[ RUN      ] attention-mmad-${operation}"
    "${build_dir}/xlite_asr_attention_mmad_probe_runner" \
        "${operation}" "${warmup}" "${iterations}" >"${log}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] attention-mmad-${operation}"; cat "${log}"
    else
        echo "[  FAILED  ] attention-mmad-${operation} (recorded; continuing)"
        failures+=("${operation}:${status}:${log}")
    fi
done
echo
if [[ ${executed} -eq 0 ]]; then echo "Unknown MMAD case: ${case_filter}" >&2; exit 2; fi
echo "ASR attention MMAD probe summary: $((executed-${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
    echo "================================================================================"
    echo "AGGREGATED FAILURES"
    for failure in "${failures[@]}"; do
        IFS=: read -r name status log <<<"${failure}"
        echo; echo "[${name}] exit=${status} log=${log}"; echo "--------------------------------------------------------------------------------"
        cat "${log}"
    done
    exit 1
fi
