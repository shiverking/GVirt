#!/usr/bin/env bash
set -uo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_paged_kv_stage_probe"
build_dir=${XLITE_ASR_PAGED_KV_STAGE_BUILD_DIR:-/tmp/xlite_asr_paged_kv_stage_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}; jobs=${XLITE_BUILD_JOBS:-8}; warmup=${XLITE_ASR_KV_STAGE_WARMUP:-3}; iterations=${XLITE_ASR_KV_STAGE_ITERATIONS:-20}
cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu \
    -DSOC_VERSION=Ascend310P3 -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?
export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
cases=("head0-first:0:1:0" "tail15:17:15:3" "inside-block:112:16:7" "cross-127:120:16:4" "cross-single:127:2:1" "block1:128:16:6" "last-block:2032:16:2")
failures=(); executed=0
for spec in "${cases[@]}"; do
    IFS=: read -r name start valid head <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${name}" ]]; then continue; fi
    executed=$((executed+1)); log="${build_dir}/${name}.log"; echo "[ RUN      ] ${name} start=${start} valid=${valid} head=${head}"
    "${build_dir}/xlite_asr_paged_kv_stage_probe_runner" "${start}" "${valid}" "${head}" "${warmup}" "${iterations}" >"${log}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then echo "[       OK ] ${name}"; cat "${log}"; else echo "[  FAILED  ] ${name} (recorded; continuing)"; failures+=("${name}:${status}:${log}"); fi
done
echo; if [[ ${executed} -eq 0 ]]; then echo "Unknown staging case: ${case_filter}" >&2; exit 2; fi
echo "ASR paged KV stage summary: $((executed-${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
    echo "================================================================================"; echo "AGGREGATED FAILURES"
    for failure in "${failures[@]}"; do IFS=: read -r name status log <<<"${failure}"; echo; echo "[${name}] exit=${status} log=${log}"; echo "--------------------------------------------------------------------------------"; cat "${log}"; done
    exit 1
fi
