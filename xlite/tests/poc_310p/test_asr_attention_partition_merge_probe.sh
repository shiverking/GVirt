#!/usr/bin/env bash
set -uo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_attention_partition_merge_probe"
build_dir=${XLITE_ASR_ATTN_MERGE_BUILD_DIR:-/tmp/xlite_asr_attention_partition_merge_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_ATTN_MERGE_WARMUP:-3}
iterations=${XLITE_ASR_ATTN_MERGE_ITERATIONS:-20}

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?
export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"

cases=(
    "kv513:513"
    "kv1023:1023"
    "kv1024:1024"
    "kv1025:1025"
    "kv1537:1537"
    "kv2048:2048"
    "batch20:513,575,639,703,767,831,895,959,1023,1024,1025,1153,1281,1409,1537,1665,1793,1921,1985,2048"
)
failures=()
executed=0
for spec in "${cases[@]}"; do
    IFS=: read -r name lengths <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${name}" ]]; then
        continue
    fi
    executed=$((executed + 1))
    log="${build_dir}/${name}.log"
    echo "[ RUN      ] ${name} kv_lengths=${lengths}"
    "${build_dir}/xlite_asr_attention_partition_merge_runner" \
        "${lengths}" "${warmup}" "${iterations}" >"${log}" 2>&1
    status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] ${name}"
        cat "${log}"
    else
        echo "[  FAILED  ] ${name} (recorded; continuing)"
        failures+=("${name}:${status}:${log}")
    fi
done
echo
if [[ ${executed} -eq 0 ]]; then
    echo "Unknown partition-merge case: ${case_filter}" >&2
    exit 2
fi
echo "ASR attention partition merge summary: $((executed-${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
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
