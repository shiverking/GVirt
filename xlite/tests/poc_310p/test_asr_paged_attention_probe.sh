#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_paged_attention_probe"
build_dir=${XLITE_ASR_PAGED_ATTN_BUILD_DIR:-/tmp/xlite_asr_paged_attention_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_PAGED_ATTN_WARMUP:-1}
iterations=${XLITE_ASR_PAGED_ATTN_ITERATIONS:-3}

echo "[ ASR PAGED ATTENTION PROBE ] source=${source_dir}"
echo "[ ASR PAGED ATTENTION PROBE ] build=${build_dir}"
echo "[ ASR PAGED ATTENTION PROBE ] cann=${cann_path}"
cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
failures=()
executed=0
cases=(
    "batch1-kv1:1"
    "batch1-kv16:16"
    "batch1-kv127:127"
    "batch1-kv128:128"
    "batch1-kv129:129"
    "batch1-kv512:512"
    "batch1-kv2048:2048"
    "batch2-mixed:16,128"
    "batch8-mixed:15,31,63,127,255,511,1023,2047"
    "batch20-mixed:1,16,23,31,63,127,128,129,255,256,511,512,513,767,1023,1024,1535,1792,2047,2048"
)
for spec in "${cases[@]}"; do
    IFS=: read -r name lengths <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != "${name}" ]]; then
        continue
    fi
    executed=$((executed + 1))
    log="${build_dir}/${name}.log"
    echo "[ RUN      ] ${name} lengths=${lengths}"
    "${build_dir}/xlite_asr_paged_attention_probe_runner" \
        "${lengths}" "${warmup}" "${iterations}" >"${log}" 2>&1
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
    echo "Unknown ASR paged attention case: ${case_filter}" >&2
    exit 2
fi
echo "ASR paged attention probe summary: $((executed - ${#failures[@]})) passed, ${#failures[@]} failed"
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
