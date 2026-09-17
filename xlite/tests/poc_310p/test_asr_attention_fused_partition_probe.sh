#!/usr/bin/env bash
set -uo pipefail
script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_attention_partition_chain_probe"
build_dir=${XLITE_ASR_ATTN_FUSED_BUILD_DIR:-/tmp/xlite_asr_attention_fused_partition_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_ATTN_FUSED_WARMUP:-1}
iterations=${XLITE_ASR_ATTN_FUSED_ITERATIONS:-3}

cmake -S "${source_dir}" -B "${build_dir}" -DCMAKE_BUILD_TYPE=Release \
    -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?
export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"

cases=(
    "kv1:1"
    "kv16:16"
    "kv127:127"
    "kv128:128"
    "kv129:129"
    "kv512:512"
    "kv1024:1024"
    "kv2048:2048"
    "batch2-mixed:1,2048"
    "batch8-mixed:1,16,127,128,129,512,1024,2048"
    "batch20-mixed:1,16,31,63,127,128,129,255,511,512,513,767,1023,1024,1025,1281,1537,1793,1921,2048"
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
    "${build_dir}/xlite_asr_attention_partition_chain_runner" \
        "${lengths}" "${warmup}" "${iterations}" fused >"${log}" 2>&1
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
    echo "Unknown fused-partition case: ${case_filter}" >&2
    exit 2
fi
echo "ASR attention fused partition summary: $((executed-${#failures[@]})) passed, ${#failures[@]} failed"
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
