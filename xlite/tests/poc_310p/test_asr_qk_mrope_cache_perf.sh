#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_vector_probe"
build_dir=${XLITE_ASR_QK_MROPE_BUILD_DIR:-/tmp/xlite_asr_qk_mrope_perf_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
variant_filter=${3:-compare}
repeats=${4:-3}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_QK_MROPE_WARMUP:-3}
iterations=${XLITE_ASR_QK_MROPE_ITERATIONS:-20}

if [[ ! ${repeats} =~ ^[1-9][0-9]*$ ]]; then
    echo "repeats must be a positive integer" >&2
    exit 2
fi
case "${variant_filter}" in
    baseline|grouped) variants=("${variant_filter}");;
    compare) variants=(baseline grouped);;
    *) echo "variant must be baseline, grouped or compare" >&2; exit 2;;
esac

echo "[ ASR QK MROPE CACHE PERF ] source=${source_dir}"
echo "[ ASR QK MROPE CACHE PERF ] build=${build_dir}"
echo "[ ASR QK MROPE CACHE PERF ] cann=${cann_path}"
echo "[ ASR QK MROPE CACHE PERF ] device mapping is externally owned and unchanged"

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
cases=(1 2 4 6 8 12 16 20)
if [[ ${case_filter} == quick ]]; then
    cases=(1 8 20)
elif [[ ${case_filter} != all ]]; then
    cases=("${case_filter#m}")
fi

failures=()
executed=0
for tokens in "${cases[@]}"; do
    if [[ ! ${tokens} =~ ^([1-9]|1[0-9]|20)$ ]]; then
        echo "invalid token count: ${tokens}" >&2
        exit 2
    fi
    for ((run=1; run<=repeats; run++)); do
        run_variants=("${variants[@]}")
        if [[ ${variant_filter} == compare && $((run % 2)) == 0 ]]; then
            run_variants=(grouped baseline)
        fi
        for variant in "${run_variants[@]}"; do
            executed=$((executed + 1))
            log="${build_dir}/m${tokens}-${variant}-r${run}.log"
            echo "[ RUN      ] qk-mrope-cache-m${tokens} variant=${variant} run=${run}"
            "${build_dir}/xlite_asr_vector_probe_runner" qk-mrope-cache \
                "${tokens}" "${warmup}" "${iterations}" "${variant}" \
                >"${log}" 2>&1
            status=$?
            if [[ ${status} -eq 0 ]]; then
                echo "[       OK ] qk-mrope-cache-m${tokens}"
                cat "${log}"
            else
                failures+=("m${tokens}:${status}:${log}")
                echo "[  FAILED  ] qk-mrope-cache-m${tokens} (recorded; continuing)"
            fi
        done
    done
done

echo
echo "ASR QK/MRoPE/Cache summary: $((executed - ${#failures[@]})) passed, ${#failures[@]} failed"
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

if [[ ${variant_filter} == compare ]]; then
    gate_args=()
    if [[ ${case_filter} == all ]]; then
        gate_args+=(--require-gate)
    fi
    python3 "${script_dir}/summarize_asr_qk_mrope_cache.py" \
        "${build_dir}" "${case_filter}" "${repeats}" "${gate_args[@]}" || exit $?
fi
