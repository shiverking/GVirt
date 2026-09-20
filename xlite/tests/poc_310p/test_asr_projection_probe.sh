#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/asr_projection_probe"
build_dir=${XLITE_ASR_PROJECTION_BUILD_DIR:-/tmp/xlite_asr_projection_probe_release}
cann_path=${1:-${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/cann-9.1.0-beta.1}}
case_filter=${2:-all}
variant_filter=${3:-baseline}
repeats=${4:-1}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_ASR_PROJECTION_WARMUP:-3}
iterations=${XLITE_ASR_PROJECTION_ITERATIONS:-20}
if [[ ! ${repeats} =~ ^[1-9][0-9]*$ ]]; then
    echo "repeats must be a positive integer" >&2
    exit 2
fi
case "${variant_filter}" in
    baseline|cached) variants=("${variant_filter}");;
    compare) variants=(baseline cached);;
    *) echo "variant must be baseline, cached or compare" >&2; exit 2;;
esac

echo "[ ASR LOW-LEVEL PROJECTION ] source=${source_dir}"
echo "[ ASR LOW-LEVEL PROJECTION ] build=${build_dir}"
echo "[ ASR LOW-LEVEL PROJECTION ] cann=${cann_path}"
echo "[ ASR LOW-LEVEL PROJECTION ] device mapping is externally owned and unchanged"

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release -DRUN_MODE=npu -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
cmake --build "${build_dir}" --parallel "${jobs}" || exit $?

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
failures=()
executed=0
cases=(
    "qkv-m1:1:4096:2048"
    "qkv-m2:2:4096:2048"
    "qkv-m4:4:4096:2048"
    "qkv-m6:6:4096:2048"
    "qkv-m8:8:4096:2048"
    "qkv-m12:12:4096:2048"
    "qkv-m16:16:4096:2048"
    "qkv-m20:20:4096:2048"
    "o-m1:1:2048:2048"
    "o-m2:2:2048:2048"
    "o-m4:4:2048:2048"
    "o-m6:6:2048:2048"
    "o-m8:8:2048:2048"
    "o-m12:12:2048:2048"
    "o-m16:16:2048:2048"
    "o-m20:20:2048:2048"
    "gate-up-m1:1:12288:2048"
    "gate-up-m2:2:12288:2048"
    "gate-up-m4:4:12288:2048"
    "gate-up-m6:6:12288:2048"
    "gate-up-m8:8:12288:2048"
    "gate-up-m12:12:12288:2048"
    "gate-up-m16:16:12288:2048"
    "gate-up-m20:20:12288:2048"
    "down-m1:1:2048:6144"
    "down-m2:2:2048:6144"
    "down-m4:4:2048:6144"
    "down-m6:6:2048:6144"
    "down-m8:8:2048:6144"
    "down-m12:12:2048:6144"
    "down-m16:16:2048:6144"
    "down-m20:20:2048:6144"
)

if [[ ${case_filter} == quick ]]; then
    cases=(
        "qkv-m1:1:4096:2048" "qkv-m8:8:4096:2048" "qkv-m20:20:4096:2048"
        "o-m1:1:2048:2048" "o-m8:8:2048:2048" "o-m20:20:2048:2048"
        "gate-up-m1:1:12288:2048" "gate-up-m8:8:12288:2048" "gate-up-m20:20:12288:2048"
        "down-m1:1:2048:6144" "down-m8:8:2048:6144" "down-m20:20:2048:6144"
    )
fi

for spec in "${cases[@]}"; do
    IFS=: read -r name m n k <<<"${spec}"
    if [[ "${case_filter}" != "all" && "${case_filter}" != quick &&
          "${case_filter}" != "${name}" ]]; then
        continue
    fi
    for ((run=1; run<=repeats; run++)); do
        run_variants=("${variants[@]}")
        if [[ ${variant_filter} == compare && $((run % 2)) == 0 ]]; then
            run_variants=(cached baseline)
        fi
        for variant in "${run_variants[@]}"; do
            executed=$((executed + 1))
            log="${build_dir}/${name}-${variant}-r${run}.log"
            echo "[ RUN      ] ${name} variant=${variant} run=${run} M=${m} N=${n} K=${k}"
            status=1
            for attempt in 1 2 3 4; do
                "${build_dir}/xlite_asr_projection_probe_runner" \
                    "${m}" "${n}" "${k}" "${warmup}" "${iterations}" \
                    "${variant}" >"${log}" 2>&1
                status=$?
                if [[ ${status} -eq 0 ]]; then
                    break
                fi
                if grep -Fq "aclrtSetDevice failed, aclError=507033" "${log}" &&
                   [[ ${attempt} -lt 4 ]]; then
                    echo "[  RETRY  ] ${name} variant=${variant} run=${run}: "\
                         "pre-launch logical-device initialization returned 507033 "\
                         "(attempt ${attempt}/4)"
                    sleep 1
                    continue
                fi
                break
            done
            if [[ ${status} -eq 0 ]]; then
                echo "[       OK ] ${name}"
                cat "${log}"
            else
                failures+=("${name}:${status}:${log}")
                echo "[  FAILED  ] ${name} (recorded; continuing)"
            fi
        done
    done
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
if [[ ${variant_filter} == compare ]]; then
    gate_args=()
    if [[ ${case_filter} == all ]]; then
        gate_args+=(--require-gate)
    fi
    python3 "${script_dir}/summarize_asr_projection.py" \
        "${build_dir}" "${case_filter}" "${repeats}" "${gate_args[@]}" || exit $?
fi
