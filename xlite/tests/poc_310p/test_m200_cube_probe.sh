#!/usr/bin/env bash
set -uo pipefail

script_dir=$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)
source_dir="${script_dir}/m200_cube_probe"
build_dir=${XLITE_M200_PROBE_BUILD_DIR:-/tmp/xlite_m200_cube_probe_build_release}
cann_path=${ASCEND_CANN_PACKAGE_PATH:-/usr/local/Ascend/ascend-toolkit/latest}
jobs=${XLITE_BUILD_JOBS:-8}
warmup=${XLITE_M200_PROBE_WARMUP:-3}
iterations=${XLITE_M200_PROBE_ITERATIONS:-10}

echo "[ M200 CUBE PROBE ] source=${source_dir}"
echo "[ M200 CUBE PROBE ] build=${build_dir}"
echo "[ M200 CUBE PROBE ] cann=${cann_path}"

cmake -S "${source_dir}" -B "${build_dir}" \
    -DCMAKE_BUILD_TYPE=Release \
    -DRUN_MODE=npu \
    -DSOC_VERSION=Ascend310P3 \
    -DASCEND_CANN_PACKAGE_PATH="${cann_path}" || exit $?
if ! cmake --build "${build_dir}" --parallel "${jobs}"; then
    merge_rule="${build_dir}/CMakeFiles/xlite_m200_cube_probe_kernel_merge_obj.dir/build.make"
    if [[ -f "${merge_rule}" ]]; then
        echo "[ M200 CUBE PROBE ] merge_obj rule (diagnostic):"
        sed -n '65,78p' "${merge_rule}"
    fi
    exit 1
fi

export LD_LIBRARY_PATH="${build_dir}/lib:${build_dir}:${LD_LIBRARY_PATH:-}"
failures=()
cases=(
    "qkv:4096:2048"
    "o:2048:2048"
    "gate-up:12288:2048"
    "down:2048:6144"
    "lm-head:151936:2048"
)
m_values=(1 8 20)
total=$((${#cases[@]} * ${#m_values[@]}))
for spec in "${cases[@]}"; do
    IFS=: read -r projection n k <<<"${spec}"
    for m in "${m_values[@]}"; do
        name="${projection}-m${m}"
        echo "[ RUN      ] ${name} M=${m} N=${n} K=${k}"
        log="${build_dir}/${name}.log"
        "${build_dir}/xlite_m200_cube_probe_runner" \
            "${projection}" "${m}" "${n}" "${k}" "${warmup}" "${iterations}" \
            >"${log}" 2>&1
        status=$?
        if [[ ${status} -eq 0 ]]; then
            echo "[       OK ] ${name}"
            cat "${log}"
        else
            failures+=("${name}:${status}:${log}")
            echo "[  FAILED  ] ${name} (recorded; continuing)"
        fi
    done
done

echo
echo "M200 Cube probe summary: $((total - ${#failures[@]})) passed, ${#failures[@]} failed"
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
