#!/usr/bin/env bash
set -uo pipefail

if [[ $# -lt 2 || $# -gt 4 ]]; then
    echo "usage: $0 CHECKPOINT REPORT_DIR [AUDIO_EMBEDS AUDIO_POSITIONS]" >&2
    exit 2
fi

checkpoint=$1
report_dir=$2
audio_embeds=${3:-}
audio_positions=${4:-}
stability_iters=${XLITE_STABILITY_ITERS:-1}
mkdir -p "${report_dir}"
failures=()

run_test() {
    local name=$1
    shift
    local log_file="${report_dir}/${name}.log"

    echo "[ RUN      ] ${name}"
    "$@" >"${log_file}" 2>&1
    local status=$?
    if [[ ${status} -eq 0 ]]; then
        echo "[       OK ] ${name}"
    else
        failures+=("${name}:${status}:${log_file}")
        echo "[  FAILED  ] ${name} (exit=${status}, recorded; continuing)"
    fi
}

export FORWARD_BACKEND=xlite
export XLITE_DP_SIZE=1
export XLITE_TP_SIZE=1
export XLITE_WEIGHT_NZ=0

run_test single-layer python3 tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --prompt "Single layer numerical gate." \
    --num-layers 1 \
    --decode-tokens 16 \
    --stability-iters "${stability_iters}" \
    --report "${report_dir}/single_layer.json"

run_test tokens python3 tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --prompt "Please transcribe the speech accurately." \
    --decode-tokens 16 \
    --stability-iters "${stability_iters}" \
    --report "${report_dir}/tokens.json"

run_test synthetic-129 python3 tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --input-mode synthetic \
    --prompt-tokens 129 \
    --decode-tokens 16 \
    --stability-iters "${stability_iters}" \
    --report "${report_dir}/synthetic_129.json"

if [[ -n "${audio_embeds}" || -n "${audio_positions}" ]]; then
    if [[ -z "${audio_embeds}" || -z "${audio_positions}" ]]; then
        echo "AUDIO_EMBEDS and AUDIO_POSITIONS must be supplied together" >&2
        exit 2
    fi
    run_test real-audio-embeds python3 tests/poc_310p/run_qwen3_asr_llm.py \
        --checkpoint "${checkpoint}" \
        --input-mode file \
        --embeds-file "${audio_embeds}" \
        --positions-file "${audio_positions}" \
        --decode-tokens 16 \
        --stability-iters "${stability_iters}" \
        --report "${report_dir}/real_audio_embeds.json"
fi

total_tests=3
if [[ -n "${audio_embeds}" ]]; then
    total_tests=4
fi
echo
echo "Acceptance summary: $((total_tests - ${#failures[@]})) passed, ${#failures[@]} failed"
if [[ ${#failures[@]} -ne 0 ]]; then
    echo
    echo "================================================================================"
    echo "AGGREGATED FAILURES"
    for failure in "${failures[@]}"; do
        IFS=: read -r name status log_file <<<"${failure}"
        echo
        echo "[${name}] exit=${status} log=${log_file}"
        echo "--------------------------------------------------------------------------------"
        cat "${log_file}"
    done
    exit 1
fi
