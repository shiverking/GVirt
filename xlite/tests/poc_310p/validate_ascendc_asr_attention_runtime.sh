#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 || $# -gt 3 ]]; then
    echo "usage: $0 CHECKPOINT REPORT_DIR [ASCEND_CANN_PACKAGE_PATH]" >&2
    exit 2
fi

checkpoint=$1
report_dir=$2
cann_path=${3:-/usr/local/Ascend/cann-9.1.0-beta.1}

mkdir -p "${report_dir}"

echo "[ RUN      ] fused-partition-production-source"
bash tests/poc_310p/test_asr_attention_fused_partition_probe.sh \
    "${cann_path}" all \
    2>&1 | tee "${report_dir}/fused-partition.log"
echo "[       OK ] fused-partition-production-source"

echo "[ RUN      ] runtime-ascendc-decode-attention"
XLITE_TEST_FP16_ONLY=1 \
python3 tests/kernels/attention.py --ascendc-decode-only \
    2>&1 | tee "${report_dir}/runtime-attention.log"
echo "[       OK ] runtime-ascendc-decode-attention"

echo "[ RUN      ] full28-synthetic129"
set +e
python3 tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --input-mode synthetic \
    --prompt-tokens 129 \
    --decode-tokens 16 \
    --max-seq-len 512 \
    --stability-iters 1 \
    --matmul-backend ascendc_asr \
    --decode-attention-backend ascendc_asr \
    --decode-diagnostics \
    --report "${report_dir}/full28-synthetic129.json" \
    2>&1 | tee "${report_dir}/full28-synthetic129.log"
full_model_status=${PIPESTATUS[0]}
set -e
python3 tests/poc_310p/summarize_decode_diagnostics.py \
    "${report_dir}/full28-synthetic129.json" \
    | tee "${report_dir}/decode-diagnostics-summary.log"
if [[ ${full_model_status} -ne 0 ]]; then
    echo "[  FAILED  ] full28-synthetic129 (diagnostics preserved)"
    exit "${full_model_status}"
fi
echo "[       OK ] full28-synthetic129"

echo
echo "AscendC ASR Attention Runtime validation PASS"
echo "Reports: ${report_dir}"
