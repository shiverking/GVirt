#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 2 ]]; then
    echo "usage: $0 CHECKPOINT REPORT_DIR" >&2
    exit 2
fi

checkpoint=$1
report_dir=$2
mkdir -p "${report_dir}"

export FORWARD_BACKEND=xlite
export XLITE_DP_SIZE=1
export XLITE_TP_SIZE=1
export XLITE_WEIGHT_NZ=0

prompts=(
    "Please transcribe the speech accurately."
    "Recognize this Chinese speech segment."
    "Return only the spoken words without explanation."
    "Transcribe names, numbers, and punctuation."
    "Continue decoding from the supplied audio embeddings."
)

for index in "${!prompts[@]}"; do
    python tests/poc_310p/run_qwen3_asr_llm.py \
        --checkpoint "${checkpoint}" \
        --prompt "${prompts[$index]}" \
        --decode-tokens 16 \
        --stability-iters 50 \
        --report "${report_dir}/tokens_${index}.json"
done

python tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --input-mode synthetic \
    --prompt-tokens 129 \
    --decode-tokens 16 \
    --stability-iters 50 \
    --report "${report_dir}/synthetic_129.json"
