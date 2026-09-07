#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 4 ]]; then
    echo "usage: $0 CHECKPOINT REPORT_DIR AUDIO_EMBEDS AUDIO_POSITIONS" >&2
    exit 2
fi

checkpoint=$1
report_dir=$2
audio_embeds=$3
audio_positions=$4
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

python tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --prompt "Single layer numerical gate." \
    --num-layers 1 \
    --decode-tokens 16 \
    --stability-iters 1 \
    --report "${report_dir}/single_layer.json"

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

python tests/poc_310p/run_qwen3_asr_llm.py \
    --checkpoint "${checkpoint}" \
    --input-mode file \
    --embeds-file "${audio_embeds}" \
    --positions-file "${audio_positions}" \
    --decode-tokens 16 \
    --stability-iters 50 \
    --report "${report_dir}/real_audio_embeds.json"
