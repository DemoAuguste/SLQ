#!/usr/bin/env bash
#
# Perplexity on a comma-separated list of datasets (default wikitext2,c4,fineweb_edu).
# Extra arguments are forwarded to eval/run_ppl.py.
#
# Usage:
#   scripts/run_ppl.sh <model_path> [datasets] [extra args]
#
# Example:
#   scripts/run_ppl.sh /path/to/checkpoints/Qwen3-8B-SLQ-DL wikitext2,c4
#
set -euo pipefail

MODEL="${1:?usage: $0 <model_path> [datasets] [extra args]}"
DATASETS="${2:-wikitext2,c4,fineweb_edu}"
shift $(( $# < 2 ? $# : 2 ))

for DATASET in ${DATASETS//,/ }; do
    python eval/run_ppl.py \
        --model      "${MODEL}" \
        --dataset    "${DATASET}" \
        --num_tokens 32768 \
        --seq_length 2048 \
        "$@"
done
