#!/usr/bin/env bash
#
# Benchmark-in-the-loop Task-Lossless search: lowest average bitwidth whose mean recovery
# vs BF16 is >= min_recovery, running the full lm-eval suite at every binary-search step.
# Expensive; intended for validating Algorithm 2 (scripts/run_search_tl_calibrated.sh).
# Sampling defaults are Qwen3's (see scripts/run_lmeval.sh; top_k 0 = off).
# Extra arguments are forwarded to slq_search_tl.py.
#
# Usage:
#   scripts/run_search_tl.sh <hf_model_id> <layer_dir> [min_recovery] [num_gpus] [temperature] [top_p] [top_k] [extra args]
#
# Example:
#   scripts/run_search_tl.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit 0.99 8
#
set -euo pipefail

USAGE="usage: $0 <model_id> <layer_dir> [min_recovery] [num_gpus] [temperature] [top_p] [top_k] [extra args]"
MODEL="${1:?${USAGE}}"
LAYER_DIR="${2:?${USAGE}}"
RECOVERY="${3:-0.99}"
NUM_GPUS="${4:-8}"
TEMP="${5:-0.6}"
TOP_P="${6:-0.95}"
TOP_K="${7:-20}"
shift $(( $# < 7 ? $# : 7 ))

# The 7 Table 1 scores (ifeval contributes prompt- and inst-level). The *_llama
# variants ship with lm_eval==0.4.11 (eval/requirements_lmeval.txt).
TASKS="arc_challenge_llama,gsm8k_llama,gsm8k_cot_llama,ifeval,mmlu_cot_llama,mmlu_pro_llama"

python slq_search_tl.py \
    --model        "${MODEL}" \
    --layer-dir    "${LAYER_DIR}" \
    --tasks        "${TASKS}" \
    --min-recovery "${RECOVERY}" \
    --bitwidths    2,3,4,5,6,7,8 \
    --seeds        42,1,1234 \
    --temperature  "${TEMP}" \
    --top-p        "${TOP_P}" \
    --top-k        "${TOP_K}" \
    --num-gpus     "${NUM_GPUS}" \
    "$@"
