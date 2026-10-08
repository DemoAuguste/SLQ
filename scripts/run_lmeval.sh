#!/usr/bin/env bash
#
# Evaluate a checkpoint (SLQ build or BF16) on the Table 1 benchmarks with lm-eval-harness
# via vLLM, non-greedy over 3 seeds. The model is evaluated as given (no re-quantization).
# Sampling defaults are Qwen3's; use 1.0 0.95 20 for Qwen3.5-27B and 0.6 0.9 0 (top_k 0 = off)
# for Llama-3.3-70B-Instruct. Arguments after the sixth are forwarded to eval/run_lmeval.py.
#
# Usage:
#   scripts/run_lmeval.sh <model_path> <output_dir> [num_gpus] [temperature] [top_p] [top_k] [extra args]
#
# Example:
#   scripts/run_lmeval.sh /path/to/checkpoints/Qwen3-8B-SLQ-DL results/Qwen3-8B-SLQ-DL 1
#
set -euo pipefail

USAGE="usage: $0 <model_path> <output_dir> [num_gpus] [temperature] [top_p] [top_k] [extra args]"
MODEL="${1:?${USAGE}}"
OUT_DIR="${2:?${USAGE}}"
NUM_GPUS="${3:-1}"
TEMP="${4:-0.6}"
TOP_P="${5:-0.95}"
TOP_K="${6:-20}"
shift $(( $# < 6 ? $# : 6 ))

# The 7 Table 1 scores (ifeval contributes prompt- and inst-level). The *_llama
# variants ship with lm_eval==0.4.11 (eval/requirements_lmeval.txt).
TASKS="arc_challenge_llama,gsm8k_llama,gsm8k_cot_llama,ifeval,mmlu_cot_llama,mmlu_pro_llama"

python eval/run_lmeval.py \
    --model         "${MODEL}" \
    --tasks         "${TASKS}" \
    --seeds         42,1,1234 \
    --temperature   "${TEMP}" \
    --top-p         "${TOP_P}" \
    --top-k         "${TOP_K}" \
    --num-gpus      "${NUM_GPUS}" \
    --output-dir    "${OUT_DIR}" \
    "$@"
