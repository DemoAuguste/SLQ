#!/usr/bin/env bash
#
# Task-Lossless search via single-point calibration (Algorithm 2, App. A.3) with the 2x
# out-of-regime guardrail. The anchor recovery is computed from the eval/run_lmeval.py
# results.json of the BF16 model and of the uniform cal_bitwidth model built from layer_dir
# (see README). Extra arguments are forwarded to slq_search_dl.py.
#
# Usage:
#   scripts/run_search_tl_calibrated.sh <hf_model_id> <layer_dir> <bf16_results.json> <anchor_results.json> [min_recovery] [cal_bitwidth] [extra args]
#
# Example:
#   scripts/run_search_tl_calibrated.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit \
#       results/Qwen3-8B-bf16/results.json results/Qwen3-8B-uniform4/results.json 0.99 4
#
set -euo pipefail

USAGE="usage: $0 <model_id> <layer_dir> <bf16_results.json> <anchor_results.json> [min_recovery] [cal_bitwidth] [extra args]"
MODEL="${1:?${USAGE}}"
LAYER_DIR="${2:?${USAGE}}"
BF16_RESULTS="${3:?${USAGE}}"
ANCHOR_RESULTS="${4:?${USAGE}}"
RECOVERY="${5:-0.99}"
CAL_BITWIDTH="${6:-4}"
shift $(( $# < 6 ? $# : 6 ))

python slq_search_dl.py \
    --model                  "${MODEL}" \
    --layer-dir              "${LAYER_DIR}" \
    --sensitivity-method     shapley \
    --mode                   tl \
    --calibration-bitwidth   "${CAL_BITWIDTH}" \
    --calibration-results    "${BF16_RESULTS}" "${ANCHOR_RESULTS}" \
    --min-recovery           "${RECOVERY}" \
    --calibration-data       fineweb_edu \
    --calibration-samples    512 \
    --calibration-seq-length 2048 \
    --top-k                  10 \
    "$@"
