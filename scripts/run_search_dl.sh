#!/usr/bin/env bash
#
# Distribution-Lossless search: lowest average bitwidth whose measured EAR >= target (Sec. 3.3).
# Each binary-search step measures EAR with one forward pass; add --use-predicted-constraints
# for the purely predicted variant. All bitwidths in layer_dir are searched unless --bitwidths
# is given (e.g. --bitwidths 4,8). Extra arguments are forwarded to slq_search_dl.py.
#
# Usage:
#   scripts/run_search_dl.sh <hf_model_id> <layer_dir> [ear_target] [extra args]
#
# Example:
#   scripts/run_search_dl.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit 0.99
#
set -euo pipefail

MODEL="${1:?usage: $0 <model_id> <layer_dir> [ear_target] [extra args]}"
LAYER_DIR="${2:?usage: $0 <model_id> <layer_dir> [ear_target] [extra args]}"
EAR_TARGET="${3:-0.99}"
shift $(( $# < 3 ? $# : 3 ))

python slq_search_dl.py \
    --model                  "${MODEL}" \
    --layer-dir              "${LAYER_DIR}" \
    --sensitivity-method     shapley \
    --min-ear                "${EAR_TARGET}" \
    --calibration-data       fineweb_edu \
    --calibration-samples    512 \
    --calibration-seq-length 2048 \
    --top-k                  10 \
    --bitwidth-tolerance     0.05 \
    "$@"
