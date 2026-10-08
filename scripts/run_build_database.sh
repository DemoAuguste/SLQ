#!/usr/bin/env bash
#
# Build the Shapley sensitivity database (Algorithm 1) and the linear estimator
# that slq_search_dl.py and slq_search_tl.py read. Extra arguments are forwarded.
#
# Usage:
#   scripts/run_build_database.sh <hf_model_id> <layer_dir> [extra args]
#
# Example:
#   scripts/run_build_database.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit
#
set -euo pipefail

MODEL="${1:?usage: $0 <model_id> <layer_dir> [extra args]}"
LAYER_DIR="${2:?usage: $0 <model_id> <layer_dir> [extra args]}"
shift 2

# vLLM fuses these, so they must share a bitwidth at serving time.
LAYER_GROUPS='mlp.gate_proj,mlp.up_proj self_attn.k_proj,self_attn.q_proj,self_attn.v_proj'

python slq_build_database.py \
    --model                  "${MODEL}" \
    --layer-dir              "${LAYER_DIR}" \
    --calibration-data       fineweb_edu \
    --calibration-samples    512 \
    --calibration-seq-length 2048 \
    --batch-size             4 \
    --top-k                  10 \
    --seed                   42 \
    --num-noise-samples      15 \
    --num-permutations       50 \
    --shapley-bitwidths      2,3,4,5,6,7,8 \
    --layer-groups           "${LAYER_GROUPS}" \
    "$@"
