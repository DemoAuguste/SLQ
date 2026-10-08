#!/usr/bin/env bash
#
# Assemble a checkpoint for vLLM serving from a per-layer bitwidth config, or a uniform
# checkpoint when the third argument is a bitwidth. Extra arguments are forwarded to
# slq_build_checkpoint.py.
#
# Usage:
#   scripts/run_build_checkpoint.sh <hf_model_id> <layer_dir> <config_txt | bits> <output_dir> [extra args]
#
# Example:
#   scripts/run_build_checkpoint.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit config.txt $SLQ_DATA/checkpoints
#   scripts/run_build_checkpoint.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit 4 $SLQ_DATA/checkpoints --quant-format fake
#
set -euo pipefail

USAGE="usage: $0 <model_id> <layer_dir> <config_txt | bits> <output_dir> [extra args]"
MODEL="${1:?${USAGE}}"
LAYER_DIR="${2:?${USAGE}}"
CONFIG="${3:?${USAGE}}"
OUT_DIR="${4:?${USAGE}}"
shift 4

if [[ "${CONFIG}" =~ ^[0-9]+$ ]]; then
    SOURCE=(--uniform "${CONFIG}")
else
    SOURCE=(--config "${CONFIG}")
fi

python slq_build_checkpoint.py \
    --model        "${MODEL}" \
    --layer-dir    "${LAYER_DIR}" \
    "${SOURCE[@]}" \
    --output-dir   "${OUT_DIR}" \
    --quant-format humming_online \
    "$@"
