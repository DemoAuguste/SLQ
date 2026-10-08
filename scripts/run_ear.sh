#!/usr/bin/env bash
#
# Compute EAR and top-K KL on the calibration set (512 x 2048 fineweb_edu tokens)
# for bitwidth configs applied to the BF16 model from a layer database.
# Arguments after the config files (e.g. --uniform-bitwidths 4) are forwarded to eval/run_ear.py.
#
# Usage:
#   scripts/run_ear.sh <hf_model_id> <layer_dir> [<config.txt> ...] [extra args]
#
# Example:
#   scripts/run_ear.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit config_dl.txt config_tl.txt
#
set -euo pipefail

MODEL="${1:?usage: $0 <model_id> <layer_dir> [<config.txt> ...] [extra args]}"
LAYER_DIR="${2:?usage: $0 <model_id> <layer_dir> [<config.txt> ...] [extra args]}"
shift 2

python eval/run_ear.py \
    --model                  "${MODEL}" \
    --layer-dir              "${LAYER_DIR}" \
    --calibration-data       fineweb_edu \
    --calibration-samples    512 \
    --calibration-seq-length 2048 \
    --top-k                  10 \
    --configs                "$@"
