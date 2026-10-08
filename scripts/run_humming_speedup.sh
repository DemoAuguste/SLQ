#!/usr/bin/env bash
#
# Layer-replacement speedup of uniform 2..8-bit Humming kernels over BF16 on one model.
# Humming runs use one GPU with --single-gpu --compile so CUDA graphs capture the forward;
# the BF16 baseline may use several GPUs (bf16_gpus) for models that do not fit on one.
# Requires humming (pip install, or a clone at $HUMMING_ROOT, default ../humming) and
# a CUDA toolkit >= 12.8 on PATH.
#
# Usage:
#   scripts/run_humming_speedup.sh <hf_model_id> <layer_dir> [batch] [seq_len] [gpu] [bf16_gpus]
#
# Example:
#   scripts/run_humming_speedup.sh Qwen/Qwen3-8B $SLQ_DATA/Qwen3-8B/4bit 1 1 0
#
set -euo pipefail

USAGE="usage: $0 <model_id> <layer_dir> [batch] [seq_len] [gpu] [bf16_gpus]"
MODEL="${1:?${USAGE}}"
LAYER_DIR="${2:?${USAGE}}"
BATCH="${3:-1}"
SEQ_LEN="${4:-1}"            # seq=1 is decode, the regime Humming targets
GPU="${5:-0}"
BF16_GPUS="${6:-${GPU}}"

REPO_ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HUMMING_ROOT="${HUMMING_ROOT:-${REPO_ROOT}/../humming}"
if [ -d "${HUMMING_ROOT}" ]; then
    export PYTHONPATH="${HUMMING_ROOT}:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
else
    export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
fi

HUMMING_GPU="${GPU%%,*}"

# No --compile for the baseline: HF multi-device dispatch hooks break it.
echo "[bf16] model=$(basename "${MODEL}") batch=${BATCH} seq=${SEQ_LEN} gpus=${BF16_GPUS}"
CUDA_VISIBLE_DEVICES="${BF16_GPUS}" python "${REPO_ROOT}/eval/run_humming_speedup.py" \
    --model        "${MODEL}" \
    --layer-dir    "${LAYER_DIR}" \
    --uniform-bits 4 \
    --batch-size   "${BATCH}" \
    --seq-length   "${SEQ_LEN}" \
    --baseline-only

for B in 2 3 4 5 6 7 8; do
    echo "[${B}-bit] gpu=${HUMMING_GPU} batch=${BATCH} seq=${SEQ_LEN}"
    CUDA_VISIBLE_DEVICES="${HUMMING_GPU}" python "${REPO_ROOT}/eval/run_humming_speedup.py" \
        --model        "${MODEL}" \
        --layer-dir    "${LAYER_DIR}" \
        --uniform-bits "${B}" \
        --batch-size   "${BATCH}" \
        --seq-length   "${SEQ_LEN}" \
        --num-warmup   10 \
        --num-iters    30 \
        --no-baseline \
        --single-gpu \
        --compile
done
