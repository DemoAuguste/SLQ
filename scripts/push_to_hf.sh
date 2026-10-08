#!/usr/bin/env bash
#
# Upload a checkpoint built by slq_build_checkpoint.py to the HuggingFace Hub.
# Requires the `hf` CLI (huggingface-hub >= 1.0) and `hf auth login` or HF_TOKEN.
#
# Usage:
#   scripts/push_to_hf.sh <checkpoint_dir> <hf_repo_id> [--private]
#
# Example:
#   scripts/push_to_hf.sh /path/to/checkpoints/Qwen3-8B-SLQ-DL <your-org>/Qwen3-8B-SLQ-DL
#
set -euo pipefail

CKPT="${1:?usage: $0 <checkpoint_dir> <hf_repo_id> [--private]}"
REPO="${2:?usage: $0 <checkpoint_dir> <hf_repo_id> [--private]}"
PRIVATE_FLAG=""
if [ "${3:-}" = "--private" ]; then
    PRIVATE_FLAG="--private"
fi

if [ ! -f "${CKPT}/config.json" ]; then
    echo "error: ${CKPT} has no config.json; not a HF checkpoint" >&2
    exit 1
fi

echo "[push_to_hf] repo: ${REPO}"
echo "[push_to_hf] dir:  ${CKPT}"

hf repos create "${REPO}" --repo-type model ${PRIVATE_FLAG} --exist-ok

# Skip caches and dotfiles.
hf upload "${REPO}" "${CKPT}" . --repo-type model \
    --exclude '*.pyc' --exclude '__pycache__/*' --exclude '.*'

echo "[push_to_hf] done: https://huggingface.co/${REPO}"
