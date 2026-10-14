#!/usr/bin/env bash
#
# One-shot parallel GPTQ at every SLQ bitwidth {2,...,8} (asymmetric INT, group size 128).
# Writes the per-(layer, bitwidth) database read by the sensitivity and search steps.
# Extra arguments are forwarded to slq_quantize.py.
#
# Usage:
#   scripts/run_gptq.sh <hf_model_id> <save_dir> [num_gpus] [extra args]
#
# Example:
#   scripts/run_gptq.sh Qwen/Qwen3-8B $SLQ_DATA 8
#
set -euo pipefail

MODEL="${1:?usage: $0 <model_id> <save_dir> [num_gpus] [extra args]}"
SAVE_DIR="${2:?usage: $0 <model_id> <save_dir> [num_gpus] [extra args]}"
NUM_GPUS="${3:-8}"
shift $(( $# < 3 ? $# : 3 ))

CALIB_DATA="${CALIB_DATA:-fineweb_edu}"
SEQ_LEN=8192
NUM_TOKENS=$((1024 * 8192))
BITS_LIST="2 3 4 5 6 7 8"
LOAD_BITS=4
GROUP_SIZE=128

# Hybrid (linear-attention) models such as Qwen3.5 additionally quantize the
# linear_attn projections. VLM checkpoints nest the decoder under model.language_model;
# override the block paths there, e.g.:
#   BLOCK_MODULES=model.language_model.layers \
#   PRE_BLOCK_MODULES=model.language_model.embed_tokens \
#   scripts/run_gptq.sh Qwen/Qwen3.5-27B $SLQ_DATA 8

torchrun --nnodes=1 --nproc-per-node="${NUM_GPUS}" --master_port 29501 slq_quantize.py \
    --model_name_or_path "${MODEL}" \
    --quantizable_modules '.*layers.*((q|k|v|o|gate|up|down)_proj|linear_attn\.(in_proj_a|in_proj_b|in_proj_qkv|in_proj_z|out_proj))$' \
    --pre_block_modules    "${PRE_BLOCK_MODULES:-model.embed_tokens}" \
    --block_modules        "${BLOCK_MODULES:-model.layers}" \
    --calibration_data     "${CALIB_DATA}" \
    --calibration_tokens   "${NUM_TOKENS}" \
    --calibration_sequence_length "${SEQ_LEN}" \
    --bitwidth_options     ${BITS_LIST} \
    --calibration_bitwidth "${LOAD_BITS}" \
    --group_size           "${GROUP_SIZE}" \
    --low_cpu_mem_usage \
    --cpu_offload_modules \
    --cpu_offload_activations \
    --dtype                bfloat16 \
    --attn_implementation  "${ATTN_IMPL:-flash_attention_2}" \
    --save_dir             "${SAVE_DIR}" \
    "$@"
