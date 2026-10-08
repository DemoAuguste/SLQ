#!/usr/bin/env python3
"""Top-K KL and EAR of a quantized checkpoint against its base model on the calibration set."""

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loading import load_model, get_tokenizer
from src.data.loading import get_data
from src.metrics.evaluation import compute_baseline_topk, compute_kl_and_ear, compute_perplexity


def main():
    parser = argparse.ArgumentParser(description="Top-K KL and EAR between a base and a quantized model")
    parser.add_argument("--base-model", required=True)
    parser.add_argument("--quant-model", required=True)
    parser.add_argument("--dataset", default="fineweb_edu", choices=["wikitext2", "c4", "fineweb_edu"])
    parser.add_argument("--num_tokens", type=int, default=512 * 2048)
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=42, help="data sampling seed (same as the search)")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--top_k", type=int, default=10)
    parser.add_argument("--compute-ppl", action="store_true", help="also report perplexity of both models")
    parser.add_argument("--gpu", type=int, default=None, help="single GPU id (default: balanced over all GPUs)")
    args = parser.parse_args()

    device_map = {"": args.gpu} if args.gpu is not None else "balanced"
    tokenizer = get_tokenizer(args.base_model)
    data = get_data(args.dataset, args.num_tokens, args.seq_length, tokenizer, train=True, seed=args.seed)
    input_ids = torch.cat(data, dim=0)

    # The two models are loaded one after the other; only the base top-K logits are kept.
    ppl = {}
    for role, path in (("base", args.base_model), ("quant", args.quant_model)):
        model = load_model(path, dtype=args.dtype, device_map=device_map)
        if role == "base":
            topk_vals, topk_idx = compute_baseline_topk(model, input_ids, args.batch_size, args.top_k)
        else:
            mean_kl, mean_ear, n_tokens = compute_kl_and_ear(model, topk_vals, topk_idx, input_ids, args.batch_size)
        if args.compute_ppl:
            ppl[role] = compute_perplexity(model, data, args.batch_size)[:2]
        del model
        torch.cuda.empty_cache()

    print(f"\nBase model: {args.base_model}\nQuant model: {args.quant_model}")
    print(f"Dataset: {args.dataset}, tokens: {n_tokens}")
    print(f"EAR (top-{args.top_k}): {mean_ear:.6f}")
    print(f"KL (top-{args.top_k}): {mean_kl:.6f}")
    if args.compute_ppl:
        print("Perplexity (exp of mean NLL, +/- std over sequences):")
        for role, (mean, var) in ppl.items():
            print(f"  {role:<6} {mean:.4f} +/- {np.sqrt(var):.4f}")
        print(f"  Recovery: {100 * ppl['base'][0] / ppl['quant'][0]:.2f}%")


if __name__ == "__main__":
    main()
