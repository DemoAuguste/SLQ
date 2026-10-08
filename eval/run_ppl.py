#!/usr/bin/env python3
"""Perplexity on wikitext2 / c4 / fineweb_edu."""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.loading import load_model, get_tokenizer
from src.data.loading import get_data
from src.metrics.evaluation import compute_perplexity


def main():
    parser = argparse.ArgumentParser(description="Evaluate perplexity of a model")
    parser.add_argument("--model", required=True, help="HF model id or local path")
    parser.add_argument("--dataset", default="wikitext2", choices=["wikitext2", "c4", "fineweb_edu"])
    parser.add_argument("--num_tokens", type=int, default=8192, help="wikitext2 uses its full test split")
    parser.add_argument("--seq_length", type=int, default=2048)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--dtype", default="bfloat16", choices=["float16", "bfloat16"])
    parser.add_argument("--gpu", type=int, default=None, help="single GPU id (default: balanced over all GPUs)")
    args = parser.parse_args()

    device_map = {"": args.gpu} if args.gpu is not None else "balanced"
    model = load_model(args.model, dtype=args.dtype, device_map=device_map)
    tokenizer = get_tokenizer(args.model)
    data = get_data(args.dataset, args.num_tokens, args.seq_length, tokenizer, train=False)
    mean_ppl, var_ppl, _ = compute_perplexity(model, data, args.batch_size)

    print(f"\nModel: {args.model}\nDataset: {args.dataset}")
    print(f"Tokens: {len(data) * args.seq_length}")
    print(f"Perplexity: {mean_ppl:.4f} +/- {np.sqrt(var_ppl):.4f}")


if __name__ == "__main__":
    main()
