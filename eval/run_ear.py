#!/usr/bin/env python3
r"""EAR and top-K KL of per-layer bitwidth configs on the search calibration set.

Usage:
  python eval/run_ear.py --model Qwen/Qwen3-8B --layer-dir <layer_dir> \
    --configs cfg1.txt cfg2.txt --uniform-bitwidths 4,8 \
    --bitwidth-map 2:2.156,3:3.156,4:4.156,5:5.156,6:6.156,7:7.156,8:8.156
"""

import argparse
import json
import logging
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from slq_build_checkpoint import parse_config_file
from slq_search_dl import apply_config, compute_avg_bitwidth, load_calibration, parse_bitwidth_map
from src.models.loading import load_model, get_tokenizer
from src.utils.common import fix_seed, get_layer_param_counts
from src.metrics.evaluation import compute_baseline_topk, compute_kl_and_ear
from src.utils.weight_store import WeightStore

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def main():
    parser = argparse.ArgumentParser(description="EAR/KL evaluation for pre-built configs")
    parser.add_argument('--model', required=True)
    parser.add_argument('--layer-dir', required=True)
    parser.add_argument('--configs', nargs='*', default=[])
    parser.add_argument('--uniform-bitwidths', default='', help="comma-separated uniform bitwidths to add")
    parser.add_argument('--bitwidth-map', default='')
    parser.add_argument('--calibration-data', default='fineweb_edu')
    parser.add_argument('--calibration-samples', type=int, default=512)
    parser.add_argument('--calibration-seq-length', type=int, default=2048)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--top-k', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--dtype', default='bfloat16', choices=['float16', 'bfloat16'])
    parser.add_argument('--output', default=None)
    args = parser.parse_args()

    fix_seed(args.seed)
    bitwidth_map = parse_bitwidth_map(args.bitwidth_map)
    layer_dir = Path(args.layer_dir)

    weight_store = WeightStore(str(layer_dir)).load()
    layer_names = weight_store.get_layer_names()

    configs_to_eval = [(Path(p).stem, parse_config_file(p)[0]) for p in args.configs]
    configs_to_eval += [(f"uniform_{bw}bit", {name: int(bw) for name in layer_names})
                        for bw in args.uniform_bitwidths.split(',') if bw.strip()]

    # Configs are applied one after another without restoring BF16 weights,
    # so every config must assign every layer.
    for name, config in configs_to_eval:
        missing = set(layer_names) - set(config)
        if missing:
            raise ValueError(f"Config {name} does not assign {len(missing)} layers, "
                             f"e.g. {sorted(missing)[:3]}")

    model = load_model(args.model, dtype=args.dtype)
    tokenizer = get_tokenizer(args.model)
    param_counts = get_layer_param_counts(model, layer_names)

    calibration_data, calibration_masks = load_calibration(args, tokenizer)
    baseline_topk_vals, baseline_topk_idx = compute_baseline_topk(
        model, calibration_data, args.batch_size, args.top_k
    )

    results = []
    for name, config in configs_to_eval:
        avg_bw = compute_avg_bitwidth(config, param_counts, bitwidth_map)

        apply_config(model, config, weight_store)
        kl, ear, _ = compute_kl_and_ear(
            model, baseline_topk_vals, baseline_topk_idx,
            calibration_data, args.batch_size, masks=calibration_masks
        )

        results.append({'name': name, 'avg_bw': avg_bw, 'kl': kl, 'ear': ear})

    logger.info(f"{'Config':<50} {'Bits':>6} {'KL':>12} {'EAR':>10}")
    for r in results:
        logger.info(f"{r['name']:<50} {r['avg_bw']:>6.3f} {r['kl']:>12.6f} {r['ear']:>10.6f}")

    out = args.output or str(layer_dir / "ear_results.json")
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    with open(out, 'w') as f:
        json.dump({'model': args.model, 'results': results}, f, indent=2)
    logger.info(f"Saved: {out}")


if __name__ == '__main__':
    main()
