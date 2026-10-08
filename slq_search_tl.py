#!/usr/bin/env python3
r"""Benchmark-in-the-loop TL search (lm-eval at every step); Algorithm 2 is slq_search_dl.py --mode tl.

Usage:
    python slq_search_tl.py --model Qwen/Qwen3-8B --layer-dir <layer_dir> \
        --tasks arc_challenge_llama,gsm8k_llama,mmlu_cot_llama --min-recovery 0.99 --num-gpus 8
"""

import argparse
import json
import logging
import shutil
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, List

from slq_search_dl import (
    bisect_budget,
    compute_avg_bitwidth,
    distribution_lines,
    extract_layer_groups,
    find_database,
    get_actual_bitwidth,
    load_sensitivity,
    parse_bitwidth_map,
    solve_allocation,
    write_config,
)
from src.sensitivity.database import ErrorDatabase, ShapleySensitivityData
from src.utils.weight_store import WeightStore

sys.path.insert(0, str(Path(__file__).parent / 'eval'))
from run_lmeval import TASK_METRICS, build_gen_kwargs, collect_results, run_lmeval

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def build_checkpoint(model_id: str, config: Dict[str, int], weight_store: WeightStore, checkpoint_dir: Path):
    """Copy the HF snapshot and patch the fake-quantized weights into its safetensors files."""
    from huggingface_hub import snapshot_download
    from safetensors import safe_open
    from safetensors.torch import save_file

    snapshot_path = Path(model_id) if Path(model_id).is_dir() else Path(snapshot_download(model_id))
    shutil.rmtree(checkpoint_dir, ignore_errors=True)
    shutil.copytree(snapshot_path, checkpoint_dir)

    index_path = checkpoint_dir / "model.safetensors.index.json"
    if index_path.exists():
        with open(index_path) as f:
            weight_map = json.load(f)["weight_map"]
    else:
        with safe_open(str(checkpoint_dir / "model.safetensors"), framework="pt") as f:
            weight_map = {k: "model.safetensors" for k in f.keys()}

    shards: Dict[str, Dict[str, str]] = {}
    for layer_name in config:
        # VLM checkpoints (e.g. Qwen3.5) nest the decoder under model.language_model.
        candidates = [layer_name + ".weight",
                      layer_name.replace("model.layers.", "model.language_model.layers.") + ".weight"]
        hf_key = next((k for k in candidates if k in weight_map), None)
        if hf_key is None:
            raise KeyError(f"No checkpoint tensor for quantized layer {layer_name}")
        shards.setdefault(weight_map[hf_key], {})[hf_key] = layer_name

    for shard_file, keys in shards.items():
        shard_path = checkpoint_dir / shard_file
        with safe_open(str(shard_path), framework="pt") as f:
            metadata = f.metadata() or {}
            tensors = {k: f.get_tensor(k) for k in f.keys()}
        # transformers refuses safetensors files without a "format" entry.
        metadata.setdefault("format", "pt")
        for hf_key, layer_name in keys.items():
            w = weight_store.get_layer_weight(layer_name, config[layer_name])
            tensors[hf_key] = w.to(dtype=tensors[hf_key].dtype).contiguous()
        save_file(tensors, str(shard_path), metadata=metadata)
    logger.info(f"Checkpoint built at {checkpoint_dir}")


def evaluate(model_path: str, eval_dir: Path, args, tasks: List[str], seeds: List[int], gen_kwargs: str):
    """Run lm-eval for each seed; return (per-label mean over seeds, per-seed scores)."""
    per_seed = []
    for seed in seeds:
        seed_dir, success = run_lmeval(
            model_path=model_path, tasks=tasks, output_dir=eval_dir, seed=seed, gen_kwargs=gen_kwargs,
            tensor_parallel_size=args.num_gpus, gpu_memory_utilization=args.gpu_memory_utilization,
            max_model_len=args.max_model_len, limit=args.limit, enable_thinking=not args.no_thinking, max_gen_toks=args.max_gen_toks,
        )
        scores = collect_results(seed_dir) if success else {}
        if scores:
            per_seed.append({"seed": seed, "scores": scores})
        else:
            logger.error(f"lm_eval produced no scores for seed {seed}")
    labels = {label for r in per_seed for label in r["scores"]}
    avg = {}
    for label in labels:
        vals = [r["scores"][label] for r in per_seed if label in r["scores"]]
        avg[label] = sum(vals) / len(vals)
    return avg, per_seed


def compute_recovery(scores: Dict[str, float], baseline: Dict[str, float], tasks: List[str]):
    """Return (mean recovery, per-label recovery) of candidate vs. baseline scores."""
    labels = {label for task, _, label in TASK_METRICS if task in tasks}
    per_task = {label: scores[label] / baseline[label]
                for label in labels if label in scores and baseline.get(label, 0) > 0}
    if labels - per_task.keys():
        logger.warning(f"Labels excluded from recovery: {sorted(labels - per_task.keys())}")
    avg = sum(per_task.values()) / len(per_task) if per_task else 0.0
    return avg, per_task


def main():
    parser = argparse.ArgumentParser(description="Lowest average bitwidth whose lm-eval recovery meets --min-recovery")
    parser.add_argument('--model', type=str, required=True, help="HF model id or local checkpoint directory")
    parser.add_argument('--layer-dir', type=str, required=True, help="Directory with quantized layers and databases")
    parser.add_argument('--shapley-db', type=str, default=None,
                        help="Shapley database (default: the single shapley_*.json in --layer-dir)")
    parser.add_argument('--kl-weight', type=float, default=0.0)
    parser.add_argument('--ear-weight', type=float, default=1.0)
    parser.add_argument('--tasks', type=str, required=True, help="Comma-separated lm_eval tasks")
    parser.add_argument('--min-recovery', type=float, required=True, help="Required mean score ratio vs. BF16")
    parser.add_argument('--seeds', type=str, default='42,1,1234', help="Comma-separated lm_eval seeds")
    parser.add_argument('--baseline-json', type=str, default=None,
                        help="BF16 results.json from eval/run_lmeval.py (skips the baseline run)")
    parser.add_argument('--bitwidths', type=str, default='2,3,4,5,6,7,8')
    parser.add_argument('--bitwidth-map', type=str, default='', help="'4:4.156,8:8.156' (INT asym g128, Table 7)")
    parser.add_argument('--bitwidth-tolerance', type=float, default=0.1)
    parser.add_argument('--num-gpus', type=int, default=8)
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8)
    parser.add_argument('--max-model-len', type=int, default=32768)
    parser.add_argument('--max-gen-toks', type=int, default=4096)
    parser.add_argument('--limit', type=float, default=None, help="lm_eval --limit (fraction or count per task)")
    # Defaults are Qwen3's (Table 1); Qwen3.5 uses temperature 1.0, Llama-3.3 top_p 0.9 and --top-k 0.
    parser.add_argument('--temperature', type=float, default=0.6)
    parser.add_argument('--top-p', type=float, default=0.95)
    parser.add_argument('--top-k', type=int, default=20, help="<= 0 disables top-k")
    parser.add_argument('--no-thinking', action='store_true')
    parser.add_argument('--work-dir', type=str, default=None, help="Default: <layer-dir>/search_tl_work")
    parser.add_argument('--output', type=str, default=None, help="Output config path")
    args = parser.parse_args()

    layer_dir = Path(args.layer_dir)
    bitwidth_map = parse_bitwidth_map(args.bitwidth_map)
    tasks = [t.strip() for t in args.tasks.split(',')]
    eval_seeds = [int(s) for s in args.seeds.split(',')]
    gen_kwargs = build_gen_kwargs(args)

    error_db = ErrorDatabase.load(str(layer_dir / "error_db.json"))
    if not error_db.param_counts:
        raise ValueError("Parameter counts missing from error_db.json; rebuild it with slq_build_database.py")
    shapley_data = load_sensitivity(find_database(layer_dir, 'shapley', args.shapley_db), ShapleySensitivityData)
    specified_bw = {int(b) for b in args.bitwidths.split(',')}
    available_bw = sorted(specified_bw & set(error_db.get_common_bitwidths()) & set(map(int, shapley_data.bitwidths)))
    if not available_bw:
        parser.error("no --bitwidths entry is in both --layer-dir and the Shapley database")
    logger.info(f"Available bitwidths: {available_bw}")

    weight_store = WeightStore(str(layer_dir)).load()
    layer_groups = extract_layer_groups(shapley_data)

    work_base = Path(args.work_dir) if args.work_dir else layer_dir / "search_tl_work"
    work_base.mkdir(parents=True, exist_ok=True)

    if args.baseline_json:
        with open(args.baseline_json) as f:
            baseline_scores = json.load(f)['averages']
    else:
        baseline_scores, baseline_per_seed = evaluate(
            args.model, work_base / "baseline" / "eval", args, tasks, eval_seeds, gen_kwargs)
        if not baseline_scores:
            logger.error("Baseline evaluation failed for all seeds")
            return 1
        with open(work_base / "baseline_results.json", 'w') as f:
            json.dump({"seeds": eval_seeds, "per_seed": baseline_per_seed, "averages": baseline_scores}, f, indent=2)
    logger.info(f"Baseline scores: {baseline_scores}")

    history = []
    best_checkpoint = None

    def solve(budget):
        return solve_allocation(error_db, shapley_data, budget, available_bw, args.kl_weight, args.ear_weight,
                                bitwidth_map, layer_groups)

    def evaluate_config(config):
        nonlocal best_checkpoint
        step = len(history) + 1
        actual_bw = compute_avg_bitwidth(config, error_db.param_counts, bitwidth_map)
        logger.info(f"Step {step}: average bitwidth {actual_bw:.4f}, " + ", ".join(distribution_lines(config)))
        step_work = work_base / f"step{step}_bw{actual_bw:.2f}"
        checkpoint = step_work / "checkpoint"
        build_checkpoint(args.model, config, weight_store, checkpoint)
        avg_scores, per_seed_scores = evaluate(str(checkpoint), step_work / "eval", args, tasks, eval_seeds, gen_kwargs)
        avg_rec, per_task_rec = compute_recovery(avg_scores, baseline_scores, tasks)
        # A failed evaluation counts as not meeting the constraint.
        satisfies = bool(avg_scores) and avg_rec >= args.min_recovery
        logger.info(f"Step {step}: recovery {avg_rec:.4f} per task {per_task_rec}, satisfies={satisfies}")
        result = {"step": step, "actual_bw": actual_bw, "avg_recovery": avg_rec, "per_task_recovery": per_task_rec,
                  "avg_scores": avg_scores, "per_seed_scores": per_seed_scores, "satisfies": satisfies,
                  "distribution": dict(Counter(config.values()))}
        history.append(result)
        with open(step_work / "step_result.json", 'w') as f:
            json.dump(result, f, indent=2)
        # Keep only the best candidate's checkpoint on disk (each one is a full model copy).
        if satisfies:
            if best_checkpoint is not None:
                shutil.rmtree(best_checkpoint, ignore_errors=True)
            best_checkpoint = checkpoint
        else:
            shutil.rmtree(checkpoint, ignore_errors=True)
        return result

    bw_range = [get_actual_bitwidth(f(available_bw), bitwidth_map) for f in (min, max)]
    best_config, best = bisect_budget(solve, evaluate_config, lambda r: r["satisfies"], *bw_range,
                                      args.bitwidth_tolerance)

    history_path = work_base / "search_history.json"
    summary = {"args": vars(args), "baseline": baseline_scores, "history": history}
    if not best["satisfies"]:
        with open(history_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.error(f"No configuration reaches recovery {args.min_recovery}; history in {history_path}")
        return 1

    best_bitwidth, best_recovery = best["actual_bw"], best["avg_recovery"]
    logger.info(f"Best bitwidth {best_bitwidth:.4f}, recovery {best_recovery:.4f}, checkpoint {best_checkpoint}")

    output_path = args.output or str(
        layer_dir / f"config_task_{'-'.join(tasks)}_rec{args.min_recovery}_{len(eval_seeds)}seeds"
                    f"_bw{best_bitwidth:.2f}_bws{'-'.join(map(str, available_bw))}.txt"
    )
    scores = "".join(f"\n#   {label}: {score:.2f} ({baseline_scores.get(label, float('nan')):.2f})"
                     for label, score in sorted(best["avg_scores"].items()))
    write_config(output_path, {
        'Model': args.model, 'Tasks': ','.join(tasks),
        'Min recovery': args.min_recovery, 'Eval seeds': eval_seeds,
        'Weights': f"kl={args.kl_weight}, ear={args.ear_weight}", 'Available bitwidths': available_bw,
        'Average bitwidth': f"{best_bitwidth:.4f}", 'Recovery': f"{best_recovery:.4f}",
        'Binary search steps': len(history), 'Scores (baseline)': scores,
    }, best_config)

    summary["best"] = {"bitwidth": best_bitwidth, "recovery": best_recovery, "scores": best["avg_scores"],
                       "config_path": output_path, "checkpoint": str(best_checkpoint)}
    with open(history_path, 'w') as f:
        json.dump(summary, f, indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
