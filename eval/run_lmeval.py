#!/usr/bin/env python3
r"""Zero-shot lm-eval benchmarks (paper Tables 1, 4) via vLLM, non-greedy over several seeds.

Usage (Qwen3 sampling, the defaults):
    python eval/run_lmeval.py --model Qwen/Qwen3-8B --output-dir results/Qwen3-8B \
        --temperature 0.6 --top-p 0.95 --top-k 20 --seeds 42,1,1234
"""

import argparse
import json
import logging
import os
import re
import shutil
import statistics
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)


# (lm_eval task, metric key in results_*.json, label). ifeval yields two Table 1 columns.
TASK_METRICS: List[Tuple[str, str, str]] = [
    ("arc_challenge_llama", "exact_match,strict_match",     "ARC Chal."),
    ("gsm8k_llama",         "exact_match,strict_match",     "GSM8K"),
    ("gsm8k_cot_llama",     "exact_match,strict-match",     "GSM8K (CoT)"),
    ("ifeval",              "prompt_level_strict_acc,none", "IFEval Prompt"),
    ("ifeval",              "inst_level_strict_acc,none",   "IFEval Inst."),
    ("mmlu_cot_llama",      "exact_match,strict_match",     "MMLU CoT"),
    ("mmlu_pro_llama",      "exact_match,strict_match",     "MMLU Pro"),
    # Table 4 (App. C.2): MMLU-Pro answer-only, not part of the default suite.
    ("mmlu_pro",            "exact_match,custom-extract",   "MMLU Pro (0-shot)"),
]

DEFAULT_TASKS = "arc_challenge_llama,gsm8k_llama,gsm8k_cot_llama,ifeval,mmlu_cot_llama,mmlu_pro_llama"


def run_lmeval(
    model_path: str,
    tasks: List[str],
    output_dir: Path,
    seed: int,
    gen_kwargs: str,
    tensor_parallel_size: int,
    gpu_memory_utilization: float,
    max_model_len: int,
    max_num_seqs: Optional[int] = None,
    batch_size: str = "auto",
    lmeval_bin: str = "lm_eval",
    limit: Optional[float] = None,
    enable_thinking: bool = True,
    max_gen_toks: int = 4096,
) -> Tuple[Path, bool]:
    """Run lm_eval with the vllm backend for one seed; returns (seed_dir, success)."""
    seed_dir = output_dir / f"seed{seed}"
    # Remove stale results so old scores are never reported.
    shutil.rmtree(seed_dir, ignore_errors=True)
    seed_dir.mkdir(parents=True)

    model_args = [
        f"pretrained={model_path}",
        "dtype=auto",
        "add_bos_token=False",
        f"tensor_parallel_size={tensor_parallel_size}",
        f"gpu_memory_utilization={gpu_memory_utilization}",
        f"max_model_len={max_model_len}",
        f"max_gen_toks={max_gen_toks}",
        "enforce_eager=True",
        "mm_encoder_tp_mode=data",
        "disable_custom_all_reduce=True",
    ]
    if max_num_seqs is not None:
        model_args.append(f"max_num_seqs={max_num_seqs}")
    if not enable_thinking:
        model_args.append("enable_thinking=False")

    cmd = [
        lmeval_bin, "--model", "vllm", "--model_args", ",".join(model_args),
        "--tasks", ",".join(tasks), "--fewshot_as_multiturn", "--apply_chat_template",
        "--batch_size", batch_size, "--log_samples", "--write_out",
        "--output_path", str(seed_dir), "--seed", str(seed),
        "--gen_kwargs", f"{gen_kwargs},seed={seed}", "--show_config",
    ]
    if limit is not None:
        cmd += ["--limit", str(limit)]
    log.info(f"[seed {seed}] {' '.join(cmd)}")

    env = os.environ.copy()
    # Respect an externally set CUDA_VISIBLE_DEVICES so several runs can share a node.
    env.setdefault("CUDA_VISIBLE_DEVICES", ",".join(map(str, range(tensor_parallel_size))))
    # fork does not work with CUDA in vLLM worker processes.
    env["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    # torch.compile triggers CUBLAS errors on the Qwen3.5 hybrid architecture.
    env["VLLM_TORCH_COMPILE_LEVEL"] = "0"
    # NCCL P2P on PCIe-only nodes can cause CUBLAS_STATUS_INVALID_VALUE under TP.
    env.setdefault("NCCL_P2P_DISABLE", "1")
    env["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, env=env, bufsize=1)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    proc.wait()

    # Written after the run: lm_eval may remove the output dir on failure.
    seed_dir.mkdir(parents=True, exist_ok=True)
    log_file = seed_dir / "log.txt"
    log_file.write_text("".join(lines))
    if proc.returncode != 0:
        log.warning(f"lm_eval exited with code {proc.returncode} (seed={seed}), log: {log_file}")
        return seed_dir, False
    return seed_dir, True


def _samples(seed_dir: Path, task: str):
    """Yield (doc, target, response with <think> blocks stripped) from a task's sample JSONL files."""
    for sf in sorted(seed_dir.rglob(f"samples_{task}_*.jsonl")):
        with open(sf) as f:
            for line in f:
                d = json.loads(line)
                resp = re.sub(r"<think>.*?</think>\s*", "", d["resps"][0][0], flags=re.DOTALL)
                yield d["doc"], str(d.get("target", "")).strip(), resp


# Leading answer letter, optionally wrapped in markdown bold or brackets: C, C., **C**, **C. text**, [C].
_ARC_LETTER_RE = re.compile(r"(?:\*\*|\[)?([A-D])(?:\*\*|\]|[.\s,;:]|$)")


def _rescore(seed_dir: Path) -> Dict[str, float]:
    """Re-score ARC and IFEval from samples; lm_eval's scorers fail on thinking-model output."""
    scores = {}
    # strict_match misses bold or bracketed letters.
    arc = [bool((m := _ARC_LETTER_RE.match(r.strip())) and m.group(1) == t)
           for _, t, r in _samples(seed_dir, "arc_challenge_llama") if t]
    if arc:
        scores["ARC Chal."] = 100 * sum(arc) / len(arc)

    # IFEval constraints are checked on the full response, so thinking text breaks length/format rules.
    ifeval = list(_samples(seed_dir, "ifeval"))
    if ifeval:
        from lm_eval.tasks.ifeval.utils import (
            InputExample, agg_inst_level_acc, test_instruction_following_strict)
        outs = [test_instruction_following_strict(
                    InputExample(key=doc["key"], instruction_id_list=doc["instruction_id_list"],
                                 prompt=doc["prompt"], kwargs=doc["kwargs"]), resp)
                for doc, _, resp in ifeval]
        scores["IFEval Prompt"] = 100 * sum(o.follow_all_instructions for o in outs) / len(outs)
        scores["IFEval Inst."] = 100 * agg_inst_level_acc([o.follow_instruction_list for o in outs])
    return scores


def collect_results(seed_dir: Path) -> Dict[str, float]:
    """Return {label: score in %} from lm_eval results, with ARC and IFEval re-scored from samples."""
    result_files = sorted(seed_dir.rglob("results_*.json"))
    if not result_files:
        log.warning(f"No results_*.json found in {seed_dir}")
        return {}
    task_data = json.loads(result_files[-1].read_text()).get("results", {})
    scores = {label: 100 * float(task_data[task][metric]) for task, metric, label in TASK_METRICS
              if task_data.get(task, {}).get(metric) is not None}
    rescored = _rescore(seed_dir)
    for label, new in rescored.items():
        log.info(f"{label} re-scored: {scores.get(label, float('nan')):.2f}% -> {new:.2f}%")
    scores.update(rescored)
    return scores


def print_results(all_seed_scores: List[Dict[str, float]], model_name: str, gen_kwargs: str) -> None:
    """Print mean/std/min/max per benchmark across seeds."""
    print(f"\nResults: {model_name}\n  {gen_kwargs}\n  {len(all_seed_scores)} seed(s)\n")
    print(f"  {'Benchmark':<20}{'Mean':>8}{'Std':>8}{'Min':>8}{'Max':>8}")
    means = []
    for label in dict.fromkeys(label for _, _, label in TASK_METRICS):
        vals = [s[label] for s in all_seed_scores if label in s]
        if vals:
            means.append(statistics.fmean(vals))
            print(f"  {label:<20}{means[-1]:>8.2f}{statistics.pstdev(vals):>8.2f}"
                  f"{min(vals):>8.2f}{max(vals):>8.2f}")
    if means:
        print(f"  {'Average':<20}{statistics.fmean(means):>8.2f}\n")


def build_gen_kwargs(args) -> str:
    """lm_eval --gen_kwargs from --temperature/--top-p/--top-k (top_k <= 0 disables it)."""
    if args.temperature == 0:
        return "temperature=0,do_sample=False"
    kw = f"temperature={args.temperature},do_sample=True,top_p={args.top_p}"
    return kw + (f",top_k={args.top_k}" if args.top_k > 0 else "")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--model", required=True, help="HF model ID or local checkpoint, evaluated as-is")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--tasks", default=DEFAULT_TASKS, help="comma-separated lm_eval tasks")
    parser.add_argument("--seeds", default="42,1,1234")
    # Defaults are the Qwen3 Table 1 setting; Qwen3.5: --temperature 1.0, Llama-3.3: --top-p 0.9 --top-k 0.
    parser.add_argument("--temperature", type=float, default=0.6, help="0 = greedy")
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--top-k", type=int, default=20, help="<= 0 disables top-k")
    parser.add_argument("--no-thinking", action="store_true",
                        help="pass enable_thinking=False to the chat template")
    parser.add_argument("--num-gpus", type=int, default=None,
                        help="tensor-parallel size (default: all visible GPUs)")
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.8)
    parser.add_argument("--max-num-seqs", type=int, default=None)
    parser.add_argument("--max-model-len", type=int, default=32768)
    parser.add_argument("--max-gen-toks", type=int, default=4096)
    parser.add_argument("--batch-size", default="auto")
    parser.add_argument("--limit", type=float, default=None,
                        help="samples per task (count, or fraction if < 1)")
    parser.add_argument("--lmeval-bin", default="lm_eval")
    args = parser.parse_args()

    if args.num_gpus is None:
        import torch
        args.num_gpus = max(torch.cuda.device_count(), 1)

    gen_kwargs = build_gen_kwargs(args)

    output_dir = Path(args.output_dir).resolve()
    seeds = [int(s) for s in args.seeds.split(",")]
    tasks = [t.strip() for t in args.tasks.split(",")]
    log.info(f"Model: {args.model}, tasks: {tasks}, seeds: {seeds}, gen kwargs: {gen_kwargs}")

    all_seed_scores = []
    for seed in seeds:
        seed_dir, ok = run_lmeval(
            model_path=args.model, tasks=tasks, output_dir=output_dir / "lmeval", seed=seed,
            gen_kwargs=gen_kwargs, tensor_parallel_size=args.num_gpus,
            gpu_memory_utilization=args.gpu_memory_utilization, max_model_len=args.max_model_len,
            max_num_seqs=args.max_num_seqs, batch_size=args.batch_size, lmeval_bin=args.lmeval_bin,
            limit=args.limit, enable_thinking=not args.no_thinking, max_gen_toks=args.max_gen_toks,
        )
        scores = collect_results(seed_dir) if ok else {}
        if not scores:
            log.error(f"Seed {seed} failed, aborting remaining seeds")
            break
        all_seed_scores.append(scores)
        log.info(f"Seed {seed} scores: {scores}")
    if not all_seed_scores:
        sys.exit(1)

    print_results(all_seed_scores, args.model, gen_kwargs)
    labels = dict.fromkeys(label for s in all_seed_scores for label in s)
    out = {
        "seeds": seeds[:len(all_seed_scores)],
        "per_seed": [{"seed": s, "scores": sc} for s, sc in zip(seeds, all_seed_scores)],
        "averages": {label: statistics.fmean(s[label] for s in all_seed_scores if label in s)
                     for label in labels},
    }
    (output_dir / "results.json").write_text(json.dumps(out, indent=2))
    log.info(f"Results saved to {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
