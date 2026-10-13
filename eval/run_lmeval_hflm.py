#!/usr/bin/env python3
r"""Zero-shot benchmark evaluation on Ascend NPU via lm-eval HFLM (in-process).

Compares the base model against an SLQ per-layer bitwidth config applied to it. HFLM runs the
already-loaded NPU model directly, avoiding a vLLM server (the paper's non-greedy protocol).

Usage:
    python eval/run_lmeval_hflm.py --model /path/to/Qwen3-0.6B                      # BF16 baseline
    python eval/run_lmeval_hflm.py --model /path/to/Qwen3-0.6B --layer-dir <dir> \
        --config <config_dl_*.txt>                                                  # SLQ config
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch

from src.models.loading import get_tokenizer, load_model
from src.utils.device import device_type
from src.utils.weight_store import WeightStore


def eval_lm_eval(model, tokenizer, tasks, batch_size="8"):
    import lm_eval
    from lm_eval.models.huggingface import HFLM
    from lm_eval.tasks import TaskManager

    device = "npu" if device_type() == "npu" else ("cuda" if torch.cuda.is_available() else "cpu")
    try:
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size, device=device)
    except TypeError:
        # older lm_eval HFLM does not accept an "npu" device string; set it directly.
        hflm = HFLM(pretrained=model, tokenizer=tokenizer, batch_size=batch_size)
        hflm._device = torch.device(device)
    tm = TaskManager(include_defaults=True)
    res = lm_eval.simple_evaluate(hflm, tasks=tasks, batch_size=batch_size, task_manager=tm)["results"]
    return {t: res[t].get("acc_norm,none", res[t].get("acc,none")) for t in tasks}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--layer-dir", help="SLQ layer store (needed with --config / --uniform-bitwidth)")
    ap.add_argument("--config", help="per-layer bitwidth config txt; omit for the BF16 baseline")
    ap.add_argument("--uniform-bitwidth", type=int, help="apply a uniform bitwidth to every layer")
    ap.add_argument("--tasks", default="piqa,arc_easy,arc_challenge,boolq,hellaswag,winogrande")
    ap.add_argument("--batch-size", default="8")
    ap.add_argument("--output", help="optional json path for scores")
    args = ap.parse_args()

    model = load_model(args.model)
    tokenizer = get_tokenizer(args.model)

    if args.config:
        from slq_build_checkpoint import parse_config_file
        from slq_search_dl import apply_config
        config, _ = parse_config_file(args.config)
        if args.layer_dir is None:
            raise SystemExit("--layer-dir is required with --config")
        ws = WeightStore(args.layer_dir).load()
        apply_config(model, config, ws)
        print(f"[eval] applied config {args.config} ({len(config)} layers)")
    elif args.uniform_bitwidth is not None:
        from slq_search_dl import apply_config
        if args.layer_dir is None:
            raise SystemExit("--layer-dir is required with --uniform-bitwidth")
        ws = WeightStore(args.layer_dir).load()
        config = {n: args.uniform_bitwidth for n in ws.get_layer_names()}
        apply_config(model, config, ws)
        print(f"[eval] applied uniform-{args.uniform_bitwidth}bit ({len(config)} layers)")

    tasks = [t.strip() for t in args.tasks.split(",")]
    scores = eval_lm_eval(model, tokenizer, tasks, args.batch_size)
    avg = sum(scores.values()) / len(scores)
    print("[eval] scores:")
    for t in tasks:
        print(f"  {t}: {scores.get(t)}")
    print(f"[eval] average: {avg:.5f}")

    if args.output:
        json.dump({"scores": scores, "average": avg}, open(args.output, "w"), indent=2)
        print(f"[eval] saved {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())